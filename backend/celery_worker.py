from celery import Celery
import cv2
import os
import redis
import json
import subprocess
import imageio_ffmpeg
import torch
import open_clip
from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid
from ultralytics import YOLO
from agent import analyze_detection

load_dotenv()

celery_app = Celery(
    "sentinelvue",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

redis_client = redis.Redis(host="localhost", port=6379, db=0)

UPLOAD_FOLDER = "uploads"
model = YOLO("yolov8n.pt")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
clip_model.eval()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def embed_crop(crop_image):
    pil_image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
    return embedding[0].tolist()


@celery_app.task
def run_detection_task(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_filename = f"temp_{filename}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    embedded_count = 0
    embedded_points = []  # NEW: track id + class + confidence for agent analysis later

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        total_detections += len(results[0].boxes)
        frame_count += 1

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        if frame_count % 10 == 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                vector = embed_crop(crop)
                point_id = str(uuid.uuid4())

                try:
                    qdrant_client.upsert(
                        collection_name="anomaly_events",
                        points=[
                            PointStruct(
                                id=point_id,
                                vector=vector,
                                payload={
                                    "video_filename": filename,
                                    "frame": frame_count,
                                    "class_name": class_name,
                                    "confidence": confidence,
                                }
                            )
                        ]
                    )
                    embedded_count += 1
                    embedded_points.append({
                        "id": point_id,
                        "class_name": class_name,
                        "confidence": confidence,
                    })
                except Exception as e:
                    # A single network hiccup shouldn't kill the whole task —
                    # log it and keep processing the rest of the video.
                    print(f"Qdrant upsert failed for frame {frame_count}, skipping: {e}")

        if frame_count % 10 == 0 or frame_count == total_frames:
            redis_client.publish("progress_channel", json.dumps({
                "frame": frame_count,
                "total_frames": total_frames,
                "detections_so_far": total_detections
            }))

    cap.release()
    out.release()

    output_filename = f"annotated_{filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg_path, "-y",
        "-i", temp_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ], check=True)

    os.remove(temp_path)

    # --- NEW: run the agent on the 2 highest-confidence detections ---
    redis_client.publish("progress_channel", json.dumps({
        "frame": frame_count,
        "total_frames": total_frames,
        "status": "Running agent analysis..."
    }))

    agent_insights = []
    top_detections = sorted(
        embedded_points, key=lambda p: p["confidence"], reverse=True
    )[:2]

    for det in top_detections:
        try:
            assessment = analyze_detection(
                point_id=det["id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
            )
            agent_insights.append({
                "point_id": det["id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "assessment": assessment,
            })
        except Exception as e:
            print(f"Agent analysis failed for point {det['id']}: {e}")

    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "embedded_to_qdrant": embedded_count,
        "annotated_video": output_filename,
        "agent_insights": agent_insights,
    }