from celery import Celery
import cv2
import os
import redis
from ultralytics import YOLO
import json
import subprocess
import imageio_ffmpeg


celery_app = Celery(
    "sentinelvue",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# A separate, direct connection to Redis — NOT for job queuing this time,
# but for publishing progress messages, like a radio broadcast.
redis_client = redis.Redis(host="localhost", port=6379, db=0)

UPLOAD_FOLDER = "uploads"
model = YOLO("yolov8n.pt")

@celery_app.task
def run_detection_task(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # OpenCV writes to a TEMP file first — this is the browser-incompatible version
    temp_filename = f"temp_{filename}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        total_detections += len(results[0].boxes)
        frame_count += 1

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        if frame_count % 10 == 0 or frame_count == total_frames:
            redis_client.publish("progress_channel", json.dumps({
                "frame": frame_count,
                "total_frames": total_frames,
                "detections_so_far": total_detections
            }))

    cap.release()
    out.release()

    # NEW: convert the temp file to a proper browser-compatible H.264 video
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

    # Clean up the temp file — we don't need it once conversion is done
    os.remove(temp_path)

    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "annotated_video": output_filename
    }

    cap.release()
    out.release()

    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "annotated_video": output_filename
    }