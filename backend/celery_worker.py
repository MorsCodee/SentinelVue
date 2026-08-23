from celery import Celery
import cv2
import os
import redis
from ultralytics import YOLO

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

    output_filename = f"annotated_{filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

        # NEW: broadcast progress every 10 frames (not every single frame —
        # that would flood the channel with way too many messages)
        if frame_count % 10 == 0 or frame_count == total_frames:
            import json
            redis_client.publish("progress_channel", json.dumps({
                "frame": frame_count,
                "total_frames": total_frames,
                "detections_so_far": total_detections
            }))

    cap.release()
    out.release()

    return {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "annotated_video": output_filename
    }