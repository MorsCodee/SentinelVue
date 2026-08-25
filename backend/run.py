import os
import json
import redis
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO
from celery_worker import run_detection_task, celery_app
from flask_cors import CORS
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"

# Same Redis connection pattern as celery_worker.py, but this time
# Flask is the one LISTENING to the radio channel, not broadcasting on it.
redis_client = redis.Redis(host="localhost", port=6379, db=0)


@app.route("/")
def health_check():
    return {"status": "SentinelVue backend is alive"}

@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return {"error": "No video file provided"}, 400

    video = request.files["video"]
    save_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(save_path)

    return {"status": "uploaded", "filename": video.filename}, 200

@app.route("/api/detect/<filename>", methods=["POST"])
def detect(filename):
    task = run_detection_task.delay(filename)
    return {"status": "processing", "task_id": task.id}, 202

@app.route("/api/status/<task_id>", methods=["GET"])
def check_status(task_id):
    task = celery_app.AsyncResult(task_id)
    if task.state == "PENDING":
        return {"state": task.state, "status": "Waiting to be processed..."}
    elif task.state == "SUCCESS":
        return {"state": task.state, "result": task.result}
    elif task.state == "FAILURE":
        return {"state": task.state, "error": str(task.info)}
    else:
        return {"state": task.state}

@app.route("/api/video/<filename>", methods=["GET"])
def get_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# 
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

@app.route("/api/similar/<point_id>", methods=["GET"])
def find_similar(point_id):
    results = qdrant_client.query_points(
        collection_name="anomaly_events",
        query=point_id,
        limit=6,  # 6, so we can drop the first if it's the point itself
    )

    matches = []
    for point in results.points:
        if str(point.id) == point_id:
            continue  # skip itself, we only want OTHER similar detections
        matches.append({
            "id": str(point.id),
            "score": point.score,
            "video_filename": point.payload.get("video_filename"),
            "frame": point.payload.get("frame"),
            "class_name": point.payload.get("class_name"),
            "confidence": point.payload.get("confidence"),
        })

    return {"matches": matches[:5]}, 200

@socketio.on("connect")
def handle_connect():
    print("A client connected!")
    socketio.emit("server_message", {"data": "Hello from Flask, you're connected!"})

@socketio.on("disconnect")
def handle_disconnect():
    print("A client disconnected.")

# --- NEW: the background listener ---

def listen_to_redis():
    pubsub = redis_client.pubsub()
    pubsub.subscribe("progress_channel")

    for message in pubsub.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            # Relay whatever Celery shouted straight to the frontend
            socketio.emit("progress_update", data)

if __name__ == "__main__":
    # Start the listener running alongside the server, in the background,
    # instead of blocking everything else while it waits for messages
    socketio.start_background_task(listen_to_redis)
    socketio.run(app, debug=True, port=5000)