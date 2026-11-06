import sys, time, cv2
from ultralytics import YOLO

VIDEO = sys.argv[1]   # koi bhi chhoti sample video
N = 100               # kitne frames test karne hain

def load_frames():
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while len(frames) < N:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames

def bench(path, label, frames):
    model = YOLO(path, task="detect")
    for f in frames[:5]:                      # warmup
        model(f, verbose=False, device="cpu")
    t = time.perf_counter()
    for f in frames:
        model(f, verbose=False, device="cpu")
    dt = time.perf_counter() - t
    print(f"{label}: {dt/len(frames)*1000:.1f} ms/frame | {len(frames)/dt:.1f} FPS")

frames = load_frames()
YOLO("yolov8n.pt").export(format="onnx")
bench("yolov8n.pt", "PyTorch (CPU)", frames)
bench("yolov8n.onnx", "ONNX (CPU)", frames)