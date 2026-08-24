import { useState, useRef, useEffect } from "react";
import { io } from "socket.io-client";
import CornerBrackets from "./components/CornerBrackets";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("idle"); // idle | uploading | processing | done | error
  const [progress, setProgress] = useState({ frame: 0, total_frames: 0 });
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);
  const socketRef = useRef(null);

  // Set up the WebSocket connection ONCE when the component first loads,
  // not every time something re-renders — that's what this useEffect does.
  useEffect(() => {
    const socket = io("http://127.0.0.1:5000");
    socketRef.current = socket;

    socket.on("progress_update", (data) => {
      setProgress(data);
    });

    // Cleanup: close the connection if this component ever unmounts,
    // so we don't leave dangling connections behind.
    return () => socket.disconnect();
  }, []);

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (file) setSelectedFile(file);
  }

  async function handleUpload() {
    if (!selectedFile) return;
    setUploadStatus("uploading");

    const formData = new FormData();
    formData.append("video", selectedFile);

    try {
      const uploadRes = await fetch("http://127.0.0.1:5000/api/upload", {
        method: "POST",
        body: formData,
      });
      if (!uploadRes.ok) throw new Error("Upload failed");
      const uploadData = await uploadRes.json();

      // Upload succeeded — now immediately trigger detection
      setUploadStatus("processing");
      const detectRes = await fetch(
        `http://127.0.0.1:5000/api/detect/${uploadData.filename}`,
        { method: "POST" }
      );
      if (!detectRes.ok) throw new Error("Detection failed to start");

      // We don't need to do anything else here — progress will
      // arrive automatically via the WebSocket listener above.
      // We just need to poll the final result once, or wait for
      // completion — for now, let's poll status every 2 seconds.
      pollStatus(await detectRes.json());
    } catch (err) {
      console.error(err);
      setUploadStatus("error");
    }
  }

  function pollStatus({ task_id }) {
    const interval = setInterval(async () => {
      const res = await fetch(`http://127.0.0.1:5000/api/status/${task_id}`);
      const data = await res.json();

      if (data.state === "SUCCESS") {
        clearInterval(interval);
        setResult(data.result);
        setUploadStatus("done");
      } else if (data.state === "FAILURE") {
        clearInterval(interval);
        setUploadStatus("error");
      }
    }, 2000);
  }

  const percent =
    progress.total_frames > 0
      ? Math.round((progress.frame / progress.total_frames) * 100)
      : 0;

  return (
    <div className="min-h-screen bg-bg text-ink font-body">
      <header className="border-b border-line px-8 py-5">
        <h1 className="font-display font-bold uppercase tracking-widest text-2xl">
          Sentinel<span className="text-amber">Vue</span>
        </h1>
        <p className="font-mono text-xs text-muted mt-1">
          MULTI-CAMERA ANOMALY DETECTION SYSTEM
        </p>
      </header>

      <main className="p-8 flex flex-col items-center">
        <div className="relative w-full max-w-2xl mt-12 border-2 border-dashed border-line bg-panel p-16 flex flex-col items-center text-center">
          <CornerBrackets color="ink" />

          <p className="font-display uppercase tracking-wide text-lg mb-2">
            Feed Ingestion
          </p>

          {selectedFile ? (
            <p className="font-mono text-xs text-cyan mb-6">
              {selectedFile.name.toUpperCase()} — READY
            </p>
          ) : (
            <p className="font-mono text-xs text-muted mb-6">
              DRAG VIDEO FILE OR CLICK TO BROWSE — MP4, MOV
            </p>
          )}

          <input
            type="file"
            accept="video/mp4,video/quicktime"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
          />

          {!selectedFile && uploadStatus === "idle" && (
            <button
              onClick={() => fileInputRef.current.click()}
              className="font-mono text-xs uppercase tracking-wider border border-ink px-5 py-2 hover:bg-ink hover:text-panel transition-colors"
            >
              Select File
            </button>
          )}

          {selectedFile && uploadStatus === "idle" && (
            <button
              onClick={handleUpload}
              className="font-mono text-xs uppercase tracking-wider border border-amber bg-amber text-panel px-5 py-2 hover:opacity-90 transition-opacity"
            >
              Upload & Analyze
            </button>
          )}

          {(uploadStatus === "uploading" || uploadStatus === "processing") && (
            <div className="w-full">
              <p className="font-mono text-xs text-muted mb-2">
                {uploadStatus === "uploading"
                  ? "UPLOADING..."
                  : `PROCESSING — FRAME ${progress.frame} / ${progress.total_frames || "?"}`}
              </p>
              <div className="w-full h-1 bg-line">
                <div
                  className="h-1 bg-cyan transition-all duration-300"
                  style={{ width: `${percent}%` }}
                />
              </div>
            </div>
          )}

          {uploadStatus === "done" && result && (
            <div className="w-full">
              <p className="font-mono text-xs text-cyan mb-4">
                ANALYSIS COMPLETE — {result.total_detections} DETECTIONS ACROSS{" "}
                {result.frames_processed} FRAMES
              </p>

              <div className="relative border border-line">
                <CornerBrackets color="cyan" />
                <video
                  controls
                  className="w-full block"
                  src={`http://127.0.0.1:5000/api/video/${result.annotated_video}`}
                />
              </div>
            </div>
          )}

          {uploadStatus === "error" && (
            <p className="font-mono text-xs text-amber">✗ SOMETHING FAILED</p>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;