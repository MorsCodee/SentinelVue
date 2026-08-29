from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from agent import analyze_detection

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Grab one real point to test with
points, _ = client.scroll(collection_name="anomaly_events", limit=1, with_payload=True)
point = points[0]

print("Analyzing:", point.id, point.payload)

result = analyze_detection(
    point_id=str(point.id),
    class_name=point.payload.get("class_name"),
    confidence=point.payload.get("confidence"),
)

print("\n--- AGENT ASSESSMENT ---")
print(result)