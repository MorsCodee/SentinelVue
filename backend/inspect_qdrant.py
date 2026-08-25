from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Grab 5 random points just to inspect their metadata
points, _ = client.scroll(
    collection_name="anomaly_events",
    limit=5,
    with_payload=True,
    with_vectors=False,  # we don't need to print 512 numbers, just metadata
)

for point in points:
    print(point.id, point.payload)