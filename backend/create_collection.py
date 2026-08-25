from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

client.create_collection(
    collection_name="anomaly_events",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

print("Collection created!")
print(client.get_collections())