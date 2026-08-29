from dotenv import load_dotenv
import os
import json
from groq import Groq
from qdrant_client import QdrantClient

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


# --- The REAL function that actually runs when the tool is "called" ---
def search_similar_detections(point_id):
    results = qdrant_client.query_points(
        collection_name="anomaly_events",
        query=point_id,
        limit=6,
    )
    matches = []
    for point in results.points:
        if str(point.id) == point_id:
            continue
        matches.append({
            "score": round(point.score, 3),
            "class_name": point.payload.get("class_name"),
            "frame": point.payload.get("frame"),
            "confidence": point.payload.get("confidence"),
        })
    return matches[:5]


# --- A description of that tool, written for the MODEL to read ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_detections",
            "description": (
                "Search the vector memory for past detections that look visually "
                "similar to a given detection. Returns similarity scores and metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "point_id": {
                        "type": "string",
                        "description": "The unique ID of the detection to compare against"
                    }
                },
                "required": ["point_id"]
            }
        }
    }
]


def analyze_detection(point_id, class_name, confidence):
    system_prompt = (
        "You are a security analyst AI assisting a video surveillance system called "
        "SentinelVue. When given a new detection, use the search_similar_detections "
        "tool to check whether similar detections have occurred before. Then write a "
        "short, 2-3 sentence assessment: state whether this appears to be a recurring/"
        "known pattern or a novel event, and give a brief recommendation."
    )

    user_message = (
        f"New detection: class='{class_name}', confidence={confidence:.2f}, "
        f"id='{point_id}'. Check the vector memory for similar past detections "
        f"and give your assessment."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # --- CALL 1: let the model decide whether it needs the tool ---
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = search_similar_detections(args["point_id"])

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        # --- CALL 2: now let it write the real final answer, with real data ---
        final_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
        )
        return final_response.choices[0].message.content

    return response_message.content