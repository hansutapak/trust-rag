import json
from kafka import KafkaConsumer
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.config import EMBEDDING_MODEL, CHROMA_COLLECTION, CHROMA_DIR, CREDIBILITY_RULES

TOPIC = "smart_city_sensors"


def start_consumer():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers="127.0.0.1:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print("Consumer started. Waiting for messages...")
    for message in consumer:
        data = message.value
        doc = Document(
            page_content=f"Sensor: {data['sensor_id']} | Location: {data['location']} | "
            f"Metric: {data['metric']} | Reading: {data['reading']}",
            metadata={
                "timestamp": data["timestamp"],
                "credibility": CREDIBILITY_RULES.get(data["source_type"], 0.5),
                "source_type": data["source_type"],
                "document_type": "sensor",
            },
        )
        vectorstore.add_documents([doc])
        print(f"Stored: {doc.page_content[:80]}...")


if __name__ == "__main__":
    start_consumer()
