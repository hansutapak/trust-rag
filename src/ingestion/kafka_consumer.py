import json
from kafka import KafkaConsumer
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
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

    embeddings = FastEmbedEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print("Consumer started. Waiting for messages...")
    for message in consumer:
        data = message.value

        # Extract ffields from NGSI-LD format
        sensor_id = data["id"]
        source_type = data["source"]["sensorType"]
        timestamp = data["dateObserved"]
        location = data["location"]["coordinates"]
        metric = data["attributes"]["metric"]
        reading = data["attributes"]["reading"]
        credibility = CREDIBILITY_RULES.get(source_type, 0.5)

        # Page content readable by LLM
        page_content = (
            f"sensor: {sensor_id} | {metric}: {reading} | "
            f"Location: {location[1]}, {location[0]}"
        )

        doc = Document(
            page_content=page_content,
            metadata={
                "timestamp": timestamp,
                "source_type": source_type,
                "credibility": credibility,
                "document_type": "sensor",
                "entity_type": "sensor",
                "location": location,
                "id": sensor_id,
            },
        )
        vectorstore.add_documents([doc])
        print(f"Stored: {doc.page_content[:80]}...")


if __name__ == "__main__":
    start_consumer()
