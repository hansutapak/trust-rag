import hashlib
import json
import os
import time
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CREDIBILITY_RULES,
    EMBEDDING_MODEL,
)


# All vector stored
def get_vectorstore():
    embeddings = FastEmbedEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )


# json data to chromaDB
def load_json_data(max_per_file=None, batch_size=50, delay_seconds=60):
    """Load Open Data JSON files from data/processed into ChromaDB, in batches"""
    json_dir = "data/processed"
    files = {
        "trees.json": "Tree",
        "climate.json": "ClimateShelter",
        "greenpoints.json": "GreenPoint",
    }

    embeddings = FastEmbedEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    for filename, entity_type in files.items():
        filepath = os.path.join(json_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ {filename} not found, skipping.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            entities = json.load(f)

        # Apply max limits
        if max_per_file is not None:
            entities = entities[:max_per_file]

        total_loaded = 0
        for start in range(0, len(entities), batch_size):
            batch = entities[start : start + batch_size]

            documents = []
            ids = []

            for entity in batch:
                # Build a readable text summary for LLM
                name = entity.get("attributes", {}).get("name", entity.get("id"))
                attributes = entity.get("attributes", {})

                key_info = ", ".join(
                    f"{k}: {v}"
                    for k, v in list(attributes.items())
                    if v and v != "None"
                )

                page_content = f"{entity_type}: {name} | {key_info}"

                # Metadata for the TRS layer
                timestamp = entity.get("dateObserved", datetime.now().isoformat())
                source_type = entity.get("source", {}).get("sensorType", "government")
                credibility = CREDIBILITY_RULES.get(source_type, 0.5)
                location = entity.get("location", {}).get("coordinates", [])

                doc = Document(
                    page_content=page_content,
                    metadata={
                        "timestamp": timestamp,
                        "source_type": source_type,
                        "credibility": credibility,
                        "document_type": "stable"
                        if entity_type in ("ClimateShelter", "GreenPoint")
                        else "sensor",
                        "entity_type": entity_type,
                        "location": location,
                        "id": entity.get("id"),
                    },
                )
                documents.append(doc)

                # Hash-based ID to prevent duplicates
                unique_string = f"{page_content}_{entity.get('id')}"
                doc_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
                ids.append(doc_id)

            vectorstore.add_documents(documents, ids=ids)
            total_loaded += len(documents)
            print(
                f"✅ Loaded {len(documents)} {entity_type} (batch {start // batch_size + 1})"
            )

            # Sleep between batches to respect quota
            if start + batch_size < len(entities):
                print(f"Sleeping {delay_seconds}s to respect quota...")
                time.sleep(delay_seconds)

        print(f"✅ Total {entity_type} loaded: {total_loaded}")
