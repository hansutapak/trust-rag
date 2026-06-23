import hashlib
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL


def test_duplicate_prevention():
    print("🔍 Testing ChromaDB Duplicate Prevention via Hash IDs...\n")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    # 1. Check initial count
    initial_count = vectorstore._collection.count()
    print(f"1. Initial chunk count: {initial_count}")

    # 2. Create a dummy chunk and generate its deterministic hash ID
    content = "Test sensor reading: AQI 50 on Princes Street."
    source_type = "government_sensor"
    unique_string = f"{content}_{source_type}"
    chunk_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()

    doc = Document(
        page_content=content,
        metadata={
            "source_type": source_type,
            "document_type": "sensor",
            "credibility": 0.9,
            "timestamp": datetime.now().isoformat(),
        },
    )

    # 3. Add it to the database for the FIRST time
    vectorstore.add_documents(documents=[doc], ids=[chunk_id])
    count_after_first = vectorstore._collection.count()
    print(f"2. Count after 1st addition: {count_after_first}")

    # 4. Try to add the EXACT SAME chunk with the EXACT SAME ID again
    vectorstore.add_documents(documents=[doc], ids=[chunk_id])
    count_after_second = vectorstore._collection.count()
    print(f"3. Count after 2nd addition: {count_after_second}")

    # 5. Verify the result
    print("\n" + "=" * 50)
    if count_after_first == count_after_second:
        print("✅ SUCCESS: Duplicate prevention works!")
        print(
            "ChromaDB recognized the identical ID and UPSERTED (overwrote) instead of creating a duplicate."
        )
    else:
        print("❌ FAILURE: Duplicates were created. The hash method failed.")
    print("=" * 50)


if __name__ == "__main__":
    test_duplicate_prevention()
