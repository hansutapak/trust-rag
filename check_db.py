from src.ingestion.static_loader import get_vectorstore

# load documents and store in chromaDB
vectorstore = get_vectorstore()

# see the vector collection stored in ChromaDB

collection = vectorstore.get()

print(f"Total chunks stored: {len(collection['ids'])}")
print("\nAll chunks IDs:")
for id in collection["ids"]:
    print(id)

print("\nAll chuknks content:")
for doc in collection["documents"]:
    print("---")
    print(doc)

print("\nAll metadata:")
for meta in collection["metadatas"]:
    print(meta)
