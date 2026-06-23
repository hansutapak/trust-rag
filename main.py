from src.generation.rag_chain import build_rag_chain
from src.ingestion.static_loader import get_vectorstore

# load documents and store in chromaDB
vectorstore = get_vectorstore()

# build_rag_chain
rag_chain = build_rag_chain(vectorstore)

# run query
response = rag_chain.invoke(
    "Where is the climate shelter Casal de Barri La Cosa Nostra?"
)
print("TRS RAG Answer:")
print(response)
