import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import rank_chunks_by_trs


def test_trs_debug():
    """Print TRS scores for retrieved chunks, showing source and metadata."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(
        "What is the air quality on Princes Street and can I do cycling?", k=5
    )
    ranked = rank_chunks_by_trs(results)

    print("-" * 50)
    for chunk, trs in ranked:
        print(f"TRS: {trs:.4f}")
        print(f"Source: {chunk.metadata.get('source_type')}")
        print(f"Document Type: {chunk.metadata.get('document_type')}")
        print(f"Credibility: {chunk.metadata.get('credibility')}")
        print(f"Timestamp: {chunk.metadata.get('timestamp')}")
        print(f"Content: {chunk.page_content[:80]}")
        print("-" * 50)


if __name__ == "__main__":
    test_trs_debug()
