"""
Sanity check: Is TRS retrieving the exact intended documents?
No LLM calls – just pure TRS math.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ingestion.static_loader import get_vectorstore
from src.retrieval.retriever import trs_retriever

QUERIES = [
    ("AQI in Eixample", "BCN-AQ-001"),
    ("Temperature in Nou Barris", "BCN-WE-001"),
    ("Temperature near green point", "BCN-WE-001"),  # Should be Rank 1
    ("Tree species near climate shelter", "0015929AR"),  # Should be Rank 1
    ("Latest citizen report air quality", "92086029624"),  # Citizen report ID
    ("Conflict AQI-999", "BCN-AQ-001"),  # TRS should drop BCN-AQ-999
]


def main():
    vectorstore = get_vectorstore()
    print("=" * 60)
    print("🔬 TRS SANITY CHECK (No LLM, pure retrieval)")
    print("=" * 60)

    for query, expected_id in QUERIES:
        print(f"\n📌 Query: {query[:40]}...")
        print(f"   Expected top ID: {expected_id}")

        # Run TRS retrieval
        docs = trs_retriever(vectorstore, query)

        # Print top 3 IDs and scores
        print("   Top 3 TRS results:")
        for i, doc in enumerate(docs[:3], 1):
            doc_id = doc.metadata.get("id", "unknown")
            # Quick estimate: we need the actual TRS score, but trs_retriever returns docs only.
            # We'll re-rank manually just to show scores.
            from src.scoring.trs import rank_chunks_by_trs

            candidates = vectorstore.similarity_search_with_score(query, k=100)
            ranked = rank_chunks_by_trs(candidates, vectorstore)
            score = 0.0
            for d, s in ranked:
                if d.metadata.get("id") == doc_id:
                    score = s
                    break
            print(f"      {i}. ID: {doc_id} | TRS: {score:.4f}")

        # Assertion check
        top_id = docs[0].metadata.get("id", "")
        if top_id == expected_id:
            print("   ✅ PASS: TRS retrieved the correct document at Rank 1.")
        else:
            print(f"   ❌ FAIL: TRS retrieved {top_id} but expected {expected_id}.")
            print(
                "   ⚠️  This means we need to adjust decay/credibility for this query."
            )


if __name__ == "__main__":
    main()
