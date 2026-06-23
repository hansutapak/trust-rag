import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DECAY_RATES
from src.generation.rag_chain import build_rag_chain, format_doc
from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import calculate_recency_score, rank_chunks_by_trs


def debug_trs():
    vectorstore = get_vectorstore()
    query = "Casal de Barri in Nou Barris"

    # 1. Raw Chroma distances
    raw = vectorstore.similarity_search_with_score(query, k=5)
    print("RAW CHROMA DISTANCES (smaller = more similar)")
    for doc, dist in raw:
        print(
            f"  {dist:.4f} | {doc.metadata.get('source_type')}: {doc.page_content[:60]}..."
        )

    # 2. TRS ranking with accurate component display
    ranked = rank_chunks_by_trs(raw)
    print("\nTRS RANKING (similarity × recency × credibility = TRS)")
    for doc, trs in ranked:
        timestamp = doc.metadata.get("timestamp")
        credibility = doc.metadata.get("credibility", 0.5)
        doc_type = doc.metadata.get("document_type", "sensor")
        decay_rate = DECAY_RATES.get(doc_type, DECAY_RATES["sensor"])

        recency = calculate_recency_score(timestamp, decay_rate)
        # similarity derived from TRS, recency, credibility
        similarity = trs / (recency * credibility) if recency * credibility > 0 else 0
        print(
            f"  TRS={trs:.4f}  similarity≈{similarity:.4f}  recency={recency:.4f}  credibility={credibility}  {doc.metadata.get('source_type')}"
        )

    # 3. Context sent to LLM
    top_chunks = [doc for doc, _ in ranked[:3]]
    context = format_doc(top_chunks)
    print("\nCONTEXT SENT TO LLM:\n" + context)

    # 4. LLM answer
    rag_chain = build_rag_chain(vectorstore)
    answer = rag_chain.invoke(query)
    print("\nLLM ANSWER:\n" + answer)


if __name__ == "__main__":
    debug_trs()
