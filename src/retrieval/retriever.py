from src.config import RETRIEVED_K, TOP_K_CHUNKS
from src.scoring.trs import rank_chunks_by_trs


# TRS aware retriever func
def trs_retriever(vectorstore, query):
    results_with_scores = vectorstore.similarity_search_with_score(query, k=RETRIEVED_K)

    # if nothing recieved return nothing
    if not results_with_scores:
        return []

    # ranked
    ranked = rank_chunks_by_trs(results_with_scores)

    # keep top 3 chunks only
    top_chunks = [chunk for chunk, trs in ranked[:TOP_K_CHUNKS]]

    return top_chunks
