"""
RAGAS evaluation: baseline (cosine‑similarity only) vs. TRS‑enabled pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time

import pandas as pd
from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import LLM_MODEL, RETRIEVED_K, TOP_K_CHUNKS
from src.generation.rag_chain import format_doc, prompt
from src.ingestion.static_loader import get_vectorstore
from src.retrieval.retriever import trs_retriever
from src.scoring.trs import rank_chunks_by_trs

# Reduced query set (12 queries) – covers all entity types and TRS components
QUERIES = [
    # Trees
    "What species of tree is most common in Nou Barris?",
    "Where can I find a Pinus pinea in Barcelona?",
    # Climate shelters
    "Where is the nearest climate shelter in Nou Barris?",
    "What is the full address of the climate shelter in Sant Andreu?",
    # Green points
    "Where can I recycle electronics in Sants-Montjuïc?",
    "What is the address of the nearest green point to Pg Salvat Papasseit?",
    # Live sensors (recency)
    "What is the current AQI in Eixample?",
    "What is the temperature in Nou Barris right now?",
    # Citizen reports (credibility)
    "What is the latest citizen report about air quality in Barcelona?",
    # Spatial conflict
    "Is the AQI reading from sensor BCN-AQ-999 consistent with nearby sensors?",
    # Mixed / cross‑entity
    "What tree species are planted near the climate shelter in Sant Andreu?",
    "What is the temperature near the green point at C Nicaragua, 101?",
]


def build_eval_chain(vectorstore):
    """Return a simple chain that takes a query and returns the generated answer."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, llm


def run_baseline(query, vectorstore, chain):
    """Baseline: top‑k cosine chunks, no TRS."""
    docs = vectorstore.similarity_search(query, k=TOP_K_CHUNKS)
    context = format_doc(docs)
    answer = chain.invoke({"context": context, "question": query})
    trs_scores = [0.0] * len(docs)
    return answer, docs, trs_scores


def run_trs(query, vectorstore, chain):
    """TRS‑enabled: retrieve → TRS re‑rank → top‑k."""
    docs = trs_retriever(vectorstore, query)
    context = format_doc(docs)
    answer = chain.invoke({"context": context, "question": query})
    ranked = rank_chunks_by_trs(
        vectorstore.similarity_search_with_score(query, k=RETRIEVED_K),
        vectorstore,
    )
    top_trs = {doc.metadata["id"]: trs for doc, trs in ranked}
    trs_scores = [top_trs.get(doc.metadata.get("id"), 0.0) for doc in docs]
    return answer, docs, trs_scores


def evaluate():
    """Run the full evaluation and print results."""
    vectorstore = get_vectorstore()
    chain, llm = build_eval_chain(vectorstore)

    records = []
    for query in QUERIES:
        time.sleep(2)
        # Baseline
        t0 = time.time()
        ans_base, docs_base, trs_base = run_baseline(query, vectorstore, chain)
        t_base = time.time() - t0

        # TRS
        t0 = time.time()
        ans_trs, docs_trs, trs_trs = run_trs(query, vectorstore, chain)
        t_trs = time.time() - t0

        avg_trs_base = sum(trs_base) / len(trs_base) if trs_base else 0.0
        avg_trs_trs = sum(trs_trs) / len(trs_trs) if trs_trs else 0.0

        records.append(
            {
                "question": query,
                "baseline_answer": ans_base,
                "trs_answer": ans_trs,
                "baseline_contexts": [doc.page_content for doc in docs_base],
                "trs_contexts": [doc.page_content for doc in docs_trs],
                "baseline_avg_trs": avg_trs_base,
                "trs_avg_trs": avg_trs_trs,
                "baseline_latency": t_base,
                "trs_latency": t_trs,
            }
        )

    df = pd.DataFrame(records)

    # --- RAGAS evaluation -------------------------------------------------
    baseline_data = {
        "question": df["question"].tolist(),
        "answer": df["baseline_answer"].tolist(),
        "contexts": df["baseline_contexts"].tolist(),
    }
    trs_data = {
        "question": df["question"].tolist(),
        "answer": df["trs_answer"].tolist(),
        "contexts": df["trs_contexts"].tolist(),
    }

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import context_precision, faithfulness
    except ImportError:
        print("RAGAS not installed. Skipping RAGAS metrics.")
        ragas_evaluate = None

    if ragas_evaluate:
        print("Computing RAGAS metrics (may take a few minutes) …")
        baseline_dataset = Dataset.from_dict(baseline_data)
        trs_dataset = Dataset.from_dict(trs_data)

        base_result = ragas_evaluate(
            baseline_dataset,
            metrics=[faithfulness, context_precision],
            llm=llm,
        )
        trs_result = ragas_evaluate(
            trs_dataset,
            metrics=[faithfulness, context_precision],
            llm=llm,
        )
        df["baseline_faithfulness"] = base_result["faithfulness"]
        df["baseline_context_precision"] = base_result["context_precision"]
        df["trs_faithfulness"] = trs_result["faithfulness"]
        df["trs_context_precision"] = trs_result["context_precision"]

        # Temporal Faithfulness
        df["baseline_temporal_faithfulness"] = (
            df["baseline_faithfulness"] * df["baseline_avg_trs"]
        )
        df["trs_temporal_faithfulness"] = df["trs_faithfulness"] * df["trs_avg_trs"]

    # Save raw results
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/baseline_eval_results.csv", index=False)

    # --- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (mean across {} queries)".format(len(df)))
    print("=" * 60)
    if ragas_evaluate:
        print(
            f"Baseline  Faithfulness          : {df['baseline_faithfulness'].mean():.4f}"
        )
        print(f"TRS       Faithfulness          : {df['trs_faithfulness'].mean():.4f}")
        print(
            f"Baseline  Context Precision      : {df['baseline_context_precision'].mean():.4f}"
        )
        print(
            f"TRS       Context Precision      : {df['trs_context_precision'].mean():.4f}"
        )
        print(
            f"Baseline  Temporal Faithfulness  : {df['baseline_temporal_faithfulness'].mean():.4f}"
        )
        print(
            f"TRS       Temporal Faithfulness  : {df['trs_temporal_faithfulness'].mean():.4f}"
        )
    print(f"Baseline  avg latency (s)        : {df['baseline_latency'].mean():.3f}")
    print(f"TRS       avg latency (s)        : {df['trs_latency'].mean():.3f}")
    print("\nResults saved to results/eval_results.csv")
    return df


if __name__ == "__main__":
    evaluate()
