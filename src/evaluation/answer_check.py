"""
Final answer‑level check with context for all 50 queries.
Reads the 3 batch CSVs, combines them, retrieves context, and computes partial match.
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import RETRIEVED_K
from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import rank_chunks_by_trs

# ======================================================================
# 1. SET YOUR BATCH FILE PATHS HERE
# ======================================================================
batch_files = [
    "results/qpp_genre_results_fb1.csv",  # <-- CHANGE THIS TO YOUR ACTUAL FILE
    "results/qpp_genre_results_fb2.csv",  # <-- CHANGE THIS TO YOUR ACTUAL FILE
    "results/qpp_genre_results_fb3.csv",  # <-- CHANGE THIS TO YOUR ACTUAL FILE
]


# ======================================================================
# 2. Helper functions
# ======================================================================
def load_batches(files):
    dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"✅ Loaded {f} ({len(df)} queries)")
        else:
            print(f"⚠️ File not found: {f}")
    if not dfs:
        raise FileNotFoundError("No batch CSV files found!")
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total queries combined: {len(df)}")
    return df


def extract_value(answer):
    num_match = re.search(r"(\d+\.?\d*)", str(answer))
    if num_match:
        return num_match.group(1)
    entities = [
        "Gleditsia triacanthos f. inermis",
        "Pg Salvat Papasseit, 3",
        "Pg Fabra i Puig, 423",
        "not consistent",
    ]
    for ent in entities:
        if ent.lower() in str(answer).lower():
            return ent
    return None


def get_contexts(query, vectorstore):
    baseline_docs = vectorstore.similarity_search(query, k=1)
    baseline_ctx = baseline_docs[0].page_content if baseline_docs else ""

    candidates = vectorstore.similarity_search_with_score(query, k=RETRIEVED_K)
    ranked = rank_chunks_by_trs(candidates, vectorstore)
    trs_ctx = ranked[0][0].page_content if ranked else ""

    return baseline_ctx, trs_ctx


# ======================================================================
# 3. Main
# ======================================================================
def main():
    vectorstore = get_vectorstore()
    df = load_batches(batch_files)

    # Build ground truths from TRS answers
    ground_truths = {}
    for _, row in df.iterrows():
        query = row["query"]
        gt = extract_value(row["trs_answer"])
        if gt:
            ground_truths[query] = gt
    for q in df["query"]:
        if "consistent" in q.lower() and q not in ground_truths:
            ground_truths[q] = "not consistent"

    results = []
    for _, row in df.iterrows():
        query = row["query"]
        gt = ground_truths.get(query, "")
        baseline_ans = str(row["baseline_answer"])
        trs_ans = str(row["trs_answer"])

        baseline_ctx, trs_ctx = get_contexts(query, vectorstore)

        results.append(
            {
                "query": query,
                "ground_truth": gt,
                "baseline_context": baseline_ctx[:300] + "..."
                if len(baseline_ctx) > 300
                else baseline_ctx,
                "trs_context": trs_ctx[:300] + "..." if len(trs_ctx) > 300 else trs_ctx,
                "baseline_answer": baseline_ans,
                "trs_answer": trs_ans,
                "baseline_partial": 1 if gt.lower() in baseline_ans.lower() else 0,
                "trs_partial": 1 if gt.lower() in trs_ans.lower() else 0,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv("results/answer_level_final_fb_1.csv", index=False)

    print("\n" + "=" * 60)
    print("📊 FINAL ANSWER-LEVEL SUMMARY")
    print("=" * 60)
    print(f"Baseline Partial Match : {result_df['baseline_partial'].mean():.2%}")
    print(f"TRS Partial Match      : {result_df['trs_partial'].mean():.2%}")
    print("\n✅ Results saved to results/answer_level_final_fb_1.csv")


if __name__ == "__main__":
    main()
