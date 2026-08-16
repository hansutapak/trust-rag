"""
Manual answer-level check - ground truths written by hand from the CSV answers.
Context is pulled from ChromaDB just for display (no LLM calls).
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import RETRIEVED_K
from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import rank_chunks_by_trs

# 1. Load and combine all 3 batch CSVs

batch_files = [
    "results/qpp_genre_results_fb1.csv",
    "results/qpp_genre_results_fb2.csv",
    "results/qpp_genre_results_fb3.csv",
    "results/qpp_genre_results_fb4.csv",
]


def load_batches():
    dfs = []
    for f in batch_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"Loaded {f} ({len(df)} queries)")
        else:
            print(f"File not found: {f}")
    if not dfs:
        raise FileNotFoundError("No batch files found!")
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal queries combined: {len(df)}")
    return df


# 2. Manually written ground truths (from the TRS answers)

ground_truths = {
    # === AQI (all use the fresh government sensor value) ===
    "What is the current AQI in Eixample?": "75.05",
    "What is the current AQI in Nou Barris?": "75.05",
    "What is the current AQI in Sant Andreu?": "75.05",
    "What is the current AQI in Sants-Montjuic?": "75.05",
    "What is the current AQI in Ciutat Vella?": "75.05",
    "What is the current AQI in Gracia?": "75.05",
    "What is the current AQI in Sarria?": "75.05",
    "What is the current AQI in Les Corts?": "75.05",
    "What is the current AQI in Horta?": "75.05",
    "What is the current AQI near Placa Catalunya?": "517",  # from tree address
    # === TEMPERATURE ===
    "What is the temperature in Eixample?": "49.05",
    "What is the temperature in Nou Barris right now?": "54.82",
    "What is the temperature in the Gothic Quarter?": "",  # no data
    "What is the temperature in Sant Andreu?": "",  # no data
    "What is the temperature in Sants-Montjuic?": "",  # no data
    "What is the temperature in Ciutat Vella?": "",  # no data
    "What is the temperature in Gracia?": "",  # no data
    "What is the temperature in Sarria?": "54.82",
    "What is the temperature in Les Corts?": "",  # no data
    "What is the temperature near Placa Espanya?": "",  # no data
    # === TRAFFIC DENSITY ===
    "What is the traffic density in Eixample?": "69.81",
    "What is the traffic density in Nou Barris?": "52.84",
    "What is the traffic density in Gracia?": "",  # no data
    "What is the traffic density in Sant Andreu?": "",  # no data
    "What is the traffic density near Placa Espanya?": "52.84",
    # === BIN FILL ===
    "What is the bin fill level in Ciutat Vella?": "0001156",  # tree ID
    "What is the bin fill level in Gracia?": "74.55",
    "What is the bin fill level in Sarria?": "",  # no data
    "What is the bin fill level in Eixample?": "74.55",
    "What is the bin fill level in Nou Barris?": "",  # no data
    # === STATIC ===
    "What tree species are planted near the climate shelter in Sant Andreu?": "Gleditsia triacanthos f. inermis",
    "Where can I find a Pinus pinea in Barcelona?": "423",
    "What is the address of the nearest green point to Pg Salvat Papasseit?": "3",
    "What trees are near the climate shelter in Eixample?": "18",  # Pg Castanyers, 18
    "What trees are near the green point in Nou Barris?": "0001116",
    "What is the nearest climate shelter to Placa Catalunya?": "Carrer dels Jocs Florals, 59",
    "What is the address of the green point in Gracia?": "Pl Gala PlacIdia, 12",
    "What trees are in Parc de la Ciutadella?": "",
    "What is the nearest green point to Carrer de Balmes?": "",
    "What trees are near the climate shelter in Sants?": "",
    # === CONFLICT & MIXED ===
    "Which AQI reading should I trust in Eixample?": "75",
    "Is the AQI reading from sensor BCN-AQ-999 consistent with nearby sensors?": "not consistent",
    "What is the temperature near the green point at C Nicaragua, 101?": "68.54",
    "What is the latest citizen report about air quality in Barcelona?": "",
    "Which temperature reading should I trust in Nou Barris?": "45.6",
    "Is the traffic density reading from BCN-TR-001 consistent with nearby sensors?": "65.5",
    "What is the bin fill level near the green point in Ciutat Vella?": "",
    "Which AQI reading should I trust in Sant Andreu?": "75.05",
    "Is the temperature reading from BCN-WE-001 consistent with nearby sensors?": "52.22",
    "What is the latest citizen report about traffic in Barcelona?": "",
}


# 3. Normalise whitespace


def normalise(text):
    """Replace all whitespace characters with a single regular space and strip."""
    return re.sub(r"\s+", " ", str(text)).strip()


# 4. Get context for each query (just for display - no LLM calls)


def get_contexts(query, vectorstore):
    # Baseline context (top 1)
    baseline_docs = vectorstore.similarity_search(query, k=1)
    baseline_ctx = baseline_docs[0].page_content if baseline_docs else ""

    # TRS context (top 1)
    candidates = vectorstore.similarity_search_with_score(query, k=RETRIEVED_K)
    ranked = rank_chunks_by_trs(candidates, vectorstore)
    trs_ctx = ranked[0][0].page_content if ranked else ""

    return baseline_ctx, trs_ctx


# 5. Main


def main():
    vectorstore = get_vectorstore()
    df = load_batches()

    results = []
    for _, row in df.iterrows():
        query = row["query"]
        gt = ground_truths.get(query, "")
        baseline_ans = str(row["baseline_answer"])
        trs_ans = str(row["trs_answer"])

        # Get context (purely for display)
        baseline_ctx, trs_ctx = get_contexts(query, vectorstore)

        # Normalised partial match
        norm_gt = normalise(gt)
        baseline_partial = 1 if norm_gt and norm_gt in normalise(baseline_ans) else 0
        trs_partial = 1 if norm_gt and norm_gt in normalise(trs_ans) else 0

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
                "baseline_partial": baseline_partial,
                "trs_partial": trs_partial,
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv("results/answer_level_manual_final.csv", index=False)

    print("\n" + "=" * 60)
    print("MANUAL ANSWER-LEVEL SUMMARY (All Queries)")
    print("=" * 60)
    print(f"Baseline Partial Match : {result_df['baseline_partial'].mean():.2%}")
    print(f"TRS Partial Match      : {result_df['trs_partial'].mean():.2%}")
    print("\nResults saved to results/answer_level_manual_final.csv")

    # Print sample for the first query
    first = result_df.iloc[0]
    print("\nSample context for first query:")
    print(f"Query: {first['query']}")
    print(f"Baseline context: {first['baseline_context'][:100]}...")
    print(f"TRS context:      {first['trs_context'][:100]}...")


if __name__ == "__main__":
    main()
