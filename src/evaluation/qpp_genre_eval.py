"""
QPP-GenRE evaluation: exact implementation of Meng et al. (2025)
Computes RR@10 and nDCG@10 using LLM-generated relevance judgments.
Also generates final answers for answer-level checking.
"""

import math
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    LLM_MODEL_GROQ,
    # LLM_MODEL,
    RETRIEVED_K,
)
from src.generation.rag_chain import format_doc, prompt
from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import rank_chunks_by_trs

load_dotenv()


# 1. Configuration

N_JUDGE = 10  # depth for judging
QUERIES = [
    # === AQI SENSORS (10) ===
    "What is the current AQI in Eixample?",
    "What is the current AQI in Nou Barris?",
    "What is the current AQI in Sant Andreu?",
    "What is the current AQI in Sants-Montjuic?",
    "What is the current AQI in Ciutat Vella?",
    "What is the current AQI in Gracia?",
    "What is the current AQI in Sarria?",
    "What is the current AQI in Les Corts?",
    "What is the current AQI in Horta?",
    "What is the current AQI near Placa Catalunya?",
    # === TEMPERATURE SENSORS (10) ===
    "What is the temperature in Eixample?",
    "What is the temperature in Nou Barris right now?",
    "What is the temperature in the Gothic Quarter?",
    "What is the temperature in Sant Andreu?",
    "What is the temperature in Sants-Montjuic?",
    "What is the temperature in Ciutat Vella?",
    "What is the temperature in Gracia?",
    "What is the temperature in Sarria?",
    "What is the temperature in Les Corts?",
    "What is the temperature near Placa Espanya?",
    # === TRAFFIC & BIN FILL (10) ===
    "What is the traffic density in Eixample?",
    "What is the traffic density in Nou Barris?",
    "What is the traffic density in Gracia?",
    "What is the traffic density in Sant Andreu?",
    "What is the traffic density near Placa Espanya?",
    "What is the bin fill level in Ciutat Vella?",
    "What is the bin fill level in Gracia?",
    "What is the bin fill level in Sarria?",
    "What is the bin fill level in Eixample?",
    "What is the bin fill level in Nou Barris?",
    # === STATIC (TREES, SHELTERS, GREEN POINTS) (10) ===
    "What tree species are planted near the climate shelter in Sant Andreu?",
    "Where can I find a Pinus pinea in Barcelona?",
    "What is the address of the nearest green point to Pg Salvat Papasseit?",
    "What trees are near the climate shelter in Eixample?",
    "What trees are near the green point in Nou Barris?",
    "What is the nearest climate shelter to Placa Catalunya?",
    "What is the address of the green point in Gracia?",
    "What trees are in Parc de la Ciutadella?",
    "What is the nearest green point to Carrer de Balmes?",
    "What trees are near the climate shelter in Sants?",
    # === CONFLICT & MIXED (10) ===
    "Which AQI reading should I trust in Eixample?",
    "Is the AQI reading from sensor BCN-AQ-999 consistent with nearby sensors?",
    "What is the temperature near the green point at C Nicaragua, 101?",
    "What is the latest citizen report about air quality in Barcelona?",
    "Which temperature reading should I trust in Nou Barris?",
    "Is the traffic density reading from BCN-TR-001 consistent with nearby sensors?",
    "What is the bin fill level near the green point in Ciutat Vella?",
    "Which AQI reading should I trust in Sant Andreu?",
    "Is the temperature reading from BCN-WE-001 consistent with nearby sensors?",
    "What is the latest citizen report about traffic in Barcelona?",
]


# 2. Helper functions


def dcg_at_k(labels, k):
    """DCG@k for binary relevance labels (0/1)."""
    dcg = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel == 1:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def rr_at_10(labels):
    """Reciprocal Rank @10."""
    for i, rel in enumerate(labels[:10], start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0


def ndcg_at_10(labels):
    """nDCG@10 using QPP-GenRE approximation."""
    dcg = dcg_at_k(labels, 10)
    sorted_labels = sorted(labels, reverse=True)
    idcg = dcg_at_k(sorted_labels, 10)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def get_relevance_judgments(query, docs, llm, max_retries=3):
    labels = []
    for doc in docs:
        text = doc.page_content[:3000]
        prompt_text = f"Query: {query}\nPassage: {text}\nDoes the passage directly answer the query? Answer only 'Yes' or 'No'."

        time.sleep(4.5)
        for attempt in range(max_retries):
            try:
                response = llm.invoke([HumanMessage(content=prompt_text)])
                content = response.content
                if isinstance(content, list):
                    answer = " ".join([str(c) for c in content]).strip().lower()
                else:
                    answer = content.strip().lower()
                if "yes" in answer:
                    labels.append(1)
                else:
                    labels.append(0)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (attempt + 1) * 15
                    print(
                        f"\n[WARNING] Rate limited during judgment. Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    if attempt == max_retries - 1:
                        print(f"Error judging doc: {e}. Defaulting to 0.")
                        labels.append(0)
                    else:
                        time.sleep(2)
    return labels


def build_answer_chain():
    """Build the chain that generates the final answer using Gemini (via Groq)."""
    llm = ChatOpenAI(
        # model=LLM_MODEL,
        model=LLM_MODEL_GROQ,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_retries=5,
    )
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 3. Main evaluation


def main():
    vectorstore = get_vectorstore()
    llm = ChatOpenAI(
        # model=LLM_MODEL,
        model=LLM_MODEL_GROQ,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )
    answer_chain = build_answer_chain()

    retrieve_k = max(RETRIEVED_K, N_JUDGE)

    records = []
    for query in QUERIES:
        print(f"\nProcessing query: {query[:50]}...")

        # Baseline
        docs_base = vectorstore.similarity_search(query, k=N_JUDGE)
        labels_base = get_relevance_judgments(query, docs_base, llm)
        rr_base = rr_at_10(labels_base)
        ndcg_base = ndcg_at_10(labels_base)
        context_base = format_doc(docs_base[:1])
        time.sleep(4.5)
        ans_base = answer_chain.invoke({"context": context_base, "question": query})

        # TRS
        candidates = vectorstore.similarity_search_with_score(query, k=retrieve_k)
        ranked = rank_chunks_by_trs(candidates, vectorstore)
        docs_trs = [doc for doc, _ in ranked][:N_JUDGE]
        labels_trs = get_relevance_judgments(query, docs_trs, llm)
        rr_trs = rr_at_10(labels_trs)
        ndcg_trs = ndcg_at_10(labels_trs)
        context_trs = format_doc(docs_trs[:1])
        time.sleep(4.5)
        ans_trs = answer_chain.invoke({"context": context_trs, "question": query})

        # Debug prints
        print(f"\nDEBUG for query: {query[:30]}...")
        print(
            f"Baseline docs (first 3 IDs): {[doc.metadata.get('id', 'no_id') for doc in docs_base[:3]]}"
        )
        print(
            f"TRS docs (first 3 IDs): {[doc.metadata.get('id', 'no_id') for doc in docs_trs[:3]]}"
        )
        print(f"TRS top 5 scores: {[score for _, score in ranked[:5]]}")

        # Record
        records.append(
            {
                "query": query,
                "baseline_rr10": rr_base,
                "trs_rr10": rr_trs,
                "baseline_ndcg10": ndcg_base,
                "trs_ndcg10": ndcg_trs,
                "baseline_labels": labels_base,
                "trs_labels": labels_trs,
                "baseline_answer": ans_base,
                "trs_answer": ans_trs,
            }
        )

        # rate limiter
        time.sleep(6)

    # Save to CSV
    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/qpp_genre_results_fb4.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("QPP-GenRE Summary (mean across queries)")
    print("=" * 60)
    print(f"Baseline RR@10    : {df['baseline_rr10'].mean():.4f}")
    print(f"TRS      RR@10    : {df['trs_rr10'].mean():.4f}")
    print(f"Baseline nDCG@10  : {df['baseline_ndcg10'].mean():.4f}")
    print(f"TRS      nDCG@10  : {df['trs_ndcg10'].mean():.4f}")
    print("\nResults saved to results/qpp_genre_results_fb4.csv")


if __name__ == "__main__":
    main()
