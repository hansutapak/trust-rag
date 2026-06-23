<h1 style="color:#EF476F;">trust-rag</h1>

**Total Reliability Score (TRS) middleware for trustworthy Retrieval‑Augmented Generation in smart‑city Cyber‑Physical Systems.**

Prevents context poisoning in real‑time RAG by mathematically scoring every retrieved IoT chunk on **semantic similarity × exponential recency decay × source credibility**, with spatial conflict resolution.

---

## Architecture

`Open Data BCN -> ETL (NGSI‑LD JSON) -> ChromaDB -> TRS Middleware -> LLM (Gemini) -> Answer`

The TRS layer sits between the retriever and generator, acting as a safety filter that discards stale, contradictory, or low‑credibility data before generation.

---

## Stack

- **Ingestion:** Apache Kafka (simulated live streams)  
- **Storage:** ChromaDB (vector store)  
- **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`) – local, zero‑cost  
- **TRS Engine:** Python (exponential decay, Bayesian credibility, Z‑score spatial conflict)  
- **Generation:** Gemini (via LangChain)  
- **Evaluation:** RAGAS + QPP‑GenRE (nDCG, RR)  

---

## Quick Start

```bash
git clone git@github.com:hansutapak/trust-rag.git
cd trust-rag
uv sync
python set_up.py          # seeds ChromaDB from Open Data BCN
python main.py             # runs a query
python tests/debug_trs.py  # shows TRS scores and component breakdown
