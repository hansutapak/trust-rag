# Trust-rag

**The Total Reliability Score (TRS) is a type of middleware for reliable, trustworthy Retrieval-Augmented Generation (RAG) in Cyber-Physical Systems (CPSs) for smart cities.**

Prevents context poisoning in real-time RAG by mathematically weighing each retrieved IoT chunk by its degree of semantic similarity, exponential recency decay and source credibility, with spatial conflict resolution.

---

## Architecture

`Open Data BCN -> ETL (NGSI‑LD JSON) -> ChromaDB -> TRS Middleware -> LLM (Gemini) -> Answer`

The TRS layer is placed between the retriever and generator as a safety filter that filters out low credibility data as well as contradictory and stale data that may be fed into the generator.

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

## Quick Start

```bash
git clone git@github.com:hansutapak/trust-rag.git
cd trust-rag
uv sync

# 1. Generate NGSI‑LD JSON from raw CSVs (included)
python src/etl/convert.py

# 2. Seed ChromaDB
python set_up.py

# 3. Run a query (LLM response)
python main.py

# 4. Inspect TRS scores
python tests/debug_trs.py
