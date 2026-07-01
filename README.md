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
- **TRS Engine:** Python (adaptive per-source decay rates, Bayesian credibility, Z‑score spatial conflict)  
- **Generation:** Gemini (via LangChain)  
- **Evaluation:** RAGAS + QPP‑GenRE (nDCG, RR)  

---

## Quick Start

```bash
git clone git@github.com:hansutapak/trust-rag.git
cd trust-rag
uv sync

# 1. Start Colima (macOS Docker runtime)
colima start

# 2. Start Kafka
docker-compose up -d

# 3. Run the live sensor simulator (Terminal 2)
uv run python -m src.ingestion.sensor_simulation

# 4. Run the Kafka consumer (Terminal 3)
uv run python -m src.ingestion.kafka_consumer

# 5. Let data get stored for 2‑3 minutes, then stop both Terminal 2 and 3 (Ctrl+C)

# 6. Check the database
uv run python tools/inspect_db.py

# 7. Run a query (LLM response) - must relate to the stored Barcelona sensor data
python main.py

# 8. Inspect TRS scores
python tests/debug_trs.py

# 9. Verify adaptive decay
python tests/test_adaptive_decay.py

# 10. Verify Bayesian credibility
python tests/test_bayesian_credibility.py


