from dotenv import load_dotenv

load_dotenv()

# model
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# chromaDB
CHROMA_COLLECTION = "smart_city"
CHROMA_DIR = "./chroma_db"

# TRS settings
DECAY_RATES = {"sensor": 0.001, "stable": 0.0}
ADAPTIVE_DECAY_RATES = {
    # static entities (no decay)
    "Tree": 0.0,
    "ClimateShelter": 0.0,
    "GreenPoint": 0.0,
    # live sensor types (λ = 1 / shelf‑life in seconds)
    "citizen_device": 0.0083,  # T_stale = 120 s (2 min)
    "council_approved": 0.0033,  # T_stale = 300 s (5 min)
    "commercial_iot": 0.0011,  # T_stale = 900 s (15 min)
    "government_sensor": 0.0006,  # T_stale = 1800 s (30 min)
    "met_office": 0.0003,  # T_stale = 3600 s (1 h)
}
TOP_K_CHUNKS = 3
RETRIEVED_K = 10
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# credibility rules
CREDIBILITY_RULES = {
    "government_sensor": 0.9,
    "met_office": 0.9,
    "council_approved": 0.75,
    "commercial_iot": 0.6,
    "citizen_device": 0.3,
    "flagged_sensor": 0.1,
    "official_document": 1.0,
    "header": 0.5,
    "government": 0.9,
    "mixed": 0.5,
}

# Bayesian credibility priors (alpha = accurate count, beta = inaccurate count)
# These are derived from the static credibility values so that
# credibility = alpha / (alpha + beta)
INITIAL_BAYESIAN_PRIORS = {
    "government_sensor": (90, 10),  # 90% trust
    "met_office": (90, 10),  # 90% trust
    "council_approved": (75, 25),  # 75% trust
    "commercial_iot": (60, 40),  # 60% trust
    "citizen_device": (30, 70),  # 30% trusts
    "flagged_sensor": (10, 90),  # 10% trust
    "official_document": (99, 1),  # 99% trust
    "header": (50, 50),  # 50% trust
    "government": (90, 10),  # 90% trust (static data)
    "mixed": (50, 50),  # 50% trust
}

# static data path
STATIC_DATA_PATH = "data/static"
