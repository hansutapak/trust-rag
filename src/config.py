from dotenv import load_dotenv

load_dotenv()

# model
LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# chromaDB
CHROMA_COLLECTION = "smart_city"
CHROMA_DIR = "./chroma_db"

# TRS settings
DECAY_RATES = {"sensor": 0.001, "stable": 0.0}
TOP_K_CHUNKS = 3
RETRIEVED_K = 5
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# credibbility rules
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

# static data path
STATIC_DATA_PATH = "data/static"
