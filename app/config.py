import os
from dotenv import load_dotenv

# Load .env from backend/ regardless of where Python is invoked from
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"))


# --- LLM -----------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL        = "qwen/qwen3-32b"
TEMPERATURE  = 0.3
MAX_TOKENS   = 1024

# --- Paths ----------------------------------------------------------------
BASE_DIR      = _BASE_DIR
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DATA_DIR      = os.path.join(BASE_DIR, "data")
CHROMA_DIR    = os.path.join(DATA_DIR, "chroma_db")
REGISTRY_FILE = os.path.join(DATA_DIR, "indexed_files.json")

# --- Embedding ------------------------------------------------------------
EMBED_MODEL     = "all-MiniLM-L6-v2"
COLLECTION_NAME = "knowledge_base"

# --- Chunking -------------------------------------------------------------
# "semantic" splits on meaning boundaries; "fixed" uses fixed-size windows.
CHUNK_STRATEGY = "semantic"

# Fixed strategy
CHUNK_SIZE = 500
OVERLAP    = 100

# Semantic strategy
# Break where adjacent-sentence distance is in the top (100 - percentile)%.
# Higher percentile = fewer, larger chunks. MAX_CHUNK_SIZE is a hard cap.
SEMANTIC_BREAKPOINT_PERCENTILE = 95
MAX_CHUNK_SIZE                 = 1000

BATCH_SIZE = 100

# --- Retrieval ------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.4
TOP_K                = 5

# --- Bootstrap required directories ---------------------------------------
os.makedirs(DATA_DIR,      exist_ok=True)
os.makedirs(CHROMA_DIR,    exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)