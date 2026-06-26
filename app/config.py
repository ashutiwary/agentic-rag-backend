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
CHUNK_SIZE = 500
OVERLAP    = 100
BATCH_SIZE = 100

# --- Retrieval ------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.4
TOP_K                = 5

# --- Bootstrap required directories ---------------------------------------
os.makedirs(DATA_DIR,      exist_ok=True)
os.makedirs(CHROMA_DIR,    exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)