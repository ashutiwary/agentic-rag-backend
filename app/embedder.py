from __future__ import annotations
import logging
import os

# Suppress verbose HTTP and HuggingFace logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Disable HuggingFace Hub progress bars and online checks
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL

_model: SentenceTransformer = SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _model.encode(texts).tolist()


def embed_one(text: str) -> list[list[float]]:
    # Returns a (1, dim) list-of-lists — the shape ChromaDB's
    # `query_embeddings` expects for a single query.
    return _model.encode([text]).tolist()