from __future__ import annotations
import chromadb
import app.config as config
from app.config import CHROMA_DIR, COLLECTION_NAME
from app.embedder import embed_one

# --- ChromaDB client — initialised once per process -------------------
_client: chromadb.PersistentClient = chromadb.PersistentClient(
    path=CHROMA_DIR
)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def get_collection():
    """Return the shared ChromaDB collection handle."""
    return _collection


def search(query: str, n_results: int | None = None) -> list[dict]:
    """
    Retrieve the most semantically relevant document chunks
    for a given query.

    Args:
        query:     Natural language query string.
        n_results: Maximum number of results to retrieve before
                   threshold filtering.

    Returns:
        List of dicts with keys: text, score, source.
        Sorted by score descending. Empty list if no hits pass
        the similarity threshold.
    """
    if not query.strip():
        return []

    # Resolve at call time so admin /settings updates take effect.
    if n_results is None:
        n_results = config.TOP_K

    query_embedding = embed_one(query)

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    # results["documents"] is a list-of-lists (one per query)
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    hits: list[dict] = []
    for doc, distance, metadata in zip(documents, distances, metadatas):
        score = round(1 - distance, 3)
        if score >= config.SIMILARITY_THRESHOLD:
            hits.append({
                "text"  : doc,
                "score" : score,
                "source": metadata.get("source", "unknown"),
            })

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def get_stats() -> dict:
    """Return current collection statistics."""
    return {
        "total_chunks": _collection.count(),
        "collection"  : COLLECTION_NAME,
        "embed_model" : config.EMBED_MODEL,
    }