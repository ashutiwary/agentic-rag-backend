from __future__ import annotations
import os
import json
import logging
import app.config as config
from app.config import REGISTRY_FILE, DOCUMENTS_DIR
from app.loader    import load_document, SUPPORTED_FORMATS
from app.chunker   import chunk_document
from app.retriever import get_collection, get_stats
from app.embedder  import embed

logger = logging.getLogger(__name__)


# --- Registry helpers -----------------------------------------------------

def load_registry() -> dict:
    """Load the file index registry from disk."""
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict) -> None:
    """Persist the file index registry to disk."""
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def is_already_indexed(filepath: str, registry: dict) -> bool:
    """
    Return True if the file exists in the registry and its
    last-modified time has not changed since it was indexed.
    """
    if filepath not in registry:
        return False
    return registry[filepath]["mtime"] == os.path.getmtime(filepath)


# --- Core indexing --------------------------------------------------------

def index_document(filepath: str) -> int:
    """
    Load, chunk, embed, and store a single document.

    Args:
        filepath: Absolute path to the document.

    Returns:
        Number of chunks indexed. Returns 0 if document is empty.

    Raises:
        Exception: Propagates any loader or ChromaDB errors to the caller.
    """
    collection = get_collection()
    filename   = os.path.basename(filepath)

    logger.info("Indexing: %s", filename)

    text   = load_document(filepath)
    chunks = chunk_document(text)

    if not chunks:
        logger.warning("No content extracted from %s", filename)
        return 0

    # Remove any existing chunks for this file first, so re-indexing a
    # modified document replaces stale content instead of being skipped
    # by ChromaDB's duplicate-ID handling.
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Embed in batches to avoid memory spikes on large documents
    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), config.BATCH_SIZE):
        batch = chunks[i: i + config.BATCH_SIZE]
        all_embeddings.extend(embed(batch))

    ids       = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": filename, "filepath": filepath}
        for _ in chunks
    ]

    collection.add(
        documents=chunks,
        embeddings=all_embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    # Record the document in the registry so uploads/single indexes show up
    # in listings and status checks (not just folder scans).
    registry = load_registry()
    registry[filepath] = {
        "mtime"   : os.path.getmtime(filepath),
        "chunks"  : len(chunks),
        "filename": filename,
    }
    save_registry(registry)

    logger.info("Indexed %d chunks from %s", len(chunks), filename)
    return len(chunks)


def delete_document(filename: str) -> int:
    """
    Remove all chunks for a document from ChromaDB and the registry.

    Args:
        filename: Base filename (not full path).

    Returns:
        Number of chunks deleted.
    """
    collection = get_collection()

    results = collection.get(where={"source": filename})
    ids     = results["ids"]

    if ids:
        collection.delete(ids=ids)

    registry = load_registry()
    registry = {
        k: v for k, v in registry.items()
        if v.get("filename") != filename
    }
    save_registry(registry)

    logger.info("Deleted %d chunks for %s", len(ids), filename)
    return len(ids)


def index_folder(folder_path: str = DOCUMENTS_DIR) -> dict:
    """
    Scan a folder and index all new or modified documents.
    Unchanged files (matched by mtime in registry) are skipped.

    Args:
        folder_path: Directory to scan. Defaults to DOCUMENTS_DIR.

    Returns:
        Summary dict with keys: indexed, skipped, failed, total_chunks.
    """
    registry = load_registry()
    indexed  = 0
    skipped  = 0
    failed   = 0

    logger.info("Scanning folder: %s", folder_path)

    for filename in sorted(os.listdir(folder_path)):
        ext      = os.path.splitext(filename)[1].lower()
        filepath = os.path.join(folder_path, filename)

        if ext not in SUPPORTED_FORMATS:
            continue

        if is_already_indexed(filepath, registry):
            logger.info("Skipping unchanged file: %s", filename)
            skipped += 1
            continue

        try:
            # index_document persists its own registry entry.
            count = index_document(filepath)
            indexed += count
        except Exception as exc:
            logger.error("Failed to index %s: %s", filename, exc)
            failed += 1

    summary = {
        "indexed"     : indexed,
        "skipped"     : skipped,
        "failed"      : failed,
        "total_chunks": get_stats()["total_chunks"],
    }

    logger.info(
        "Indexing complete. indexed=%d skipped=%d failed=%d total=%d",
        indexed, skipped, failed, summary["total_chunks"]
    )

    return summary