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
#
# The registry maps a stable source_id to what we last indexed for it:
#   { source_id, kind, name, version, chunks }
# kind is "local" for files in the documents folder, or a connector name
# (e.g. "arxiv") for external sources. version is whatever token tells us
# the content changed: a file mtime, a remote timestamp, or a content hash.

def load_registry() -> dict:
    """Load the index registry from disk."""
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict) -> None:
    """Persist the index registry to disk."""
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def is_already_indexed(filepath: str, registry: dict) -> bool:
    """
    Return True if a local file is in the registry and its last-modified
    time has not changed since it was indexed.
    """
    entry = registry.get(filepath)
    if not entry:
        return False
    return entry.get("version") == os.path.getmtime(filepath)


# --- Core indexing --------------------------------------------------------

def index_text(
    source_id: str,
    name: str,
    text: str,
    version,
    kind: str = "local",
) -> int:
    """
    Chunk, embed, and store arbitrary text under a stable source id.

    This is the single indexing path shared by local files and external
    connectors. Any existing chunks for the same source_id are removed
    first, so re-indexing a changed document replaces stale content
    instead of duplicating it.

    Args:
        source_id: Stable unique id for the document (e.g. a filepath or
                   "arxiv:2401.01234").
        name:      Human-readable display name shown as the answer source.
        text:      Extracted document text.
        version:   Change token (mtime, remote timestamp, or content hash).
        kind:      Source category ("local" or a connector name).

    Returns:
        Number of chunks indexed. 0 if the document is empty.
    """
    collection = get_collection()

    # Replace any existing chunks for this source.
    existing = collection.get(where={"source_id": source_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunks = chunk_document(text)

    if not chunks:
        logger.warning("No content extracted from %s", name)
        registry = load_registry()
        registry.pop(source_id, None)
        save_registry(registry)
        return 0

    # Embed in batches to avoid memory spikes on large documents.
    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), config.BATCH_SIZE):
        batch = chunks[i: i + config.BATCH_SIZE]
        all_embeddings.extend(embed(batch))

    ids       = [f"{source_id}::chunk::{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": name, "source_id": source_id, "kind": kind}
        for _ in chunks
    ]

    collection.add(
        documents=chunks,
        embeddings=all_embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    registry = load_registry()
    registry[source_id] = {
        "source_id": source_id,
        "kind"     : kind,
        "name"     : name,
        "version"  : version,
        "chunks"   : len(chunks),
    }
    save_registry(registry)

    logger.info("Indexed %d chunks from %s", len(chunks), name)
    return len(chunks)


def index_document(filepath: str) -> int:
    """
    Load and index a single local file. Thin wrapper over index_text that
    reads the file and uses its path as the source id and mtime as version.
    """
    filename = os.path.basename(filepath)
    logger.info("Indexing: %s", filename)

    text = load_document(filepath)
    return index_text(
        source_id=filepath,
        name=filename,
        text=text,
        version=os.path.getmtime(filepath),
        kind="local",
    )


def delete_by_source_id(source_id: str) -> int:
    """Remove all chunks for a source id from ChromaDB and the registry."""
    collection = get_collection()

    results = collection.get(where={"source_id": source_id})
    ids     = results["ids"]
    if ids:
        collection.delete(ids=ids)

    registry = load_registry()
    if source_id in registry:
        del registry[source_id]
        save_registry(registry)

    logger.info("Deleted %d chunks for %s", len(ids), source_id)
    return len(ids)


def delete_document(filename: str) -> int:
    """Delete a local document's chunks by filename (API convenience)."""
    source_id = os.path.join(DOCUMENTS_DIR, filename)
    return delete_by_source_id(source_id)


def index_folder(folder_path: str = DOCUMENTS_DIR) -> dict:
    """
    Scan a folder and index all new or modified documents.
    Unchanged files (matched by mtime in the registry) are skipped.

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
