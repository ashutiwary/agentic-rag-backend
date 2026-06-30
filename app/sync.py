from __future__ import annotations
import logging

from app.connectors import build_connector, load_sources
from app.indexer    import index_text, delete_by_source_id, load_registry
from app.retriever  import get_stats

logger = logging.getLogger(__name__)


def sync_connector(connector, registry: dict) -> tuple[set[str], dict]:
    """
    Sync one source against a registry snapshot.

    For each listed document: skip it if its version is unchanged, otherwise
    fetch and (re)index it. Deletion is handled by the caller so that several
    sources of the same kind do not delete each other's documents.

    Returns:
        (seen_source_ids, stats) where stats has indexed, skipped, failed.
    """
    seen: set[str] = set()
    indexed = skipped = failed = 0

    for doc in connector.list_documents():
        seen.add(doc.source_id)
        entry = registry.get(doc.source_id)

        if entry and entry.get("version") == doc.version:
            skipped += 1
            continue

        try:
            text = doc.fetch()
            index_text(doc.source_id, doc.name, text, doc.version, connector.kind)
            indexed += 1
        except Exception as exc:
            logger.error("Failed to index %s: %s", doc.source_id, exc)
            failed += 1

    return seen, {"indexed": indexed, "skipped": skipped, "failed": failed}


def sync_all() -> dict:
    """
    Build every source in sources.json and sync each. Documents that vanished
    from a source are removed, but only for source kinds that were listed
    successfully this run, so a failing connector never deletes its content.
    """
    specs    = load_sources()
    registry = load_registry()

    results: list[dict] = []
    seen_by_kind: dict[str, set[str]] = {}
    listed_kinds: set[str] = set()

    for spec in specs:
        try:
            connector     = build_connector(spec)
            seen, stats   = sync_connector(connector, registry)
        except Exception as exc:
            logger.error("Skipping source %s: %s", spec, exc)
            results.append({"spec": spec, "error": str(exc)})
            continue

        listed_kinds.add(connector.kind)
        seen_by_kind.setdefault(connector.kind, set()).update(seen)
        results.append({"kind": connector.kind, **stats})

    # Remove documents that disappeared from any successfully-listed source.
    # Local files (kind "local") are never in listed_kinds, so they are safe.
    removed = 0
    for source_id, meta in list(load_registry().items()):
        kind = meta.get("kind")
        if kind in listed_kinds and source_id not in seen_by_kind.get(kind, set()):
            delete_by_source_id(source_id)
            removed += 1

    summary = {
        "sources"     : results,
        "removed"     : removed,
        "total_chunks": get_stats()["total_chunks"],
    }
    logger.info("Sync complete: %s", summary)
    return summary
