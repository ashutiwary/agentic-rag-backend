from __future__ import annotations
import logging
from fastapi    import APIRouter, HTTPException
from pydantic   import BaseModel, Field
from app.indexer  import index_folder
from app.sync      import sync_all
from app.retriever import get_stats
from app.config    import DOCUMENTS_DIR
import app.config as _config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


class SettingsUpdate(BaseModel):
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k:                int   | None = Field(default=None, ge=1, le=20)
    chunk_size:           int   | None = Field(default=None, ge=100, le=2000)
    overlap:              int   | None = Field(default=None, ge=0, le=500)


@router.get("/health")
async def health():
    db_stats = get_stats()
    return {"status": "ok", "total_chunks": db_stats["total_chunks"]}


@router.get("/stats")
async def stats():
    db_stats = get_stats()
    return {
        "database": db_stats,
        "settings": {
            "model"               : _config.MODEL,
            "embed_model"         : _config.EMBED_MODEL,
            "similarity_threshold": _config.SIMILARITY_THRESHOLD,
            "top_k"               : _config.TOP_K,
            "chunk_size"          : _config.CHUNK_SIZE,
            "overlap"             : _config.OVERLAP,
            "documents_dir"       : DOCUMENTS_DIR,
        },
    }


@router.post("/reindex")
async def reindex():
    try:
        summary = index_folder(DOCUMENTS_DIR)
    except Exception as exc:
        logger.error("Reindex failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    logger.info("Reindex complete: %s", summary)
    return {"status": "complete", "summary": summary}


@router.post("/sync")
async def sync():
    """Pull from every source in sources.json, indexing only what changed."""
    try:
        summary = sync_all()
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    logger.info("Sync complete: %s", summary)
    return {"status": "complete", "summary": summary}


@router.patch("/settings")
async def update_settings(payload: SettingsUpdate):
    updated = {}

    if payload.similarity_threshold is not None:
        _config.SIMILARITY_THRESHOLD = payload.similarity_threshold
        updated["similarity_threshold"] = payload.similarity_threshold

    if payload.top_k is not None:
        _config.TOP_K = payload.top_k
        updated["top_k"] = payload.top_k

    if payload.chunk_size is not None:
        _config.CHUNK_SIZE = payload.chunk_size
        updated["chunk_size"] = payload.chunk_size

    if payload.overlap is not None:
        _config.OVERLAP = payload.overlap
        updated["overlap"] = payload.overlap

    if not updated:
        raise HTTPException(status_code=400, detail="No valid settings provided")

    logger.info("Settings updated: %s", updated)
    return {"status": "updated", "changes": updated}