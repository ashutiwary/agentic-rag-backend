from __future__ import annotations
import logging
import os
import shutil
from fastapi       import APIRouter, HTTPException, UploadFile, File
from app.config    import DOCUMENTS_DIR
from app.indexer   import index_document, delete_document, load_registry, is_already_indexed
from app.loader    import SUPPORTED_FORMATS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/")
async def list_documents():
    registry  = load_registry()
    documents = [
        {
            "filename": meta["name"],
            "chunks"  : meta["chunks"],
            "mtime"   : meta.get("version"),
        }
        for meta in registry.values()
        if meta.get("kind") == "local"
    ]
    documents.sort(key=lambda d: d["filename"])
    return {"total": len(documents), "documents": documents}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}",
        )

    save_path = os.path.join(DOCUMENTS_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    try:
        chunk_count = index_document(save_path)
    except Exception as exc:
        os.remove(save_path)
        logger.error("Indexing failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=f"File saved but indexing failed: {exc}")

    logger.info("Uploaded and indexed: %s (%d chunks)", file.filename, chunk_count)
    return {"status": "indexed", "filename": file.filename, "chunks": chunk_count}


@router.delete("/{filename}")
async def remove_document(filename: str):
    filepath = os.path.join(DOCUMENTS_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

    try:
        deleted_chunks = delete_document(filename)
    except Exception as exc:
        logger.error("Failed to delete chunks for %s: %s", filename, exc)
        raise HTTPException(status_code=500, detail=f"Failed to remove from index: {exc}")

    try:
        os.remove(filepath)
    except Exception as exc:
        logger.warning("Chunks deleted but file removal failed: %s", exc)

    logger.info("Deleted: %s (%d chunks removed)", filename, deleted_chunks)
    return {"status": "deleted", "filename": filename, "chunks_removed": deleted_chunks}


@router.get("/{filename}/status")
async def document_status(filename: str):
    filepath       = os.path.join(DOCUMENTS_DIR, filename)
    registry       = load_registry()
    exists_on_disk = os.path.exists(filepath)
    indexed        = is_already_indexed(filepath, registry)
    return {
        "filename"      : filename,
        "exists_on_disk": exists_on_disk,
        "indexed"       : indexed,
        "chunks"        : registry.get(filepath, {}).get("chunks", 0),
    }