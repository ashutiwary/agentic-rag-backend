from __future__ import annotations
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi                 import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat                import router as chat_router
from api.documents           import router as documents_router
from api.admin               import router as admin_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(
    title       ="RAG Chatbot API",
    description ="Backend API for the RAG chatbot with admin dashboard",
    version     ="1.0.0",
    docs_url    ="/docs",
    redoc_url   ="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id", "X-Sources"],
)

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(admin_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "RAG Chatbot API",
        "version": "1.0.0",
        "docs"   : "/docs",
    }