from __future__ import annotations
import logging
import uuid
from fastapi           import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic          import BaseModel, Field
from app.rag           import RAGSession

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

_sessions: dict[str, RAGSession] = {}

# Cap in-memory sessions to avoid unbounded growth on long-running servers.
MAX_SESSIONS = 1000


def _evict_oldest(store: dict) -> None:
    """Drop the oldest inserted sessions until under the cap (insertion-ordered)."""
    while len(store) > MAX_SESSIONS:
        oldest = next(iter(store))
        del store[oldest]
        logger.info("Evicted session: %s", oldest)


def _get_or_create_session(session_id: str) -> RAGSession:
    if session_id not in _sessions:
        _sessions[session_id] = RAGSession()
        _evict_oldest(_sessions)
        logger.info("Created new session: %s", session_id)
    return _sessions[session_id]


class ChatRequest(BaseModel):
    message:    str        = Field(..., min_length=1, max_length=4096)
    session_id: str | None = Field(default=None)


@router.post("/")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    session    = _get_or_create_session(session_id)

    try:
        hits, token_stream = session.chat(request.message)
        sources            = session.extract_sources(hits)
    except Exception as exc:
        logger.error("RAG pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    def _generate():
        try:
            for token in token_stream:
                yield token
        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield f"\n[Stream error: {exc}]"

    return StreamingResponse(
        _generate(),
        media_type="text/plain",
        headers={
            "X-Session-Id": session_id,
            "X-Sources"   : ",".join(sources),
        },
    )


@router.post("/session")
async def create_session():
    session_id            = str(uuid.uuid4())
    _sessions[session_id] = RAGSession()
    _evict_oldest(_sessions)
    logger.info("Session created: %s", session_id)
    return {"session_id": session_id}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id].clear()
    logger.info("Session cleared: %s", session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/session/{session_id}/history")
async def get_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    history = [
        msg for msg in _sessions[session_id].history
        if msg["role"] != "system"
    ]
    return {"session_id": session_id, "history": history}

from app.agent import AgentSession

_agent_sessions: dict[str, AgentSession] = {}


def _get_or_create_agent_session(session_id: str) -> AgentSession:
    if session_id not in _agent_sessions:
        _agent_sessions[session_id] = AgentSession()
        _evict_oldest(_agent_sessions)
        logger.info("Created new agent session: %s", session_id)
    return _agent_sessions[session_id]


@router.post("/agent")
async def agent_chat_endpoint(request: ChatRequest):
    """
    Agentic chat endpoint.
    LLM decides which tools to call based on the question.
    Returns a streaming text response.
    """
    session_id = request.session_id or str(uuid.uuid4())
    session    = _get_or_create_agent_session(session_id)

    try:
        token_stream = session.chat(request.message)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    def _generate():
        try:
            for token in token_stream:
                yield token
        except Exception as exc:
            logger.error("Agent stream error: %s", exc)
            yield f"\n[Stream error: {exc}]"

    return StreamingResponse(
        _generate(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )