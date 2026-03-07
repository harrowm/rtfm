# Administrative endpoints (cache flush, etc.)
from fastapi import APIRouter

from src.services.cache import flush_cache
from src.services.ingestion import flush_docs
from src.services.memory import clear_session

router = APIRouter(tags=["admin"])


@router.post("/cache/flush", summary="Flush all semantic cache entries")
async def cache_flush() -> dict:
    """Delete all entries from the semantic cache. Does not affect the document index."""
    deleted = await flush_cache()
    return {"status": "ok", "deleted": deleted}


@router.post("/docs/flush", summary="Flush all ingested document chunks")
async def docs_flush() -> dict:
    """Delete all document chunks from the vector index.

    Also flushes the semantic cache since cached answers are based on the
    now-deleted documents and would otherwise continue to be served.
    """
    deleted_docs = await flush_docs()
    deleted_cache = await flush_cache()
    return {"status": "ok", "deleted_docs": deleted_docs, "deleted_cache": deleted_cache}


@router.delete("/session/{session_id}", summary="Clear a session's conversation history")
async def session_clear(session_id: str) -> dict:
    """Delete the conversation history for the given session ID."""
    await clear_session(session_id)
    return {"status": "ok", "session_id": session_id}
