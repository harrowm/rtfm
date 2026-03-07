# Administrative endpoints (cache flush, etc.)
from fastapi import APIRouter

from src.services.cache import flush_cache
from src.services.memory import clear_session

router = APIRouter(tags=["admin"])


@router.post("/cache/flush", summary="Flush all semantic cache entries")
async def cache_flush() -> dict:
    """Delete all entries from the semantic cache. Does not affect the document index."""
    deleted = await flush_cache()
    return {"status": "ok", "deleted": deleted}


@router.delete("/session/{session_id}", summary="Clear a session's conversation history")
async def session_clear(session_id: str) -> dict:
    """Delete the conversation history for the given session ID."""
    await clear_session(session_id)
    return {"status": "ok", "session_id": session_id}
