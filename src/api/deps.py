# FastAPI dependency injections
from __future__ import annotations

from fastapi import Header, HTTPException

from src.models.memory import LongTermMemory, SessionMemory
from src.services.memory import load_long_term_memory, load_session


async def get_session(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> SessionMemory | None:
    """Dependency: load session history from Redis if X-Session-Id header is present."""
    if not x_session_id:
        return None
    return await load_session(x_session_id)


async def get_long_term_memory(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> LongTermMemory | None:
    """Dependency: load long-term memory facts from Redis if X-Session-Id header is present."""
    if not x_session_id:
        return None
    return await load_long_term_memory(x_session_id)
