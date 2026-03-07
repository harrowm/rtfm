# Session and long-term memory handlers
from __future__ import annotations

import json
import logging

from src.config import get_settings
from src.models.memory import LongTermMemory, Message, SessionMemory
from src.services import ollama_client as _llm
from src.services.redis_manager import get_redis_client
from src.utils.prompts import MEMORY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

_SESSION_KEY = "session:{session_id}:messages"
_MEMORY_KEY = "memory:{session_id}:facts"


# ---------------------------------------------------------------------------
# Session memory (short-term conversation history)
# ---------------------------------------------------------------------------

async def load_session(session_id: str) -> SessionMemory:
    """Load conversation history for *session_id* from Redis."""
    client = await get_redis_client()
    key = _SESSION_KEY.format(session_id=session_id)
    raw_messages = await client.lrange(key, 0, -1)

    messages = []
    for raw in raw_messages:
        data = json.loads(raw)
        messages.append(Message(**data))

    return SessionMemory(session_id=session_id, messages=messages)


async def save_message(session_id: str, role: str, content: str) -> None:
    """Append a single message to the session history in Redis."""
    settings = get_settings()
    client = await get_redis_client()
    key = _SESSION_KEY.format(session_id=session_id)

    await client.rpush(key, json.dumps({"role": role, "content": content}))
    # Refresh TTL on every interaction
    await client.expire(key, settings.session_ttl_hours * 3600)


async def clear_session(session_id: str) -> None:
    """Delete the conversation history for *session_id*."""
    client = await get_redis_client()
    await client.delete(_SESSION_KEY.format(session_id=session_id))
    logger.info("Cleared session: %s", session_id)


# ---------------------------------------------------------------------------
# Long-term memory (persistent facts extracted from conversations)
# ---------------------------------------------------------------------------

async def load_long_term_memory(session_id: str) -> LongTermMemory:
    """Load persisted facts for *session_id* from Redis."""
    client = await get_redis_client()
    key = _MEMORY_KEY.format(session_id=session_id)
    raw_facts = await client.lrange(key, 0, -1)
    facts = [f.decode() if isinstance(f, bytes) else f for f in raw_facts]
    return LongTermMemory(session_id=session_id, facts=facts)


async def extract_and_save_memories(session_id: str, session: SessionMemory) -> list[str]:
    """Use the LLM to extract long-term facts from the session and persist them.

    Returns the list of newly extracted fact strings.
    """
    if len(session.messages) < 2:
        return []  # Not enough context to extract anything useful

    conversation_text = session.to_text()
    prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conversation_text)

    try:
        raw = await _llm.generate(prompt, temperature=0.0)
        # Extract JSON array from the response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        new_facts: list[str] = json.loads(raw[start:end])
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", exc)
        return []

    if not new_facts:
        return []

    client = await get_redis_client()
    key = _MEMORY_KEY.format(session_id=session_id)
    for fact in new_facts:
        await client.rpush(key, fact)

    logger.info("Stored %d new memories for session %s", len(new_facts), session_id)
    return new_facts
