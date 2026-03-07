# Chat endpoint with streaming support
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse

from src.api.deps import get_session, get_long_term_memory
from src.models.memory import LongTermMemory, SessionMemory

logger = logging.getLogger(__name__)


def _bg(coro) -> None:
    """Schedule a coroutine as a background task, logging any exceptions."""
    async def _run():
        try:
            await coro
        except Exception as exc:
            logger.warning("Background task failed: %s", exc)
    asyncio.create_task(_run())
from src.models.schemas import ChatRequest
from src.services.cache import cache_answer, get_cached_answer
from src.services.memory import extract_and_save_memories, save_message
from src.services.rag import answer, stream_answer
from src.utils.metrics import Timer, metrics
from src.utils.prompts import RAG_SYSTEM_PROMPT

router = APIRouter(prefix="/chat", tags=["chat"])


def _system_with_memory(ltm: LongTermMemory | None) -> str:
    """Prepend long-term memory facts to the system prompt if available."""
    if not ltm or not ltm.facts:
        return RAG_SYSTEM_PROMPT
    return RAG_SYSTEM_PROMPT + "\n\n" + ltm.to_context_string()


@router.post("", summary="Ask a question about ingested documentation")
async def chat(
    req: ChatRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    session: SessionMemory | None = Depends(get_session),
    ltm: LongTermMemory | None = Depends(get_long_term_memory),
):
    source_file = (req.filters or {}).get("source_file")
    history = session.to_history_dicts() if session else []

    if req.stream:
        async def event_stream():
            # 1. Cache check
            with Timer() as t:
                cached = await get_cached_answer(req.question)
            if cached:
                metrics.record_hit(t.elapsed)
                if x_session_id:
                    await save_message(x_session_id, "user", req.question)
                    await save_message(x_session_id, "assistant", cached)
                yield f"data: {json.dumps({'token': cached})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': [], 'cache_hit': True})}\n\n"
                return

            # 2. RAG generation — hold the done event until after we've written to cache
            collected_tokens: list[str] = []
            done_event: str | None = None
            should_cache = False
            with Timer() as t:
                async for item in stream_answer(
                    req.question,
                    top_k=req.top_k,
                    source_file=source_file,
                    history=history,
                    system_prompt=_system_with_memory(ltm),
                ):
                    if isinstance(item, dict):
                        # Hold the done sentinel; forward everything else immediately
                        if item.get("done"):
                            # Only cache real RAG answers (non-empty sources = not rejected/not-found)
                            should_cache = bool(item.get("sources"))
                            done_event = json.dumps({**item, "cache_hit": False})
                        else:
                            yield f"data: {json.dumps({**item, 'cache_hit': False})}\n\n"
                    else:
                        collected_tokens.append(item)
                        yield f"data: {json.dumps({'token': item})}\n\n"

            metrics.record_miss(t.elapsed)

            # 3. Fire session + cache writes in the background, yield done immediately
            full_answer = "".join(collected_tokens)
            if x_session_id and full_answer:
                _bg(save_message(x_session_id, "user", req.question))
                _bg(save_message(x_session_id, "assistant", full_answer))
                updated_session = session or SessionMemory(session_id=x_session_id)
                updated_session.add("user", req.question)
                updated_session.add("assistant", full_answer)
                _bg(extract_and_save_memories(x_session_id, updated_session))
            if should_cache and full_answer:
                _bg(cache_answer(req.question, full_answer))

            # Done is sent immediately — background writes continue independently
            if done_event:
                yield f"data: {done_event}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path
    with Timer() as t:
        cached = await get_cached_answer(req.question)
    if cached:
        metrics.record_hit(t.elapsed)
        if x_session_id:
            await save_message(x_session_id, "user", req.question)
            await save_message(x_session_id, "assistant", cached)
        return {"answer": cached, "sources": [], "cache_hit": True, "elapsed_seconds": round(t.elapsed, 2)}

    with Timer() as t:
        result = await answer(
            req.question,
            top_k=req.top_k,
            source_file=source_file,
            history=history,
            system_prompt=_system_with_memory(ltm),
        )

    metrics.record_miss(t.elapsed)

    if x_session_id and result.sources:
        _bg(save_message(x_session_id, "user", req.question))
        _bg(save_message(x_session_id, "assistant", result.answer))
        updated_session = session or SessionMemory(session_id=x_session_id)
        updated_session.add("user", req.question)
        updated_session.add("assistant", result.answer)
        _bg(extract_and_save_memories(x_session_id, updated_session))

    if result.sources:
        _bg(cache_answer(req.question, result.answer))

    return {
        "answer": result.answer,
        "sources": result.sources,
        "cache_hit": False,
        "elapsed_seconds": result.elapsed_seconds,
    }
