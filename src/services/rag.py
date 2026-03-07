# RAG pipeline orchestration
from __future__ import annotations

import logging
import struct
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

from src.config import get_settings
from src.services import ollama_client as _llm
from src.services.redis_manager import get_redis_client, _docs_schema
from src.utils.prompts import RAG_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    content: str
    source_file: str
    source_name: str
    chunk_id: str
    score: float


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[str]          # unique source_file values
    chunks: list[RetrievedChunk]
    cache_hit: bool = False
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def retrieve(
    question: str,
    *,
    top_k: int = 5,
    source_file: str | None = None,
) -> tuple[list[RetrievedChunk], float, float]:
    """Return (chunks, embed_secs, retrieve_secs)."""
    """Embed *question* and return the top-k most similar document chunks.

    Args:
        question: The user's question.
        top_k: Number of chunks to retrieve.
        source_file: Optional metadata filter — restrict results to one file.

    Returns:
        List of :class:`RetrievedChunk` ordered by similarity (best first).
    """
    settings = get_settings()
    client = await get_redis_client()

    schema = IndexSchema.from_dict(_docs_schema(settings.embedding_dims))
    index = AsyncSearchIndex(schema=schema, redis_client=client)

    t_embed_start = time.perf_counter()
    question_embedding = await _llm.embed(question)
    embed_secs = time.perf_counter() - t_embed_start

    blob = struct.pack(f"{len(question_embedding)}f", *question_embedding)
    filters = f"@source_file:{{{source_file}}}" if source_file else None

    query = VectorQuery(
        vector=blob,
        vector_field_name="embedding",
        return_fields=["content", "source_file", "source_name", "chunk_id"],
        num_results=top_k,
        filter_expression=filters,
    )

    t_retrieve_start = time.perf_counter()
    results = await index.query(query)
    retrieve_secs = time.perf_counter() - t_retrieve_start

    chunks = [
        RetrievedChunk(
            content=r["content"],
            source_file=r["source_file"],
            source_name=r["source_name"],
            chunk_id=r["chunk_id"],
            score=float(r.get("vector_distance", 1.0)),
        )
        for r in results
    ]

    logger.debug("Retrieved %d chunks for question: %s", len(chunks), question[:80])
    return chunks, embed_secs, retrieve_secs


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk.source_file}]\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> str:
    context = _build_context(chunks)
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
        )
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        history=history_text or "(no prior conversation)",
        question=question,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def answer(
    question: str,
    *,
    top_k: int = 5,
    source_file: str | None = None,
    history: list[dict] | None = None,
) -> RAGResult:
    """Run the full RAG pipeline and return a complete answer.

    Args:
        question: The user's question.
        top_k: Number of chunks to retrieve.
        source_file: Optional metadata filter.
        history: Conversation history as list of ``{role, content}`` dicts.

    Returns:
        A :class:`RAGResult` with the answer and retrieved sources.
    """
    t0 = time.perf_counter()

    chunks, embed_secs, retrieve_secs = await retrieve(question, top_k=top_k, source_file=source_file)

    if not chunks:
        return RAGResult(
            question=question,
            answer="I couldn't find any relevant documentation to answer your question.",
            sources=[],
            chunks=[],
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )

    prompt = _build_prompt(question, chunks, history)
    t_gen = time.perf_counter()
    answer_text = await _llm.generate(prompt, system=RAG_SYSTEM_PROMPT)
    generate_secs = time.perf_counter() - t_gen
    total_secs = time.perf_counter() - t0

    timing: StepTiming = {
        "embed_secs": round(embed_secs, 3),
        "retrieve_secs": round(retrieve_secs, 3),
        "generate_secs": round(generate_secs, 3),
        "ttft_secs": None,
        "total_secs": round(total_secs, 3),
    }
    metrics.record_step_timing(timing)
    logger.info(
        "RAG timing — embed=%.2fs retrieve=%.2fs generate=%.2fs total=%.2fs",
        embed_secs, retrieve_secs, generate_secs, total_secs,
    )

    sources = list(dict.fromkeys(c.source_file for c in chunks))  # preserve order, dedupe

    return RAGResult(
        question=question,
        answer=answer_text,
        sources=sources,
        chunks=chunks,
        elapsed_seconds=round(total_secs, 2),
    )


async def stream_answer(
    question: str,
    *,
    top_k: int = 5,
    source_file: str | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[str | dict]:
    """Run the RAG pipeline and stream tokens as they are generated.

    Yields:
        - ``str`` tokens during generation
        - A final ``dict`` sentinel: ``{"done": True, "sources": [...]}``

    Example usage in a FastAPI SSE endpoint::

        async for item in stream_answer(question):
            if isinstance(item, dict):
                # final metadata
            else:
                yield f"data: {json.dumps({'token': item})}\\n\\n"
    """
    t0 = time.perf_counter()
    chunks, embed_secs, retrieve_secs = await retrieve(question, top_k=top_k, source_file=source_file)

    if not chunks:
        yield "I couldn't find any relevant documentation to answer your question."
        yield {"done": True, "sources": [], "timing": None}
        return

    prompt = _build_prompt(question, chunks, history)
    sources = list(dict.fromkeys(c.source_file for c in chunks))

    t_gen = time.perf_counter()
    ttft_secs: float | None = None
    async for token in _llm.stream(prompt, system=RAG_SYSTEM_PROMPT):
        if ttft_secs is None:
            ttft_secs = time.perf_counter() - t_gen
        yield token

    generate_secs = time.perf_counter() - t_gen
    total_secs = time.perf_counter() - t0

    timing: StepTiming = {
        "embed_secs": round(embed_secs, 3),
        "retrieve_secs": round(retrieve_secs, 3),
        "generate_secs": round(generate_secs, 3),
        "ttft_secs": round(ttft_secs, 3) if ttft_secs is not None else None,
        "total_secs": round(total_secs, 3),
    }
    metrics.record_step_timing(timing)
    logger.info(
        "RAG timing (stream) — embed=%.2fs retrieve=%.2fs ttft=%.2fs generate=%.2fs total=%.2fs",
        embed_secs, retrieve_secs, ttft_secs or 0, generate_secs, total_secs,
    )

    yield {"done": True, "sources": sources, "timing": timing}
