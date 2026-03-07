# Semantic caching logic
from __future__ import annotations

import logging
import struct
import time

from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

from src.config import get_settings
from src.services import ollama_client as _llm
from src.services.redis_manager import get_redis_client, _cache_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------

async def get_cached_answer(question: str) -> str | None:
    """Return a cached answer if a semantically similar question exists.

    Embeds *question*, searches the ``rag_cache`` index for a near-duplicate,
    and returns the stored answer if the cosine distance is within the
    configured threshold. Returns ``None`` on a cache miss.
    """
    settings = get_settings()
    client = await get_redis_client()

    schema = IndexSchema.from_dict(_cache_schema(settings.embedding_dims))
    index = AsyncSearchIndex(schema=schema, redis_client=client)

    question_embedding = await _llm.embed(question)
    blob = struct.pack(f"{len(question_embedding)}f", *question_embedding)

    query = VectorQuery(
        vector=blob,
        vector_field_name="embedding",
        return_fields=["question", "answer", "vector_distance"],
        num_results=1,
    )

    results = await index.query(query)

    if not results:
        logger.info("Cache EMPTY (no entries) for: %s", question[:80])
        return None

    top = results[0]
    distance = float(top.get("vector_distance", 1.0))

    if distance <= settings.cache_distance_threshold:
        logger.info(
            "Cache HIT (distance=%.4f <= threshold=%.4f) for: %s",
            distance,
            settings.cache_distance_threshold,
            question[:80],
        )
        return top["answer"]

    logger.info(
        "Cache MISS (distance=%.4f > threshold=%.4f) for: %s",
        distance,
        settings.cache_distance_threshold,
        question[:80],
    )
    return None


# ---------------------------------------------------------------------------
# Cache write
# ---------------------------------------------------------------------------

async def cache_answer(question: str, answer: str) -> None:
    """Store a question-answer pair in the semantic cache.

    The entry is stored with a TTL defined by ``CACHE_TTL_SECONDS`` in config.
    """
    settings = get_settings()
    client = await get_redis_client()

    question_embedding = await _llm.embed(question)
    blob = struct.pack(f"{len(question_embedding)}f", *question_embedding)

    import hashlib
    key = "cache:" + hashlib.sha1(question.encode()).hexdigest()[:16]

    mapping = {
        "question": question,
        "answer": answer,
        "created_at": int(time.time()),
        "embedding": blob,
    }

    await client.hset(key, mapping=mapping)
    if settings.cache_ttl_seconds > 0:
        await client.expire(key, settings.cache_ttl_seconds)

    logger.info("Cached answer for: %s", question[:80])


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

async def flush_cache() -> int:
    """Delete all entries from the semantic cache index.

    Returns the number of keys deleted.
    """
    client = await get_redis_client()
    keys = [k async for k in client.scan_iter("cache:*")]
    if keys:
        await client.delete(*keys)
    logger.info("Flushed %d cache entries", len(keys))
    return len(keys)
