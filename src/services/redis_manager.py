# Redis connection and index management
from __future__ import annotations

import logging

import redis.asyncio as aioredis
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema

from src.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index schemas
# ---------------------------------------------------------------------------

def _docs_schema(embedding_dims: int) -> dict:
    """Schema for document chunk storage and vector search."""
    return {
        "index": {
            "name": "docs",
            "prefix": "doc",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "content",     "type": "text"},
            {"name": "source_file", "type": "tag"},
            {"name": "source_name", "type": "tag"},
            {"name": "chunk_id",    "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "dims": embedding_dims,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }


def _cache_schema(embedding_dims: int) -> dict:
    """Schema for semantic cache storage and vector search."""
    return {
        "index": {
            "name": "rag_cache",
            "prefix": "cache",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "question",   "type": "text"},
            {"name": "answer",     "type": "text"},
            {"name": "created_at", "type": "numeric"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "dims": embedding_dims,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_redis_client: aioredis.Redis | None = None


async def get_redis_client() -> aioredis.Redis:
    """Return the shared async Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,  # keep bytes for vector fields
        )
        logger.info("Redis client created", extra={"url": settings.redis_url})
    return _redis_client


async def close_redis_client() -> None:
    """Close the shared Redis client (call on application shutdown)."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis client closed")


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

async def ensure_indexes() -> None:
    """Create the docs and rag_cache vector indexes if they don't exist."""
    settings = get_settings()
    client = await get_redis_client()

    for schema_dict in [
        _docs_schema(settings.embedding_dims),
        _cache_schema(settings.embedding_dims),
    ]:
        schema = IndexSchema.from_dict(schema_dict)
        index = AsyncSearchIndex(schema=schema, redis_client=client)

        if not await index.exists():
            await index.create(overwrite=False)
            logger.info("Created Redis index: %s", schema.index.name)
        else:
            logger.info("Redis index already exists: %s", schema.index.name)


async def drop_index(name: str, *, delete_documents: bool = False) -> None:
    """Drop a named index (and optionally its documents). Useful in tests."""
    client = await get_redis_client()
    settings = get_settings()

    schema_dict = (
        _docs_schema(settings.embedding_dims)
        if name == "docs"
        else _cache_schema(settings.embedding_dims)
    )
    schema = IndexSchema.from_dict(schema_dict)
    index = AsyncSearchIndex(schema=schema, redis_client=client)
    await index.delete(drop=delete_documents)
    logger.info("Dropped Redis index: %s (documents deleted: %s)", name, delete_documents)
