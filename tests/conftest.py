# Test fixtures and shared configuration
from __future__ import annotations

import json
import math
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

import src.services.ollama_client as _ollama_mod
import src.services.redis_manager as _rm_mod

from src.main import app


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires live Redis and Ollama services")


# ---------------------------------------------------------------------------
# Fake embedding helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIMS = 1024


def fake_embedding(seed: int = 0) -> list[float]:
    """Return a deterministic unit-norm-ish float vector for testing."""
    v = [math.sin(seed + i) for i in range(EMBEDDING_DIMS)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def fake_embedding_blob(seed: int = 0) -> bytes:
    emb = fake_embedding(seed)
    return struct.pack(f"{len(emb)}f", *emb)


# ---------------------------------------------------------------------------
# Core service mocks
# Patch functions at their SOURCE module (src.services.ollama_client) so all
# consumers that access them via `from src.services import ollama_client as _llm`
# see the mock through the module attribute lookup.
# For Redis, inject directly into the singleton `_redis_client`.
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embed():
    """Patch ollama_client.embed at the source module."""
    m = AsyncMock(return_value=fake_embedding(0))
    with patch.object(_ollama_mod, "embed", m):
        yield m


@pytest.fixture
def mock_embed_batch():
    """Patch ollama_client.embed_batch at the source module."""
    async def _batch(texts):
        return [fake_embedding(i) for i in range(len(texts))]

    m = AsyncMock(side_effect=_batch)
    with patch.object(_ollama_mod, "embed_batch", m):
        yield m


@pytest.fixture
def mock_generate():
    """Patch ollama_client.generate at the source module."""
    m = AsyncMock(return_value="This is a test answer from the LLM.")
    with patch.object(_ollama_mod, "generate", m):
        yield m


@pytest.fixture
def mock_redis():
    """Inject a mock Redis client directly into the redis_manager singleton."""
    store: dict = {}

    client = AsyncMock()

    async def _ping():
        return True

    async def _hset(key, mapping=None, **kwargs):
        store[key] = mapping or kwargs
        return 1

    async def _hgetall(key):
        return store.get(key, {})

    async def _rpush(key, *values):
        store.setdefault(key, [])
        store[key].extend(values)
        return len(store[key])

    async def _lrange(key, start, end):
        lst = store.get(key, [])
        return lst[start: None if end == -1 else end + 1]

    async def _delete(*keys):
        for k in keys:
            store.pop(k, None)
        return len(keys)

    async def _expire(key, ttl):
        return 1

    async def _scan_iter(pattern):
        prefix = pattern.rstrip("*")
        for k in list(store.keys()):
            if k.startswith(prefix):
                yield k

    pipeline_mock = MagicMock()
    pipeline_mock.hset = AsyncMock()
    pipeline_mock.execute = AsyncMock(return_value=[])

    client.ping = AsyncMock(side_effect=_ping)
    client.hset = AsyncMock(side_effect=_hset)
    client.hgetall = AsyncMock(side_effect=_hgetall)
    client.rpush = AsyncMock(side_effect=_rpush)
    client.lrange = AsyncMock(side_effect=_lrange)
    client.delete = AsyncMock(side_effect=_delete)
    client.expire = AsyncMock(side_effect=_expire)
    client.scan_iter = _scan_iter
    client.pipeline = MagicMock(return_value=pipeline_mock)

    # Inject directly into the singleton — all consumers call get_redis_client()
    # which checks _redis_client first; if set, it returns it without connecting.
    old = _rm_mod._redis_client
    _rm_mod._redis_client = client
    try:
        yield client
    finally:
        _rm_mod._redis_client = old


# ---------------------------------------------------------------------------
# FastAPI async test client
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_client():
    """Async HTTPX client against the FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
