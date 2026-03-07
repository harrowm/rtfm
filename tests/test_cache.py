# Tests for semantic cache
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.services.cache import cache_answer, flush_cache, get_cached_answer


class TestGetCachedAnswer:
    @pytest.mark.asyncio
    async def test_hit_within_threshold(self, mock_embed, mock_redis):
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=[
            {"question": "How do I auth?", "answer": "Use an API key.", "vector_distance": "0.05"}
        ])
        with patch("src.services.cache.AsyncSearchIndex", return_value=mock_index):
            result = await get_cached_answer("How do I authenticate?")
        assert result == "Use an API key."

    @pytest.mark.asyncio
    async def test_miss_above_threshold(self, mock_embed, mock_redis):
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=[
            {"question": "Unrelated", "answer": "Irrelevant.", "vector_distance": "0.80"}
        ])
        with patch("src.services.cache.AsyncSearchIndex", return_value=mock_index):
            result = await get_cached_answer("How do I authenticate?")
        assert result is None

    @pytest.mark.asyncio
    async def test_miss_empty_index(self, mock_embed, mock_redis):
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=[])
        with patch("src.services.cache.AsyncSearchIndex", return_value=mock_index):
            result = await get_cached_answer("Any question")
        assert result is None


class TestCacheAnswer:
    @pytest.mark.asyncio
    async def test_stores_question_and_answer(self, mock_embed, mock_redis):
        await cache_answer("What is Redis?", "Redis is an in-memory store.")
        mock_redis.hset.assert_called_once()
        mapping = mock_redis.hset.call_args.kwargs.get("mapping", {})
        assert mapping.get("answer") == "Redis is an in-memory store."
        assert mapping.get("question") == "What is Redis?"

    @pytest.mark.asyncio
    async def test_sets_ttl_on_entry(self, mock_embed, mock_redis):
        await cache_answer("TTL question?", "TTL answer.")
        mock_redis.expire.assert_called_once()


class TestFlushCache:
    @pytest.mark.asyncio
    async def test_flush_deletes_all_cache_keys(self, mock_redis):
        # Pre-populate the mock store so scan_iter yields keys
        async def _scan(*_, **__):
            for key in ["cache:abc", "cache:def"]:
                yield key
        mock_redis.scan_iter = _scan

        count = await flush_cache()
        assert count == 2
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_empty_cache_returns_zero(self, mock_redis):
        async def _empty(*_, **__):
            return
            yield  # noqa: unreachable — makes this an async generator
        mock_redis.scan_iter = _empty

        count = await flush_cache()
        assert count == 0
