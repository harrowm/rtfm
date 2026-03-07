# Tests for RAG pipeline
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.services.rag import RetrievedChunk, _build_context, _build_prompt, answer, retrieve


def _make_chunk(content="Test content", source="doc.md") -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        source_file=source,
        source_name="doc",
        chunk_id="0",
        score=0.05,
    )


# ---------------------------------------------------------------------------
# Prompt construction (pure functions, no I/O)
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    def test_build_context_single_chunk(self):
        ctx = _build_context([_make_chunk()])
        assert "doc.md" in ctx
        assert "Test content" in ctx

    def test_build_context_multiple_chunks_have_separator(self):
        chunks = [_make_chunk(f"Content {i}", f"file{i}.md") for i in range(3)]
        ctx = _build_context(chunks)
        assert "---" in ctx
        assert "Content 0" in ctx and "Content 2" in ctx

    def test_build_prompt_includes_all_sections(self):
        history = [{"role": "user", "content": "Previous question"}]
        prompt = _build_prompt("New question?", [_make_chunk()], history)
        assert "New question?" in prompt
        assert "Previous question" in prompt
        assert "doc.md" in prompt

    def test_build_prompt_no_history_shows_placeholder(self):
        prompt = _build_prompt("What is this?", [_make_chunk()], None)
        assert "no prior conversation" in prompt


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_mapped_chunks(self, mock_embed, mock_redis):
        fake_results = [{
            "content": "Auth requires an API key.",
            "source_file": "api.md",
            "source_name": "api",
            "chunk_id": "0",
            "vector_distance": "0.05",
        }]
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=fake_results)

        with patch("src.services.rag.AsyncSearchIndex", return_value=mock_index):
            chunks, embed_secs, retrieve_secs = await retrieve("How do I authenticate?")

        assert len(chunks) == 1
        assert chunks[0].source_file == "api.md"
        assert embed_secs >= 0
        assert retrieve_secs >= 0
        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_index_returns_empty_list(self, mock_embed, mock_redis):
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=[])

        with patch("src.services.rag.AsyncSearchIndex", return_value=mock_index):
            chunks, embed_secs, retrieve_secs = await retrieve("Anything?")

        assert chunks == []
        assert embed_secs >= 0
        assert retrieve_secs >= 0


# ---------------------------------------------------------------------------
# answer()
# ---------------------------------------------------------------------------

class TestAnswer:
    @pytest.mark.asyncio
    async def test_no_chunks_returns_fallback_message(self, mock_embed, mock_redis):
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=[])

        with patch("src.services.rag.AsyncSearchIndex", return_value=mock_index):
            result = await answer("What is the meaning of life?")

        assert "couldn't find" in result.answer.lower()
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_with_chunks_calls_generate(self, mock_embed, mock_generate, mock_redis):
        fake_results = [{
            "content": "The answer is 42.",
            "source_file": "guide.md",
            "source_name": "guide",
            "chunk_id": "0",
            "vector_distance": "0.02",
        }]
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=fake_results)

        with patch("src.services.rag.AsyncSearchIndex", return_value=mock_index):
            result = await answer("What is the answer?")

        assert result.answer == "This is a test answer from the LLM."
        assert "guide.md" in result.sources
        mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_sources_deduplicated(self, mock_embed, mock_generate, mock_redis):
        fake_results = [
            {"content": f"Chunk {i}", "source_file": "guide.md",
             "source_name": "guide", "chunk_id": str(i), "vector_distance": "0.02"}
            for i in range(3)
        ]
        mock_index = AsyncMock()
        mock_index.query = AsyncMock(return_value=fake_results)

        with patch("src.services.rag.AsyncSearchIndex", return_value=mock_index):
            result = await answer("Tell me about the guide")

        assert result.sources.count("guide.md") == 1
