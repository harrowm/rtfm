# Tests for document ingestion pipeline
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.ingestion import ingest_file, ingest_text, load_text
from src.utils.chunking import split_text, token_count


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_short_text_produces_single_chunk(self):
        chunks = split_text("Hello world", chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_produces_multiple_chunks(self):
        text = " ".join(["word"] * 600)
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunks_overlap(self):
        text = " ".join([str(i) for i in range(300)])
        chunks = split_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= 2
        end_of_first = set(chunks[0].split()[-10:])
        start_of_second = set(chunks[1].split()[:10])
        assert len(end_of_first & start_of_second) > 0

    def test_empty_string(self):
        chunks = split_text("", chunk_size=500, chunk_overlap=50)
        assert chunks == [] or chunks == [""]

    def test_token_count_basic(self):
        assert token_count("hello world") == 2

    def test_token_count_empty(self):
        assert token_count("") == 0


# ---------------------------------------------------------------------------
# load_text
# ---------------------------------------------------------------------------

class TestLoadText:
    def test_load_markdown(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nSome content.")
        assert "Title" in load_text(f)

    def test_load_txt(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Plain text content")
        assert load_text(f) == "Plain text content"

    def test_load_html_strips_tags(self, tmp_path):
        f = tmp_path / "doc.html"
        f.write_text("<html><body><h1>Hello</h1><p>World</p></body></html>")
        text = load_text(f)
        assert "Hello" in text
        assert "<" not in text

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake bytes")
        with pytest.raises(ValueError, match="Unsupported"):
            load_text(f)


# ---------------------------------------------------------------------------
# ingest_file / ingest_text
# ---------------------------------------------------------------------------

class TestIngestFile:
    @pytest.mark.asyncio
    async def test_ingest_markdown_returns_metadata(self, tmp_path, mock_embed_batch, mock_redis):
        f = tmp_path / "guide.md"
        f.write_text("# Guide\n\n" + "This is a sentence. " * 50)

        pipeline_mock = MagicMock()
        pipeline_mock.hset = AsyncMock()
        pipeline_mock.execute = AsyncMock(return_value=[])
        mock_redis.pipeline.return_value = pipeline_mock

        result = await ingest_file(f, source_name="guide")

        assert result["source_name"] == "guide"
        assert result["source_file"] == "guide.md"
        assert result["chunks_processed"] >= 1
        assert result["elapsed_seconds"] >= 0
        mock_embed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_empty_file_raises(self, tmp_path, mock_embed_batch, mock_redis):
        f = tmp_path / "empty.md"
        f.write_text("   ")
        with pytest.raises(ValueError, match="No extractable text"):
            await ingest_file(f)

    @pytest.mark.asyncio
    async def test_ingest_text_direct(self, mock_embed_batch, mock_redis):
        pipeline_mock = MagicMock()
        pipeline_mock.hset = AsyncMock()
        pipeline_mock.execute = AsyncMock(return_value=[])
        mock_redis.pipeline.return_value = pipeline_mock

        result = await ingest_text("Hello " * 100, source_name="test-doc")
        assert result["source_name"] == "test-doc"
        assert result["chunks_processed"] >= 1
