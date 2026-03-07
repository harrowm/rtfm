# Document loading, chunking, embedding
from __future__ import annotations

import hashlib
import logging
import struct
import time
from pathlib import Path

import redis.asyncio as aioredis
from bs4 import BeautifulSoup

from src.services import ollama_client as _llm
from src.services.redis_manager import get_redis_client
from src.utils.chunking import split_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_text(file_path: Path) -> str:
    """Load a document file and return its plain text content.

    Supported formats:
    - ``.md`` / ``.txt``  — returned as-is
    - ``.html`` / ``.htm`` — tags stripped via BeautifulSoup
    - ``.pdf``             — text extracted page-by-page via pypdf
    """
    suffix = file_path.suffix.lower()

    if suffix in (".md", ".txt"):
        return file_path.read_text(encoding="utf-8")

    if suffix in (".html", ".htm"):
        raw = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator="\n")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required for PDF ingestion") from exc
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Redis storage helpers
# ---------------------------------------------------------------------------

def _chunk_key(source_file: str, chunk_index: int) -> str:
    digest = hashlib.sha1(f"{source_file}:{chunk_index}".encode()).hexdigest()[:12]
    return f"doc:{digest}"


async def _store_chunks(
    client: aioredis.Redis,
    chunks: list[str],
    embeddings: list[list[float]],
    source_file: str,
    source_name: str,
) -> int:
    """Write chunks and their embeddings to Redis as hash entries."""
    pipe = client.pipeline(transaction=False)
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        key = _chunk_key(source_file, idx)
        # Encode embedding as 4-byte little-endian floats (FLOAT32 binary blob)
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        pipe.hset(
            key,
            mapping={
                "content": chunk,
                "source_file": source_file,
                "source_name": source_name,
                "chunk_id": str(idx),
                "embedding": blob,
            },
        )
    await pipe.execute()
    return len(chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_file(
    file_path: Path,
    source_name: str | None = None,
    source_file: str | None = None,
) -> dict:
    """Ingest a document file into Redis.

    Steps:
    1. Load and extract plain text from the file.
    2. Split into overlapping token-aware chunks.
    3. Embed all chunks in a single batched Ollama call.
    4. Store chunks + embeddings in Redis under the ``docs`` index.

    Args:
        file_path: Path to the document file.
        source_name: Human-readable label (defaults to the filename stem).

    Returns:
        A dict with ``source_file``, ``source_name``, ``chunks_processed``,
        and ``elapsed_seconds``.
    """
    t0 = time.perf_counter()
    source_name = source_name or file_path.stem
    source_file = source_file or file_path.name

    logger.info("Ingesting %s (%s)", source_file, source_name)

    # 1. Load
    text = load_text(file_path)
    if not text.strip():
        raise ValueError(f"No extractable text found in {file_path}")

    # 2. Chunk
    chunks = split_text(text)
    logger.info("%d chunks created from %s", len(chunks), source_file)

    # 3. Embed (single batched call)
    embeddings = await _llm.embed_batch(chunks)

    # 4. Store
    client = await get_redis_client()
    stored = await _store_chunks(client, chunks, embeddings, source_file, source_name)

    elapsed = time.perf_counter() - t0
    logger.info("Ingested %d chunks in %.2fs", stored, elapsed)

    return {
        "source_file": source_file,
        "source_name": source_name,
        "chunks_processed": stored,
        "elapsed_seconds": round(elapsed, 2),
    }


async def ingest_text(
    text: str,
    source_name: str,
    source_file: str | None = None,
) -> dict:
    """Ingest a raw text string directly (no file required).

    Useful for ingesting content fetched from URLs or generated programmatically.
    """
    t0 = time.perf_counter()
    source_file = source_file or f"{source_name}.txt"

    chunks = split_text(text)
    embeddings = await _llm.embed_batch(chunks)
    client = await get_redis_client()
    stored = await _store_chunks(client, chunks, embeddings, source_file, source_name)

    elapsed = time.perf_counter() - t0
    return {
        "source_file": source_file,
        "source_name": source_name,
        "chunks_processed": stored,
        "elapsed_seconds": round(elapsed, 2),
    }


async def flush_docs() -> int:
    """Delete all document chunks from the docs index.

    Returns the number of keys deleted.
    """
    client = await get_redis_client()
    keys = [k async for k in client.scan_iter("doc:*")]
    if keys:
        await client.delete(*keys)
    logger.info("Flushed %d document chunks", len(keys))
    return len(keys)
