# Token-aware text splitting
from __future__ import annotations

import tiktoken

from src.config import get_settings


def _encoder() -> tiktoken.Encoding:
    # cl100k_base is used by most modern models including Qwen and bge-m3
    return tiktoken.get_encoding("cl100k_base")


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """Split *text* into overlapping token-capped chunks.

    Args:
        text: Raw text to split.
        chunk_size: Maximum tokens per chunk (defaults to config value).
        chunk_overlap: Token overlap between consecutive chunks (defaults to config value).

    Returns:
        List of text chunk strings.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    enc = _encoder()
    token_ids = enc.encode(text)

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(token_ids):
            break
        start += chunk_size - chunk_overlap

    return chunks


def token_count(text: str) -> int:
    """Return the number of tokens in *text*."""
    return len(_encoder().encode(text))
