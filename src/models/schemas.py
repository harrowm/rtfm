# Pydantic request/response models
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = None
    stream: bool = True
    top_k: int = Field(default=3, ge=1, le=20)
    filters: dict | None = None  # e.g. {"source_file": "api-reference.md"}


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    cache_hit: bool
    elapsed_seconds: float


class IngestResponse(BaseModel):
    status: str
    source_file: str
    source_name: str
    chunks_processed: int
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    redis: bool
    models: dict[str, bool]


class CacheFlushResponse(BaseModel):
    status: str
    deleted: int
