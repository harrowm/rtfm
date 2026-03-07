# Configuration management
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Models
    llm_model: str = "qwen:14b"
    embedding_model: str = "bge-m3"
    embedding_dims: int = 1024

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Semantic cache
    cache_distance_threshold: float = 0.15
    cache_ttl_seconds: int = 86400  # 24 hours

    # Session memory
    session_ttl_hours: int = 24

    # LLM performance tuning (passed directly to Ollama)
    # num_ctx: KV-cache context window. Smaller = less memory + faster time-to-first-token.
    # 4096 comfortably fits top_k=5 chunks × 500 tokens + prompt overhead.
    llm_num_ctx: int = 4096
    # num_gpu: layers to offload to GPU. 99 = offload everything (recommended for Apple Silicon).
    llm_num_gpu: int = 99
    # num_thread: CPU threads for generation. 0 = let Ollama decide (usually optimal).
    llm_num_thread: int = 0

    # Application
    log_level: str = "info"
    app_host: str = "127.0.0.1"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance. Import and call this everywhere config is needed."""
    return Settings()
