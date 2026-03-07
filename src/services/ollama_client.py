# Unified LLM and embedding client
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import ollama

from src.config import get_settings

logger = logging.getLogger(__name__)


def _async_client() -> ollama.AsyncClient:
    """Return an async Ollama client pointed at the configured base URL."""
    settings = get_settings()
    return ollama.AsyncClient(host=settings.ollama_base_url)


def _llm_options(temperature: float) -> dict:
    """Build the Ollama options dict from config settings."""
    settings = get_settings()
    opts: dict = {"temperature": temperature}
    if settings.llm_num_ctx > 0:
        opts["num_ctx"] = settings.llm_num_ctx
    if settings.llm_num_gpu >= 0:
        opts["num_gpu"] = settings.llm_num_gpu
    if settings.llm_num_thread > 0:
        opts["num_thread"] = settings.llm_num_thread
    return opts


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

async def embed(text: str) -> list[float]:
    """Embed a single string using the configured embedding model.

    Returns a flat list of floats with length equal to ``EMBEDDING_DIMS``.
    """
    settings = get_settings()
    client = _async_client()

    response = await client.embed(
        model=settings.embedding_model,
        input=text,
    )
    # ollama >= 0.4: response.embeddings is a list[list[float]]
    return response.embeddings[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in a single Ollama call."""
    settings = get_settings()
    client = _async_client()

    response = await client.embed(
        model=settings.embedding_model,
        input=texts,
    )
    return response.embeddings


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

async def generate(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Generate a complete (non-streaming) response from the LLM.

    Args:
        prompt: The user message.
        system: Optional system prompt to prepend.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).

    Returns:
        The full text response as a string.
    """
    settings = get_settings()
    client = _async_client()

    messages: list[ollama.Message] = []
    if system:
        messages.append(ollama.Message(role="system", content=system))
    messages.append(ollama.Message(role="user", content=prompt))

    response = await client.chat(
        model=settings.llm_model,
        messages=messages,
        options=_llm_options(temperature),
        keep_alive="-1",
    )
    return response.message.content


async def stream(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    """Stream tokens from the LLM as they are generated.

    Yields individual text chunks (tokens) as they arrive. Callers can
    forward these directly to a Server-Sent Events response.

    Example::

        async for token in stream("How do I authenticate?"):
            print(token, end="", flush=True)
    """
    settings = get_settings()
    client = _async_client()

    messages: list[ollama.Message] = []
    if system:
        messages.append(ollama.Message(role="system", content=system))
    messages.append(ollama.Message(role="user", content=prompt))

    async for chunk in await client.chat(
        model=settings.llm_model,
        messages=messages,
        options=_llm_options(temperature),
        keep_alive="-1",
        stream=True,
    ):
        token = chunk.message.content
        if token:
            yield token


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def warm_models() -> None:
    """Load LLM and embedding models into GPU memory at startup.

    Sends a minimal request to each model so the first real user request
    doesn't pay the cold-load penalty (~15-20s for a 14B model).
    """
    settings = get_settings()
    client = _async_client()
    logger.info("Warming up models (this loads them into GPU memory)...")
    try:
        # Warm embedding model
        await client.embed(model=settings.embedding_model, input="warmup")
        logger.info("Embedding model '%s' loaded.", settings.embedding_model)
    except Exception as exc:
        logger.warning("Could not warm embedding model: %s", exc)
    try:
        # Warm LLM — minimal prompt, keep_alive=-1 pins it in memory permanently
        await client.chat(
            model=settings.llm_model,
            messages=[ollama.Message(role="user", content="hi")],
            options=_llm_options(temperature=0.0),
            keep_alive="-1",
        )
        logger.info("LLM '%s' loaded and pinned in GPU memory.", settings.llm_model)
    except Exception as exc:
        logger.warning("Could not warm LLM: %s", exc)


async def keepalive_loop() -> None:
    """Background task: ping Ollama every 10 minutes to prevent model eviction."""
    settings = get_settings()
    client = _async_client()
    while True:
        await asyncio.sleep(600)  # 10 minutes
        try:
            await client.chat(
                model=settings.llm_model,
                messages=[ollama.Message(role="user", content=".")],
                options={"num_predict": 1},
                keep_alive="-1",
            )
            logger.debug("Keep-alive ping sent to Ollama.")
        except Exception as exc:
            logger.warning("Keep-alive ping failed: %s", exc)


async def check_models() -> dict[str, bool]:
    """Return availability status for the configured LLM and embedding models.

    Useful for the /health endpoint and startup validation.
    """
    settings = get_settings()
    client = _async_client()

    try:
        response = await client.list()
        available = {m.model for m in response.models}
    except Exception as exc:
        logger.warning("Could not reach Ollama: %s", exc)
        return {settings.llm_model: False, settings.embedding_model: False}

    def _present(name: str) -> bool:
        # Ollama names may include ":latest" when listed even if pulled with a tag
        return any(name in m for m in available)

    return {
        settings.llm_model: _present(settings.llm_model),
        settings.embedding_model: _present(settings.embedding_model),
    }
