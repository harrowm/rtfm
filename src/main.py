# FastAPI application entrypoint
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.endpoints.chat import router as chat_router
from src.api.endpoints.ingest import router as ingest_router
from src.api.endpoints.metrics import router as metrics_router
from src.api.endpoints.admin import router as admin_router
from src.services.ollama_client import check_models
from src.services.redis_manager import close_redis_client, ensure_indexes, get_redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    yield
    await close_redis_client()


app = FastAPI(
    title="RTFM Agent",
    description="AI documentation assistant with RAG, semantic caching, and memory",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict:
    redis_ok = False
    try:
        client = await get_redis_client()
        redis_ok = await client.ping()
    except Exception:
        pass

    models = await check_models()

    return {
        "status": "ok" if redis_ok and all(models.values()) else "degraded",
        "redis": redis_ok,
        "models": models,
    }
