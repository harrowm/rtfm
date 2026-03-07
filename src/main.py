# FastAPI application entrypoint
from fastapi import FastAPI

app = FastAPI(
    title="RTFM Agent",
    description="AI documentation assistant with RAG, semantic caching, and memory",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
