# Observability metrics endpoint
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from src.utils.metrics import metrics

router = APIRouter(tags=["observability"])


@router.get("/metrics", summary="Prometheus-format metrics")
async def get_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.to_prometheus(), media_type="text/plain")


@router.get("/metrics/json", summary="Metrics as JSON")
async def get_metrics_json() -> dict:
    last_timing = metrics.last_step_timing()
    recent = metrics.recent_step_timings(10)
    return {
        "cache_hits": metrics.cache_hits,
        "cache_misses": metrics.cache_misses,
        "hit_rate": round(metrics.hit_rate(), 4),
        "avg_latency_seconds": round(metrics.avg_latency(), 4),
        "total_requests": metrics.total_requests,
        "last_request_timing": last_timing,
        "recent_timings": recent,
    }
