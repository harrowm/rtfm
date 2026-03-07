# Redis vector index initialization
import asyncio
import sys

sys.path.insert(0, ".")  # run from project root: uv run python scripts/create_indexes.py

from src.services.redis_manager import ensure_indexes, close_redis_client


async def main() -> None:
    print("Creating Redis vector indexes...")
    await ensure_indexes()
    await close_redis_client()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
