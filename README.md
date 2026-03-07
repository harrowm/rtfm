# RTFM Agent

A local AI documentation assistant. Ingest your docs, then ask questions. The agent retrieves the most relevant chunks using vector similarity search, generates answers with a local LLM, caches semantically similar queries, and tracks conversation context across sessions.

Runs entirely on your machine — no cloud APIs required.

## How it works

1. **Ingest** — Upload Markdown, text, or HTML files. The agent splits them into ~500-token chunks, embeds each chunk using BGE-M3, and stores the vectors in Redis.
2. **Ask** — Submit a question via `POST /chat`. The agent embeds your question, retrieves the top-K matching chunks, builds a prompt with conversation history and long-term memory, and streams the answer back.
3. **Cache** — After answering, the question + answer pair is stored in a Redis HNSW index. Future questions within a cosine distance of `0.15` return the cached answer immediately, skipping LLM inference.
4. **Remember** — Conversation history is stored per `X-Session-Id` header. After each exchange, the LLM extracts durable facts into a long-term memory store that persists across sessions.

## Stack

| Layer | Component |
|---|---|
| API | FastAPI 0.115+ with async lifespan |
| LLM + embeddings | Ollama — `qwen:14b` (Q4_K_M) and `bge-m3` (1024-dim) |
| Vector store + cache | Redis Stack 7.2 with HNSW indexes via RedisVL |
| Config | pydantic-settings with `.env` file |
| Chunking | tiktoken (`cl100k_base`) token-aware splitter |
| Tests | pytest + pytest-asyncio (43 tests, no live services needed) |

## System requirements

- **macOS** with Apple Silicon (arm64)
- **24 GB unified memory** recommended — models use ~11 GB at runtime
- **Docker Desktop** 4.25+ (arm64)
- **Python 3.12** (managed by `uv`)
- **Homebrew**

## Quick start

### 1. Install system dependencies

```bash
brew install uv ollama
# Docker Desktop: https://www.docker.com/products/docker-desktop/
```

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/harrowm/rtfm.git
cd rtfm
uv sync
```

### 3. Start everything

```bash
./scripts/dev.sh
```

`dev.sh` does the following in order:

1. Pulls `qwen:14b` and `bge-m3` via Ollama if not already present (~11 GB on first run)
2. Launches Docker Desktop if it is not running and waits for the daemon to be ready
3. Starts Redis Stack via Docker Compose and polls until `redis-cli ping` returns `PONG`
4. Creates the HNSW vector indexes in Redis
5. Starts `uvicorn` at **http://localhost:8000**

> **First run:** Model downloads can take several minutes. Subsequent starts skip the download and complete in seconds.

Interactive API docs: **http://localhost:8000/docs**

## Configuration

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `LLM_MODEL` | `qwen:14b` | Chat completion model |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding model |
| `EMBEDDING_DIMS` | `1024` | Must match the embedding model output |
| `CHUNK_SIZE` | `500` | Max tokens per document chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between consecutive chunks |
| `CACHE_DISTANCE_THRESHOLD` | `0.15` | Cosine distance below which a cached answer is returned |
| `CACHE_TTL_SECONDS` | `86400` | Cache entry lifetime (24 h) |
| `SESSION_TTL_HOURS` | `24` | Session history lifetime |
| `LOG_LEVEL` | `info` | Uvicorn log level |

## API reference

### `GET /health`

Returns live connectivity status.

```json
{
  "status": "ok",
  "redis": true,
  "models": { "qwen:14b": true, "bge-m3": true }
}
```

---

### `POST /ingest`

Upload a documentation file to the vector index.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | yes | `.md`, `.txt`, `.html`, `.htm`, or `.pdf` |
| `source_name` | string | no | Label stored with each chunk (defaults to filename) |

**Response**

```json
{
  "status": "success",
  "source_file": "getting-started.md",
  "source_name": "getting-started.md",
  "chunks_processed": 42,
  "elapsed_seconds": 3.14
}
```

**Example**

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@docs/getting-started.md" \
  -F "source_name=getting-started"
```

---

### `POST /chat`

Ask a question. Returns a streaming SSE or non-streaming JSON response.

**Request body**

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | — | Your question (1–4096 chars) |
| `stream` | bool | `true` | Stream tokens via SSE |
| `top_k` | int | `5` | Chunks to retrieve (1–20) |
| `filters` | object | `null` | Optional metadata filter, e.g. `{"source_file": "api.md"}` |

Pass `X-Session-Id: <id>` as a request header to enable session and long-term memory.

**Streaming response** (`stream: true`, default)

```
data: {"token": "To authenticate"}
data: {"token": " with the API, include an API key"}
data: {"done": true, "sources": ["api-reference.md"], "cache_hit": false}
```

**Non-streaming response** (`stream: false`)

```json
{
  "answer": "To authenticate with the API, include an API key in the Authorization header.",
  "sources": ["api-reference.md"],
  "cache_hit": false,
  "elapsed_seconds": 1.82
}
```

**Examples**

```bash
# Streaming with session memory
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: my-session" \
  -d '{"question": "How do I install the SDK?"}'

# Non-streaming, filtered to a specific doc
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is rate limiting?", "stream": false, "filters": {"source_file": "api.md"}}'
```

---

### `GET /metrics`

Prometheus-format metrics.

```
# HELP rtfm_cache_hits_total Total semantic cache hits
# TYPE rtfm_cache_hits_total counter
rtfm_cache_hits_total 127
# HELP rtfm_cache_misses_total Total semantic cache misses
# TYPE rtfm_cache_misses_total counter
rtfm_cache_misses_total 43
```

### `GET /metrics/json`

Same data as JSON.

```json
{
  "cache_hits": 127,
  "cache_misses": 43,
  "hit_rate": 0.747,
  "avg_latency_seconds": 1.24,
  "total_requests": 170
}
```

---

### `POST /cache/flush`

Delete all semantic cache entries. Does not affect the document index.

```bash
curl -X POST http://localhost:8000/cache/flush
# {"status": "ok", "deleted": 12}
```

---

### `DELETE /session/{session_id}`

Clear the conversation history for a session.

```bash
curl -X DELETE http://localhost:8000/session/my-session
# {"status": "ok", "session_id": "my-session"}
```

---

## Tests

The test suite mocks all external services — no running Redis or Ollama needed.

```bash
uv run pytest tests/ -v
```

43 tests across 4 files:

| File | Covers |
|---|---|
| `tests/test_ingestion.py` | Text loading, token-aware chunking, embed + store pipeline |
| `tests/test_rag.py` | Prompt assembly, chunk retrieval, answer generation |
| `tests/test_cache.py` | Cache hit/miss logic, TTL storage, flush |
| `tests/test_memory.py` | Session serialisation, long-term fact extraction |

## Project structure

```
rtfm/
├── pyproject.toml           # Dependencies and pytest config
├── uv.lock                  # Deterministic lockfile
├── .python-version          # Python 3.12
├── .env.example             # Environment variable template
├── docker-compose.yml       # Redis Stack (arm64)
│
├── src/
│   ├── main.py              # FastAPI app, router registration, lifespan
│   ├── config.py            # pydantic-settings Settings class
│   ├── models/
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── memory.py        # SessionMemory, LongTermMemory models
│   ├── services/
│   │   ├── ollama_client.py # embed(), embed_batch(), generate(), stream(), check_models()
│   │   ├── redis_manager.py # Singleton Redis client, HNSW index lifecycle
│   │   ├── ingestion.py     # load → chunk → embed → store pipeline
│   │   ├── rag.py           # retrieve(), answer(), stream_answer()
│   │   ├── cache.py         # Semantic cache get/set/flush
│   │   └── memory.py        # Session and long-term memory persistence
│   ├── utils/
│   │   ├── chunking.py      # tiktoken-based text splitter
│   │   ├── prompts.py       # RAG and memory-extraction prompt templates
│   │   └── metrics.py       # In-process metrics singleton
│   └── api/
│       ├── deps.py          # FastAPI dependency injectors (session, LTM)
│       └── endpoints/
│           ├── ingest.py    # POST /ingest
│           ├── chat.py      # POST /chat (SSE + non-streaming)
│           ├── metrics.py   # GET /metrics, GET /metrics/json
│           └── admin.py     # POST /cache/flush, DELETE /session/{id}
│
├── tests/
│   ├── conftest.py          # Shared fixtures: mock Redis singleton, mock Ollama
│   ├── test_ingestion.py
│   ├── test_rag.py
│   ├── test_cache.py
│   └── test_memory.py
│
└── scripts/
    ├── dev.sh               # One-command dev environment startup
    └── create_indexes.py    # Standalone index initialisation script
```

## Development

### Add dependencies

```bash
uv add package-name          # production
uv add --dev pytest-cov      # dev-only
```

### Start the server manually

Requires Redis on `localhost:6379` and Ollama on `localhost:11434` to already be running.

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

## Troubleshooting

**`/health` returns `"status": "degraded"`**

```bash
ollama list
ollama pull qwen:14b
ollama pull bge-m3
```

**Redis connection refused**

```bash
docker ps | grep redis
docker compose up -d
redis-cli ping          # should return PONG
```

**High memory usage**

On a 16 GB machine, set `LLM_MODEL=qwen:7b` in `.env`.

**Docker Desktop not starting**

```bash
open -a "Docker"
# wait ~30 s then re-run ./scripts/dev.sh
```

## License

Educational and personal use. Component licenses:

- Redis Stack: RSALv2 / SSPL
- Ollama: MIT
- Qwen models: https://ollama.com/library/qwen
- BGE-M3: Apache 2.0
