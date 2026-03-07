# RTFM Agent

A full-stack AI documentation assistant that ingests technical documentation, answers questions using retrieval-augmented generation (RAG), caches semantically similar queries to reduce inference costs, and maintains both session-based and long-term memory. Built for local execution on Apple Silicon using Python, Redis Stack, LangChain, and Ollama.

## System Requirements

- **Hardware**: Apple M4 MacBook Air with 24GB unified memory (recommended configuration)
- **Operating System**: macOS Sonoma 14.0 or later
- **Docker Desktop**: Version 4.25+ with Apple Silicon (arm64) support enabled
- **Python**: 3.12 (managed via uv)

The 24GB memory configuration provides sufficient headroom to run the complete stack locally:
- Qwen 3.5 14B parameter model (quantized): ~9-10 GB
- BGE-M3 embedding model: ~1.5 GB runtime
- Redis Stack container: ~500 MB
- Python runtime and application: ~1-2 GB
- macOS and background processes: ~3-4 GB

## Prerequisites: Homebrew Installation

Install required system dependencies using Homebrew:

```bash
# Install Homebrew if not already present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install core dependencies
brew install uv docker ollama redis

# Install optional development tools
brew install curl jq git
```

Verify installations:

```bash
uv --version
docker --version
ollama --version
redis-cli --version
```

## Architecture Overview

### Component Diagram

```
+--------------------------------------------------+
|              Mac M4 MacBook Air                  |
+--------------------------------------------------+
|  +-------------+  +-------------+  +-----------+ |
|  |   Ollama    |  |   Docker    |  |  Python   | |
|  |             |  |             |  |  App      | |
|  | - qwen:14b  |  | - Redis     |  | - FastAPI| |
|  |   (Q4_K_M)  |  |   Stack     |  | - LangChain| |
|  | - bge-m3    |  |   (arm64)   |  | - RedisVL | |
|  +------+------+  +------+------+  +-----+-----+ |
|         |                |                |       |
|         | gRPC/HTTP      | TCP:6379       | HTTP  |
|         v                v                v       |
|  +-------------+  +-------------+  +-----------+ |
|  | LLM         |  | Vector      |  | REST API  | |
|  | Inference   |  | Index +     |  | Endpoints:| |
|  | (Local)     |  | Cache +     |  | - POST /ingest |
|  |             |  | Sessions    |  | - POST /chat   |
|  |             |  | + Memory    |  | - GET  /metrics|
|  +-------------+  +-------------+  +-----------+ |
+--------------------------------------------------+
```

### Data Flow: RAG Query Processing

1. User submits question via POST /chat endpoint
2. System checks semantic cache for similar prior queries
3. If cache miss: embed question using BGE-M3 via Ollama
4. Perform vector similarity search against Redis document index
5. Retrieve session history and long-term memories from Redis
6. Construct prompt with retrieved context, conversation history, and user memories
7. Generate answer using Qwen 3.5:14b via Ollama
8. Stream response to client, then cache result and update memory stores

### Key Technical Decisions

| Component | Selection | Rationale |
|-----------|-----------|-----------|
| LLM | Qwen 3.5:14b via Ollama (Q4_K_M quantization) | Strong reasoning capability with manageable memory footprint on 24GB M4 Air |
| Embedding Model | BGE-M3 via Ollama | 1024-dimensional vectors, strong multilingual performance, Apache 2.0 license |
| Vector Database | Redis Stack 7.2+ | Native arm64 Docker support, integrated vector search, caching, and session management |
| Application Framework | FastAPI + LangChain + RedisVL | Async support, type safety, high-level abstractions for RAG patterns |
| Package Management | uv | Native Apple Silicon wheel resolution, significantly faster dependency installation, lower memory usage during resolution |

## Project Structure

```
rtfm-agent/
├── pyproject.toml          # Project metadata and dependencies (uv-managed)
├── uv.lock                 # Deterministic dependency lockfile
├── .python-version         # Python version specification (3.12)
├── docker-compose.yml      # Redis Stack and optional Agent Memory Server
├── README.md               # This file
│
├── src/
│   ├── __init__.py
│   ├── main.py             # FastAPI application entrypoint
│   ├── config.py           # Configuration management
│   ├── models/
│   │   ├── schemas.py      # Pydantic request/response models
│   │   └── memory.py       # Session and long-term memory models
│   ├── services/
│   │   ├── ollama_client.py    # Unified LLM and embedding client
│   │   ├── redis_manager.py    # Redis connection and index management
│   │   ├── ingestion.py        # Document loading, chunking, embedding
│   │   ├── rag.py              # RAG pipeline orchestration
│   │   ├── cache.py            # Semantic caching logic
│   │   └── memory.py           # Session and long-term memory handlers
│   ├── utils/
│   │   ├── chunking.py         # Token-aware text splitting
│   │   ├── prompts.py          # System prompts for RAG and memory extraction
│   │   └── metrics.py          # Performance and cache metrics tracking
│   └── api/
│       ├── endpoints/
│       │   ├── ingest.py       # Document ingestion endpoint
│       │   ├── chat.py         # Chat endpoint with streaming support
│       │   ├── metrics.py      # Observability metrics endpoint
│       │   └── admin.py        # Administrative endpoints (cache flush, etc.)
│       └── deps.py             # FastAPI dependency injections
│
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_rag.py
│   ├── test_cache.py
│   └── test_memory.py
│
├── docs/
│   ├── sample_docs/        # Test documentation files (Markdown, HTML)
│   └── api.md              # OpenAPI specification
│
└── scripts/
    ├── dev.sh              # Development environment startup script
    ├── create_indexes.py   # Redis vector index initialization
    └── benchmark.py        # Local performance testing utilities
```

## Setup Instructions

### Step 1: Clone and Initialize Project

```bash
git clone <repository-url> rtfm-agent
cd rtfm-agent

# Initialize uv project environment
uv python install 3.12
uv venv --python 3.12
```

### Step 2: Install Python Dependencies

```bash
# Install all dependencies including development tools
uv sync --all-extras
```

### Step 3: Pull Required Ollama Models

```bash
# Pull the primary language model (approximately 8-9 GB download)
ollama pull qwen:14b

# Pull the embedding model (approximately 1.2 GB download)
ollama pull bge-m3

# Verify models are available
ollama list
```

### Step 4: Start Redis Stack via Docker Compose

```bash
# Start Redis Stack container (arm64 native image)
docker compose up -d redis-stack

# Verify Redis connectivity
redis-cli ping
# Expected response: PONG
```

### Step 5: Create Redis Vector Indexes

```bash
# Run index initialization script
uv run python scripts/create_indexes.py
```

This creates two Redis indexes:
- `docs`: For storing document chunks with vector embeddings and metadata
- `rag_cache`: For semantic caching of question-answer pairs

### Step 6: Start the Application

```bash
# Using the development script (recommended)
./scripts/dev.sh

# Or manually:
source .venv/bin/activate
uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000 --workers 1
```

The API will be available at http://localhost:8000. Interactive API documentation is available at http://localhost:8000/docs.

## Development Workflow

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Quality Checks

```bash
# Linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Auto-formatting
uv run ruff format src/ tests/
```

### Adding New Dependencies

```bash
# Add a production dependency
uv add package-name

# Add a development-only dependency
uv add --dev pytest-cov

# Update all dependencies to latest compatible versions
uv lock --upgrade
```

### Environment Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Key configuration variables:

```env
# Redis connection
REDIS_URL=redis://localhost:6379

# Ollama endpoint
OLLAMA_BASE_URL=http://localhost:11434

# Model configuration
LLM_MODEL=qwen:14b
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMS=1024

# Application settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CACHE_DISTANCE_THRESHOLD=0.15
SESSION_TTL_HOURS=24
```

## API Endpoints

### Document Ingestion

```http
POST /ingest
Content-Type: multipart/form-data

Parameters:
- file: Markdown, text, or HTML documentation file
- source_name: Optional identifier for the document

Response:
{
  "status": "success",
  "chunks_processed": 42,
  "source_file": "getting-started.md"
}
```

### Chat Query

```http
POST /chat
Content-Type: application/json

Request:
{
  "question": "How do I authenticate with the API?",
  "session_id": "optional-session-identifier",
  "filters": {
    "source_file": "api-reference.md"  // Optional metadata filter
  }
}

Response (streaming via Server-Sent Events):
data: {"token": "To authenticate", "source": null}
data: {"token": " with the API, include", "source": null}
data: {"token": " an API key in the", "source": ["api-reference.md:Authentication"]}
data: {"done": true}
```

### Metrics

```http
GET /metrics

Response (Prometheus format):
# HELP rtfm_cache_hits_total Total number of semantic cache hits
# TYPE rtfm_cache_hits_total counter
rtfm_cache_hits_total 127

# HELP rtfm_cache_misses_total Total number of semantic cache misses
# TYPE rtfm_cache_misses_total counter
rtfm_cache_misses_total 43

# HELP rtfm_request_latency_seconds Request processing latency
# TYPE rtfm_request_latency_seconds histogram
rtfm_request_latency_seconds_bucket{le="0.5"} 89
rtfm_request_latency_seconds_bucket{le="5.0"} 156
rtfm_request_latency_seconds_bucket{le="+Inf"} 170
```

### Administrative Endpoints

```http
POST /cache/flush
# Clears all entries from the semantic cache

GET /health
# Returns service health status and component connectivity
```

## Performance Considerations for M4 MacBook Air

### Memory Management

- The application runs with a single Uvicorn worker (`--workers 1`) to prevent memory contention on the 24GB configuration
- Redis is configured with a 1GB memory limit and LRU eviction policy to prevent unbounded growth
- Document chunking uses 500-token chunks with 50-token overlap to balance retrieval quality against embedding computation cost

### Quantization Strategy

- Qwen 3.5:14b uses Q4_K_M quantization by default via Ollama, providing optimal quality-to-memory ratio
- For development iterations where speed is prioritized over final answer quality, consider temporarily using `qwen:7b` or `llama3:8b`

### Caching Configuration

- Semantic cache distance threshold set to 0.15 (cosine distance) provides a balance between hit rate and answer relevance
- Cache entries have a 24-hour TTL to prevent stale answers while maintaining useful reuse
- Monitor cache hit rate via `/metrics` endpoint; adjust threshold based on observed query patterns

### Monitoring Resource Usage

The application includes optional memory monitoring via psutil. Enable verbose logging to track resource usage during development:

```bash
LOG_LEVEL=debug uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

## Testing Guidelines

### Unit Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test category
uv run pytest tests/test_rag.py -v
```

### Integration Testing

```bash
# Start test environment
docker compose up -d redis-stack
uv run pytest tests/ --integration

# Test document ingestion with sample files
curl -X POST http://localhost:8000/ingest \
  -F "file=@docs/sample_docs/getting-started.md" \
  -F "source_name=getting-started"

# Verify data in Redis
redis-cli FT.INFO docs
redis-cli HGETALL doc:<key>
```

### Query Testing Scenarios

1. **Direct knowledge question**: Ask a question explicitly answered in ingested documentation. Verify response accuracy and proper source citation.

2. **Out-of-scope question**: Ask a question not covered by available documentation. Verify the system acknowledges insufficient context rather than hallucinating.

3. **Cross-document question**: Ask a question requiring information from multiple source files. Verify the system retrieves and synthesizes context appropriately.

4. **Semantic cache validation**: Ask the same question twice, then ask a semantically similar rephrasing. Verify the second and third requests return faster responses from cache.

5. **Session memory validation**: Ask a follow-up question using pronouns or references to prior context. Verify the system maintains conversational continuity.

6. **Hybrid search validation**: Ask a question scoped to a specific document using metadata filters. Verify results are constrained to the specified source.

## Troubleshooting

### Ollama Model Loading Issues

If model loading fails or is excessively slow:

```bash
# Check Ollama service status
ollama serve  # Run in separate terminal if not already running

# Verify model availability
ollama list

# Re-pull model if corrupted
ollama rm qwen:14b
ollama pull qwen:14b
```

### Redis Connection Failures

```bash
# Verify Docker container status
docker ps | grep redis

# Check Redis logs
docker logs rtfm-redis

# Test direct connectivity
redis-cli ping
```

### High Memory Usage

If the system becomes unresponsive during heavy usage:

1. Monitor memory usage via Activity Monitor or `uv run python -c "import psutil; print(psutil.virtual_memory())"`
2. Reduce concurrent requests or batch size in ingestion pipeline
3. Consider temporarily switching to a smaller LLM for development iterations
4. Increase Redis eviction aggressiveness: `redis-cli CONFIG SET maxmemory-policy allkeys-lru`

### Docker Apple Silicon Compatibility

Ensure Docker Desktop is configured for Apple Silicon:

1. Open Docker Desktop settings
2. Navigate to "General" tab
3. Verify "Use Apple Virtualization framework" is enabled
4. Confirm containers show "arch: arm64" in `docker ps` output

## License

This project is intended for educational and personal use. Ensure compliance with licenses of all included components:

- Redis Stack: RSALv2 / SSPL
- Ollama: MIT
- LangChain: MIT
- Qwen models: Check specific model license at https://ollama.com/library/qwen
- BGE-M3: Apache 2.0

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make changes and ensure tests pass: `uv run pytest tests/`
4. Run linting and formatting: `uv run ruff check . && uv run ruff format .`
5. Submit a pull request with descriptive commit messages

## Acknowledgments

This project implements concepts from the Redis "RTFM For Me Agent" coding challenge. Redis Stack provides the vector search, caching, and session management infrastructure. Ollama enables local execution of open-weight language models on Apple Silicon hardware.