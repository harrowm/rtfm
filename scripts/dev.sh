#!/usr/bin/env bash
set -e

echo "🔧 Starting RTFM Agent dev environment on Mac M4..."

# 1. Determine models from .env (fallback to defaults if not set)
LLM_MODEL=$(grep -E '^LLM_MODEL=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' || echo "qwen2.5:7b")
EMBEDDING_MODEL=$(grep -E '^EMBEDDING_MODEL=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' || echo "bge-m3")

echo "Using LLM: $LLM_MODEL, Embedding: $EMBEDDING_MODEL"

if ! ollama list | grep -q "$LLM_MODEL"; then
    echo "Pulling $LLM_MODEL (this may take a few minutes)..."
    ollama pull "$LLM_MODEL"
fi

if ! ollama list | grep -q "$EMBEDDING_MODEL"; then
    echo "Pulling $EMBEDDING_MODEL..."
    ollama pull "$EMBEDDING_MODEL"
fi

# 2. Ensure Docker Desktop is running
if ! docker info >/dev/null 2>&1; then
    echo "🐳 Docker is not running. Launching Docker Desktop..."
    open -a "Docker"
    echo "⏳ Waiting for Docker to be ready (this may take ~30 seconds)..."
    until docker info >/dev/null 2>&1; do
        sleep 2
        printf "."
    done
    echo ""
    echo "✅ Docker is ready."
fi

# 3. Start Redis Stack
echo "🗄️  Starting Redis Stack..."
docker compose up -d redis-stack

# Wait for Redis to be ready
until redis-cli ping 2>/dev/null | grep -q PONG; do
    echo "⏳ Waiting for Redis..."
    sleep 1
done
echo "✅ Redis is up and responding to PING."

# 4. Kill any existing process on port 8000
if lsof -ti :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 in use — stopping previous server..."
    lsof -ti :8000 | xargs kill -9
    sleep 1
    echo "Previous API server stopped."
fi
if lsof -ti :8501 >/dev/null 2>&1; then
    echo "Port 8501 in use - stopping previous UI server..."
    lsof -ti :8501 | xargs kill -9
    sleep 1
    echo "Previous UI server stopped."
fi

# 5. Activate uv environment
source .venv/bin/activate

# 6. Start Streamlit UI in the background
echo "Starting Streamlit UI on http://localhost:8501"
uv run streamlit run scripts/ui.py --server.headless true --server.port 8501 &
STREAMLIT_PID=$!

# Open browser only once FastAPI /health returns 200 (model warm-up complete)
(
    echo "Waiting for API to be ready..."
    until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        sleep 2
    done
    echo "API ready — opening browser."
    open "http://localhost:8501"
) &

# Kill both processes cleanly on Ctrl+C
trap "echo Shutting down...; kill $STREAMLIT_PID 2>/dev/null; exit 0" INT TERM

# 7. Start FastAPI server in the foreground
echo "Starting FastAPI server on http://localhost:8000"
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 --workers 1