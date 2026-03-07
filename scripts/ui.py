"""
RTFM Agent — Streamlit UI
Run with: uv run streamlit run scripts/ui.py
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Generator

import psutil
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:8000"


def base_url() -> str:
    return st.session_state.get("base_url", DEFAULT_BASE_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RTFM Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 RTFM Agent")
st.caption("Local AI documentation assistant — RAG · Semantic cache · Session memory")

# ---------------------------------------------------------------------------
# Sidebar — connection settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    url_input = st.text_input("API base URL", value=DEFAULT_BASE_URL)
    st.session_state["base_url"] = url_input

    st.markdown("---")
    st.markdown("**Quick links**")
    st.markdown(f"[Swagger UI]({url_input}/docs)  |  [ReDoc]({url_input}/redoc)")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render_timing(timing: dict | None, elapsed: float | None) -> None:
    """Render a compact timing breakdown under a chat message."""
    if timing:
        embed = timing.get("embed_secs", 0)
        retrieve = timing.get("retrieve_secs", 0)
        ttft = timing.get("ttft_secs")
        generate = timing.get("generate_secs", 0)
        total = timing.get("total_secs", 0)
        parts = [
            f"embed **{embed:.2f}s**",
            f"retrieve **{retrieve:.2f}s**",
        ]
        if ttft is not None:
            parts.append(f"ttft **{ttft:.2f}s**")
        parts.append(f"generate **{generate:.2f}s**")
        parts.append(f"total **{total:.2f}s**")
        st.caption("⏱ " + " · ".join(parts))
    elif elapsed is not None:
        st.caption(f"⏱ total **{elapsed:.2f}s**")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_status, tab_chat, tab_ingest, tab_metrics, tab_admin = st.tabs(
    ["🟢 Status", "💬 Chat", "📂 Ingest", "📊 Metrics", "🛠 Admin"]
)

# ===========================================================================
# STATUS tab
# ===========================================================================
with tab_status:
    st.subheader("Service health")

    if st.button("Refresh", key="refresh_health"):
        st.session_state.pop("health_data", None)

    try:
        resp = requests.get(f"{base_url()}/health", timeout=5)
        data = resp.json()
        st.session_state["health_data"] = data
    except Exception as exc:
        st.error(f"Could not reach API: {exc}")
        data = None

    if data:
        overall = data.get("status", "unknown")
        colour = "🟢" if overall == "ok" else "🔴"
        st.markdown(f"### {colour} Overall status: `{overall}`")

        col1, col2 = st.columns(2)
        with col1:
            redis_ok = data.get("redis", False)
            st.metric("Redis", "✅ connected" if redis_ok else "❌ down")
        with col2:
            models: dict = data.get("models", {})
            for model_name, ok in models.items():
                st.metric(model_name, "✅ ready" if ok else "❌ missing")

        with st.expander("Raw JSON"):
            st.json(data)

    # Memory stats (psutil reads local machine — same as where models run)
    st.markdown("---")
    st.markdown("### System memory")
    vm = psutil.virtual_memory()
    used_gb = vm.used / 1024**3
    total_gb = vm.total / 1024**3
    avail_gb = vm.available / 1024**3
    pct = vm.percent

    m1, m2, m3 = st.columns(3)
    m1.metric("Used", f"{used_gb:.1f} GB", help="Includes GPU (unified memory)")
    m2.metric("Available", f"{avail_gb:.1f} GB")
    m3.metric("Total", f"{total_gb:.1f} GB")
    colour = "normal" if pct < 80 else "inverse"
    st.progress(int(pct), text=f"{pct:.0f}% in use")
    if pct > 85:
        st.warning(
            f"Memory pressure is high ({pct:.0f}%). Ollama may be offloading "
            "model layers to CPU, causing slow TTFT. Consider using a smaller model "
            "or closing other apps."
        )

    # Ollama loaded models
    st.markdown("### Ollama GPU state")
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
        )
        st.code(result.stdout or "(no models loaded)", language="text")
    except Exception as exc:
        st.caption(f"Could not run ollama ps: {exc}")

# ===========================================================================
# CHAT tab
# ===========================================================================
with tab_chat:
    st.subheader("Ask a question")

    # Chat history stored in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Controls
    with st.expander("⚙️ Chat options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            session_id = st.text_input(
                "Session ID",
                value="streamlit-session",
                help="Enables conversation memory across questions",
            )
            top_k = st.slider("Top-K chunks", min_value=1, max_value=20, value=5)
        with col2:
            source_filter = st.text_input(
                "Filter by source file",
                placeholder="e.g. api-reference.md",
                help="Leave blank to search all documents",
            )
            use_stream = st.checkbox("Stream response", value=True)

    # Display history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))
            if msg.get("cache_hit"):
                st.caption("⚡ Cache hit")
            _render_timing(msg.get("timing"), msg.get("elapsed"))

    # Input
    if question := st.chat_input("Ask a question about your docs…"):
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["messages"].append({"role": "user", "content": question})

        # Build request
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-Id"] = session_id
        payload: dict = {
            "question": question,
            "stream": use_stream,
            "top_k": top_k,
        }
        if source_filter.strip():
            payload["filters"] = {"source_file": source_filter.strip()}

        # Stream or non-stream
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            sources: list[str] = []
            cache_hit = False
            elapsed = None
            timing = None

            if use_stream:
                # SSE streaming
                collected: list[str] = []
                try:
                    with requests.post(
                        f"{base_url()}/chat",
                        json=payload,
                        headers=headers,
                        stream=True,
                        timeout=120,
                    ) as r:
                        r.raise_for_status()
                        for raw_line in r.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            if "token" in event:
                                collected.append(event["token"])
                                answer_placeholder.markdown("".join(collected) + "▌")
                            if event.get("done"):
                                sources = event.get("sources", [])
                                cache_hit = event.get("cache_hit", False)
                                timing = event.get("timing")
                                break
                    answer_placeholder.markdown("".join(collected))
                    full_answer = "".join(collected)
                except Exception as exc:
                    st.error(f"Streaming error: {exc}")
                    full_answer = ""
            else:
                # Non-streaming
                try:
                    r = requests.post(
                        f"{base_url()}/chat",
                        json=payload,
                        headers=headers,
                        timeout=120,
                    )
                    r.raise_for_status()
                    result = r.json()
                    full_answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    cache_hit = result.get("cache_hit", False)
                    elapsed = result.get("elapsed_seconds")
                    answer_placeholder.markdown(full_answer)
                except Exception as exc:
                    st.error(f"Request error: {exc}")
                    full_answer = ""

            if sources:
                st.caption("Sources: " + ", ".join(sources))
            if cache_hit:
                st.caption("⚡ Cache hit")
            _render_timing(timing, elapsed)

        # Save to history
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
                "cache_hit": cache_hit,
                "timing": timing,
                "elapsed": elapsed,
            }
        )

    if st.session_state["messages"] and st.button("🗑 Clear chat history", key="clear_chat"):
        st.session_state["messages"] = []
        st.rerun()

# ===========================================================================
# INGEST tab
# ===========================================================================
with tab_ingest:
    st.subheader("Upload a document")
    st.markdown(
        "Supported formats: **Markdown** (`.md`), **plain text** (`.txt`), "
        "**HTML** (`.html`, `.htm`), **PDF** (`.pdf`)"
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["md", "txt", "html", "htm", "pdf"],
        key="ingest_file",
    )
    source_name = st.text_input(
        "Source name (optional)",
        placeholder="e.g. API Reference",
        help="Label stored with each chunk — defaults to the filename",
    )

    if st.button("📤 Upload & Ingest", disabled=uploaded_file is None):
        with st.spinner("Ingesting…"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                data = {}
                if source_name.strip():
                    data["source_name"] = source_name.strip()

                resp = requests.post(
                    f"{base_url()}/ingest",
                    files=files,
                    data=data,
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                st.success(
                    f"✅ Ingested **{result['source_file']}** — "
                    f"{result['chunks_processed']} chunks in {result['elapsed_seconds']:.2f}s"
                )
                with st.expander("Full response"):
                    st.json(result)
            except requests.HTTPError as exc:
                st.error(f"HTTP {exc.response.status_code}: {exc.response.text}")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ===========================================================================
# METRICS tab
# ===========================================================================
with tab_metrics:
    st.subheader("Runtime metrics")

    col_refresh, col_auto = st.columns([1, 3])
    with col_refresh:
        refresh_metrics = st.button("🔄 Refresh", key="refresh_metrics")
    with col_auto:
        auto_refresh = st.checkbox("Auto-refresh every 5 s", value=False, key="auto_refresh")

    if refresh_metrics or auto_refresh:
        try:
            resp = requests.get(f"{base_url()}/metrics/json", timeout=5)
            resp.raise_for_status()
            m = resp.json()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total requests", m.get("total_requests", 0))
            c2.metric("Cache hits", m.get("cache_hits", 0))
            c3.metric("Cache misses", m.get("cache_misses", 0))
            c4.metric("Hit rate", f"{m.get('hit_rate', 0):.1%}")
            c5.metric("Avg latency", f"{m.get('avg_latency_seconds', 0):.2f}s")

            # Per-step timing breakdown
            last = m.get("last_request_timing")
            recent = m.get("recent_timings", [])

            if last:
                st.markdown("---")
                st.markdown("### ⏱ Last request — step breakdown")
                t1, t2, t3, t4, t5 = st.columns(5)
                t1.metric("Embed", f"{last.get('embed_secs', 0):.2f}s",
                          help="Time to embed the question with bge-m3")
                t2.metric("Retrieve", f"{last.get('retrieve_secs', 0):.2f}s",
                          help="Redis HNSW vector search time")
                t3.metric("TTFT", f"{last.get('ttft_secs') or 0:.2f}s",
                          help="Time to first token from LLM (streaming only)")
                t4.metric("Generate", f"{last.get('generate_secs', 0):.2f}s",
                          help="Total LLM inference time")
                t5.metric("Total", f"{last.get('total_secs', 0):.2f}s")

                if last.get("generate_secs", 0) > 15:
                    st.warning(
                        "**LLM is the bottleneck** (`generate` > 15s). Tips:\n"
                        "- Set `LLM_NUM_CTX=2048` in `.env` to reduce KV-cache size (fastest win)\n"
                        "- Switch to a smaller model: `LLM_MODEL=qwen2.5:7b` in `.env`\n"
                        "- Reduce `top_k` in Chat options to send less context to the LLM\n"
                        "- Ensure Ollama is using the GPU: check `ollama ps` in a terminal"
                    )
                elif last.get("embed_secs", 0) > 3:
                    st.warning(
                        "**Embedding is slow** (`embed` > 3s). Check Ollama is running "
                        "and the bge-m3 model is loaded (`ollama ps`)."
                    )
                else:
                    st.success("Timing looks healthy.")

            if recent:
                st.markdown("---")
                st.markdown("### 📈 Recent requests (last 10)")
                import pandas as pd
                df = pd.DataFrame(recent)[
                    ["embed_secs", "retrieve_secs", "ttft_secs", "generate_secs", "total_secs"]
                ].rename(columns={
                    "embed_secs": "Embed (s)",
                    "retrieve_secs": "Retrieve (s)",
                    "ttft_secs": "TTFT (s)",
                    "generate_secs": "Generate (s)",
                    "total_secs": "Total (s)",
                })
                st.dataframe(df.style.format("{:.2f}"), width='stretch')
                st.bar_chart(df[["Embed (s)", "Retrieve (s)", "Generate (s)"]])

            with st.expander("Prometheus text format"):
                prom_resp = requests.get(f"{base_url()}/metrics", timeout=5)
                st.code(prom_resp.text, language="text")

            with st.expander("Raw JSON"):
                st.json(m)

        except Exception as exc:
            st.error(f"Could not fetch metrics: {exc}")

        if auto_refresh:
            time.sleep(5)
            st.rerun()
    else:
        st.info("Click **Refresh** or enable auto-refresh to load metrics.")

# ===========================================================================
# ADMIN tab
# ===========================================================================
with tab_admin:
    st.subheader("Administration")

    # --- Cache flush ---
    st.markdown("### 🗑 Flush semantic cache")
    st.caption("Deletes all cached question/answer pairs. The document index is not affected.")

    if st.button("Flush cache", type="primary", key="flush_cache"):
        try:
            resp = requests.post(f"{base_url()}/cache/flush", timeout=10)
            resp.raise_for_status()
            result = resp.json()
            st.success(f"✅ Flushed {result.get('deleted', 0)} cache entries.")
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.markdown("---")

    # --- Docs flush ---
    st.markdown("### 📄 Flush document index")
    st.caption("Deletes all ingested document chunks. Use this to remove stale or duplicate chunks before re-ingesting.")

    if st.button("Flush documents", type="primary", key="flush_docs"):
        try:
            resp = requests.post(f"{base_url()}/docs/flush", timeout=10)
            resp.raise_for_status()
            result = resp.json()
            st.success(f"✅ Flushed {result.get('deleted', 0)} document chunks.")
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.markdown("---")

    # --- Session clear ---
    st.markdown("### 🧹 Clear session history")
    st.caption("Deletes the conversation history for a given session ID.")

    session_to_clear = st.text_input(
        "Session ID to clear",
        placeholder="e.g. streamlit-session",
        key="session_to_clear",
    )

    if st.button("Clear session", disabled=not session_to_clear.strip(), key="clear_session_btn"):
        try:
            resp = requests.delete(
                f"{base_url()}/session/{session_to_clear.strip()}",
                timeout=10,
            )
            resp.raise_for_status()
            st.success(f"✅ Session '{session_to_clear}' cleared.")
        except Exception as exc:
            st.error(f"Error: {exc}")

    st.markdown("---")

    # --- Raw API explorer ---
    st.markdown("### 🔍 Raw API explorer")
    st.caption("Send a custom GET request and inspect the JSON response.")

    raw_path = st.text_input("Path", value="/health", key="raw_path", placeholder="/health")
    if st.button("Send GET", key="raw_get"):
        try:
            resp = requests.get(f"{base_url()}{raw_path}", timeout=10)
            st.code(f"HTTP {resp.status_code}", language="text")
            try:
                st.json(resp.json())
            except Exception:
                st.code(resp.text, language="text")
        except Exception as exc:
            st.error(f"Error: {exc}")
