# Tests for session and long-term memory
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

import src.services.ollama_client as _ollama_mod

from src.models.memory import LongTermMemory, SessionMemory
from src.services.memory import (
    clear_session,
    extract_and_save_memories,
    load_long_term_memory,
    load_session,
    save_message,
)


# ---------------------------------------------------------------------------
# Model unit tests (pure, no I/O)
# ---------------------------------------------------------------------------

class TestSessionMemoryModel:
    def test_add_message(self):
        session = SessionMemory(session_id="s1")
        session.add("user", "Hello")
        session.add("assistant", "Hi there!")
        assert len(session.messages) == 2

    def test_to_history_dicts(self):
        session = SessionMemory(session_id="s1")
        session.add("user", "Hi")
        assert session.to_history_dicts() == [{"role": "user", "content": "Hi"}]

    def test_to_text(self):
        session = SessionMemory(session_id="s1")
        session.add("user", "Question")
        session.add("assistant", "Answer")
        text = session.to_text()
        assert "User: Question" in text
        assert "Assistant: Answer" in text


class TestLongTermMemoryModel:
    def test_to_context_string_empty(self):
        assert LongTermMemory(session_id="s1").to_context_string() == ""

    def test_to_context_string_with_facts(self):
        ltm = LongTermMemory(session_id="s1", facts=["Prefers Python", "Uses Redis"])
        ctx = ltm.to_context_string()
        assert "Prefers Python" in ctx
        assert ctx.startswith("## What I know")


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_save_message_calls_rpush(self, mock_redis):
        await save_message("sess-1", "user", "Hello")
        mock_redis.rpush.assert_called_once()
        payload = json.loads(mock_redis.rpush.call_args.args[1])
        assert payload == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_save_message_sets_ttl(self, mock_redis):
        await save_message("sess-1", "user", "ping")
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_session_deserializes_messages(self, mock_redis):
        mock_redis.lrange = AsyncMock(return_value=[
            json.dumps({"role": "user", "content": "Hello"}),
            json.dumps({"role": "assistant", "content": "Hi back"}),
        ])
        session = await load_session("sess-1")
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].content == "Hi back"

    @pytest.mark.asyncio
    async def test_load_empty_session(self, mock_redis):
        mock_redis.lrange = AsyncMock(return_value=[])
        session = await load_session("nonexistent")
        assert session.messages == []

    @pytest.mark.asyncio
    async def test_clear_session_deletes_key(self, mock_redis):
        await clear_session("sess-1")
        mock_redis.delete.assert_called_once()
        assert "sess-1" in mock_redis.delete.call_args.args[0]


# ---------------------------------------------------------------------------
# Long-term memory persistence
# ---------------------------------------------------------------------------

class TestLongTermMemoryPersistence:
    @pytest.mark.asyncio
    async def test_load_facts_from_redis(self, mock_redis):
        mock_redis.lrange = AsyncMock(return_value=["Prefers dark mode", "Uses Mac M4"])
        ltm = await load_long_term_memory("sess-1")
        assert "Prefers dark mode" in ltm.facts
        assert "Uses Mac M4" in ltm.facts

    @pytest.mark.asyncio
    async def test_extract_and_save_memories(self, mock_redis):
        session = SessionMemory(session_id="sess-1")
        session.add("user", "I prefer Python and use a Mac M4.")
        session.add("assistant", "Got it, I'll keep Python examples.")

        with patch.object(
            _ollama_mod, "generate",
            new=AsyncMock(return_value='["Prefers Python", "Uses Mac M4"]'),
        ):
            facts = await extract_and_save_memories("sess-1", session)

        assert "Prefers Python" in facts
        assert "Uses Mac M4" in facts
        assert mock_redis.rpush.called

    @pytest.mark.asyncio
    async def test_extract_skips_single_message_sessions(self, mock_redis):
        session = SessionMemory(session_id="sess-1")
        session.add("user", "Hi")  # only 1 message — threshold not met
        facts = await extract_and_save_memories("sess-1", session)
        assert facts == []
        mock_redis.rpush.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_handles_malformed_llm_response(self, mock_redis):
        session = SessionMemory(session_id="sess-1")
        session.add("user", "Something")
        session.add("assistant", "Something else")

        with patch.object(
            _ollama_mod, "generate",
            new=AsyncMock(return_value="Sorry, I cannot extract facts."),
        ):
            facts = await extract_and_save_memories("sess-1", session)

        assert facts == []
        mock_redis.rpush.assert_not_called()
