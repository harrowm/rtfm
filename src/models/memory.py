# Session and long-term memory models
from __future__ import annotations

from pydantic import BaseModel


class Message(BaseModel):
    """A single turn in a conversation."""
    role: str   # "user" or "assistant"
    content: str


class SessionMemory(BaseModel):
    """Short-term memory: the conversation history for one session."""
    session_id: str
    messages: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def to_history_dicts(self) -> list[dict]:
        """Return history in the format expected by the RAG prompt builder."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_text(self) -> str:
        """Render the conversation as plain text for LLM memory extraction."""
        return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in self.messages)


class LongTermMemory(BaseModel):
    """Persistent facts extracted from past conversations."""
    session_id: str
    facts: list[str] = []

    def to_context_string(self) -> str:
        """Format facts as a bullet list for injection into the system prompt."""
        if not self.facts:
            return ""
        items = "\n".join(f"- {f}" for f in self.facts)
        return f"## What I know about you\n{items}"
