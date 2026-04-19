from typing import Optional
from pydantic import BaseModel


# ── Requests ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


# ── Responses ───────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    interrupted: bool = False           # True when graph is paused for human input
    interrupt_message: Optional[str] = None  # The question/prompt shown to user


class SessionResponse(BaseModel):
    session_id: str


class MessageItem(BaseModel):
    role: str       # "user" | "assistant" | "tool"
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]
