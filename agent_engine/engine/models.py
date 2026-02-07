"""Core data models — no internal dependencies, only Pydantic + stdlib."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound envelope (adapter → engine)
# ---------------------------------------------------------------------------

class EventEnvelope(BaseModel):
    """Adapter-agnostic incoming request."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    user_id: str = "anonymous"
    role: str = "user"  # user | operator | admin
    text: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Outbound events (engine → adapter)
# ---------------------------------------------------------------------------

class EngineEventType(str, Enum):
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRIEVE = "retrieve"
    TRACE = "trace"
    FINAL = "final"
    ERROR = "error"


class EngineEvent(BaseModel):
    type: EngineEventType
    data: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = ""
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session(BaseModel):
    session_id: str
    tenant_id: str = "default"
    user_id: str = "anonymous"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    selected_skill: str | None = None
    trace_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# RAG chunk
# ---------------------------------------------------------------------------

class DocChunk(BaseModel):
    id: str
    text: str
    source: str
    score: float


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    """A single tool/function call requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


class LLMResult(BaseModel):
    """Complete (non-streaming) LLM response."""
    content: str | None = None
    tool_calls: list[ToolCallRequest] | None = None
