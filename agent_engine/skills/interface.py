"""Skill ABC — defines system prompt, tool allowlist, and optional pre-retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_engine.engine.models import DocChunk
from agent_engine.memory.interface import Memory


class Skill(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def allowed_tools(self) -> list[str]: ...

    async def pre_retrieve(self, text: str, memory: Memory) -> list[DocChunk]:
        """Optional RAG pre-fetch before the LLM call. Override to enable."""
        return []
