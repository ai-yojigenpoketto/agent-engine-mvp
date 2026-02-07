"""Memory (RAG) interface — depends only on engine.models.DocChunk."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_engine.engine.models import DocChunk


class Memory(ABC):
    """Async retrieval interface.

    Swap to a real vector store by implementing this ABC.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        session_id: str | None = None,
    ) -> list[DocChunk]: ...
