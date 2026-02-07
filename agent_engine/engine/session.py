"""Session store — ABC + in-memory implementation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from agent_engine.engine.models import Session


class SesStore(ABC):
    """Async session persistence interface.

    Swap to Redis/Postgres by implementing this ABC.
    """

    @abstractmethod
    async def get(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def save(self, session: Session) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...


class InMemorySesStore(SesStore):
    """Dict-backed store — suitable for single-process dev/test."""

    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self._store.get(session_id)

    async def save(self, session: Session) -> None:
        session.updated_at = time.time()
        self._store[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
