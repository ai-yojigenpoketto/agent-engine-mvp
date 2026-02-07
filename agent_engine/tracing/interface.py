"""TraceCollector ABC — no internal deps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TraceCollector(ABC):
    """Collects structured trace events for observability."""

    @abstractmethod
    async def emit(self, trace_id: str, event_type: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    async def flush(self, trace_id: str) -> None: ...
