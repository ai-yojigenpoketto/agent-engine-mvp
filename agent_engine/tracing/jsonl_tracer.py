"""JSONL file-based trace collector."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_engine.tracing.interface import TraceCollector


class JSONLTraceCollector(TraceCollector):
    """Writes trace events to ``./traces/{trace_id}.jsonl``.

    Events are buffered in memory and flushed at the end of each
    ``AgentEngine.handle()`` call.
    """

    def __init__(self, trace_dir: str = "./traces") -> None:
        self._dir = Path(trace_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, list[dict[str, Any]]] = {}

    async def emit(self, trace_id: str, event_type: str, data: dict[str, Any]) -> None:
        entry = {
            "ts": time.time(),
            "trace_id": trace_id,
            "event": event_type,
            **data,
        }
        self._buffers.setdefault(trace_id, []).append(entry)

    async def flush(self, trace_id: str) -> None:
        entries = self._buffers.pop(trace_id, [])
        if not entries:
            return
        path = self._dir / f"{trace_id}.jsonl"
        with open(path, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
