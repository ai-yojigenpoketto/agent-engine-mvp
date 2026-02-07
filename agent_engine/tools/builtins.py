"""Built-in example tools: log_search and kb_query."""

from __future__ import annotations

from pydantic import BaseModel

from agent_engine.memory.interface import Memory
from agent_engine.tools.registry import ToolDef


# ---------------------------------------------------------------------------
# log_search — searches mock infrastructure logs
# ---------------------------------------------------------------------------

class LogSearchInput(BaseModel):
    query: str


class LogSearchOutput(BaseModel):
    results: list[str]


_MOCK_LOGS = [
    "2024-01-15 10:01 GPU0: temperature 85C, utilization 98%",
    "2024-01-15 10:02 GPU1: ECC error detected, count=3",
    "2024-01-15 10:03 GPU0: CUDA OOM at batch_size=128",
    "2024-01-15 10:04 GPU2: driver version mismatch warning",
    "2024-01-15 10:05 GPU1: NVLink error, peer GPU3 unreachable",
    "2024-01-15 10:06 GPU3: fan speed 0 RPM — possible failure",
    "2024-01-15 10:07 GPU0: Xid 79 — GPU has fallen off the bus",
]


async def _log_search_handler(inp: LogSearchInput) -> dict:
    words = inp.query.lower().split()
    hits = [log for log in _MOCK_LOGS if any(w in log.lower() for w in words)]
    return {"results": hits or ["No matching logs found."]}


LOG_SEARCH_TOOL = ToolDef(
    name="log_search",
    description="Search infrastructure logs for GPU/system events matching a query.",
    input_model=LogSearchInput,
    output_model=LogSearchOutput,
    handler=_log_search_handler,
    allowed_roles={"user", "operator", "admin"},
)


# ---------------------------------------------------------------------------
# kb_query — calls Memory.retrieve and returns snippets
# ---------------------------------------------------------------------------

class KBQueryInput(BaseModel):
    query: str


class KBQueryOutput(BaseModel):
    snippets: list[str]


def make_kb_query_tool(memory: Memory) -> ToolDef:
    """Factory — binds a *Memory* instance into the tool handler."""

    async def _kb_query_handler(inp: KBQueryInput) -> dict:
        chunks = await memory.retrieve(inp.query, k=3)
        return {"snippets": [c.text for c in chunks]}

    return ToolDef(
        name="kb_query",
        description="Query the knowledge base for relevant documentation snippets.",
        input_model=KBQueryInput,
        output_model=KBQueryOutput,
        handler=_kb_query_handler,
        allowed_roles={"user", "operator", "admin"},
    )
