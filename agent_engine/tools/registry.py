"""Tool registry with Pydantic v2 schemas, RBAC, timeout, retry, and audit."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from agent_engine.tracing.interface import TraceCollector

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Registration record for a single tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[..., Awaitable[Any]]
    allowed_roles: set[str] = field(default_factory=lambda: {"user", "operator", "admin"})
    timeout: float = 30.0
    max_retries: int = 1


class ToolRegistry:
    """Central tool store with RBAC, timeout/retry, and tracing hooks."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    # -- registration -------------------------------------------------------

    def register(self, tool_def: ToolDef) -> None:
        self._tools[tool_def.name] = tool_def
        logger.info("Registered tool %s (roles=%s)", tool_def.name, tool_def.allowed_roles)

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_for_role(self, role: str) -> list[ToolDef]:
        return [t for t in self._tools.values() if role in t.allowed_roles]

    # -- OpenAI function-calling schemas ------------------------------------

    def openai_schemas(self, role: str, allowed_tools: list[str]) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas filtered by role + skill allowlist."""
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            if role not in tool.allowed_roles:
                continue
            if tool.name not in allowed_tools:
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_model.model_json_schema(),
                },
            })
        return schemas

    # -- execution ----------------------------------------------------------

    async def execute(
        self,
        name: str,
        role: str,
        input_data: dict[str, Any],
        trace_collector: TraceCollector | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        if role not in tool.allowed_roles:
            raise PermissionError(f"Role '{role}' is not allowed to use tool '{name}'")

        validated_input = tool.input_model(**input_data)

        last_exc: Exception | None = None
        for attempt in range(1, tool.max_retries + 2):  # +2 because range is exclusive
            try:
                t0 = time.time()
                raw = await asyncio.wait_for(
                    tool.handler(validated_input),
                    timeout=tool.timeout,
                )
                latency = time.time() - t0

                # Validate output
                if isinstance(raw, dict):
                    validated = tool.output_model(**raw)
                elif isinstance(raw, BaseModel):
                    validated = raw
                else:
                    validated = tool.output_model.model_validate(raw)

                logger.info(
                    "tool=%s attempt=%d latency=%.3fs OK",
                    name, attempt, latency,
                )
                if trace_collector and trace_id:
                    await trace_collector.emit(trace_id, "tool_exec", {
                        "tool": name,
                        "attempt": attempt,
                        "latency_ms": round(latency * 1000, 2),
                        "status": "ok",
                    })

                return validated.model_dump()

            except Exception as exc:
                last_exc = exc
                logger.warning("tool=%s attempt=%d error=%s", name, attempt, exc)
                if trace_collector and trace_id:
                    await trace_collector.emit(trace_id, "tool_exec", {
                        "tool": name,
                        "attempt": attempt,
                        "status": "error",
                        "error": str(exc),
                    })

        raise last_exc  # type: ignore[misc]
