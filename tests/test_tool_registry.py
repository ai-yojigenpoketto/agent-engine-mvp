"""Tests for ToolRegistry — permissions, execution, timeout, schemas."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from agent_engine.tools.registry import ToolDef, ToolRegistry


# -- helpers ----------------------------------------------------------------

class EchoInput(BaseModel):
    msg: str


class EchoOutput(BaseModel):
    echo: str


async def _echo_handler(inp: EchoInput) -> dict:
    return {"echo": inp.msg}


async def _slow_handler(inp: EchoInput) -> dict:
    await asyncio.sleep(5)
    return {"echo": inp.msg}


def _make_echo_tool(**overrides) -> ToolDef:
    defaults = dict(
        name="echo",
        description="Echoes input",
        input_model=EchoInput,
        output_model=EchoOutput,
        handler=_echo_handler,
        allowed_roles={"user", "operator", "admin"},
    )
    defaults.update(overrides)
    return ToolDef(**defaults)


# -- tests ------------------------------------------------------------------

class TestToolRegistryPermissions:
    """Requirement: test_tool_registry_permissions."""

    async def test_user_blocked_from_admin_tool(self):
        registry = ToolRegistry()
        registry.register(_make_echo_tool(name="admin_only", allowed_roles={"admin"}))

        with pytest.raises(PermissionError, match="not allowed"):
            await registry.execute("admin_only", "user", {"msg": "hi"})

    async def test_admin_can_use_admin_tool(self):
        registry = ToolRegistry()
        registry.register(_make_echo_tool(name="admin_only", allowed_roles={"admin"}))

        result = await registry.execute("admin_only", "admin", {"msg": "hi"})
        assert result == {"echo": "hi"}

    async def test_operator_role(self):
        registry = ToolRegistry()
        registry.register(_make_echo_tool(name="ops_tool", allowed_roles={"operator", "admin"}))

        result = await registry.execute("ops_tool", "operator", {"msg": "check"})
        assert result["echo"] == "check"

        with pytest.raises(PermissionError):
            await registry.execute("ops_tool", "user", {"msg": "check"})

    async def test_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="not found"):
            await registry.execute("nonexistent", "admin", {"msg": "hi"})


class TestToolExecution:
    async def test_basic_execution(self, tool_registry):
        result = await tool_registry.execute("log_search", "user", {"query": "GPU"})
        assert "results" in result
        assert isinstance(result["results"], list)

    async def test_kb_query_returns_snippets(self, tool_registry):
        result = await tool_registry.execute("kb_query", "user", {"query": "GPU temperature"})
        assert "snippets" in result
        assert len(result["snippets"]) > 0

    async def test_timeout(self):
        registry = ToolRegistry()
        registry.register(_make_echo_tool(name="slow", handler=_slow_handler, timeout=0.1))

        with pytest.raises(asyncio.TimeoutError):
            await registry.execute("slow", "user", {"msg": "hi"})

    async def test_retry_on_failure(self):
        call_count = 0

        async def _flaky(inp: EchoInput) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient failure")
            return {"echo": inp.msg}

        registry = ToolRegistry()
        registry.register(_make_echo_tool(name="flaky", handler=_flaky, max_retries=2))

        result = await registry.execute("flaky", "user", {"msg": "ok"})
        assert result == {"echo": "ok"}
        assert call_count == 2


class TestOpenAISchemas:
    def test_schemas_filter_by_role_and_allowlist(self, tool_registry):
        schemas = tool_registry.openai_schemas(role="user", allowed_tools=["log_search"])
        names = [s["function"]["name"] for s in schemas]
        assert "log_search" in names
        assert "kb_query" not in names

    def test_schemas_empty_when_no_match(self, tool_registry):
        schemas = tool_registry.openai_schemas(role="user", allowed_tools=["nonexistent"])
        assert schemas == []
