"""Shared fixtures for agent_engine tests."""

from __future__ import annotations

import pytest

from agent_engine.engine.session import InMemorySesStore
from agent_engine.memory.in_memory import InMemoryMemory
from agent_engine.skills.doc_qa import DocQASkill
from agent_engine.skills.gpu_diagnosis import GPUDiagnosisSkill
from agent_engine.skills.router import Router
from agent_engine.tools.builtins import LOG_SEARCH_TOOL, make_kb_query_tool
from agent_engine.tools.registry import ToolRegistry
from agent_engine.tracing.jsonl_tracer import JSONLTraceCollector


@pytest.fixture
def memory():
    return InMemoryMemory()


@pytest.fixture
def tool_registry(memory):
    registry = ToolRegistry()
    registry.register(LOG_SEARCH_TOOL)
    registry.register(make_kb_query_tool(memory))
    return registry


@pytest.fixture
def router():
    skills = {
        "doc_qa": DocQASkill(),
        "gpu_diagnosis": GPUDiagnosisSkill(),
    }
    return Router(skills)


@pytest.fixture
def ses_store():
    return InMemorySesStore()


@pytest.fixture
def trace_collector(tmp_path):
    return JSONLTraceCollector(trace_dir=str(tmp_path / "traces"))
