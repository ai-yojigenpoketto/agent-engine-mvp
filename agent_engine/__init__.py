"""agent_engine — minimal agent runtime with session, tools, memory, and tracing.

Usage::

    from agent_engine import create_engine

    engine = create_engine()
    async for event in engine.handle(envelope):
        print(event)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ (no-op if file missing)

from agent_engine.engine.agent import AgentEngine
from agent_engine.engine.llm import DemoMockLLMClient, OpenAILLMClient
from agent_engine.engine.models import EngineEvent, EngineEventType, EventEnvelope
from agent_engine.engine.session import InMemorySesStore
from agent_engine.memory.in_memory import InMemoryMemory
from agent_engine.skills.doc_qa import DocQASkill
from agent_engine.skills.gpu_diagnosis import GPUDiagnosisSkill
from agent_engine.skills.router import Router
from agent_engine.tools.builtins import LOG_SEARCH_TOOL, make_kb_query_tool
from agent_engine.tools.registry import ToolRegistry
from agent_engine.tracing.jsonl_tracer import JSONLTraceCollector

__all__ = [
    "AgentEngine",
    "EngineEvent",
    "EngineEventType",
    "EventEnvelope",
    "create_engine",
]


def create_engine(
    *,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    trace_dir: str = "./traces",
    use_mock_llm: bool | None = None,
) -> AgentEngine:
    """Wire all components and return a ready-to-use AgentEngine.

    Environment variables (all optional):
      OPENAI_API_KEY   — required for real LLM calls
      OPENAI_MODEL     — default ``gpt-4o-mini``
      USE_MOCK_LLM     — set to ``1`` to use the demo mock
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    model = openai_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    mock = use_mock_llm if use_mock_llm is not None else os.environ.get("USE_MOCK_LLM") == "1"

    # -- components --
    ses_store = InMemorySesStore()
    memory = InMemoryMemory()
    trace_collector = JSONLTraceCollector(trace_dir)

    tool_registry = ToolRegistry()
    tool_registry.register(LOG_SEARCH_TOOL)
    tool_registry.register(make_kb_query_tool(memory))

    skills = {
        "doc_qa": DocQASkill(),
        "gpu_diagnosis": GPUDiagnosisSkill(),
    }
    router = Router(skills)

    if mock or not api_key:
        llm_client = DemoMockLLMClient()
    else:
        llm_client = OpenAILLMClient(api_key=api_key, model=model)

    return AgentEngine(
        ses_store=ses_store,
        tool_registry=tool_registry,
        memory=memory,
        router=router,
        llm_client=llm_client,
        trace_collector=trace_collector,
    )
