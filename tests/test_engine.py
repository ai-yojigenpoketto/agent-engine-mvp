"""Tests for AgentEngine — tool-call roundtrip and end-to-end flow."""

from __future__ import annotations

import pytest

from agent_engine.engine.agent import AgentEngine
from agent_engine.engine.llm import MockLLMClient
from agent_engine.engine.models import (
    EngineEventType,
    EventEnvelope,
    LLMResult,
    ToolCallRequest,
)


class TestEngineToolCallRoundtrip:
    """Requirement: test_engine_loop_tool_call_roundtrip."""

    async def test_tool_call_then_final(
        self, ses_store, tool_registry, memory, router, trace_collector
    ):
        mock_llm = MockLLMClient([
            # Iteration 0: LLM requests log_search
            LLMResult(tool_calls=[
                ToolCallRequest(id="tc-1", name="log_search", arguments={"query": "GPU error"}),
            ]),
            # Iteration 1: LLM returns final content
            LLMResult(content="GPU1 has ECC errors. Check VRAM health."),
        ])

        engine = AgentEngine(
            ses_store=ses_store,
            tool_registry=tool_registry,
            memory=memory,
            router=router,
            llm_client=mock_llm,
            trace_collector=trace_collector,
        )

        envelope = EventEnvelope(
            text="/gpu GPU error on node-5",
            session_id="test-session",
        )

        events = [e async for e in engine.handle(envelope)]
        types = [e.type for e in events]

        # Must contain these event types in order
        assert EngineEventType.RETRIEVE in types
        assert EngineEventType.TOOL_CALL in types
        assert EngineEventType.TOOL_RESULT in types
        assert EngineEventType.TOKEN in types
        assert EngineEventType.FINAL in types

        # Final event should carry the answer text
        final = next(e for e in events if e.type == EngineEventType.FINAL)
        assert "ECC errors" in final.data["text"]

        # Tool result should contain log_search output
        tool_result = next(e for e in events if e.type == EngineEventType.TOOL_RESULT)
        assert tool_result.data["name"] == "log_search"
        assert "results" in tool_result.data["result"]

        # LLM should have been called exactly twice
        assert mock_llm.call_count == 2

    async def test_session_persists_messages(
        self, ses_store, tool_registry, memory, router, trace_collector
    ):
        mock_llm = MockLLMClient([
            LLMResult(content="Hello!"),
        ])

        engine = AgentEngine(
            ses_store=ses_store,
            tool_registry=tool_registry,
            memory=memory,
            router=router,
            llm_client=mock_llm,
            trace_collector=trace_collector,
        )

        envelope = EventEnvelope(text="hi", session_id="persist-test")
        _ = [e async for e in engine.handle(envelope)]

        session = await ses_store.get("persist-test")
        assert session is not None
        assert len(session.messages) >= 3  # system + user + assistant
        assert session.selected_skill == "doc_qa"

    async def test_max_iterations_error(
        self, ses_store, tool_registry, memory, router, trace_collector
    ):
        """If the LLM keeps requesting tools and never returns content."""
        # Return tool calls for every iteration (more than MAX_ITERATIONS)
        responses = [
            LLMResult(tool_calls=[
                ToolCallRequest(id=f"tc-{i}", name="log_search", arguments={"query": "loop"}),
            ])
            for i in range(10)
        ]
        mock_llm = MockLLMClient(responses)

        engine = AgentEngine(
            ses_store=ses_store,
            tool_registry=tool_registry,
            memory=memory,
            router=router,
            llm_client=mock_llm,
            trace_collector=trace_collector,
        )

        envelope = EventEnvelope(text="/gpu infinite loop test", session_id="loop-test")
        events = [e async for e in engine.handle(envelope)]
        types = [e.type for e in events]

        assert EngineEventType.ERROR in types
        error_event = [e for e in events if e.type == EngineEventType.ERROR][-1]
        assert "Max iterations" in error_event.data.get("error", "")
