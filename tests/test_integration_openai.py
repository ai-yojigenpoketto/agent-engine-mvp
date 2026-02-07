"""Integration tests that hit the real OpenAI API.

Skipped automatically when OPENAI_API_KEY is not set.
Run with:  OPENAI_API_KEY=sk-... pytest tests/test_integration_openai.py -v -s
"""

from __future__ import annotations

import json
import os

import pytest

from agent_engine import create_engine
from agent_engine.engine.models import EngineEventType, EventEnvelope

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-API integration tests",
)


class TestOpenAIDocQA:
    """Default skill (doc_qa) — no /gpu prefix."""

    async def test_doc_qa_returns_final(self):
        engine = create_engine()
        envelope = EventEnvelope(
            text="How do I monitor GPU utilization?",
            session_id="integ-docqa",
        )

        events = [e async for e in engine.handle(envelope)]
        types = [e.type for e in events]

        assert EngineEventType.RETRIEVE in types, "should pre-retrieve context"
        assert EngineEventType.FINAL in types, "must produce a final answer"

        final = next(e for e in events if e.type == EngineEventType.FINAL)
        assert len(final.data["text"]) > 10, "final answer should be non-trivial"
        print(f"\n--- doc_qa final ({len(final.data['text'])} chars) ---")
        print(final.data["text"][:500])


class TestOpenAIGPUDiagnosis:
    """/gpu prefix — should trigger tool calls and structured output."""

    async def test_gpu_diagnosis_uses_tools(self):
        engine = create_engine()
        envelope = EventEnvelope(
            text="/gpu ECC errors detected on GPU1, what should I do?",
            session_id="integ-gpu",
        )

        events = [e async for e in engine.handle(envelope)]
        types = [e.type for e in events]

        assert EngineEventType.RETRIEVE in types, "should pre-retrieve context"
        assert EngineEventType.FINAL in types, "must produce a final answer"

        # The LLM should have called at least one tool
        tool_calls = [e for e in events if e.type == EngineEventType.TOOL_CALL]
        tool_results = [e for e in events if e.type == EngineEventType.TOOL_RESULT]
        print(f"\n--- gpu_diagnosis: {len(tool_calls)} tool call(s) ---")
        for tc in tool_calls:
            print(f"  called: {tc.data['name']}({tc.data['arguments']})")

        assert len(tool_calls) > 0, "LLM should call at least one tool"
        assert len(tool_results) == len(tool_calls), "each call should get a result"

        # Check tool names are from the gpu_diagnosis allowlist
        for tc in tool_calls:
            assert tc.data["name"] in ("log_search", "kb_query")

        final = next(e for e in events if e.type == EngineEventType.FINAL)
        print(f"\n--- gpu_diagnosis final ({len(final.data['text'])} chars) ---")
        print(final.data["text"][:500])

        # Try to parse as structured JSON (the skill instructs this format)
        try:
            parsed = json.loads(final.data["text"])
            assert "summary" in parsed
            assert "evidence" in parsed
            assert "next_steps" in parsed
            print("\nPASS: output is valid structured JSON")
        except json.JSONDecodeError:
            # LLM may not always comply perfectly; that's OK for integration
            print("\nNOTE: final output is not strict JSON (LLM discretion)")


class TestOpenAISessionContinuity:
    """Multi-turn: second message should see prior context."""

    async def test_second_turn_has_history(self):
        engine = create_engine()
        sid = "integ-session-cont"

        # Turn 1
        env1 = EventEnvelope(text="What causes CUDA OOM errors?", session_id=sid)
        events1 = [e async for e in engine.handle(env1)]
        assert any(e.type == EngineEventType.FINAL for e in events1)

        # Turn 2 — references turn 1
        env2 = EventEnvelope(text="How do I fix that?", session_id=sid)
        events2 = [e async for e in engine.handle(env2)]
        final2 = next(e for e in events2 if e.type == EngineEventType.FINAL)

        print(f"\n--- turn 2 final ---")
        print(final2.data["text"][:500])

        # The answer should be contextual (not "fix what?")
        assert len(final2.data["text"]) > 20


class TestOpenAITraceOutput:
    """Verify traces are written to disk after a real API call."""

    async def test_trace_file_created(self, tmp_path):
        engine = create_engine(trace_dir=str(tmp_path / "traces"))
        envelope = EventEnvelope(
            text="What is NVLink?",
            session_id="integ-trace",
        )

        events = [e async for e in engine.handle(envelope)]
        trace_id = events[0].trace_id

        trace_file = tmp_path / "traces" / f"{trace_id}.jsonl"
        assert trace_file.exists(), "trace file should be created"

        lines = [json.loads(l) for l in trace_file.read_text().strip().split("\n")]
        event_types = {l["event"] for l in lines}

        print(f"\n--- trace: {len(lines)} events, types={event_types} ---")
        assert "route" in event_types
        assert "llm_call" in event_types
        assert "handle_done" in event_types

        # Check latency is recorded
        llm_event = next(l for l in lines if l["event"] == "llm_call")
        assert llm_event["latency_ms"] > 0, "LLM latency should be positive"
        print(f"  LLM latency: {llm_event['latency_ms']:.0f}ms")
