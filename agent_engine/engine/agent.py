"""AgentEngine — the core runtime loop."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

from agent_engine.engine.llm import LLMClient
from agent_engine.engine.models import (
    EngineEvent,
    EngineEventType,
    EventEnvelope,
    Session,
)
from agent_engine.engine.session import SesStore
from agent_engine.memory.interface import Memory
from agent_engine.skills.router import Router
from agent_engine.tools.registry import ToolRegistry
from agent_engine.tracing.interface import TraceCollector

logger = logging.getLogger(__name__)


class AgentEngine:
    """Public API: ``async for event in engine.handle(envelope): ...``"""

    MAX_ITERATIONS: int = 6
    # TODO: token budget placeholder — enforce per-request token ceiling

    def __init__(
        self,
        ses_store: SesStore,
        tool_registry: ToolRegistry,
        memory: Memory,
        router: Router,
        llm_client: LLMClient,
        trace_collector: TraceCollector,
    ) -> None:
        self._ses = ses_store
        self._tools = tool_registry
        self._memory = memory
        self._router = router
        self._llm = llm_client
        self._trace = trace_collector

    # ------------------------------------------------------------------
    # Public handle
    # ------------------------------------------------------------------

    async def handle(self, envelope: EventEnvelope) -> AsyncIterator[EngineEvent]:
        trace_id = envelope.trace_id
        t_start = time.time()

        # 1. Session ---------------------------------------------------
        session = await self._ses.get(envelope.session_id)
        if session is None:
            session = Session(
                session_id=envelope.session_id,
                tenant_id=envelope.tenant_id,
                user_id=envelope.user_id,
                trace_id=trace_id,
            )

        # 2. Route -----------------------------------------------------
        skill, cleaned_text = self._router.route(envelope.text)
        session.selected_skill = skill.name
        await self._trace.emit(trace_id, "route", {
            "skill": skill.name,
            "original_text": envelope.text,
            "cleaned_text": cleaned_text,
        })

        # 3. Pre-retrieve (RAG) ----------------------------------------
        t_ret = time.time()
        chunks = await skill.pre_retrieve(cleaned_text, self._memory)
        if chunks:
            await self._trace.emit(trace_id, "retrieve", {
                "count": len(chunks),
                "latency_ms": round((time.time() - t_ret) * 1000, 2),
            })
            yield EngineEvent(
                type=EngineEventType.RETRIEVE,
                data={"chunks": [c.model_dump() for c in chunks]},
                trace_id=trace_id,
            )

        # 4. Build messages --------------------------------------------
        messages = list(session.messages)  # copy history

        system_content = skill.system_prompt()
        if chunks:
            ctx = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
            system_content += f"\n\nRelevant context:\n{ctx}"

        # Upsert system message at index 0
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": system_content}
        else:
            messages.insert(0, {"role": "system", "content": system_content})

        messages.append({"role": "user", "content": cleaned_text})

        # 5. Tool schemas for this role + skill -------------------------
        tool_schemas = self._tools.openai_schemas(
            role=envelope.role,
            allowed_tools=skill.allowed_tools(),
        )

        # 6. LLM loop --------------------------------------------------
        for iteration in range(self.MAX_ITERATIONS):
            t_llm = time.time()
            result = await self._llm.generate(
                messages,
                tools=tool_schemas or None,
            )
            llm_latency = time.time() - t_llm
            await self._trace.emit(trace_id, "llm_call", {
                "iteration": iteration,
                "latency_ms": round(llm_latency * 1000, 2),
                "has_tool_calls": bool(result.tool_calls),
            })

            # -- tool calls ---------------------------------------------
            if result.tool_calls:
                # Append assistant message with tool_calls
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in result.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for tc in result.tool_calls:
                    yield EngineEvent(
                        type=EngineEventType.TOOL_CALL,
                        data={"id": tc.id, "name": tc.name, "arguments": tc.arguments},
                        trace_id=trace_id,
                    )

                    try:
                        tool_out = await self._tools.execute(
                            name=tc.name,
                            role=envelope.role,
                            input_data=tc.arguments,
                            trace_collector=self._trace,
                            trace_id=trace_id,
                        )
                    except Exception as exc:
                        tool_out = {"error": str(exc)}
                        yield EngineEvent(
                            type=EngineEventType.ERROR,
                            data={"tool": tc.name, "error": str(exc)},
                            trace_id=trace_id,
                        )

                    yield EngineEvent(
                        type=EngineEventType.TOOL_RESULT,
                        data={"id": tc.id, "name": tc.name, "result": tool_out},
                        trace_id=trace_id,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_out),
                    })

                continue  # next LLM iteration

            # -- content (final answer) ---------------------------------
            if result.content:
                # Yield word-level token events (simulated streaming)
                words = result.content.split(" ")
                for i, word in enumerate(words):
                    token_text = word if i == len(words) - 1 else word + " "
                    yield EngineEvent(
                        type=EngineEventType.TOKEN,
                        data={"text": token_text},
                        trace_id=trace_id,
                    )

                yield EngineEvent(
                    type=EngineEventType.FINAL,
                    data={"text": result.content},
                    trace_id=trace_id,
                )
                messages.append({"role": "assistant", "content": result.content})
                break

        else:
            # Exhausted MAX_ITERATIONS without a final answer
            yield EngineEvent(
                type=EngineEventType.ERROR,
                data={"error": f"Max iterations ({self.MAX_ITERATIONS}) reached without final answer"},
                trace_id=trace_id,
            )

        # 7. Persist session -------------------------------------------
        session.messages = messages
        session.trace_id = trace_id
        await self._ses.save(session)

        # 8. Flush traces ----------------------------------------------
        await self._trace.emit(trace_id, "handle_done", {
            "total_latency_ms": round((time.time() - t_start) * 1000, 2),
        })
        await self._trace.flush(trace_id)
