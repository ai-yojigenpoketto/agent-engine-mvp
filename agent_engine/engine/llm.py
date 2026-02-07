"""LLM client — ABC, OpenAI implementation, and mocks."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from agent_engine.engine.models import LLMResult, ToolCallRequest

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM interface. Returns a complete (non-streaming) response.

    Token-level streaming is simulated in the engine by splitting content
    into word chunks — keeps this interface simple and testable.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult: ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        # Late import so the rest of the package works without openai installed
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if choice.message.tool_calls:
            tc_list = [
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]
            return LLMResult(tool_calls=tc_list)

        return LLMResult(content=choice.message.content or "")


# ---------------------------------------------------------------------------
# Test mock — deterministic, pre-loaded responses
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Returns pre-configured responses in order. Used in unit tests."""

    def __init__(self, responses: list[LLMResult]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    async def generate(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        if self._call_index >= len(self._responses):
            return LLMResult(content="[mock responses exhausted]")
        result = self._responses[self._call_index]
        self._call_index += 1
        return result

    @property
    def call_count(self) -> int:
        return self._call_index


# ---------------------------------------------------------------------------
# Demo mock — context-aware, for running without an API key
# ---------------------------------------------------------------------------

class DemoMockLLMClient(LLMClient):
    """Demonstrates the full tool-calling loop without a real LLM.

    Behaviour:
    1. If the last message is a tool result → return a summary.
    2. If tools are available → call the first tool with the user query.
    3. Otherwise → return a generic text response.
    """

    async def generate(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        last = messages[-1] if messages else {}

        # After a tool result, produce a final answer
        if last.get("role") == "tool":
            content = last.get("content", "")
            try:
                parsed = json.loads(content)
                # For gpu_diagnosis skill, return structured JSON
                if any("gpu" in m.get("content", "").lower() for m in messages if m.get("role") == "system"):
                    return LLMResult(content=json.dumps({
                        "summary": "Mock diagnosis based on retrieved data.",
                        "evidence": list(parsed.get("results", parsed.get("snippets", ["no data"]))),
                        "next_steps": ["Check GPU hardware", "Update drivers", "Review thermal configuration"],
                    }, indent=2))
            except (json.JSONDecodeError, AttributeError):
                pass
            return LLMResult(content=f"Based on the gathered information: {content[:200]}")

        # If tools are available, call the first one
        if tools:
            tool_name = tools[0]["function"]["name"]
            user_text = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user" and m.get("content")),
                "query",
            )
            return LLMResult(tool_calls=[
                ToolCallRequest(id="demo-tc-1", name=tool_name, arguments={"query": user_text}),
            ])

        return LLMResult(content="This is a demo response. Set OPENAI_API_KEY for real LLM output.")
