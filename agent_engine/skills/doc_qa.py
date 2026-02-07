"""doc_qa skill — general documentation Q&A with kb_query."""

from __future__ import annotations

from agent_engine.engine.models import DocChunk
from agent_engine.memory.interface import Memory
from agent_engine.skills.interface import Skill


class DocQASkill(Skill):
    @property
    def name(self) -> str:
        return "doc_qa"

    def system_prompt(self) -> str:
        return (
            "You are a helpful documentation assistant. Answer questions based on "
            "the provided context. If the context is insufficient, say so. "
            "Use the kb_query tool to search for additional information if needed."
        )

    def allowed_tools(self) -> list[str]:
        return ["kb_query"]

    async def pre_retrieve(self, text: str, memory: Memory) -> list[DocChunk]:
        return await memory.retrieve(text, k=3)
