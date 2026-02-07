"""gpu_diagnosis skill — structured GPU issue diagnosis."""

from __future__ import annotations

from agent_engine.engine.models import DocChunk
from agent_engine.memory.interface import Memory
from agent_engine.skills.interface import Skill


class GPUDiagnosisSkill(Skill):
    @property
    def name(self) -> str:
        return "gpu_diagnosis"

    def system_prompt(self) -> str:
        return (
            "You are a GPU infrastructure diagnosis expert. Analyze the user's "
            "GPU issue using available tools. Search logs with log_search and "
            "the knowledge base with kb_query to gather evidence.\n\n"
            "IMPORTANT: Your final response MUST be valid JSON with exactly "
            "this structure:\n"
            "{\n"
            '  "summary": "brief diagnosis summary",\n'
            '  "evidence": ["evidence item 1", "evidence item 2"],\n'
            '  "next_steps": ["recommended action 1", "recommended action 2"]\n'
            "}"
        )

    def allowed_tools(self) -> list[str]:
        return ["log_search", "kb_query"]

    async def pre_retrieve(self, text: str, memory: Memory) -> list[DocChunk]:
        return await memory.retrieve(text, k=3)
