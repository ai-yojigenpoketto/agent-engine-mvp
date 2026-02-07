"""Skill router — picks a skill based on command prefix."""

from __future__ import annotations

from agent_engine.skills.interface import Skill


class Router:
    """Maps incoming text to a skill via command prefixes.

    Prefix rules (checked in order):
      /gpu ...  => gpu_diagnosis
      *         => doc_qa  (default)
    """

    PREFIX_MAP: dict[str, str] = {
        "/gpu": "gpu_diagnosis",
    }

    def __init__(self, skills: dict[str, Skill], default_skill: str = "doc_qa") -> None:
        self._skills = skills
        self._default = default_skill

    def route(self, text: str) -> tuple[Skill, str]:
        """Return ``(skill, cleaned_text)``."""
        stripped = text.strip()
        for prefix, skill_name in self.PREFIX_MAP.items():
            if stripped.lower().startswith(prefix):
                cleaned = stripped[len(prefix):].strip()
                skill = self._skills.get(skill_name)
                if skill is not None:
                    return skill, cleaned or stripped
        return self._skills[self._default], stripped
