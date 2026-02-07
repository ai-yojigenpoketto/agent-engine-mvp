"""Tests for Router skill selection."""

from __future__ import annotations

import pytest


class TestRouterSkillSelection:
    """Requirement: test_router_skill_selection."""

    def test_gpu_prefix_routes_to_gpu_diagnosis(self, router):
        skill, text = router.route("/gpu my GPU is overheating")
        assert skill.name == "gpu_diagnosis"
        assert text == "my GPU is overheating"

    def test_gpu_prefix_case_insensitive(self, router):
        skill, text = router.route("/GPU ECC error count rising")
        assert skill.name == "gpu_diagnosis"
        assert text == "ECC error count rising"

    def test_default_routes_to_doc_qa(self, router):
        skill, text = router.route("how do I monitor GPU usage?")
        assert skill.name == "doc_qa"
        assert text == "how do I monitor GPU usage?"

    def test_gpu_prefix_only_returns_original(self, router):
        skill, text = router.route("/gpu")
        assert skill.name == "gpu_diagnosis"
        # When only the prefix is provided, return original text
        assert text == "/gpu"

    def test_whitespace_handling(self, router):
        skill, text = router.route("  /gpu   temperature high  ")
        assert skill.name == "gpu_diagnosis"
        assert text == "temperature high"

    def test_non_prefix_with_gpu_word(self, router):
        """'/gpu' must be a prefix, not just present in the text."""
        skill, _ = router.route("tell me about gpu errors")
        assert skill.name == "doc_qa"

    def test_skill_allowed_tools(self, router):
        gpu_skill, _ = router.route("/gpu test")
        assert "log_search" in gpu_skill.allowed_tools()
        assert "kb_query" in gpu_skill.allowed_tools()

        doc_skill, _ = router.route("test")
        assert "kb_query" in doc_skill.allowed_tools()
        assert "log_search" not in doc_skill.allowed_tools()
