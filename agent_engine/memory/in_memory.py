"""In-memory keyword-overlap retrieval — MVP stand-in for vector search."""

from __future__ import annotations

from agent_engine.engine.models import DocChunk
from agent_engine.memory.interface import Memory

SAMPLE_CORPUS: list[DocChunk] = [
    DocChunk(
        id="1",
        text="GPU temperature should not exceed 83C under sustained load. "
             "If it does, check fan speeds and thermal paste.",
        source="gpu_ops_manual.md",
        score=0.0,
    ),
    DocChunk(
        id="2",
        text="ECC errors on NVIDIA GPUs can indicate failing VRAM. "
             "Run nvidia-smi -q -d ECC to check error counts.",
        source="gpu_troubleshooting.md",
        score=0.0,
    ),
    DocChunk(
        id="3",
        text="CUDA Out of Memory errors can be resolved by reducing batch size, "
             "enabling gradient checkpointing, or using mixed precision training.",
        source="ml_ops_guide.md",
        score=0.0,
    ),
    DocChunk(
        id="4",
        text="NVLink errors between GPUs may indicate a hardware issue. "
             "Reseat the NVLink bridge and run nvidia-smi nvlink -s.",
        source="gpu_troubleshooting.md",
        score=0.0,
    ),
    DocChunk(
        id="5",
        text="Driver version mismatches between CUDA toolkit and GPU driver "
             "can cause runtime errors. Use nvidia-smi and nvcc --version to verify.",
        source="driver_guide.md",
        score=0.0,
    ),
    DocChunk(
        id="6",
        text="To monitor GPU utilization in real-time, use nvidia-smi dmon "
             "or gpustat for a cleaner output.",
        source="monitoring_guide.md",
        score=0.0,
    ),
]


class InMemoryMemory(Memory):
    """Keyword-overlap scorer over a static corpus."""

    def __init__(self, corpus: list[DocChunk] | None = None) -> None:
        self._corpus = corpus if corpus is not None else list(SAMPLE_CORPUS)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        session_id: str | None = None,
    ) -> list[DocChunk]:
        query_words = set(query.lower().split())
        scored: list[DocChunk] = []
        for chunk in self._corpus:
            text_words = set(chunk.text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                score = round(overlap / max(len(query_words), 1), 4)
                scored.append(chunk.model_copy(update={"score": score}))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]
