# CLAUDE.md — AgentEngine MVP

## What This Is

A from-scratch AI agent runtime (SDK) in Python. No LangChain, no LlamaIndex. The core engine is embeddable and transport-agnostic, with thin adapters for FastAPI SSE and CLI JSON.

## Architecture Rules (DO NOT VIOLATE)

- **`engine/` must NEVER import FastAPI, uvicorn, or any adapter-specific code.** The engine is a pure async library.
- **`adapters/` are thin translators only.** No business logic in adapters — they convert `EventEnvelope` → engine and `EngineEvent` → SSE/JSON.
- **All tool I/O uses Pydantic v2 models.** Every `ToolDef` has an `input_model` and `output_model`.
- **Interfaces (ABCs) before implementations.** `SesStore`, `Memory`, `LLMClient`, `Skill`, `TraceCollector` are all abstract. Add new implementations, don't modify the ABCs without good reason.

## Project Layout

```
agent_engine/
├── engine/          # Core runtime — models, session, LLM client, agent loop
│   ├── models.py    # ALL data models live here (EventEnvelope, EngineEvent, Session, DocChunk, LLMResult, ToolCallRequest)
│   ├── session.py   # SesStore ABC + InMemorySesStore
│   ├── llm.py       # LLMClient ABC + OpenAILLMClient + MockLLMClient + DemoMockLLMClient
│   └── agent.py     # AgentEngine.handle() — the core loop
├── tools/
│   ├── registry.py  # ToolRegistry (register, execute, RBAC, timeout, retry, openai_schemas)
│   └── builtins.py  # log_search + kb_query tool definitions
├── memory/
│   ├── interface.py  # Memory ABC
│   └── in_memory.py  # InMemoryMemory (keyword overlap scorer)
├── skills/
│   ├── interface.py       # Skill ABC (name, system_prompt, allowed_tools, pre_retrieve)
│   ├── doc_qa.py          # Default skill — uses kb_query
│   ├── gpu_diagnosis.py   # /gpu skill — uses log_search + kb_query, structured JSON output
│   └── router.py          # Router — prefix-based dispatch ("/gpu" → gpu_diagnosis, default → doc_qa)
├── tracing/
│   ├── interface.py       # TraceCollector ABC
│   └── jsonl_tracer.py    # Writes to ./traces/{trace_id}.jsonl
└── adapters/
    ├── web_fastapi/app.py  # POST /chat → SSE stream, GET /health
    └── cli/main.py         # argv/stdin → JSON lines to stdout
```

## Key Patterns

- **Factory wiring**: `agent_engine/__init__.py` has `create_engine()` which wires all components. Both adapters call this.
- **Event flow**: `EventEnvelope` in → `AgentEngine.handle()` → yields `EngineEvent` (async iterator) → adapter translates to output format.
- **LLM loop**: Up to 6 iterations. If LLM returns `tool_calls` → execute tools → append results → call LLM again. If LLM returns `content` → yield token events → yield final → break.
- **RBAC is two-layer**: `role ∈ tool.allowed_roles` AND `tool.name ∈ skill.allowed_tools()`. Both must pass.
- **Traces are buffered**: Events accumulate in memory per `trace_id`, flushed to disk at the end of `handle()`.

## Dev Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Tests (unit only — no API key needed)
pytest tests/test_tool_registry.py tests/test_engine.py tests/test_router.py -v

# Tests (full — requires OPENAI_API_KEY in .env)
pytest -v -s

# Run web server
uvicorn agent_engine.adapters.web_fastapi.app:app --port 8000

# Run CLI
python -m agent_engine.adapters.cli.main "/gpu ECC error on GPU1"

# Docker
docker compose up --build
```

## Environment Variables

- `OPENAI_API_KEY` — required for real LLM calls (loaded from `.env` via python-dotenv)
- `OPENAI_MODEL` — default `gpt-4o-mini`
- `USE_MOCK_LLM=1` — uses DemoMockLLMClient (no API key needed, demonstrates full tool-call flow)

## Testing Conventions

- Unit tests use `MockLLMClient` with pre-loaded `LLMResult` responses — deterministic, no network.
- Integration tests in `test_integration_openai.py` hit the real API — auto-skipped when `OPENAI_API_KEY` is unset.
- Fixtures are in `tests/conftest.py`: `memory`, `tool_registry`, `router`, `ses_store`, `trace_collector`.
- All tests are async (`asyncio_mode = "auto"` in pyproject.toml).

## When Adding New Features

- **New tool**: Create Pydantic `Input`/`Output` models in `tools/builtins.py`, define a `ToolDef`, register it in `create_engine()`.
- **New skill**: Implement `Skill` ABC in `skills/`, add to the `skills` dict and `Router.PREFIX_MAP` in `create_engine()`.
- **New session backend**: Implement `SesStore` ABC, swap in `create_engine()`.
- **New memory backend**: Implement `Memory` ABC, swap in `create_engine()`.
- **New LLM provider**: Implement `LLMClient` ABC (just the `generate` method), swap in `create_engine()`.
- **New tracer**: Implement `TraceCollector` ABC, swap in `create_engine()`.

## Style

- Type hints everywhere. Use `from __future__ import annotations`.
- Structured logging via `logging.getLogger(__name__)`.
- No emojis in code or docs unless explicitly asked.
- Keep it minimal — don't add abstractions for one-time operations.
