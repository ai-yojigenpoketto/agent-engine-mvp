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

## Original Prompt

The prompt used to generate this project (for reference / reproducibility):

```
You are an expert staff-level AI infrastructure engineer. Build an MVP "AgentEngine" (agent runtime) as a reusable Python package, plus thin adapters (Web SSE + CLI JSON). The goal is to create an embeddable runtime (SDK-like) that standardizes session, tools, memory (RAG interface), and tracing. Keep it minimal but production-minded.

REPO TARGET
- Create a new repo folder: agent_engine_mvp/
- Python 3.11
- Use uv OR venv (prefer uv) and provide commands.
- Provide a single docker-compose.yml for local run (optional but nice). At minimum: local run scripts.

HARD REQUIREMENTS (must-have)
1) Separation of concerns:
   - engine/ MUST NOT import FastAPI/CLI-specific code.
   - adapters/ are thin: translate input -> EventEnvelope and translate EngineEvents -> output (SSE/JSON).
2) AgentEngine public API:
   - AgentEngine.handle(event: EventEnvelope) -> AsyncIterator[EngineEvent]
   - EngineEvent types include: token, tool_call, tool_result, retrieve, trace, final, error.
3) Session:
   - Minimal SessionStore with in-memory dict for MVP, but designed with an interface so it can be swapped to Redis/Postgres later.
   - Session contains: session_id, tenant_id, user_id, message history, selected_skill, trace_id.
4) Tools:
   - Implement ToolRegistry with:
     - tool schemas via Pydantic v2 (input/output models)
     - allowlist / RBAC by "role" (for MVP: roles = user, operator, admin)
     - timeout + retry (simple)
     - audit logging + tracing hooks
   - Provide 2 example tools:
     - log_search(query: str) -> {results: list[str]} (mocked data for MVP)
     - kb_query(query: str) -> {snippets: list[str]} (calls Memory.retrieve)
5) Memory (RAG interface):
   - Define Memory interface: retrieve(query, k, session_id) -> list[DocChunk]
   - Provide a simple in-memory implementation with a tiny sample corpus.
   - Provide chunk structure: {id, text, source, score}
6) Router/Skills:
   - Define Skill interface: name, system_prompt(), allowed_tools(), pre_retrieve(text)
   - Provide two skills:
     - doc_qa (uses kb_query)
     - gpu_diagnosis (uses log_search + kb_query; outputs structured JSON: {summary, evidence[], next_steps[]})
   - Router picks skill by command prefix:
     - "/gpu ..." => gpu_diagnosis
     - default => doc_qa
7) LLM:
   - Use OpenAI API with function calling OR a mock LLM for tests.
   - Provide env var config:
     - OPENAI_API_KEY
     - OPENAI_MODEL (default gpt-4o-mini)
   - Implement a robust loop:
     - max_iterations = 6
     - token budget placeholder
     - if LLM requests tool call => execute tool, append tool result, continue
     - final => stream final tokens and yield final event
8) Streaming:
   - Web adapter: FastAPI endpoint POST /chat that streams SSE from EngineEvents (token events become SSE "token").
   - CLI adapter: reads JSON from stdin or args; prints JSON lines of EngineEvents.
9) Tracing:
   - Implement TraceCollector interface.
   - MVP implementation: write JSONL to ./traces/{trace_id}.jsonl with timestamps.
   - Every tool call, retrieval, and LLM response should emit trace events with latency.
10) Tests:
   - pytest + pytest-asyncio
   - At least:
     - test_tool_registry_permissions
     - test_engine_loop_tool_call_roundtrip (with mocked LLM)
     - test_router_skill_selection

OUTPUTS / DELIVERABLES
A) Code in agent_engine_mvp/ with clean structure:
   - engine/ (core runtime)
   - tools/
   - memory/
   - skills/
   - tracing/
   - adapters/web_fastapi/
   - adapters/cli/
   - tests/
B) A README.md with:
   - Setup instructions (uv/venv)
   - How to run web SSE server
   - How to run CLI
   - Example curl + example CLI usage
   - Example trace file content
C) A "Verification checklist" section in README showing the exact steps to confirm:
   - SSE streams tokens
   - Tool call works
   - Trace file contains events
   - Router picks /gpu

CONSTRAINTS
- Keep it minimal. No databases, no Redis, no Docker unless simple.
- No LangChain/LlamaIndex. Implement small custom runtime.
- Clear type hints, structured logging, and comments.

START NOW
1) Scaffold the directory structure.
2) Implement the core interfaces and AgentEngine.handle.
3) Implement tools + memory + skills + router.
4) Implement adapters (FastAPI SSE + CLI JSON).
5) Add tests.
6) Write README with commands and examples.
```
