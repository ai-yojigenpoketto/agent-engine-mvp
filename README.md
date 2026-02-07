# AgentEngine MVP

Minimal, embeddable agent runtime with session management, tool registry (RBAC + timeout/retry), memory (RAG interface), skill routing, and JSONL tracing.

**No LangChain. No LlamaIndex. Pure Python.**

## Architecture

```
Input (HTTP/CLI)
  → Adapter (translate to EventEnvelope)
  → AgentEngine.handle(envelope) → AsyncIterator[EngineEvent]
      → Router (pick skill by prefix)
      → Skill (system_prompt, allowed_tools, pre_retrieve)
      → LLM Loop (max 6 iterations):
          1. Call LLM with function-calling schemas
          2. tool_calls? → validate RBAC → execute → yield events → loop
          3. content?    → yield token events → yield final → break
      → Save session + flush traces
  → Adapter (translate EngineEvents → SSE / JSON lines)
```

### Separation of Concerns

| Layer | Imports | Does NOT import |
|-------|---------|-----------------|
| `engine/` | models, tools, memory, skills, tracing | FastAPI, CLI, any adapter |
| `adapters/web_fastapi/` | FastAPI, engine | CLI code |
| `adapters/cli/` | engine, stdlib | FastAPI |

## Project Structure

```
agent_engine_mvp/
├── pyproject.toml
├── agent_engine/
│   ├── __init__.py              # create_engine() factory
│   ├── engine/
│   │   ├── models.py            # EventEnvelope, EngineEvent, Session, DocChunk, LLMResult
│   │   ├── session.py           # SesStore ABC + InMemorySesStore
│   │   ├── llm.py               # LLMClient ABC + OpenAI + Mock + DemoMock
│   │   └── agent.py             # AgentEngine — the core loop
│   ├── tools/
│   │   ├── registry.py          # ToolRegistry (Pydantic schemas, RBAC, timeout, retry)
│   │   └── builtins.py          # log_search, kb_query
│   ├── memory/
│   │   ├── interface.py         # Memory ABC
│   │   └── in_memory.py         # InMemoryMemory (keyword scorer + sample corpus)
│   ├── skills/
│   │   ├── interface.py         # Skill ABC
│   │   ├── doc_qa.py            # DocQA skill (uses kb_query)
│   │   ├── gpu_diagnosis.py     # GPUDiagnosis skill (log_search + kb_query → structured JSON)
│   │   └── router.py            # Router (prefix-based skill selection)
│   ├── tracing/
│   │   ├── interface.py         # TraceCollector ABC
│   │   └── jsonl_tracer.py      # JSONL file tracer → ./traces/{trace_id}.jsonl
│   └── adapters/
│       ├── web_fastapi/app.py   # POST /chat → SSE stream
│       └── cli/main.py          # stdin/argv → JSON lines
└── tests/
    ├── conftest.py
    ├── test_tool_registry.py    # Permissions, execution, timeout, retry, schemas
    ├── test_engine.py           # Tool-call roundtrip, session persistence, max iterations
    └── test_router.py           # Skill selection by prefix
```

## Setup

### With uv (recommended)

```bash
cd agent_engine_mvp
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### With venv

```bash
cd agent_engine_mvp
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### With Docker

```bash
cd agent_engine_mvp

# Add your API key to .env
echo "OPENAI_API_KEY=sk-..." > .env

# Build and run
docker compose up --build

# In another terminal:
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "/gpu ECC error on GPU1", "session_id": "s1"}'

# Stop
docker compose down
```

Traces are persisted to `./traces/` on the host via bind mount.

To run with the demo mock (no API key):

```bash
# Override the env var
OPENAI_API_KEY= USE_MOCK_LLM=1 docker compose up --build
```

## Run Tests

```bash
pytest -v
```

Expected: 20 tests pass (permissions, tool roundtrip, router selection, etc.)

## Run the Web SSE Server

```bash
# With .env (recommended — reads OPENAI_API_KEY from .env automatically):
uvicorn agent_engine.adapters.web_fastapi.app:app --port 8000

# Or export explicitly:
export OPENAI_API_KEY=sk-...
uvicorn agent_engine.adapters.web_fastapi.app:app --port 8000

# Without API key (demo mock):
USE_MOCK_LLM=1 uvicorn agent_engine.adapters.web_fastapi.app:app --port 8000
```

### Example curl

```bash
# Default skill (doc_qa)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I monitor GPU usage?", "session_id": "s1"}'

# GPU diagnosis skill (triggered by /gpu prefix)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "/gpu ECC error on GPU1", "session_id": "s2"}'
```

SSE events arrive as:

```
event: retrieve
data: {"type": "retrieve", "data": {"chunks": [...]}, ...}

event: tool_call
data: {"type": "tool_call", "data": {"name": "log_search", "arguments": {"query": "..."}}, ...}

event: tool_result
data: {"type": "tool_result", "data": {"name": "log_search", "result": {"results": [...]}}, ...}

event: token
data: {"type": "token", "data": {"text": "Based "}, ...}

event: final
data: {"type": "final", "data": {"text": "Based on the logs..."}, ...}
```

## Run the CLI

```bash
# With real OpenAI:
export OPENAI_API_KEY=sk-...
python -m agent_engine.adapters.cli.main "How do I check GPU temperature?"

# GPU diagnosis:
USE_MOCK_LLM=1 python -m agent_engine.adapters.cli.main "/gpu GPU temperature high on node-3"

# From stdin JSON:
echo '{"text": "/gpu ECC errors"}' | USE_MOCK_LLM=1 python -m agent_engine.adapters.cli.main
```

Output is one JSON object per line (JSON lines):

```json
{"type": "retrieve", "data": {"chunks": [...]}, "trace_id": "...", "timestamp": ...}
{"type": "tool_call", "data": {"name": "log_search", ...}, ...}
{"type": "tool_result", "data": {"result": {"results": [...]}}, ...}
{"type": "token", "data": {"text": "Based "}, ...}
{"type": "final", "data": {"text": "..."}, ...}
```

## Example Trace File

After a request, `./traces/{trace_id}.jsonl` contains:

```jsonl
{"ts": 1706000000.1, "trace_id": "abc-123", "event": "route", "skill": "gpu_diagnosis", "original_text": "/gpu ECC error", "cleaned_text": "ECC error"}
{"ts": 1706000000.2, "trace_id": "abc-123", "event": "retrieve", "count": 3, "latency_ms": 0.05}
{"ts": 1706000000.3, "trace_id": "abc-123", "event": "llm_call", "iteration": 0, "latency_ms": 120.5, "has_tool_calls": true}
{"ts": 1706000000.4, "trace_id": "abc-123", "event": "tool_exec", "tool": "log_search", "attempt": 1, "latency_ms": 2.1, "status": "ok"}
{"ts": 1706000000.5, "trace_id": "abc-123", "event": "llm_call", "iteration": 1, "latency_ms": 95.3, "has_tool_calls": false}
{"ts": 1706000000.6, "trace_id": "abc-123", "event": "handle_done", "total_latency_ms": 220.1}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for real LLM calls |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `USE_MOCK_LLM` | `0` | Set to `1` to use demo mock (no API key needed) |

## Programmatic Usage (SDK)

```python
from agent_engine import create_engine, EventEnvelope

engine = create_engine(openai_api_key="sk-...", openai_model="gpt-4o-mini")

envelope = EventEnvelope(
    text="/gpu ECC error on node-5",
    session_id="my-session",
    tenant_id="acme",
    user_id="alice",
    role="operator",
)

async for event in engine.handle(envelope):
    match event.type.value:
        case "token":
            print(event.data["text"], end="", flush=True)
        case "final":
            print()  # newline after streaming
        case "tool_call":
            print(f"[calling {event.data['name']}...]")
        case "error":
            print(f"ERROR: {event.data}")
```

## Extending

| Component | How to Extend |
|-----------|---------------|
| **Session store** | Implement `SesStore` ABC (e.g., Redis, Postgres) |
| **Memory** | Implement `Memory` ABC (e.g., pgvector, Pinecone) |
| **Tools** | Create `ToolDef` with Pydantic input/output models, register on `ToolRegistry` |
| **Skills** | Implement `Skill` ABC, add to router |
| **Tracing** | Implement `TraceCollector` ABC (e.g., OpenTelemetry, Datadog) |
| **LLM** | Implement `LLMClient` ABC (e.g., Anthropic, local models) |

## Verification Checklist

Run these steps to confirm everything works:

### 1. SSE streams tokens

```bash
USE_MOCK_LLM=1 uvicorn agent_engine.adapters.web_fastapi.app:app --port 8000 &
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "hello", "session_id": "test1"}'
# Expect: event: token lines followed by event: final
kill %1
```

### 2. Tool call works

```bash
USE_MOCK_LLM=1 python -m agent_engine.adapters.cli.main "/gpu GPU error" 2>/dev/null \
  | python -c "import sys,json; events=[json.loads(l) for l in sys.stdin]; types=[e['type'] for e in events]; assert 'tool_call' in types and 'tool_result' in types; print('PASS: tool_call + tool_result found')"
```

### 3. Trace file contains events

```bash
USE_MOCK_LLM=1 python -m agent_engine.adapters.cli.main "/gpu test" > /dev/null 2>&1
ls traces/*.jsonl && echo "PASS: trace file exists"
cat traces/*.jsonl | python -c "import sys,json; lines=[json.loads(l) for l in sys.stdin]; events={l['event'] for l in lines}; assert 'route' in events and 'tool_exec' in events and 'handle_done' in events; print(f'PASS: {len(lines)} trace events with types: {events}')"
```

### 4. Router picks /gpu

```bash
pytest tests/test_router.py -v -k "gpu_prefix"
# Expect: 2 passed (test_gpu_prefix_routes_to_gpu_diagnosis, test_gpu_prefix_case_insensitive)
```

### 5. Full test suite

```bash
pytest -v
# Expect: 20 unit tests pass + 4 integration tests (pass with API key, skip without)
```

### 6. Docker

```bash
docker compose up --build -d
curl -s http://localhost:8000/health
# Expect: {"status":"ok"}
docker compose down
```
