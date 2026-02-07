"""FastAPI SSE adapter — thin translation layer, no business logic."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent_engine import create_engine
from agent_engine.engine.models import EventEnvelope

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    engine = create_engine()
    app = FastAPI(title="AgentEngine API", version="0.1.0")

    @app.post("/chat")
    async def chat(request: Request) -> StreamingResponse:
        body = await request.json()
        envelope = EventEnvelope(**body)

        async def sse_stream():
            async for event in engine.handle(envelope):
                payload = json.dumps(event.model_dump(), default=str)
                yield f"event: {event.type.value}\ndata: {payload}\n\n"

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return app


# Module-level instance for ``uvicorn agent_engine.adapters.web_fastapi.app:app``
app = create_app()


def serve() -> None:
    """Entry-point for ``agent-web`` console script."""
    import uvicorn

    uvicorn.run(
        "agent_engine.adapters.web_fastapi.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
