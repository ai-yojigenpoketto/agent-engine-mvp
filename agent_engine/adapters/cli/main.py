"""CLI JSON-lines adapter — reads text from argv/stdin, prints EngineEvents as JSON."""

from __future__ import annotations

import asyncio
import json
import sys

from agent_engine import create_engine
from agent_engine.engine.models import EventEnvelope


async def run_cli(text: str, session_id: str = "cli-default") -> None:
    engine = create_engine()
    envelope = EventEnvelope(text=text, session_id=session_id)
    async for event in engine.handle(envelope):
        print(json.dumps(event.model_dump(), default=str), flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            print("Usage: agent-cli <text>  OR  echo '{\"text\":\"...\"}' | agent-cli", file=sys.stderr)
            sys.exit(1)
        try:
            data = json.loads(raw)
            text = data.get("text", raw)
        except json.JSONDecodeError:
            text = raw

    asyncio.run(run_cli(text))


if __name__ == "__main__":
    main()
