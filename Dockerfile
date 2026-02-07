FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[web]"

# Copy source
COPY agent_engine/ agent_engine/

EXPOSE 8000

CMD ["uvicorn", "agent_engine.adapters.web_fastapi.app:app", "--host", "0.0.0.0", "--port", "8000"]
