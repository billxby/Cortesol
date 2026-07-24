# Cortesol — stateful FastAPI + SSE belief-graph UI (the live demo).
# Runs the app from the source tree (it reads data/ via package-relative paths),
# exactly like local `uvicorn cortesol.ui.app:app`.
FROM python:3.12-slim

WORKDIR /app

# Runtime deps only. The Freesolo/Flash training SDKs (the "train" extra in
# pyproject) are NOT needed to serve the UI, so we keep the image slim.
RUN pip install --no-cache-dir \
    "networkx>=3.3" "numpy>=2.0" "pydantic>=2.7" \
    "fastapi>=0.111" "uvicorn[standard]>=0.30" "sse-starlette>=2.1" \
    "anthropic>=0.117" "openai>=1.40"

COPY cortesol ./cortesol
COPY data ./data

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
EXPOSE 8000
# Render (and most PaaS) inject $PORT.
CMD ["sh", "-c", "uvicorn cortesol.ui.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
