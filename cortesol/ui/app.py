"""FastAPI + SSE backend for the live demo (area C).

Streams EventResults (belief diffs) to the browser as they are committed. Serves
static/index.html. Launch with `make run-ui`. Reference: Graph Propagation §5.

Endpoints (planned):
  GET /            -> the vis-network page
  GET /stream      -> SSE of per-event {id, conf, impact, audit}
  POST /event      -> inject one event into the running KB (demo control)
  POST /discredit  -> trigger a retraction cascade (demo beat 4)
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Cortesol")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# TODO: /stream (SSE), /event, /discredit, and mount static/.
