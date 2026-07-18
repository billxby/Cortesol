"""Authenticated public compatibility gateway for the Cortesol Ollama policy."""

from __future__ import annotations

import asyncio
import hmac
import os
import re
import time
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path
from pydantic import BaseModel, Field

from .ollama import DEFAULT_OLLAMA_MODEL, OllamaProposalClient

MAX_INPUT_CHARS = 65_536
EVIDENCE_ID_RE = re.compile(r"(?m)^evidence_id=([^\s]+)")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=MAX_INPUT_CHARS)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=8)
    temperature: float = 0.0
    max_tokens: int = Field(default=256, ge=1, le=1024)
    adapter_revision: str | None = None


def _expected_run_id() -> str:
    return os.environ.get("CORTESOL_RUN_ID", "cortesol-plan-b")


def _require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("CORTESOL_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="gateway API key is not configured")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def _model_client() -> OllamaProposalClient:
    return OllamaProposalClient(
        os.environ.get("CORTESOL_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        timeout=int(os.environ.get("CORTESOL_OLLAMA_TIMEOUT", "180")),
        retries=int(os.environ.get("CORTESOL_OLLAMA_RETRIES", "1")),
    )


def _extract_input(request: ChatRequest) -> tuple[str, str]:
    # Caller-supplied system/assistant messages are deliberately ignored. The
    # gateway owns its system policy; only the final user data block is used.
    users = [message.content for message in request.messages if message.role == "user"]
    if not users:
        raise HTTPException(status_code=422, detail="one user message is required")
    input_text = users[-1]
    match = EVIDENCE_ID_RE.search(input_text)
    if match is None:
        raise HTTPException(status_code=422, detail="input lacks the current evidence_id")
    return input_text, match.group(1)


app = FastAPI(
    title="Cortesol Proposal Policy",
    description="Schema-locked Plan B serving facade for the Cortesol proposer.",
    version="0.1.0-plan-b",
    docs_url=None,
    redoc_url=None,
)
_inference_slots = asyncio.Semaphore(int(os.environ.get("CORTESOL_MAX_CONCURRENT", "1")))


@app.get("/health")
async def health() -> dict[str, str]:
    available = await asyncio.to_thread(_model_client().is_available)
    if not available:
        raise HTTPException(status_code=503, detail="proposal model unavailable")
    return {"status": "ok"}


@app.post("/v1/runs/{run_id}/chat", dependencies=[Depends(_require_auth)])
async def freesolo_compatible_chat(
    request: ChatRequest,
    run_id: Annotated[str, Path(max_length=128)],
) -> dict:
    """Drop-in response shape for ``FreesoloExtractor``."""
    if run_id != _expected_run_id():
        raise HTTPException(status_code=404, detail="unknown model deployment")
    input_text, evidence_id = _extract_input(request)
    started = time.monotonic()
    async with _inference_slots:
        try:
            proposed = await asyncio.to_thread(
                _model_client().propose_text, input_text, evidence_id
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    content = proposed.model_dump_json(exclude={"think"})
    return {
        "id": f"cortesol-{evidence_id}",
        "object": "chat.completion",
        "model": _expected_run_id(),
        "created": int(time.time()),
        "latency_ms": round((time.monotonic() - started) * 1000),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
