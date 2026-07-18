"""FastAPI + SSE backend for the live demo (area C).

Streams EventResults (belief diffs) to the browser as they are committed. Serves
static/index.html. Launch with `make run-ui`. Reference: Graph Propagation §5.

This backend is READ-ONLY over belief: it never sets `claim.ell`. Every state
change goes through the real lifecycle — `pipeline.process_event` (which runs the
quarantine → screen → validate → engine → propagate path) and
`propagate.discredit_source`. The UI observes; the ledger disposes.

Endpoints:
  GET  /            -> the vis-network page
  GET  /stream      -> SSE of {graph, event, cascade} messages
  POST /event       -> inject the next stream event into the running KB
  POST /discredit   -> trigger a retraction cascade (the money-shot demo beat)
  POST /reset       -> reseed the KB back to the skeptical prior
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .. import pipeline
from ..core import propagate
from ..core.config import EDGE_INFLUENCE, PRIOR_C_0
from ..core.kb import KB
from ..core.mathx import sigmoid
from ..core.results import EventResult
from ..core.schema import Edge, EdgeType, RawEvent
from ..eval.replay import seed_kb
from ..sim.world import World

app = FastAPI(title="Cortesol")

_HERE = Path(__file__).parent
_STATIC = _HERE / "static"
_STREAM_PATH = _HERE.parents[1] / "data" / "streams" / "eval_seed42.jsonl"
_DEMO_SEED = 42


# --------------------------------------------------------------------------
# Demo state — one running KB driven through the real pipeline.
# --------------------------------------------------------------------------


def _seed_demo_edges(kb: KB) -> None:
    """Seed a small typed topology so the graph has structure to propagate over.

    Edges are derived ONLY from public world identities (peptide -> its binder /
    efficacy claim), never from any latent truth. `efficacy depends_on binding` is
    the load-bearing relation for the retraction cascade; a couple of supports /
    contradicts edges give the graph variety and exercise the edge styling.
    """
    def add(src: str, dst: str, etype: str, w: float) -> None:
        if kb.get_claim(src) is None or kb.get_claim(dst) is None or src == dst:
            return
        eid = f"edge:{src}->{dst}:{etype}"
        if eid not in kb.edges:
            kb.add_edge(Edge(id=eid, src=src, dst=dst, type=EdgeType(etype), weight=w))

    # efficacy(P) depends_on binding(P) — the biological dependency.
    binder = {}
    for cid in kb.claims:
        if cid.startswith("c_bind_"):
            pep = cid.split("_")[2]  # c_bind_P5_CXCR4 -> P5
            binder[pep] = cid
    for cid in list(kb.claims):
        if cid.startswith("c_efficacy_"):
            pep = cid.split("_")[2]
            if pep in binder:
                add(binder[pep], cid, "depends_on", 1.0)

    # supports web among true binders + one contradicts, purely for topology.
    order = [binder[p] for p in ("P1", "P3", "P5") if p in binder]
    for a, b in zip(order, order[1:], strict=False):
        add(a, b, "supports", 0.8)
    if "P4" in binder and "P5" in binder:
        add(binder["P4"], binder["P5"], "contradicts", 1.0)


class DemoState:
    def __init__(self) -> None:
        self.events: list[RawEvent] = [
            RawEvent(**json.loads(line))
            for line in _STREAM_PATH.read_text().splitlines()
            if line.strip()
        ]
        self.extractor = pipeline._default_extractor()
        self.lock = asyncio.Lock()
        self.subscribers: set[asyncio.Queue] = set()
        self.reset()

    def reset(self) -> None:
        self.kb = seed_kb(World(_DEMO_SEED), self.events)
        _seed_demo_edges(self.kb)
        self.cursor = 0

    # -- graph serialization (read-only view of the KB) --

    def _impact(self, claim_id: str) -> float:
        score = 0.0
        for e in self.kb.live_edges():
            if e.src == claim_id:
                score += abs(EDGE_INFLUENCE.get(e.type.value, 0.0)) * e.weight
        return score

    def graph_payload(self, moved: list[str] | None = None) -> dict:
        moved_set = set(moved or [])
        nodes = []
        for c in self.kb.claims.values():
            nodes.append(
                {
                    "id": c.id,
                    "label": c.id.replace("c_bind_", "").replace("c_efficacy_", "~"),
                    "text": c.text,
                    "c": round(c.c, 4),
                    "u": round(c.u, 4),
                    "r": c.r,
                    "s": c.s,
                    "status": c.status.value,
                    "impact": round(self._impact(c.id), 3),
                    "moved": c.id in moved_set,
                }
            )
        edges = [
            {
                "id": e.id,
                "from": e.src,
                "to": e.dst,
                "type": e.type.value,
                "weight": e.weight,
            }
            for e in self.kb.live_edges()
        ]
        sources = [
            {"id": s.id, "tier": s.tier, "discredited": s.discredited}
            for s in sorted(self.kb.sources.values(), key=lambda x: x.id)
        ]
        return {
            "type": "graph",
            "nodes": nodes,
            "edges": edges,
            "sources": sources,
            "cursor": self.cursor,
            "total": len(self.events),
            "prior": PRIOR_C_0,
        }

    async def broadcast(self, message: dict) -> None:
        for q in list(self.subscribers):
            await q.put(message)


STATE = DemoState()


# --------------------------------------------------------------------------
# Message builders
# --------------------------------------------------------------------------


def _delta_view(kb: KB, deltas) -> list[dict]:
    out = []
    for d in deltas:
        out.append(
            {
                "claim_id": d.claim_id,
                "c_before": round(sigmoid(d.before_ell), 4),
                "c_after": round(sigmoid(d.after_ell), 4),
                "d_c": round(sigmoid(d.after_ell) - sigmoid(d.before_ell), 4),
                "cause": d.cause,
                "op": d.op,
            }
        )
    return out


def _event_message(event: RawEvent, result: EventResult) -> dict:
    src = STATE.kb.get_source(event.source_id)
    accepted = [
        {"detail": a.detail, "op": a.op}
        for a in result.audit
        if a.kind in ("commit", "flag_ood")
    ]
    rejected = [
        {"detail": a.detail, "op": a.op}
        for a in result.audit
        if a.kind == "reject"
    ]
    # recompute the deterministic red flags for display (cheap, no state change)
    from ..ingest.quarantine import quarantine
    from ..ingest.screen import screen

    red_flags = screen(quarantine(event), STATE.kb)

    return {
        "type": "event",
        "event_id": event.id,
        "t": event.t,
        "source_id": event.source_id,
        "source_tier": src.tier if src else "unknown",
        "raw_text": event.raw_text,  # DATA — rendered as text, never markup
        "fields": event.fields,
        "accepted": accepted,
        "rejected": rejected,
        "red_flags": red_flags,
        "deltas": _delta_view(STATE.kb, result.deltas),
        "dirty": result.dirty_claims,
        "truth": (event.sim_meta.event_class.value if event.sim_meta else None),
        "graph": STATE.graph_payload(moved=result.dirty_claims),
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/training")
def training_dashboard() -> FileResponse:
    return FileResponse(_STATIC / "training.html")


@app.get("/api/training")
async def training_metrics() -> dict:
    from ..train.tracker import training_snapshot

    return await asyncio.to_thread(training_snapshot)


@app.get("/stream")
async def stream(request: Request) -> EventSourceResponse:
    queue: asyncio.Queue = asyncio.Queue()
    STATE.subscribers.add(queue)
    await queue.put(STATE.graph_payload())  # initial snapshot on connect

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield {"data": json.dumps(msg)}
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            STATE.subscribers.discard(queue)

    return EventSourceResponse(gen())


@app.post("/event")
async def next_event() -> dict:
    async with STATE.lock:
        if STATE.cursor >= len(STATE.events):
            return {"status": "done", "cursor": STATE.cursor}
        event = STATE.events[STATE.cursor]
        result = pipeline.process_event(STATE.kb, event, STATE.extractor)
        STATE.cursor += 1
        message = _event_message(event, result)
        await STATE.broadcast(message)
        return {"status": "ok", "cursor": STATE.cursor, "event_id": event.id}


@app.post("/discredit")
async def discredit(payload: dict) -> dict:
    source_id = (payload or {}).get("source_id")
    if not source_id or STATE.kb.get_source(source_id) is None:
        return {"status": "error", "reason": "unknown source_id"}
    async with STATE.lock:
        deltas = propagate.discredit_source(STATE.kb, source_id)
        moved = sorted({d.claim_id for d in deltas})
        message = {
            "type": "cascade",
            "source_id": source_id,
            "deltas": _delta_view(STATE.kb, deltas),
            "dirty": moved,
            "graph": STATE.graph_payload(moved=moved),
        }
        await STATE.broadcast(message)
        return {"status": "ok", "source_id": source_id, "claims_moved": len(moved)}


@app.post("/reset")
async def reset() -> dict:
    async with STATE.lock:
        STATE.reset()
        await STATE.broadcast(STATE.graph_payload())
        return {"status": "ok"}


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
