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
from ..adapters import cortex
from ..core import propagate
from ..core.config import EDGE_INFLUENCE, PRIOR_C_0
from ..core.kb import KB
from ..core.mathx import sigmoid
from ..core.results import EventResult
from ..core.schema import Edge, EdgeType, RawEvent
from ..eval.baselines import ModelBaseline
from ..eval.replay import seed_kb
from ..ingest.extract import FakeExtractor, PaperFakeExtractor, _env, flash_available
from ..sim.world import World

app = FastAPI(title="Cortesol")

_HERE = Path(__file__).parent
_STATIC = _HERE / "static"
_SIM_STREAM_PATH = _HERE.parents[1] / "data" / "streams" / "eval_seed42.jsonl"
_PAPERS_PATH = _HERE.parents[1] / "data" / "papers" / "raw_papers.jsonl"
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


def _node_label(claim_id: str) -> str:
    """Short node label. Sim ids are `c_bind_P1_MC4R`; real-paper ids are longer
    slugs — trim both to something legible."""
    lbl = claim_id
    if lbl.startswith("c_bind_"):
        lbl = lbl[len("c_bind_"):]
    elif lbl.startswith("c_efficacy_"):
        lbl = "~" + lbl[len("c_efficacy_"):]
    elif lbl.startswith("c_"):
        lbl = "~" + lbl[2:]
    lbl = lbl.replace("_", " ")
    return lbl if len(lbl) <= 26 else lbl[:24] + "…"


class DemoState:
    """One running KB, parameterised by two independent switches:
      * data_source ∈ {sim, papers} — the simulator stream vs. real PubMed papers.
      * extractor_kind ∈ {fake, flash_stock, flash_tuned} — deterministic stand-in
        vs. the live Freesolo model (stock or tuned). Belief still moves ONLY
        through the engine regardless of which extractor proposes the ops."""

    def __init__(self) -> None:
        self.data_source = "papers"  # landing view = the real papers database
        self.extractor_kind = "fake"
        self.lock = asyncio.Lock()
        self.subscribers: set[asyncio.Queue] = set()
        self.reset()

    def _build_extractor(self):
        if self.extractor_kind == "flash_stock":
            return ModelBaseline("engine+stock", _env("FLASH_MODEL_STOCK"))
        if self.extractor_kind == "flash_tuned":
            return ModelBaseline("engine+tuned", _env("FLASH_MODEL_TUNED"))
        # fake: pick the stand-in matching the data source
        return PaperFakeExtractor() if self.data_source == "papers" else FakeExtractor()

    def reset(self) -> None:
        if self.data_source == "papers":
            self.events = cortex.load_stream(_PAPERS_PATH)
            self.kb = cortex.seed_kb_from_papers(self.events)
        else:
            self.events = [
                RawEvent(**json.loads(line))
                for line in _SIM_STREAM_PATH.read_text().splitlines()
                if line.strip()
            ]
            self.kb = seed_kb(World(_DEMO_SEED), self.events)
            _seed_demo_edges(self.kb)
        self.extractor = self._build_extractor()
        self.cursor = 0
        self.history: list[dict] = []  # recent event/cascade messages, replayed on connect
        # Story arc: the graph starts EMPTY and builds up. A claim node is
        # "revealed" only once a committed result has touched it — so Act 1 is raw
        # results arriving into a blank canvas, Act 2 is the graph forming.
        self.revealed: set[str] = set()

    def reveal(self, claim_ids) -> None:
        self.revealed.update(cid for cid in claim_ids if cid in self.kb.claims)

    # -- graph serialization (read-only view of the KB) --

    def _impact(self, claim_id: str) -> float:
        score = 0.0
        for e in self.kb.live_edges():
            if e.src == claim_id:
                score += abs(EDGE_INFLUENCE.get(e.type.value, 0.0)) * e.weight
        return score

    # -- the browse list (the "papers database" view) --

    def _document(self, idx: int, ev: RawEvent) -> dict:
        f = ev.fields
        if self.data_source == "papers":
            title = f.get("title") or ev.raw_text.split("\n", 1)[0]
            journal = f.get("journal") or ev.source_id
            year = str(f.get("year") or "")
            authors = f.get("authors") or []
            tags = [t for t in (f.get("peptide"), f.get("primary_target")) if t]
        else:
            title = ev.raw_text
            journal = ev.source_id
            year = ""
            authors = []
            tags = [t for t in (f.get("metric"), f.get("assay")) if t]
        authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        status = "done" if idx < self.cursor else ("active" if idx == self.cursor else "pending")
        return {
            "idx": idx,
            "title": title,
            "journal": journal,
            "year": year,
            "authors": authors_str,
            "tags": tags,
            "status": status,
        }

    def documents(self) -> list[dict]:
        return [self._document(i, e) for i, e in enumerate(self.events)]

    def graph_payload(self, moved: list[str] | None = None, include_documents: bool = True) -> dict:
        moved_set = set(moved or [])
        # Only revealed claims are drawn — the graph grows as results land.
        nodes = []
        for c in self.kb.claims.values():
            if c.id not in self.revealed:
                continue
            nodes.append(
                {
                    "id": c.id,
                    "label": _node_label(c.id),
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
            if e.src in self.revealed and e.dst in self.revealed
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
            "revealed": len(self.revealed),
            "claims_total": len(self.kb.claims),
            "prior": PRIOR_C_0,
            "data_source": self.data_source,
            "extractor_kind": self.extractor_kind,
            "flash_available": flash_available(),
            "flash_tuned_available": bool(_env("FLASH_MODEL_TUNED")),
            "documents": self.documents() if include_documents else None,
        }

    async def broadcast(self, message: dict, remember: bool = False) -> None:
        if remember:
            self.history.append(message)
            del self.history[:-60]  # keep the last 60 steps for late joiners
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
        "graph": STATE.graph_payload(moved=result.dirty_claims, include_documents=False),
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


@app.get("/paper/{idx}")
def paper(idx: int) -> dict:
    """Full metadata + abstract for one document, for the browse-view reader."""
    if not (0 <= idx < len(STATE.events)):
        return {"error": "out of range"}
    e = STATE.events[idx]
    f = e.fields
    parts = e.raw_text.split("\n\n", 1)
    title = f.get("title") or parts[0]
    abstract = parts[1] if len(parts) > 1 else ("" if f.get("title") else parts[0])
    return {
        "idx": idx,
        "title": title,
        "abstract": abstract,
        "journal": f.get("journal") or e.source_id,
        "year": str(f.get("year") or ""),
        "authors": f.get("authors") or [],
        "peptide": f.get("peptide"),
        "target": f.get("primary_target"),
        "pmid": (f.get("pmid") or e.id.replace("pmid_", "")),
        "status": (
            "done" if idx < STATE.cursor else ("active" if idx == STATE.cursor else "pending")
        ),
    }


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
    for msg in STATE.history:  # replay recent steps so a (re)load rebuilds the log
        await queue.put(msg)

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
        STATE.reveal(result.dirty_claims)  # committed claims join the graph
        message = _event_message(event, result)
        await STATE.broadcast(message, remember=True)
        return {"status": "ok", "cursor": STATE.cursor, "event_id": event.id}


@app.post("/discredit")
async def discredit(payload: dict) -> dict:
    source_id = (payload or {}).get("source_id")
    if not source_id or STATE.kb.get_source(source_id) is None:
        return {"status": "error", "reason": "unknown source_id"}
    async with STATE.lock:
        deltas = propagate.discredit_source(STATE.kb, source_id)
        moved = sorted({d.claim_id for d in deltas})
        STATE.reveal(moved)
        message = {
            "type": "cascade",
            "source_id": source_id,
            "deltas": _delta_view(STATE.kb, deltas),
            "dirty": moved,
            "graph": STATE.graph_payload(moved=moved, include_documents=False),
        }
        await STATE.broadcast(message, remember=True)
        return {"status": "ok", "source_id": source_id, "claims_moved": len(moved)}


@app.post("/source")
async def set_source(payload: dict) -> dict:
    """Switch the data source (sim | papers) and rebuild the KB from scratch."""
    mode = (payload or {}).get("mode")
    if mode not in ("sim", "papers"):
        return {"status": "error", "reason": "mode must be 'sim' or 'papers'"}
    async with STATE.lock:
        STATE.data_source = mode
        STATE.reset()
        await STATE.broadcast(STATE.graph_payload())
        return {"status": "ok", "data_source": mode, "total": len(STATE.events)}


@app.post("/extractor")
async def set_extractor(payload: dict) -> dict:
    """Switch the extractor (fake | flash_stock | flash_tuned). Flash options need
    FLASH_API_KEY (and flash_tuned needs FLASH_MODEL_TUNED). Rebuilds the KB so the
    run is clean from event 0."""
    kind = (payload or {}).get("kind")
    if kind not in ("fake", "flash_stock", "flash_tuned"):
        return {"status": "error", "reason": "unknown extractor kind"}
    if kind.startswith("flash") and not flash_available():
        return {"status": "error", "reason": "FLASH_API_KEY not set"}
    if kind == "flash_tuned" and not _env("FLASH_MODEL_TUNED"):
        return {"status": "error", "reason": "FLASH_MODEL_TUNED not set"}
    async with STATE.lock:
        STATE.extractor_kind = kind
        STATE.reset()
        await STATE.broadcast(STATE.graph_payload())
        return {"status": "ok", "extractor_kind": kind}


@app.post("/reset")
async def reset() -> dict:
    async with STATE.lock:
        STATE.reset()
        await STATE.broadcast(STATE.graph_payload())
        return {"status": "ok"}


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
