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
import os
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .. import bootstrap, pipeline
from ..adapters import cortex
from ..core import propagate
from ..core.assessment import EvidenceAssessment
from ..core.config import EDGE_INFLUENCE, PRIOR_C_0, SOURCE_PRIORS
from ..core.domains import (
    get_active_domain,
    list_domains,
    register_domain,
    set_active_domain,
)
from ..core.kb import KB
from ..core.mathx import sigmoid
from ..core.results import EventResult
from ..core.schema import Edge, EdgeType, RawEvent, Source
from ..eval.replay import seed_kb
from ..ingest.extract import (
    FakeExtractor,
    FreesoloExtractor,
    GenericFakeExtractor,
    PaperFakeExtractor,
    _env,
    _load_dotenv,
    flash_available,
)
from ..sim.world import World
from . import domainsmith, research

app = FastAPI(title="Cortesol")

_HERE = Path(__file__).parent
_STATIC = _HERE / "static"
_SIM_STREAM_PATH = _HERE.parents[1] / "data" / "streams" / "eval_seed42.jsonl"
_PAPERS_PATH = _HERE.parents[1] / "data" / "papers" / "raw_papers.jsonl"
_ROBUSTNESS_PATH = _HERE.parents[1] / "data" / "papers" / "robustness_pack.jsonl"
_PER_PEPTIDE = 3  # foundation size per peptide (mirrors `make bootstrap`)
_DEMO_SEED = 42
# Server-side pace between papers during "Ingest all papers" autoplay. The loop runs
# on the server (no per-paper browser round-trip), so this is the ONLY throttle and
# it's purely for watchability — set to 0.0 to rip through as fast as the engine allows.
_PLAY_PACE_S = 0.1


def _load_robustness() -> list[RawEvent]:
    """The curated adversarial stress-test pack (hype, contradiction, injection,
    fraud) that targets already-established claims. Paper-shaped, so it flows
    through the exact same lifecycle; belief still moves only through the engine."""
    if not _ROBUSTNESS_PATH.exists():
        return []
    return cortex.load_stream(_ROBUSTNESS_PATH)


def _use_foundation() -> bool:
    """Whether to open the papers demo on the pre-built foundation snapshot.
    Read live (not cached at import) so /reset picks up an env change."""
    return os.environ.get("CORTESOL_FOUNDATION", "").strip().lower() in {"1", "true", "yes", "on"}


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
        self.extractor_kind = "freesolo"
        # Imported (LLM-drafted) field of knowledge: when data_source == "domain",
        # the KB is a fresh graph seeded from `active_draft` and the ACTIVE domain is
        # the custom one. Peptides stays the default; `/source` restores it.
        self.domain_name = "peptides"
        self.active_draft: domainsmith.DomainDraft | None = None
        self._seed_report: dict = {"seeded": [], "rejected": []}
        self.lock = asyncio.Lock()
        self.subscribers: set[asyncio.Queue] = set()
        # "Ingest all papers" runs as a server-side background loop (no per-paper
        # browser round-trip). `autoplay` is the stop flag the loop checks BETWEEN
        # events, so pausing/switching never tears a half-committed event.
        self.autoplay = False
        self.play_task: asyncio.Task | None = None
        # Story-arc phase boundaries over the papers stream (see reset()):
        #   [0, n_foundation)         Act 1 — the chosen "ground truth" papers
        #   [n_foundation, n_corpus)  Act 2 — the rest of the corpus (corroboration)
        #   [n_corpus, len(events))   Act 3 — the adversarial robustness pack
        self.n_foundation = 0
        self.n_corpus = 0
        self.foundation_ids: set[str] = set()
        self.reset()

    def _build_extractor(self):
        """THE extractor is the live Freesolo checkpoint — the model proposes ops,
        the engine disposes. Reads FREESOLO_RUN_ID / FREESOLO_API_KEY / FLASH_API_URL
        from .env. Falls back to the offline heuristic only if creds are absent, so
        the demo never hard-crashes."""
        # An imported field uses the domain-agnostic offline proposer: the live
        # Freesolo checkpoint is peptide-trained, so it must NOT propose over a
        # non-peptide graph. The generalist model is the faithful upgrade later.
        if self.data_source == "domain":
            self.extractor_kind = "offline"
            return GenericFakeExtractor()
        _load_dotenv()
        run_id = _env("FREESOLO_RUN_ID")
        api_key = _env("FREESOLO_API_KEY")
        if run_id and api_key:
            try:
                self.extractor_kind = "freesolo"
                return FreesoloExtractor(
                    run_id, api_key=api_key, api_url=_env("FLASH_API_URL"), timeout=30
                )
            except Exception:
                pass
        # creds missing/invalid — keep the UI alive with the offline stand-in
        self.extractor_kind = "offline"
        return PaperFakeExtractor() if self.data_source == "papers" else FakeExtractor()

    def reset(self) -> None:
        self._foundation_loaded = False
        self.foundation_ids = set()
        if self.data_source == "papers":
            robust = _load_robustness()
            # Foundation mode (CORTESOL_FOUNDATION=1): open on a pre-established KB —
            # confidence earned by replaying the curated foundation through the engine
            # (see cortesol/bootstrap.py). Act 1 is already done; the stream is the
            # held-out corpus (Act 2) followed by the robustness pack (Act 3).
            loaded = bootstrap.load_foundation() if _use_foundation() else None
            if loaded is not None:
                self.kb, holdout = loaded
                self.events = holdout + robust
                self.n_foundation = 0  # already established in the snapshot
                self.n_corpus = len(holdout)
                self._foundation_loaded = True
            else:
                # The guided 3-act arc, built live from an empty graph. Order the
                # corpus foundation-first so Act 1 is the curated "ground truth"
                # papers, Act 2 the rest, Act 3 the adversarial pack. Selection is
                # identical to `make bootstrap` (highest-reliability venues/peptide).
                corpus = cortex.load_stream(_PAPERS_PATH)
                foundation, holdout = bootstrap.select_foundation(corpus, _PER_PEPTIDE)
                self.foundation_ids = {e.id for e in foundation}
                self.events = foundation + holdout + robust
                self.n_foundation = len(foundation)
                self.n_corpus = len(foundation) + len(holdout)
                # keep t aligned to stream position (used as the per-doc key)
                for i, e in enumerate(self.events):
                    e.t = i
                self.kb = cortex.seed_kb_from_papers(self.events)
        elif self.data_source == "domain":
            # An imported field: a FRESH graph seeded structurally from the reviewed
            # draft (claim NODES at the skeptical prior — belief still moves only
            # through the engine). No preloaded event stream; evidence arrives via the
            # Red-team paste, so the audience watches belief form from nothing.
            self.kb = KB()
            if self.active_draft is not None:
                self._seed_report = domainsmith.seed_claims_into(
                    self.kb, self.active_draft, source_id=f"curator:{self.domain_name}"
                )
            self.events = []
            self.n_foundation = 0
            self.n_corpus = 0
        else:
            self.events = [
                RawEvent(**json.loads(line))
                for line in _SIM_STREAM_PATH.read_text().splitlines()
                if line.strip()
            ]
            self.kb = seed_kb(World(_DEMO_SEED), self.events)
            _seed_demo_edges(self.kb)
            # the sim stream has no foundation/robustness acts — one flat phase
            self.n_foundation = 0
            self.n_corpus = len(self.events)
        # Research-tab papers are appended to self.events beyond this boundary. The
        # choreographed demo stream is everything up to stream_len; pinning it here
        # means chat-added papers never inflate the "0/91" counter and are never
        # auto-ingested by Ingest-next / Auto-play.
        self.stream_len = len(self.events)
        self.research_ids: set[str] = set()
        self.research_verdicts: dict[int, str] = {}
        # Research-tab provenance: each chat query is a "session" that fetched papers
        # and moved (or created) some claims. research_sessions maps a session id to
        # its query text + the papers/claims it produced; claim_sessions is the
        # inverse index (claim -> the queries that touched it) so the graph can
        # highlight the branch a single query grew.
        self.research_sessions: dict[str, dict] = {}
        self.claim_sessions: dict[str, set[str]] = {}
        self.extractor = self._build_extractor()
        # Always keep a deterministic, network-free proposer on hand. If the live
        # model call times out or errors mid-run, we degrade to this for that one
        # event so the demo keeps flowing — belief still moves ONLY through the
        # engine, just from a deterministic proposal instead of the model's.
        if self.data_source == "domain":
            self.fallback = GenericFakeExtractor()
        elif self.data_source == "papers":
            self.fallback = PaperFakeExtractor()
        else:
            self.fallback = FakeExtractor()
        self.cursor = 0
        self.history: list[dict] = []  # recent event/cascade messages, replayed on connect
        # Story arc: the graph starts EMPTY and builds up. A claim node is
        # "revealed" only once a committed result has touched it — so Act 1 is raw
        # results arriving into a blank canvas, Act 2 is the graph forming.
        self.revealed: set[str] = set()
        # Foundation mode inverts Act 1: the established belief graph is already
        # there when the demo opens, and holdout results revise it. Reveal every
        # seeded claim so the audience sees the foundation before stepping.
        if self._foundation_loaded:
            self.revealed = set(self.kb.claims)
        # An imported field opens with its whole ontology visible at the prior, so
        # the audience sees the field's claim graph immediately and then watches each
        # pasted result earn confidence through the engine.
        if self.data_source == "domain":
            self.revealed = set(self.kb.claims)

    def activate_domain(self, draft: domainsmith.DomainDraft) -> dict:
        """Register + activate an LLM-drafted field, rebuild the KB around it, and
        seed its ontology (structural NODES at the prior — never belief). Switches
        the demo to an empty event stream driven by the offline GenericFakeExtractor.
        Returns a summary of the now-active field. Caller holds STATE.lock."""
        domain = domainsmith.to_domain(draft)
        register_domain(domain)
        set_active_domain(domain)
        self.domain_name = domain.name
        self.active_draft = draft
        self.data_source = "domain"
        self.reset()
        return {
            "name": domain.name,
            "label": domain.label,
            "entity_types": list(domain.entity_types),
            "property_types": list(domain.property_types),
            "seeded": list(self._seed_report.get("seeded", [])),
            "rejected": list(self._seed_report.get("rejected", [])),
            "n_claims": len(self.kb.claims),
        }

    def reveal(self, claim_ids) -> None:
        self.revealed.update(cid for cid in claim_ids if cid in self.kb.claims)

    # -- research tab: add chat-found papers to the library + graph --

    def add_research_papers(
        self, papers: list[dict], *, session_id: str | None = None, query: str | None = None
    ) -> dict:
        """Add chat-found papers to the Library + belief graph and assess them.

        Mirrors cortex.load_stream's RawEvent shaping, merges the public entities
        into the running KB (additive, idempotent — earned belief is never
        overwritten), then runs each paper through the REAL lifecycle so the ENGINE,
        not the chat model, assigns confidence. Belief still moves only through
        pipeline.process_event; this only reads `.c` back afterwards.

        When `session_id` is given, the new papers and every claim they moved are
        recorded under that research session (tagged onto the RawEvent + indexed in
        research_sessions / claim_sessions) so the query's branch can be traced in
        the Library and highlighted in the graph.

        Returns {claims, papers, moved, session_id}: a per-claim confidence readout
        for the agent, per-paper verdict cards for the client, the moved claim ids
        for a live graph broadcast, and the session it was filed under. The caller
        holds STATE.lock."""
        from ..ingest.quarantine import quarantine
        from ..ingest.screen import screen

        # the peptides this batch is about — used to ground the readout even when the
        # papers turn out to be already-in-library duplicates (so the agent still gets
        # real confidences to answer with, not an empty result).
        input_peptides = {
            str(p.get("peptide") or "").strip().lower() for p in papers if p.get("peptide")
        }
        existing_ids = {e.id for e in self.events}
        new_events: list[RawEvent] = []
        for p in papers:
            pmid = str(p.get("pmid") or "").strip()
            eid = f"pmid_{pmid}" if pmid else f"research_{len(self.events) + len(new_events)}"
            if eid in existing_ids:
                continue  # already in the library (e.g. found twice) — don't duplicate
            existing_ids.add(eid)
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            new_events.append(
                RawEvent(
                    id=eid,
                    t=len(self.events) + len(new_events),
                    source_id=cortex._journal_source_id(p.get("journal")),
                    raw_text=f"{title}\n\n{abstract}".strip(),
                    fields={
                        "peptide": p.get("peptide"),
                        "primary_target": p.get("primary_target"),
                        "journal": p.get("journal"),
                        "year": p.get("year"),
                        "title": title,
                        "authors": (p.get("authors") or [])[:6],
                        "pmid": pmid,
                        "doi": p.get("doi"),
                        "research_query": query,
                        "session_id": session_id,
                    },
                    sim_meta=None,
                )
            )

        # Register the new library entries, then seed their public entities into the
        # running KB (a Source per venue, a c_bind claim per peptide/target, edges).
        for ev in new_events:
            ev.t = len(self.events)
            self.events.append(ev)
            self.research_ids.add(ev.id)
        cortex.merge_papers_into_kb(self.kb, new_events)

        # Run each paper through the real lifecycle with the deterministic offline
        # proposer (real papers are prose — PaperFakeExtractor reads the abstract).
        proposer = PaperFakeExtractor()
        touched: set[str] = set()
        papers_out: list[dict] = []
        for ev in new_events:
            result = pipeline.process_event(self.kb, ev, proposer)
            touched.update(result.dirty_claims)
            verdict = _verdict_of(result)
            self.research_verdicts[ev.t] = verdict
            f = ev.fields
            papers_out.append(
                {
                    "idx": ev.t,
                    "pmid": f.get("pmid"),
                    "doi": f.get("doi"),
                    "title": f.get("title"),
                    "journal": f.get("journal"),
                    "year": str(f.get("year") or ""),
                    "verdict": verdict,
                    "red_flags": screen(quarantine(ev), self.kb),
                }
            )
        self.reveal(touched)

        # File this batch under its research session so the query's branch is traceable.
        if session_id:
            sess = self.research_sessions.setdefault(
                session_id,
                {"id": session_id, "query": query or "", "paper_idxs": [], "claim_ids": set()},
            )
            if query:
                sess["query"] = query
            sess["paper_idxs"].extend(ev.t for ev in new_events)
            sess["claim_ids"].update(touched)
            for cid in touched:
                self.claim_sessions.setdefault(cid, set()).add(session_id)

        # Confidence readout — the claims this batch moved PLUS every existing claim
        # about the queried peptides, so the agent gets grounded numbers even when the
        # papers were deduped or only some claims moved. Read straight off the ledger.
        readout_ids = set(touched)
        if input_peptides:
            for cid, claim in self.kb.claims.items():
                pep_tag = next(
                    (
                        t.split(":", 1)[1].lower()
                        for t in claim.ontology_tags
                        if t.startswith("peptide:")
                    ),
                    None,
                )
                if pep_tag and pep_tag in input_peptides:
                    readout_ids.add(cid)

        claims_out: list[dict] = []
        for cid in sorted(readout_ids):
            claim = self.kb.get_claim(cid)
            if claim is None:
                continue
            claims_out.append(
                {
                    "claim": claim.text,
                    "confidence": round(claim.c, 3),
                    "supporting": claim.r,
                    "contradicting": claim.s,
                    "status": claim.status.value,
                    "stance": _stance(claim.c),
                }
            )
        return {
            "claims": claims_out,
            "papers": papers_out,
            "moved": sorted(touched),
            "session_id": session_id,
        }

    def research_sessions_view(self) -> list[dict]:
        """Serialize the research sessions for the client's history sidebar + branch
        highlighting: query text, paper count, and the claim ids each query touched."""
        return [
            {
                "id": s["id"],
                "query": s["query"],
                "n_papers": len(s["paper_idxs"]),
                "claims": sorted(s["claim_ids"]),
            }
            for s in self.research_sessions.values()
        ]

    # -- graph serialization (read-only view of the KB) --

    def _impact(self, claim_id: str) -> float:
        score = 0.0
        for e in self.kb.live_edges():
            if e.src == claim_id:
                score += abs(EDGE_INFLUENCE.get(e.type.value, 0.0)) * e.weight
        return score

    # -- the browse list (the "papers database" view) --

    def phase_of(self, idx: int) -> str:
        """Which act a stream position belongs to (papers mode). Foundation is the
        curated ground-truth slice, corroboration the rest of the corpus, robustness
        the adversarial pack; research papers are appended live from the chat tab."""
        if 0 <= idx < len(self.events) and self.events[idx].id in self.research_ids:
            return "research"
        if self.data_source != "papers":
            return "corroboration"
        if idx < self.n_foundation:
            return "foundation"
        if idx < self.n_corpus:
            return "corroboration"
        return "robustness"

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
        phase = self.phase_of(idx)
        # Research papers were run through the engine at add-time, so they are always
        # appraised (they sit past the choreographed cursor, which never reaches them).
        if phase == "research":
            status = "done"
        else:
            status = (
                "done" if idx < self.cursor
                else ("active" if idx == self.cursor else "pending")
            )
        return {
            "idx": idx,
            "title": title,
            "journal": journal,
            "year": year,
            "authors": authors_str,
            "tags": tags,
            "status": status,
            "phase": phase,
            # ids for the "look it up" link (papers mode only; None in sim mode)
            "pmid": f.get("pmid") or (ev.id[5:] if ev.id.startswith("pmid_") else None),
            "doi": f.get("doi"),
            "source_kind": "researched" if ev.id in self.research_ids else "preloaded",
            "research_query": f.get("research_query"),
            "session_id": f.get("session_id"),
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
                    "sessions": sorted(self.claim_sessions.get(c.id, ())),
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
            "total": self.stream_len,
            "n_foundation": self.n_foundation,
            "n_corpus": self.n_corpus,
            "phase": self.phase_of(self.cursor),
            "foundation_built": self.cursor >= self.n_foundation,
            "revealed": len(self.revealed),
            "claims_total": len(self.kb.claims),
            "prior": PRIOR_C_0,
            "data_source": self.data_source,
            "active_domain": {
                "name": get_active_domain().name,
                "label": get_active_domain().label,
                "custom": self.data_source == "domain",
            },
            "extractor_kind": self.extractor_kind,
            "flash_available": flash_available(),
            "flash_tuned_available": bool(_env("FLASH_MODEL_TUNED")),
            "gemini_available": research.gemini_available(),
            "documents": self.documents() if include_documents else None,
            "research_verdicts": self.research_verdicts,
            "research_sessions": self.research_sessions_view(),
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


def _event_message(event: RawEvent, result: EventResult, reasoning: dict | None = None) -> dict:
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
        "source_tier": src.tier if src else "reputable",
        "raw_text": event.raw_text,  # DATA — rendered as text, never markup
        "fields": event.fields,
        "accepted": accepted,
        "rejected": rejected,
        "red_flags": red_flags,
        "reasoning": reasoning or {},
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
        "doi": f.get("doi"),
        "status": (
            "done" if idx < STATE.cursor else ("active" if idx == STATE.cursor else "pending")
        ),
    }


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


def _process_resilient(event: RawEvent) -> tuple[EventResult, bool, dict]:
    """Run one event through the real lifecycle, degrading to the offline proposer
    if the live model call fails (timeout / network / bad gateway). Returns the
    EventResult, whether the fallback was used, and a display-only `reasoning` dict
    (the model's filled 'form' + per-op quantization steps). The engine still
    disposes; only the *proposal* source changes, so this never sets belief."""
    ctx = pipeline.prepare_event(STATE.kb, event)
    used_fallback = False
    try:
        proposed = STATE.extractor.extract(ctx)
    except Exception:  # timeout, URLError, gateway error — keep the demo alive
        used_fallback = True
        proposed = STATE.fallback.extract(ctx)
    reasoning: dict = {}
    # The tuned live model emits a neutral EvidenceAssessment that a deterministic
    # judge compiles into ops; the offline fallback proposes ProposedOps directly.
    # Dispatch on type so both the live checkpoint and the fallback drive the same
    # ledger — belief still moves ONLY through the engine.
    if isinstance(proposed, EvidenceAssessment):
        result = pipeline.commit_assessment(STATE.kb, ctx, proposed, reasoning=reasoning)
    else:
        result = pipeline.commit_proposal(STATE.kb, ctx, proposed, reasoning=reasoning)
    live = isinstance(STATE.extractor, FreesoloExtractor) and not used_fallback
    reasoning["model"] = (
        f"Freesolo {STATE.extractor.run_id}" if live else "offline heuristic (fallback)"
    )
    reasoning["live"] = live
    return result, used_fallback, reasoning


def _process_offline(event: RawEvent) -> tuple[EventResult, dict]:
    """Deterministic, network-free processing for bulk autoplay — the offline
    proposer reads the abstract and proposes ops; the engine disposes (belief still
    moves only through the engine). No live model call, so 'Ingest all papers' rips
    through the corpus instead of waiting ~seconds per paper on the network. Mirrors
    what `_build_foundation_sync` does for the foundation replay."""
    ctx = pipeline.prepare_event(STATE.kb, event)
    proposed = STATE.fallback.extract(ctx)
    reasoning: dict = {}
    result = pipeline.commit_proposal(STATE.kb, ctx, proposed, reasoning=reasoning)
    reasoning["model"] = "offline heuristic (bulk ingest)"
    reasoning["live"] = False
    return result, reasoning


async def _advance_one(fast: bool = False) -> dict:
    """Process the next stream event through the real lifecycle and broadcast it.
    Shared write path for both ingest controls. `fast=False` (single-step 'Ingest
    next paper') uses the live model with an offline fallback; `fast=True` (bulk
    autoplay) uses the deterministic offline proposer so it doesn't crawl through a
    per-paper network call. Either way, belief moves ONLY through the engine."""
    async with STATE.lock:
        if STATE.cursor >= STATE.stream_len:
            return {"status": "done", "cursor": STATE.cursor}
        event = STATE.events[STATE.cursor]
        # blocking work runs off the event loop so SSE pings / other requests stay live
        if fast:
            result, reasoning = await asyncio.to_thread(_process_offline, event)
        else:
            result, used_fallback, reasoning = await asyncio.to_thread(_process_resilient, event)
            # Reflect per-event reality: only flip to "offline" when THIS event actually
            # fell back. If the live checkpoint answered, flip back to "freesolo" so the
            # badge stops lying. (Bulk ingest is deterministic-by-design, not a fallback,
            # so it deliberately leaves the badge alone.)
            if isinstance(STATE.extractor, FreesoloExtractor):
                STATE.extractor_kind = "offline" if used_fallback else "freesolo"
        STATE.cursor += 1
        STATE.reveal(result.dirty_claims)  # committed claims join the graph
        message = _event_message(event, result, reasoning)
        await STATE.broadcast(message, remember=True)
        return {"status": "ok", "cursor": STATE.cursor, "event_id": event.id}


@app.post("/event")
async def next_event() -> dict:
    """Ingest the next single paper — the 'Ingest next paper' button (live model)."""
    return await _advance_one(fast=False)


async def _autoplay_loop() -> None:
    """Drain the remaining stream server-side, broadcasting each event over SSE so
    the graph animates with NO per-paper browser round-trip. `autoplay` is checked
    only BETWEEN events, so a pause / source-switch never interrupts a half-committed
    event — belief and cursor stay consistent."""
    try:
        while STATE.autoplay:
            r = await _advance_one(fast=True)  # deterministic + fast; single-step stays live
            if r["status"] == "done" or not STATE.autoplay:
                break
            if _PLAY_PACE_S > 0:
                await asyncio.sleep(_PLAY_PACE_S)
    finally:
        STATE.autoplay = False
        STATE.play_task = None


@app.post("/play")
async def play() -> dict:
    """Start 'Ingest all papers' as a server-side loop. Idempotent — a second call
    while already playing is a no-op."""
    if STATE.cursor >= STATE.stream_len:
        return {"status": "done", "cursor": STATE.cursor}
    if not (STATE.autoplay and STATE.play_task and not STATE.play_task.done()):
        STATE.autoplay = True
        STATE.play_task = asyncio.create_task(_autoplay_loop())
    return {"status": "playing", "cursor": STATE.cursor}


@app.post("/pause")
async def pause() -> dict:
    """Stop autoplay. The loop halts AFTER the current event commits (never mid-flight),
    so nothing is left torn."""
    STATE.autoplay = False
    return {"status": "paused", "cursor": STATE.cursor}


# --------------------------------------------------------------------------
# Paste-in ingest — drop ANY untrusted text and watch the full lifecycle run.
# The generic entry the red-team panel and domain import build on. Text is DATA:
# an injection or fraudulent report is capped/screened/rejected, never obeyed.
# --------------------------------------------------------------------------

_INGEST_SEQ = [0]


def _ingest_text_sync(text: str, tier: str, source_id: str) -> tuple[RawEvent, EventResult, dict]:
    """Quarantine a pasted, untrusted blob and run it through the REAL lifecycle
    against the running KB — the same path a stream event takes (live model with an
    offline fallback). Belief still moves only through the engine; the source tier
    sets the cap `log(τ/φ)`, so a low-trust paste can barely move belief no matter
    how sensational its content."""
    if STATE.kb.get_source(source_id) is None:
        STATE.kb.add_source(Source.from_tier(source_id, tier))
    _INGEST_SEQ[0] += 1
    event = RawEvent(
        id=f"paste_{_INGEST_SEQ[0]}",
        t=-1,  # out-of-band paste: never a stream index, so it can't relabel a Library card
        source_id=source_id,
        raw_text=text,
        fields={"source_tier": tier},
        sim_meta=None,
    )
    result, _used_fallback, reasoning = _process_resilient(event)
    reasoning["ingest"] = "pasted text"
    return event, result, reasoning


@app.post("/ingest")
async def ingest(request: Request) -> dict:
    """Paste ANY untrusted text and run the full 7-step lifecycle live:
    quarantine → retrieve → extract → screen → validate → commit → publish. Emits
    the same SSE `event` message a stream paper does, so the graph animates and the
    reasoning log fills. Body: {text, source_tier?, source_id?}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    text = (body.get("text") or "").strip()
    if not text:
        return {"status": "error", "detail": "empty text"}
    tier = str(body.get("source_tier") or "unknown")
    if tier not in SOURCE_PRIORS:
        tier = "unknown"
    source_id = str(body.get("source_id") or f"pasted:{tier}")

    async with STATE.lock:
        event, result, reasoning = await asyncio.to_thread(
            _ingest_text_sync, text, tier, source_id
        )
        STATE.reveal(result.dirty_claims)
        message = _event_message(event, result, reasoning)
        await STATE.broadcast(message, remember=True)

    return {
        "status": "ok",
        "event_id": event.id,
        "source_tier": tier,
        "accepted": message["accepted"],
        "rejected": message["rejected"],
        "red_flags": message["red_flags"],
        "deltas": message["deltas"],
        "reasoning": message["reasoning"],
    }


_CALIB_SEEDS = (202, 303, 404, 505, 606)  # held-out, unseen by training/fixtures


@app.get("/calibration")
def calibration() -> dict:
    """Prove it's calibrated. Replay several fresh HELD-OUT simulator streams (which
    carry ground truth) through Cortesol vs the gullible and stubborn baselines, and
    score each: Brier↓ / ECE↓ / injection-ASR↓ / fraud-accepted↓ / max Δc↓, plus a
    pooled reliability diagram for Cortesol. Pooling multiple seeds fills the diagram
    (one small world has only ~8 scorable claims). Fully READ-ONLY — throwaway KBs,
    never the live demo graph; sim_meta is read only here, as the eval harness (PD6)."""
    from ..eval import metrics as M
    from ..eval.baselines import GullibleBot, StubbornBot
    from ..eval.replay import HELDOUT_LENGTH, build_ground_truth, seed_kb
    from ..sim.events import emit_stream

    builders = [("Cortesol", FakeExtractor), ("Gullible", GullibleBot), ("Stubborn", StubbornBot)]
    pairs: dict[str, list] = {name: [] for name, _ in builders}
    asr_by: dict[str, list] = {name: [] for name, _ in builders}
    fraud_by: dict[str, list] = {name: [] for name, _ in builders}
    dc_by: dict[str, list] = {name: [] for name, _ in builders}
    n_events = 0

    for seed in _CALIB_SEEDS:
        world = World(seed)
        events = emit_stream(seed, HELDOUT_LENGTH)
        gt = build_ground_truth(events)
        n_events += len(events)
        for name, factory in builders:
            kb = seed_kb(world, events)
            try:
                results = list(pipeline.replay_stream(kb, events, factory()))
            except Exception:
                results = []
            pairs[name].extend((kb.claims[cid].c, y) for cid, y in gt.items() if cid in kb.claims)
            asr_by[name].append(M.asr(results, events))
            fraud_by[name].append(M.fraud_accepted_rate(results, events))
            dc_by[name].append(M.max_confidence_shift(results))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    rows = [
        {
            "system": name,
            "brier": round(M.brier_pairs(pairs[name]), 4),
            "ece": round(M.ece_pairs(pairs[name]), 4),
            "asr": round(_mean(asr_by[name]), 4),
            "fraud": round(_mean(fraud_by[name]), 4),
            "dc_max": round(max(dc_by[name]) if dc_by[name] else 0.0, 4),
        }
        for name, _ in builders
    ]
    return {
        "seeds": list(_CALIB_SEEDS),
        "n_events": n_events,
        "n_scored": len(pairs["Cortesol"]),
        "rows": rows,
        "reliability": M.reliability_pairs(pairs["Cortesol"], bins=10),
    }


@app.get("/claim/{claim_id}")
def claim_detail(claim_id: str) -> dict:
    """The 'why did this move?' cause chain for one claim — the audit inspector.
    Returns the full belief trajectory (every Δℓ with its human-readable cause), the
    current subjective-logic opinion (c, u, r/s), ontology tags, and the live edges
    touching it. READ-ONLY over the live KB; nothing here moves belief."""
    cl = STATE.kb.get_claim(claim_id)
    if cl is None:
        return {"error": "unknown claim", "id": claim_id}
    trajectory = [
        {
            "t": p.t,
            "ell": round(p.ell, 4),
            "c": round(sigmoid(p.ell), 4),
            "cause": p.cause,
        }
        for p in cl.trajectory
    ]
    edges = [
        {
            "id": e.id,
            "src": e.src,
            "dst": e.dst,
            "type": e.type.value,
            "weight": round(e.weight, 3),
            "direction": "out" if e.src == claim_id else "in",
        }
        for e in STATE.kb.live_edges()
        if e.src == claim_id or e.dst == claim_id
    ]
    op = cl.opinion
    return {
        "id": cl.id,
        "text": cl.text,
        "tags": cl.ontology_tags,
        "status": cl.status.value,
        "c": round(cl.c, 4),
        "u": round(cl.u, 4),
        "r": cl.r,
        "s": cl.s,
        "opinion": {k: round(v, 4) for k, v in op.items()},
        "trajectory": trajectory,
        "edges": edges,
    }


def _build_foundation_sync() -> tuple[list[str], list[dict]]:
    """Replay the curated ground-truth slice (events[0:n_foundation]) through the
    real lifecycle with the deterministic proposer, exactly as `make bootstrap`
    does. Belief moves ONLY through the engine — we cache its output, we never set
    ℓ. Returns (revealed claim ids, a leaderboard of what got established)."""
    ext = STATE.fallback  # PaperFakeExtractor in papers mode — deterministic, offline
    touched: set[str] = set()
    for i in range(STATE.n_foundation):
        result = pipeline.process_event(STATE.kb, STATE.events[i], ext)
        touched.update(result.dirty_claims)
    STATE.cursor = STATE.n_foundation
    STATE.reveal(touched)
    established = [c for c in STATE.kb.claims.values() if c.trajectory]
    established.sort(key=lambda c: c.ell, reverse=True)
    top = [
        {"claim_id": c.id, "text": c.text, "c": round(sigmoid(c.ell), 3), "r": c.r, "s": c.s}
        for c in established[:8]
    ]
    return sorted(touched), top


@app.post("/build-foundation")
async def build_foundation() -> dict:
    """Act 1 — establish the ground truth. Replay the chosen foundation papers so
    the initial belief graph forms from empty. No-op if already built or if the KB
    opened pre-established (CORTESOL_FOUNDATION mode)."""
    async with STATE.lock:
        if STATE.data_source != "papers" or STATE.n_foundation == 0:
            return {"status": "noop", "reason": "no foundation slice to build"}
        if STATE.cursor >= STATE.n_foundation:
            return {"status": "noop", "reason": "foundation already established"}
        revealed, top = await asyncio.to_thread(_build_foundation_sync)
        n_established = len([c for c in STATE.kb.claims.values() if c.trajectory])
        message = {
            "type": "foundation",
            "n_foundation": STATE.n_foundation,
            "n_established": n_established,
            "top": top,
            "graph": STATE.graph_payload(moved=revealed, include_documents=True),
        }
        await STATE.broadcast(message, remember=True)
        return {"status": "ok", "n_foundation": STATE.n_foundation, "n_established": n_established}


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
    STATE.autoplay = False  # stop any running autoplay before rebuilding the stream/KB
    async with STATE.lock:
        # Restore the peptide field whenever we return to a built-in corpus, so
        # switching away from an imported field always brings the peptide demo back.
        set_active_domain("peptides")
        STATE.domain_name = "peptides"
        STATE.active_draft = None
        STATE.data_source = mode
        STATE.reset()
        await STATE.broadcast(STATE.graph_payload())
        return {"status": "ok", "data_source": mode, "total": len(STATE.events)}


# --------------------------------------------------------------------------
# Import any field of knowledge — describe a field in plain English, an LLM DRAFTS
# an ontology, the human reviews/activates it, and the SAME gated engine tracks
# calibrated belief in the new field. A Domain is TRUSTED CONFIG authored via a
# human-in-the-loop, NEVER belief: seed claims are structural NODES at the skeptical
# prior (kb.add_claim), and confidence still moves only through the engine.
# --------------------------------------------------------------------------


@app.get("/domains")
def domains() -> dict:
    """List every registered field so the UI can show/switch which one is active."""
    active = get_active_domain().name
    return {
        "active": active,
        "domains": [
            {"name": d.name, "label": d.label, "active": d.name == active}
            for d in list_domains()
        ],
    }


@app.post("/domain/draft")
async def domain_draft(payload: dict) -> dict:
    """DRAFT (do NOT activate) a domain ontology from a plain-English description.
    Returns {source: 'llm'|'fallback', draft}. The LLM proposes; nothing is applied
    and no belief is touched — the human reviews the draft before activating."""
    description = str((payload or {}).get("description") or "").strip()
    if not description:
        return {"status": "error", "detail": "describe the field first"}
    source, draft = await domainsmith.draft_domain(description)
    return {
        "status": "ok",
        "source": source,
        "draft": draft.model_dump(),
        "problems": domainsmith.validate_draft(draft),
        "gemini_available": research.gemini_available(),
    }


@app.post("/domain/activate")
async def domain_activate(payload: dict) -> dict:
    """Activate a reviewed draft: build the Domain, register + set it active, rebuild
    the KB fresh, seed the draft's claims (ontology-gated: out-of-namespace tags are
    rejected, never seeded), and switch the demo to the new field with the offline
    GenericFakeExtractor. Broadcasts a fresh graph over SSE. READ-ONLY over belief —
    seed claims are structural adds at the skeptical prior."""
    raw = (payload or {}).get("draft")
    if not isinstance(raw, dict):
        return {"status": "error", "detail": "missing draft"}
    try:
        draft = domainsmith.DomainDraft.model_validate(raw)
    except Exception as exc:  # bad shape from the editor
        return {"status": "error", "detail": f"invalid draft: {exc}"}
    if not draft.entity_types or not draft.property_types:
        return {"status": "error", "detail": "declare at least one entity and one property type"}
    STATE.autoplay = False  # stop any running autoplay before rebuilding the KB
    async with STATE.lock:
        summary = STATE.activate_domain(draft)
        await STATE.broadcast(STATE.graph_payload())
    return {"status": "ok", "domain": summary}


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
    STATE.autoplay = False  # stop any running autoplay before rebuilding the KB
    async with STATE.lock:
        STATE.extractor_kind = kind
        STATE.reset()
        await STATE.broadcast(STATE.graph_payload())
        return {"status": "ok", "extractor_kind": kind}


@app.post("/warmup")
async def warmup() -> dict:
    """Pre-warm the live Freesolo checkpoint and confirm it answers. The adapter
    is spun down when idle, so the first call can take ~90s; this lets the demo
    absorb that cold start on a button press instead of on the first ingest. On
    success we (re)mark the extractor live; on failure we honestly show offline."""
    ext = STATE.extractor
    if not isinstance(ext, FreesoloExtractor):
        return {
            "status": "offline",
            "reason": "No live Freesolo checkpoint configured — set FREESOLO_RUN_ID "
            "and FREESOLO_API_KEY in .env, then Reset.",
        }
    info = await asyncio.to_thread(ext.warmup)
    async with STATE.lock:
        STATE.extractor_kind = "freesolo" if info.get("ok") else "offline"
        await STATE.broadcast(STATE.graph_payload())
    return {"status": "ok" if info.get("ok") else "error", **info}


@app.post("/reset")
async def reset() -> dict:
    async with STATE.lock:
        STATE.reset()
        await STATE.broadcast(STATE.graph_payload())
        return {"status": "ok"}


# --------------------------------------------------------------------------
# Ask the belief graph — a grounded Q&A over the ledger. The ANSWER is read
# straight from committed belief (confidence, evidence counts, provenance,
# trajectory); the model never authors it. This is the epistemic angle made
# interactive: you can interrogate what the system believes and why.
# --------------------------------------------------------------------------

_SRC_RE = re.compile(r"from (\S+) \[")
_STANCE = [
    (0.75, "Strongly supported"),
    (0.60, "Supported"),
    (0.45, "Uncertain"),
    (0.30, "Doubtful"),
    (0.00, "Refuted"),
]


def _stance(c: float) -> str:
    for thresh, label in _STANCE:
        if c >= thresh:
            return label
    return "Refuted"


def _answer_claim(claim) -> dict:
    """Turn one belief-graph claim into a grounded answer: the confidence the
    ledger holds, the evidence behind it, where it came from, and how it moved."""
    from ..core.mathx import sigmoid as _sig

    c = _sig(claim.ell)
    # provenance: distinct sources named in this claim's audit trajectory
    sources: list[str] = []
    for pt in claim.trajectory:
        m = _SRC_RE.search(pt.cause or "")
        if m and m.group(1) not in sources:
            sources.append(m.group(1))
    moves = len(claim.trajectory)
    # every claim begins at the skeptical prior; the trajectory only stores
    # post-move points, so the honest baseline is the prior itself.
    c_start = PRIOR_C_0
    return {
        "claim_id": claim.id,
        "text": claim.text,
        "confidence": round(c, 4),
        "uncertainty": round(claim.u, 4),
        "stance": _stance(c),
        "status": claim.status.value,
        "r": claim.r,
        "s": claim.s,
        "moves": moves,
        "c_start": round(c_start, 4),
        "sources": sources[:6],
    }


@app.post("/ask")
async def ask(payload: dict) -> dict:
    """Answer a natural-language question about the peptides in the KB, grounded in
    committed belief. Lexical retrieval finds the most relevant claims; the answer
    is their confidence + evidence + provenance — never generated prose."""
    from ..retrieval import _score, _tokens

    question = ((payload or {}).get("question") or "").strip()
    if not question:
        return {"status": "error", "reason": "empty question"}
    query = _tokens(question)
    ranked = sorted(
        STATE.kb.claims.values(),
        key=lambda c: (_score(c, query), c.id),
        reverse=True,
    )
    hits = [c for c in ranked if _score(c, query) > 0][:4]
    if not hits:
        return {"status": "ok", "question": question, "answers": [], "grounded": True}
    return {
        "status": "ok",
        "question": question,
        "answers": [_answer_claim(c) for c in hits],
        "grounded": True,
    }


# --------------------------------------------------------------------------
# Grounded generative report — the LLM proposes the LAYOUT; the ledger supplies
# the FACTS. The model emits a schema-constrained ReportSpec that may only
# reference claim ids from a compact KB summary; a deterministic renderer
# (ui/reportspec.py) binds every figure to committed belief. The report
# structurally cannot assert anything the graph doesn't support ("truth in, truth
# out"). READ-ONLY over belief; falls back to a deterministic default report on
# any failure or when GEMINI_API_KEY is absent, so it always works offline.
# --------------------------------------------------------------------------


def _report_summary_ids() -> list[str]:
    """The claim ids a report may reference: the revealed/active claims (the graph
    the audience is actually looking at), falling back to every claim if nothing has
    been revealed yet. Ranked by confidence so the summary leads with the strongest."""
    ids = [cid for cid in STATE.revealed if cid in STATE.kb.claims] or list(STATE.kb.claims)
    return sorted(ids, key=lambda cid: (-STATE.kb.claims[cid].c, cid))


@app.post("/report/generate")
async def report_generate(payload: dict) -> dict:
    """Generate a grounded report. The model proposes a ReportSpec (layout, bound to
    claim ids from the KB summary); we validate it with Pydantic, run the groundedness
    immune check, and render it against the live ledger. On ANY failure — no key, model
    error, bad JSON, schema violation — we fall back to a deterministic default report
    built from the top claims, so the feature always produces something grounded."""
    from .reportspec import (
        ReportSpec,
        compact_kb_summary,
        default_report,
        groundedness,
        render_report,
    )

    prompt = ((payload or {}).get("prompt") or "").strip()
    kb = STATE.kb
    summary_ids = _report_summary_ids()
    summary = compact_kb_summary(kb, claim_ids=summary_ids)

    spec: ReportSpec | None = None
    source = "fallback"
    if prompt and research.gemini_available():
        raw = await research.propose_report_spec(prompt, summary, ReportSpec.model_json_schema())
        if raw is not None:
            try:
                spec = ReportSpec.model_validate(raw)
                source = "llm"
            except Exception:
                spec = None  # schema violation -> deterministic fallback below
    if spec is None:
        spec = default_report(kb)
        source = "fallback"

    gnd = groundedness(spec, kb)
    rendered = render_report(spec, kb)
    return {
        "status": "ok",
        "source": source,
        "prompt": prompt,
        "spec": spec.model_dump(),
        "rendered": rendered,
        "groundedness": gnd,
        "gemini_available": research.gemini_available(),
    }


# --------------------------------------------------------------------------
# Research chat — a Gemini agent grounded in the belief graph (see ui/research.py).
# Two tools: find_papers (PubMed) and add_to_belief_graph (adds to the Library and
# runs the real lifecycle). The chat model proposes which papers to fetch and
# assess; belief still moves ONLY through the engine (pipeline.process_event).
# --------------------------------------------------------------------------


def _verdict_of(result: EventResult) -> str:
    """Coarse outcome of one processed event, mirroring the frontend's verdictOf:
    committed (belief moved), flagged out-of-distribution, or refused."""
    kinds = [a.kind for a in result.audit]
    if "commit" in kinds:
        return "commit"
    if "flag_ood" in kinds:
        return "flag"
    if "reject" in kinds:
        return "reject"
    return "commit"  # accepted no-op / propagation only


def _make_add_research_papers(session_id: str | None, query: str | None):
    """Build the tool callback for research.run_chat, binding this turn's session id +
    query. It mutates the KB under the lock, files the papers under the session, then
    pushes a live graph + Library update to every /stream subscriber so the other tabs
    (Library, Belief graph) reflect the new belief and the new branch immediately."""

    async def _add(papers: list[dict]) -> dict:
        async with STATE.lock:
            readout = await asyncio.to_thread(
                STATE.add_research_papers, papers, session_id=session_id, query=query
            )
        await STATE.broadcast(
            STATE.graph_payload(moved=readout.get("moved"), include_documents=True)
        )
        return readout

    return _add


def _latest_user_query(messages: list[dict]) -> str:
    """The most recent user turn — used as the research session's title/query."""
    for m in reversed(messages or []):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip()
    return ""


@app.post("/research/chat")
async def research_chat(payload: dict) -> StreamingResponse:
    """Stream a Research-tab chat turn as NDJSON (one JSON object per line). The
    client POSTs {messages:[{role,content}, …]} and reads the body with
    fetch()+ReadableStream (EventSource is GET-only, so it can't carry the history).
    The server is stateless per request apart from the in-turn paper cache."""
    messages = (payload or {}).get("messages") or []
    session_id = ((payload or {}).get("session_id") or "").strip() or None
    query = ((payload or {}).get("query") or "").strip() or _latest_user_query(messages)
    add_papers = _make_add_research_papers(session_id, query)

    async def gen():
        try:
            async for evt in research.run_chat(messages, add_papers=add_papers):
                yield json.dumps(evt) + "\n"
        except Exception as exc:  # never leave the stream hanging
            yield json.dumps({"type": "error", "text": str(exc)}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
