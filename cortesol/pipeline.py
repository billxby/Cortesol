"""The orchestrator — the per-event lifecycle. THE GLUE (logical area C).

This is the seam that wires areas A and B together. It owns no belief math and no
model — it sequences the frozen-contract calls and assembles the EventResult for
the audit log / SSE / eval. Because every dependency is behind an interface, this
runs today with FakeExtractor and the real engine.

The lifecycle (System Architecture §update-lifecycle):
  1. quarantine   event.untrusted_view() -> Evidence            (area A)
  2. retrieve     top-k claims + neighborhood -> Context        (area C)
  3. extract      Context -> ProposedOps                        (area B, the model)
  4. screen       deterministic red flags -> evidence.red_flags (area A)
  5. validate     ProposedOps -> accepted / rejected            (area A, the gate)
  6. commit+propagate  engine.apply + propagate over dirty set  (area A)
  7. publish      diff -> audit log -> SSE -> UI, snapshot KB    (area C)
"""

from __future__ import annotations

import os
from pathlib import Path

from .core import config, engine, propagate, validator
from .core.context import Context
from .core.kb import KB
from .core.ops import (
    AddClaim,
    AddEdge,
    ApplyEvidence,
    FlagOOD,
    InvalidateEdge,
    ProposedOps,
    Reject,
)
from .core.results import AuditEntry, EventResult
from .core.schema import RawEvent, Source
from .ingest.quarantine import quarantine
from .ingest.screen import screen
from .retrieval import retrieve


def _default_extractor():
    """The offline default is `FakeExtractor` (deterministic, no network), so
    tests never touch a model. Select `flash`, `freesolo`, or `ollama` through
    `CORTESOL_EXTRACTOR` for a live backend."""
    backend = os.environ.get("CORTESOL_EXTRACTOR", "fake").lower()
    if backend == "flash":
        from .eval.baselines import ModelBaseline

        return ModelBaseline("engine+flash")
    if backend == "ollama":
        from .serve.ollama import OllamaExtractor

        return OllamaExtractor()
    if backend in {"freesolo", "remote"}:
        from .ingest.extract import FreesoloExtractor

        run_id = os.environ.get("FREESOLO_RUN_ID")
        if not run_id:
            raise RuntimeError("FREESOLO_RUN_ID is required for the remote extractor")
        return FreesoloExtractor(run_id)
    from .ingest.extract import FakeExtractor

    return FakeExtractor()


def _op_summary(op) -> str:
    if isinstance(op, ApplyEvidence):
        return f"APPLY_EVIDENCE {op.direction}{op.strength} -> {op.claim_id}"
    if isinstance(op, AddClaim):
        return f"ADD_CLAIM {op.text!r}"
    if isinstance(op, AddEdge):
        return f"ADD_EDGE {op.src} -{op.type.value}-> {op.dst}"
    if isinstance(op, InvalidateEdge):
        return f"INVALIDATE_EDGE {op.edge_id}"
    if isinstance(op, FlagOOD):
        return f"FLAG_OOD {op.reason}"
    if isinstance(op, Reject):
        return f"REJECT ({op.reason})"
    return op.op


def prepare_event(kb: KB, event: RawEvent) -> Context:
    """Quarantine and retrieve the only context visible to the model.

    Screening intentionally does not happen here: deterministic red flags are a
    ledger-side safety mechanism, not labels that may leak into model inputs.
    """
    kb.event_cursor = event.t  # trajectory points on this event carry its index
    evidence = quarantine(event)
    if evidence.source_id not in kb.sources:
        tier = str(evidence.fields.get("source_tier") or config.DEFAULT_SOURCE_TIER)
        kb.add_source(Source.from_tier(evidence.source_id, tier))
    return retrieve(kb, event.untrusted_view(), evidence)


def commit_proposal(
    kb: KB,
    ctx: Context,
    proposed: ProposedOps,
    *,
    snapshot_dir=None,
) -> EventResult:
    """Screen, validate, and commit a proposal through the real ledger."""
    evidence = ctx.evidence
    event = ctx.event
    # The screen runs after extraction but before validation/commit. This keeps
    # red flags out of the prompt while guaranteeing they affect every update.
    screen(evidence, kb)
    vr = validator.validate(kb, proposed, evidence)

    # 6. commit accepted ops + propagate the ripple over the dirty neighborhood
    deltas = []
    seeds: set[str] = set()
    audit: list[AuditEntry] = []

    created: set[str] = set()  # claim ids materialised by ADD_CLAIM this event
    for op in vr.accepted:
        produced = engine.apply(kb, op, evidence)
        deltas.extend(produced)
        for d in produced:
            seeds.add(d.claim_id)
        kind = "commit"
        if isinstance(op, FlagOOD):
            kind = "flag_ood"
        elif isinstance(op, Reject):
            kind = "reject"
        elif isinstance(op, AddClaim):
            # engine.apply materialised a new node (no delta); surface it so the UI
            # reveals it and retrieval can find it on later events.
            created.add(engine.claim_id_for(op.text))
        audit.append(
            AuditEntry(
                t=event.t,
                kind=kind,
                detail=_op_summary(op),
                op=op.op,
                delta=produced[0] if produced else None,
            )
        )

    for rj in vr.rejected:
        audit.append(
            AuditEntry(
                t=event.t,
                kind="reject",
                detail=f"{_op_summary(rj.op)} — REJECTED: {rj.reason}",
                op=rj.op.op,
            )
        )

    dirty = kb.dirty_neighborhood(seeds) if seeds else set()
    ripple = propagate.propagate(kb, dirty)
    deltas.extend(ripple)
    for d in ripple:
        audit.append(AuditEntry(t=event.t, kind="propagate", detail=d.cause, delta=d))

    # 7. publish — snapshot so a crash loses at most one event
    if snapshot_dir is not None:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        kb.snapshot(Path(snapshot_dir) / "kb_latest.json")

    return EventResult(
        event_id=event.id,
        t=event.t,
        validation=vr,
        deltas=deltas,
        audit=audit,
        dirty_claims=sorted({d.claim_id for d in deltas} | created),
    )


def process_event(kb: KB, event: RawEvent, extractor=None, snapshot_dir=None) -> EventResult:
    """Run one event through the 7-step lifecycle and return its EventResult.
    `extractor` defaults to FakeExtractor so the loop runs with no model."""
    extractor = extractor or _default_extractor()
    ctx = prepare_event(kb, event)
    proposed = extractor.extract(ctx) if hasattr(extractor, "extract") else extractor(ctx)
    return commit_proposal(kb, ctx, proposed, snapshot_dir=snapshot_dir)


def replay_stream(kb: KB, events: list[RawEvent], extractor=None) -> list[EventResult]:
    """Process a whole stream in order. Used by the eval harness and the demo."""
    extractor = extractor or _default_extractor()
    return [process_event(kb, ev, extractor) for ev in events]
