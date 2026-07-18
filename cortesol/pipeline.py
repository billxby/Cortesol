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

from pathlib import Path

from .core import engine, propagate, validator
from .core.kb import KB
from .core.ops import AddClaim, AddEdge, ApplyEvidence, FlagOOD, InvalidateEdge, Reject
from .core.results import AuditEntry, EventResult
from .core.schema import RawEvent
from .ingest.quarantine import quarantine
from .ingest.screen import screen
from .retrieval import retrieve


def _default_extractor():
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


def process_event(kb: KB, event: RawEvent, extractor=None, snapshot_dir=None) -> EventResult:
    """Run one event through the 7-step lifecycle and return its EventResult.
    `extractor` defaults to FakeExtractor so the loop runs with no model."""
    extractor = extractor or _default_extractor()
    kb.event_cursor = event.t  # trajectory points on this event carry its index

    # 1. quarantine (operates on untrusted_view internally — gold never leaks)
    evidence = quarantine(event)
    # 2. retrieve a bounded context around the untrusted event
    ctx = retrieve(kb, event.untrusted_view(), evidence)
    # 3. extract — the model proposes ops (it does not write)
    proposed = extractor.extract(ctx)
    # 4. screen — deterministic red flags, regardless of the model
    screen(evidence, kb)
    # 5. validate — the sole write gate
    vr = validator.validate(kb, proposed, evidence)

    # 6. commit accepted ops + propagate the ripple over the dirty neighborhood
    deltas = []
    seeds: set[str] = set()
    audit: list[AuditEntry] = []

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
        dirty_claims=sorted({d.claim_id for d in deltas}),
    )


def replay_stream(kb: KB, events: list[RawEvent], extractor=None) -> list[EventResult]:
    """Process a whole stream in order. Used by the eval harness and the demo."""
    extractor = extractor or _default_extractor()
    return [process_event(kb, ev, extractor) for ev in events]
