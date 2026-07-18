"""The orchestrator — the per-event lifecycle. THE GLUE (logical area C).

This is the seam that wires areas A and B together. It owns no belief math and no
model — it sequences the frozen-contract calls and assembles the EventResult for
the audit log / SSE / eval. Because every dependency is behind an interface, this
runs today with FakeExtractor and a stubbed engine, and swaps in the real ones as
they land.

The lifecycle (System Architecture §update-lifecycle):
  1. quarantine   event.untrusted_view() -> Evidence            (area A)
  2. screen       deterministic red flags -> evidence.red_flags (area A)
  3. retrieve     top-k claims + neighborhood -> Context        (area C)
  4. extract      Context -> ProposedOps                        (area B, the model)
  5. validate     ProposedOps -> accepted / rejected            (area A, the gate)
  6. commit+propagate  engine.apply + propagate over dirty set  (area A)
  7. publish      diff -> audit log -> SSE -> UI, snapshot KB    (area C)
"""

from __future__ import annotations

from .core import config
from .core.context import Context
from .core.engine import apply
from .core.kb import KB
from .core.ops import AddEdge, ApplyEvidence, InvalidateEdge, ProposedOps
from .core.propagate import propagate
from .core.results import AuditEntry, EventResult
from .core.schema import RawEvent, Source
from .core.validator import validate
from .ingest.quarantine import quarantine
from .ingest.screen import screen
from .retrieval import retrieve


def prepare_event(kb: KB, event: RawEvent) -> Context:
    """Create the only model-visible context for an event."""
    public = event.untrusted_view()
    evidence = quarantine(public)
    if evidence.source_id not in kb.sources:
        tier = str(evidence.fields.get("source_tier") or config.DEFAULT_SOURCE_TIER)
        kb.add_source(Source.from_tier(evidence.source_id, tier))
    screen(evidence, kb)
    return retrieve(kb, public, evidence)


def commit_proposal(kb: KB, ctx: Context, proposed: ProposedOps) -> EventResult:
    """Validate and commit one proposal through the deterministic ledger."""
    verdict = validate(kb, proposed, ctx.evidence)
    deltas = []
    audit: list[AuditEntry] = []
    dirty: set[str] = set()
    for rejected in verdict.rejected:
        audit.append(
            AuditEntry(
                t=ctx.event.t,
                kind="reject",
                detail=rejected.reason,
                op=rejected.op.op,
            )
        )
    for op in verdict.accepted:
        produced = apply(kb, op, ctx.evidence)
        deltas.extend(produced)
        if isinstance(op, ApplyEvidence):
            dirty.add(op.claim_id)
        elif isinstance(op, AddEdge):
            dirty.update((op.src, op.dst))
        elif isinstance(op, InvalidateEdge):
            edge = kb.edges.get(op.edge_id)
            if edge:
                dirty.update((edge.src, edge.dst))
        audit.append(
            AuditEntry(
                t=ctx.event.t,
                kind="flag_ood" if op.op == "FLAG_OOD" else "commit",
                detail=f"accepted {op.op}",
                op=op.op,
                delta=produced[0] if produced else None,
            )
        )
    ripple = propagate(kb, kb.dirty_neighborhood(dirty)) if dirty else []
    deltas.extend(ripple)
    audit.extend(
        AuditEntry(t=ctx.event.t, kind="propagate", detail=d.cause, delta=d) for d in ripple
    )
    result = EventResult(
        event_id=ctx.event.id,
        t=ctx.event.t,
        validation=verdict,
        deltas=deltas,
        audit=audit,
        dirty_claims=sorted(dirty),
    )
    kb.event_cursor += 1
    return result


def process_event(kb: KB, event: RawEvent, extractor=None) -> EventResult:
    """Run one event through the 7-step lifecycle and return its EventResult.
    `extractor` defaults to FakeExtractor so the loop runs with no model."""
    if extractor is None:
        from .ingest.extract import FakeExtractor

        extractor = FakeExtractor()
    ctx = prepare_event(kb, event)
    if hasattr(extractor, "extract"):
        proposed = extractor.extract(ctx)
    else:
        proposed = extractor(ctx)
    return commit_proposal(kb, ctx, proposed)


def replay_stream(kb: KB, events: list[RawEvent], extractor=None) -> list[EventResult]:
    """Process a whole stream in order. Used by the eval harness and the demo."""
    return [process_event(kb, event, extractor=extractor) for event in events]
