"""The orchestrator — the per-event lifecycle. THE GLUE (logical area C).

This is the seam that wires areas A and B together. It owns no belief math and no
model — it sequences the frozen-contract calls and assembles the EventResult for
the audit log / SSE / eval. Because every dependency is behind an interface, this
runs today with FakeExtractor and a stubbed engine, and swaps in the real ones as
they land.

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

from .core.kb import KB
from .core.results import EventResult
from .core.schema import RawEvent


def process_event(kb: KB, event: RawEvent, extractor=None) -> EventResult:
    """Run one event through the 7-step lifecycle and return its EventResult.
    `extractor` defaults to FakeExtractor so the loop runs with no model."""
    ...  # TODO: sequence the steps above; snapshot kb after commit


def replay_stream(kb: KB, events: list[RawEvent], extractor=None) -> list[EventResult]:
    """Process a whole stream in order. Used by the eval harness and the demo."""
    ...  # TODO
