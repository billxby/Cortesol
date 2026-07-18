"""Graph propagation + the retraction cascade.

Logical area A (Ground-Truth Engine). Deterministic, no LLM.
Reference: Research/"Graph Propagation and GNNs" §1-2, "Belief Revision" §TMS.

TruthFinder source<->claim loop + typed belief propagation over the dirty k-hop
neighborhood (config.EDGE_INFLUENCE / BP_* knobs). Discrediting a source sinks
everything it supported — the demo money shot plain RAG can't do.
"""

from __future__ import annotations

from .kb import KB
from .results import Delta


def propagate(kb: KB, dirty: set[str]) -> list[Delta]:
    """Recompute dependent claims in the dirty neighborhood; return ripple Deltas."""
    ...  # TODO


def discredit_source(kb: KB, source_id: str) -> list[Delta]:
    """Retraction cascade: flag the source, propagate the un-belief downstream."""
    ...  # TODO
