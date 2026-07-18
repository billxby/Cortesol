"""Retrieval — build the bounded Context the extractor reads (area C).

Embed the incoming event -> top-k relevant claims + their 1-hop neighborhood +
source records (config.RETRIEVE_TOP_K). Must fit CONTEXT_TOKEN_BUDGET. Both the
live pipeline and the GRPO environment call this, so keep it dependency-light; a
trivial tag/recency version works before embeddings land.

Reference: System Architecture §lifecycle step 2.
"""

from __future__ import annotations

import re

from .core import config
from .core.context import Context
from .core.kb import KB
from .core.schema import Evidence, RawEvent


def retrieve(kb: KB, event: RawEvent, evidence: Evidence, k: int | None = None) -> Context:
    """Select the top-k relevant claims + neighborhood and pack a Context."""
    if event.sim_meta is not None:
        raise ValueError("retrieval accepts only an untrusted event view")
    limit = config.RETRIEVE_TOP_K if k is None else max(0, k)
    haystack = f"{event.raw_text} {event.fields}".lower()
    tokens = set(re.findall(r"[a-z0-9_]+", haystack))

    def relevance(claim) -> tuple[float, str]:
        claim_tokens = set(
            re.findall(
                r"[a-z0-9_]+", f"{claim.id} {claim.text} {' '.join(claim.ontology_tags)}".lower()
            )
        )
        overlap = len(tokens & claim_tokens)
        return (float(overlap) + 0.001 * len(claim.trajectory), claim.id)

    ranked = sorted(kb.claims.values(), key=relevance, reverse=True)
    selected = ranked[:limit]
    selected_ids = {claim.id for claim in selected}
    edges = [
        edge for edge in kb.live_edges() if edge.src in selected_ids or edge.dst in selected_ids
    ]
    neighbor_ids = {
        endpoint for edge in edges for endpoint in (edge.src, edge.dst) if endpoint in kb.claims
    }
    for claim_id in sorted(neighbor_ids - selected_ids):
        selected.append(kb.claims[claim_id])
    sources = []
    source = kb.get_source(evidence.source_id)
    if source is not None:
        sources.append(source)
    return Context(event=event, evidence=evidence, claims=selected, sources=sources, edges=edges)
