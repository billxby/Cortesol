"""The belief graph container — shared mutable state.

FROZEN CONTRACT for the public method signatures (steward: Branch 1). The KB is
the ONLY thing that holds belief state. It wraps a NetworkX graph plus the typed
records and knows how to snapshot itself to JSON (restartability is the solo
dev's insurance — System Architecture §repo-layout).

IMPORTANT ownership rule: only the engine/validator/propagate (Branch 1) mutate
belief via the methods here. The orchestrator and eval READ the KB and drive it
through the engine; they never poke `claim.ell` directly. (Prime Directive PD1.)

This container is implemented (data + snapshot). Belief-moving logic lives in
engine.py / propagate.py, which call these mutators.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .config import CONTRACT_VERSION, PROPAGATE_KHOP
from .schema import Claim, Edge, Source, TrajectoryPoint


class KB:
    def __init__(self) -> None:
        self.claims: dict[str, Claim] = {}
        self.sources: dict[str, Source] = {}
        self.edges: dict[str, Edge] = {}
        self.graph = nx.MultiDiGraph()  # nodes = claim ids; typed edges mirror self.edges
        self.event_cursor: int = 0  # increments per committed event

    # -- claims --

    def add_claim(self, claim: Claim) -> Claim:
        self.claims[claim.id] = claim
        self.graph.add_node(claim.id)
        return claim

    def get_claim(self, claim_id: str) -> Claim | None:
        return self.claims.get(claim_id)

    def move_belief(self, claim_id: str, delta_ell: float, cause: str) -> None:
        """The ONLY belief mutator. Appends to the trajectory for the audit log.
        Callers (engine/propagate) are responsible for having already capped
        `delta_ell` at DELTA_MAX — the validator guarantees this upstream."""
        cl = self.claims[claim_id]
        cl.ell += delta_ell
        cl.trajectory.append(TrajectoryPoint(t=self.event_cursor, ell=cl.ell, cause=cause))

    # -- sources --

    def add_source(self, source: Source) -> Source:
        self.sources[source.id] = source
        return source

    def get_source(self, source_id: str) -> Source | None:
        return self.sources.get(source_id)

    # -- edges (bi-temporal: invalidate, never delete) --

    def add_edge(self, edge: Edge) -> Edge:
        self.edges[edge.id] = edge
        self.graph.add_edge(edge.src, edge.dst, key=edge.id, type=edge.type.value)
        return edge

    def invalidate_edge(self, edge_id: str, at: int | None = None) -> None:
        e = self.edges[edge_id]
        e.invalid_at = at if at is not None else self.event_cursor

    def live_edges(self) -> list[Edge]:
        return [e for e in self.edges.values() if e.live]

    # -- neighborhood (drives streaming re-propagation) --

    def dirty_neighborhood(self, seeds: set[str], k: int = PROPAGATE_KHOP) -> set[str]:
        """The k-hop claim neighborhood around a set of just-touched claims —
        the only region propagate.py needs to recompute per event (Residual BP,
        Graph Propagation §2)."""
        seen: set[str] = set(seeds)
        frontier = set(seeds)
        for _ in range(k):
            nxt: set[str] = set()
            for cid in frontier:
                if cid in self.graph:
                    nxt.update(self.graph.successors(cid))
                    nxt.update(self.graph.predecessors(cid))
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return seen

    # -- snapshot / restore --

    def snapshot(self, path: str | Path) -> None:
        """Write the whole KB to JSON. Called after every committed event so a
        crash never loses more than one event of work."""
        blob = {
            "contract_version": CONTRACT_VERSION,
            "event_cursor": self.event_cursor,
            "claims": {k: v.model_dump() for k, v in self.claims.items()},
            "sources": {k: v.model_dump() for k, v in self.sources.items()},
            "edges": {k: v.model_dump() for k, v in self.edges.items()},
        }
        Path(path).write_text(json.dumps(blob, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> KB:
        blob = json.loads(Path(path).read_text())
        kb = cls()
        kb.event_cursor = blob.get("event_cursor", 0)
        for k, v in blob["sources"].items():
            kb.add_source(Source(**v))
        for k, v in blob["claims"].items():
            kb.add_claim(Claim(**v))
        for k, v in blob["edges"].items():
            kb.add_edge(Edge(**v))
        return kb
