"""Domainsmith — draft ANY field of knowledge into a `core.domains.Domain` (area C).

This realises the headline vision: a researcher describes a field in plain English,
an LLM DRAFTS a domain ontology, a human reviews/activates it, and the SAME gated
belief engine then tracks calibrated belief in that field.

Mission-loyalty rules baked in here:
  * A Domain is TRUSTED CONFIG authored via a human-in-the-loop — NEVER belief. The
    LLM proposes an ontology (entity/property types, scope, seed propositions); the
    human reviews and activates it. Nothing here sets confidence.
  * Belief still moves ONLY through the engine. `seed_claims_into` creates claim
    NODES structurally at the skeptical prior (via `kb.add_claim`) — exactly like
    ADD_CLAIM. It never assigns `ell`; confidence is still EARNED when evidence
    later flows through the pipeline.
  * The ontology gate: a seed claim may only carry tags whose namespace is one of
    the draft's DECLARED entity/property types (mirroring the validator's
    `_tag_in_ontology`). Out-of-namespace tags are rejected/flagged, never seeded —
    an LLM can propose an ontology, but it cannot smuggle in an off-ontology claim.

Everything here is pure/deterministic EXCEPT `draft_domain`'s optional LLM call,
which degrades to `_fallback_draft` (a deterministic keyword builder) when there is
no GEMINI_API_KEY or on any error — so "import a field" always works offline.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from ..core.domains import Domain
from ..core.engine import claim_id_for
from ..core.kb import KB
from ..core.schema import Claim, Source
from . import research

# --------------------------------------------------------------------------
# The draft schema — the constrained shape the LLM (or the fallback) fills in.
# --------------------------------------------------------------------------


class SeedClaim(BaseModel):
    """One proposition to seed into the field's graph. It becomes a claim NODE at
    the skeptical prior — never a set confidence. `tags` must sit within the draft's
    declared namespaces (the ontology gate)."""

    text: str
    tags: list[str] = Field(default_factory=list)


class DomainDraft(BaseModel):
    """An LLM- or fallback-drafted field of knowledge, pending human review. Maps
    onto a `core.domains.Domain` via `to_domain`. Only `name`/`label` +
    entity/property types are load-bearing; the rest refine the screen and seed the
    graph. This is TRUSTED CONFIG, not belief."""

    name: str
    label: str
    entity_types: list[str] = Field(default_factory=list)
    property_types: list[str] = Field(default_factory=list)
    out_of_scope_examples: list[str] = Field(default_factory=list)
    out_of_scope_markers: list[str] = Field(default_factory=list)
    # metric -> [min, max]; either bound may be null (JSON) -> None.
    plausible_value_bounds: dict[str, list[float | None]] = Field(default_factory=dict)
    seed_claims: list[SeedClaim] = Field(default_factory=list)


# The injectable LLM seam: (description, json_schema) -> parsed dict | None.
DraftLLM = Callable[[str, dict], Awaitable[dict | None]]


# --------------------------------------------------------------------------
# Slug / keyword helpers (pure, deterministic).
# --------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[a-z0-9]+")
# Deliberately small: enough to strip filler from a one-line field description so the
# fallback keys on the meaningful nouns. Not a linguistics project.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "is", "are",
        "with", "how", "what", "does", "do", "about", "field", "knowledge", "study",
        "studies", "research", "science", "effect", "effects", "its", "their", "this",
        "that", "into", "from", "any", "all", "we", "track", "tracking", "belief",
        "claims", "claim", "which", "whether", "impact", "impacts", "affect", "affects",
        "influence", "influences", "consumption",
    }
)


def _slug(s: str) -> str:
    """`Coffee Bean!` -> `coffee_bean`. Empty -> `x` (never an empty namespace)."""
    out = _SLUG_RE.sub("_", (s or "").strip().lower()).strip("_")
    return out or "x"


def _keywords(description: str) -> list[str]:
    """Ordered, de-duplicated significant tokens from a free-text description."""
    seen: list[str] = []
    for w in _WORD_RE.findall((description or "").lower()):
        if len(w) > 2 and w not in _STOPWORDS and w not in seen:
            seen.append(w)
    return seen


# --------------------------------------------------------------------------
# The ontology gate — which seed tags are in the draft's declared namespaces.
# --------------------------------------------------------------------------


def draft_namespaces(draft: DomainDraft) -> frozenset[str]:
    """The in-scope namespaces a seed claim tag may use — the draft's entity +
    property types. Mirrors `Domain.in_scope_namespaces`."""
    return frozenset(draft.entity_types) | frozenset(draft.property_types)


def _tag_ok(tag: str, namespaces: frozenset[str]) -> bool:
    """A tag is `<namespace>:<slug>` with namespace declared by the draft."""
    return ":" in tag and tag.split(":", 1)[0] in namespaces


def out_of_namespace_tags(draft: DomainDraft) -> list[str]:
    """Every seed-claim tag that is NOT in the draft's declared namespaces — the
    ontology-gate violations. Empty list == the draft is internally grounded."""
    ns = draft_namespaces(draft)
    bad: list[str] = []
    for sc in draft.seed_claims:
        bad += [t for t in sc.tags if not _tag_ok(t, ns)]
    return bad


def validate_draft(draft: DomainDraft) -> list[str]:
    """Human-readable problems that should block a clean activation. A non-empty
    list does not crash the build — the route surfaces it and seeding skips the
    offending claims — but it is the gate's verdict on the draft's groundedness."""
    problems: list[str] = []
    if not draft.entity_types:
        problems.append("no entity types declared")
    if not draft.property_types:
        problems.append("no property types declared")
    bad = out_of_namespace_tags(draft)
    if bad:
        problems.append(
            "seed-claim tags outside the declared ontology: " + ", ".join(sorted(set(bad)))
        )
    return problems


# --------------------------------------------------------------------------
# Draft -> Domain builder + structural seeding (NEVER belief).
# --------------------------------------------------------------------------


def to_domain(draft: DomainDraft) -> Domain:
    """Build the frozen `Domain` from a reviewed draft. Only config — no belief.

    `plausible_value_bounds` becomes the generic, field-independent screen bound
    (metric -> (min, max)); a domain with none simply skips the check, so a
    non-chemistry field never trips a peptide-shaped flag."""
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for metric, pair in (draft.plausible_value_bounds or {}).items():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            lo, hi = pair
            lo_f = float(lo) if isinstance(lo, (int, float)) else None
            hi_f = float(hi) if isinstance(hi, (int, float)) else None
            bounds[str(metric)] = (lo_f, hi_f)
    return Domain(
        name=_slug(draft.name),
        label=draft.label.strip() or draft.name.strip() or "Custom field",
        entity_types=tuple(draft.entity_types),
        property_types=tuple(draft.property_types),
        out_of_scope_examples=tuple(draft.out_of_scope_examples),
        out_of_scope_markers=tuple(m.lower() for m in draft.out_of_scope_markers),
        plausible_value_bounds=bounds,
    )


def seed_claims_into(kb: KB, draft: DomainDraft, source_id: str = "curator") -> dict:
    """Add the draft's seed claims to `kb` as NODES at the skeptical prior, plus a
    provenance Source for the human curator. STRUCTURAL only — like ADD_CLAIM, this
    never sets `ell`; every seeded claim starts at PRIOR_C_0 with high u, and
    confidence still moves only through the engine.

    The ontology gate runs here: a seed claim is seeded only when EVERY tag sits in
    the draft's declared namespaces; a claim with any out-of-namespace tag is
    rejected (skipped), so the LLM cannot introduce an off-ontology proposition.

    Returns {seeded, rejected}: the claim ids materialised and the (text, bad_tags)
    of every rejected claim."""
    if kb.get_source(source_id) is None:
        # A curator record so provenance exists; it never moves belief (no evidence
        # flows at seed time) — the tier is cosmetic.
        kb.add_source(Source.from_tier(source_id, "reputable"))

    namespaces = draft_namespaces(draft)
    seeded: list[str] = []
    rejected: list[dict] = []
    for sc in draft.seed_claims:
        text = (sc.text or "").strip()
        if not text:
            continue
        bad = [t for t in sc.tags if not _tag_ok(t, namespaces)]
        if bad or not sc.tags:
            rejected.append({"text": text, "bad_tags": bad or ["(no tags)"]})
            continue
        cid = claim_id_for(text)
        if cid not in kb.claims:
            # Default ell == logit(PRIOR_C_0): the prior. We do NOT pass ell. (PD2.)
            kb.add_claim(Claim(id=cid, text=text, ontology_tags=list(sc.tags)))
            seeded.append(cid)
    return {"seeded": seeded, "rejected": rejected}


# --------------------------------------------------------------------------
# The deterministic fallback (no LLM) — a minimal sensible field from keywords.
# --------------------------------------------------------------------------


def _fallback_draft(description: str) -> DomainDraft:
    """Build a minimal, valid `DomainDraft` from a description using only keyword
    heuristics — the offline path when there is no GEMINI_API_KEY or the LLM errors.
    Pure and deterministic: the same description always yields the same draft.

    The first significant token becomes the field's core ENTITY type; the next few
    become tracked PROPERTY types; and one seed claim per property links the two, so
    the seeded graph is in-scope by construction (every tag sits in a declared
    namespace) and the offline GenericFakeExtractor has something to attach to."""
    kws = _keywords(description)
    subject = _slug(kws[0]) if kws else "topic"
    props = []
    for k in kws[1:6]:
        s = _slug(k)
        if s != subject and s not in props:
            props.append(s)
    if not props:
        props = ["property"]

    label = (description or "").strip()
    label = (label[:69] + "…") if len(label) > 70 else (label or f"{subject.title()} research")

    entity_types = [subject, "source"]
    property_types = props
    seed_claims = [
        SeedClaim(
            text=f"{subject} affects {p}".replace("_", " "),
            tags=[f"{subject}:{subject}", f"{p}:observed"],
        )
        for p in props[:3]
    ]
    return DomainDraft(
        name=subject,
        label=label,
        entity_types=entity_types,
        property_types=property_types,
        out_of_scope_examples=[
            f"claims with no {subject.replace('_', ' ')} entity",
            "off-topic or unrelated claims",
        ],
        out_of_scope_markers=[],
        plausible_value_bounds={},
        seed_claims=seed_claims,
    )


def _normalize_draft(draft: DomainDraft) -> DomainDraft:
    """Clean an arbitrary (LLM-produced) draft into a consistent, valid shape:
    slug the name + every namespace, align seed-tag namespaces to the slugged
    types, and guarantee non-empty entity/property lists. Idempotent, so running it
    over `_fallback_draft` output changes nothing."""
    ents = _dedupe(_slug(t) for t in draft.entity_types)
    props = _dedupe(_slug(t) for t in draft.property_types)
    if not ents:
        ents = ["entity"]
    if not props:
        props = ["property"]

    def _fix_tag(tag: str) -> str:
        if ":" in tag:
            ns, slug = tag.split(":", 1)
            return f"{_slug(ns)}:{_slug(slug)}"
        return f"{_slug(tag)}:general"

    seeds = [
        SeedClaim(text=(sc.text or "").strip(), tags=[_fix_tag(t) for t in sc.tags])
        for sc in draft.seed_claims
        if (sc.text or "").strip()
    ]
    return DomainDraft(
        name=_slug(draft.name),
        label=(draft.label or "").strip() or f"{_slug(draft.name).title()} research",
        entity_types=ents,
        property_types=props,
        out_of_scope_examples=[e.strip() for e in draft.out_of_scope_examples if e.strip()],
        out_of_scope_markers=_dedupe(
            m.strip().lower() for m in draft.out_of_scope_markers if m.strip()
        ),
        plausible_value_bounds=dict(draft.plausible_value_bounds or {}),
        seed_claims=seeds,
    )


def _dedupe(items) -> list[str]:
    seen: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.append(it)
    return seen


# --------------------------------------------------------------------------
# The public draft entry point — LLM with a deterministic fallback.
# --------------------------------------------------------------------------


async def draft_domain(description: str, llm: DraftLLM | None = None) -> tuple[str, DomainDraft]:
    """Draft a field of knowledge from a plain-English description.

    Asks the Gemini seam (constrained to `DomainDraft.model_json_schema()`) to
    propose an ontology; on no key / any error / a schema violation, falls back to
    the deterministic `_fallback_draft`. Returns (source, draft) where source is
    "llm" or "fallback". Read-only over belief — this only DRAFTS config; activation
    and seeding happen later and still never set confidence.

    `llm` is injectable for testing; the default is `research.propose_domain_draft`,
    which itself returns None (no network) when no key is configured."""
    call = llm if llm is not None else research.propose_domain_draft
    raw = None
    try:
        raw = await call(description, DomainDraft.model_json_schema())
    except Exception:
        raw = None
    if raw is not None:
        try:
            return "llm", _normalize_draft(DomainDraft.model_validate(raw))
        except Exception:
            pass  # bad shape -> deterministic fallback
    return "fallback", _fallback_draft(description)
