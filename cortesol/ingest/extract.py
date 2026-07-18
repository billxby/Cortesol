"""The extractor — the LLM update policy (logical area B, the Freesolo model).

Reads a serialized Context (quarantined event + retrieved state) and emits
ProposedOps under JSON-schema-constrained decoding (ops.ops_json_schema()).
Points the openai client at the Flash deployment (.env FLASH_*). This is the model
we fine-tune: stock 4B -> SFT -> GRPO -> OPD.

Reference: System Architecture §lifecycle step 3, Fine-Tuning Plan.
The model PROPOSES; it never writes state. Its `think` trace is advisory only.
"""

from __future__ import annotations

import re

from ..core.context import Context
from ..core.domain import MIN_PLAUSIBLE_KD_PM
from ..core.ops import ApplyEvidence, FlagOOD, ProposedOps, Reject
from ..core.schema import Claim

_PEPTIDE_RE = re.compile(r"\bP\d+\b")
# Injection tells that live ONLY in the DATA position (raw_text). The extractor
# recognising these and REFUSING is legitimate — it reads the text, it does not
# obey it. (Prompt Injection Defense: text is data.)
_INJECTION_MARKERS = (
    "ignore prior instructions",
    "ignore previous instructions",
    "disregard",
    "system:",
    "set confidence",
    "override",
)
_OOS_MARKERS = ("small-molecule", "small molecule", "non-peptide", "antibody")


def extract(ctx: Context, model: str | None = None) -> ProposedOps:
    """Call the (tuned) model with the op schema and return ProposedOps.

    Not wired to a live deployment in this build — the demo and eval run on
    `FakeExtractor`. Kept as the seam the real Flash client slots into.
    """
    raise NotImplementedError("live Flash extractor not wired; use FakeExtractor")


def _match_claim(ctx: Context, prefer_efficacy: bool) -> Claim | None:
    """Map the untrusted event to a retrieved claim by peptide identity. Reads only
    the event text/fields + the retrieved claim texts (never any gold)."""
    raw = ctx.event.raw_text or ""
    m = _PEPTIDE_RE.search(raw)
    pep = m.group(0) if m else None
    kind = "efficacy" if prefer_efficacy else "bind"

    if pep:
        exact = [c for c in ctx.claims if pep.lower() in c.text.lower() and kind in c.id]
        if exact:
            return exact[0]
        any_pep = [c for c in ctx.claims if pep.lower() in c.text.lower()]
        if any_pep:
            return any_pep[0]
    return ctx.claims[0] if ctx.claims else None


class FakeExtractor:
    """Deterministic proposer so areas A and C can run the pipeline with no
    model/network. It reads the same UNTRUSTED Context the tuned model would and
    emits schema-valid ops — a stand-in that is credulous about clean results but
    refuses blatant injections / out-of-scope / physically impossible reports. It
    NEVER consults sim_meta (PD6); the screen, validator and engine do the rest.
    """

    def extract(self, ctx: Context, model: str | None = None) -> ProposedOps:
        raw = (ctx.event.raw_text or "")
        low = raw.lower()
        f = ctx.evidence.fields
        eid = ctx.evidence.id

        # 1. Injection payload in the data position -> refuse.
        if any(marker in low for marker in _INJECTION_MARKERS):
            return ProposedOps(
                think="raw_text carries an instruction-like payload; text is data, refusing.",
                ops=[Reject(evidence_id=eid, reason="injection")],
            )

        metric = f.get("metric")
        value = f.get("value")

        # 2. Physically impossible binding affinity -> unverifiable.
        if metric == "Kd" and f.get("units") == "nM" and isinstance(value, (int, float)):
            if value * 1000.0 < MIN_PLAUSIBLE_KD_PM:  # sub-picomolar, past diffusion limit
                return ProposedOps(
                    think=f"Kd={value} nM is below the diffusion limit; unverifiable.",
                    ops=[Reject(evidence_id=eid, reason="unverifiable")],
                )

        prefer_efficacy = metric == "percent_inhibition"

        # 3. Out of scope: no peptide entity anywhere -> flag, do not force-fit.
        has_peptide = bool(_PEPTIDE_RE.search(raw))
        if not has_peptide or any(marker in low for marker in _OOS_MARKERS):
            return ProposedOps(
                think="no peptide entity / small-molecule claim; outside the ontology.",
                ops=[FlagOOD(payload=raw[:200], reason="no peptide entity in scope")],
            )

        claim = _match_claim(ctx, prefer_efficacy)
        if claim is None:
            return ProposedOps(
                think="could not map the report to a known claim.",
                ops=[FlagOOD(payload=raw[:200], reason="unmapped claim")],
            )

        # 4. Direction + strength from the structured fields (not from any label).
        if prefer_efficacy:
            direction = "+"
            strength = "weak"  # functional preclinical efficacy — hype-prone, weak by default
        else:
            # binding: low Kd == binds (+), very high Kd == fails to bind (-)
            if isinstance(value, (int, float)) and value > 3000.0:
                direction = "-"
                strength = "moderate"
            else:
                direction = "+"
                p = f.get("p")
                n = f.get("n")
                strong = (
                    isinstance(p, (int, float)) and p < 0.01
                    and isinstance(n, int) and n >= 3
                    and f.get("control_peptide") is True
                    and isinstance(f.get("purity_pct"), (int, float))
                    and f.get("purity_pct") >= 95.0
                )
                strength = "strong" if strong else "moderate"

        return ProposedOps(
            think=f"{claim.id}: {metric}={value} -> APPLY_EVIDENCE {direction}{strength}.",
            ops=[
                ApplyEvidence(
                    claim_id=claim.id,
                    direction=direction,
                    strength=strength,
                    evidence_id=eid,
                )
            ],
        )
