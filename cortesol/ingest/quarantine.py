"""Quarantine + spotlighting — untrusted text handling (area A, deterministic).

Reference: Research/"Prompt Injection Defense" §Layer-1.
Datamark/delimit incoming text, strip anything resembling system delimiters, and
guarantee it only ever occupies a DATA position downstream — never an instruction.
Input is always RawEvent.untrusted_view() (gold already stripped, PD6).
"""

from __future__ import annotations

from ..core.domain import correlation_group
from ..core.schema import Evidence, RawEvent
from .fieldparse import parse_fields


def quarantine(event: RawEvent) -> Evidence:
    """Wrap/clean the untrusted event and parse it into a structured Evidence.

    The raw text is preserved verbatim as DATA (the screen and UI display it; it is
    never re-executed or placed in an instruction position). Structured `fields`
    are copied as-is — they are what the engine actually reads — and the n_eff
    correlation group (lab x method x dataset) is computed up front.

    When an event carries NO structured metric (real papers, whose measurements
    live in the abstract prose), the deterministic field parser fills the gaps from
    `raw_text`. Existing fields always win, so the simulator path — which always
    ships a `metric` — is never re-parsed and stays byte-identical.
    """
    uv = event.untrusted_view()
    fields = dict(uv.fields)
    if not fields.get("metric"):
        parsed = parse_fields(uv.raw_text)
        for k, v in parsed.items():
            fields.setdefault(k, v)  # never override an existing structured field

    return Evidence(
        id=uv.id,
        source_id=uv.source_id,
        raw_text=uv.raw_text,
        fields=fields,
        correlation_group=correlation_group(fields),
    )
