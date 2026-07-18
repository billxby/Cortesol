"""Quarantine + spotlighting — untrusted text handling (area A, deterministic).

Reference: Research/"Prompt Injection Defense" §Layer-1.
Datamark/delimit incoming text, strip anything resembling system delimiters, and
guarantee it only ever occupies a DATA position downstream — never an instruction.
Input is always RawEvent.untrusted_view() (gold already stripped, PD6).
"""

from __future__ import annotations

from ..core.domain import correlation_group
from ..core.schema import Evidence, RawEvent


def quarantine(event: RawEvent) -> Evidence:
    """Wrap/clean the untrusted event and parse it into a structured Evidence.

    The raw text is preserved verbatim as DATA (the screen and UI display it; it is
    never re-executed or placed in an instruction position). Structured `fields`
    are copied as-is — they are what the engine actually reads — and the n_eff
    correlation group (lab x method x dataset) is computed up front.
    """
    uv = event.untrusted_view()
    return Evidence(
        id=uv.id,
        source_id=uv.source_id,
        raw_text=uv.raw_text,
        fields=dict(uv.fields),
        correlation_group=correlation_group(uv.fields),
    )
