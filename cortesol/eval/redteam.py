"""Red-team battery (area C) — the injection story, quantified.

Reference: Prompt Injection Defense §red-team-battery.
  1. direct-override phrasings   2. MINJA-style delayed poison
  3. authority spoof             4. social pressure ("are you sure?" x5)
  5. correlated hype flood (tests n_eff)
Measure ASR (any op exceeding gold) + Δconfidence under attack (should be ~0).
The validator's bound means worst case is one capped, auditable nudge.
"""

from __future__ import annotations

from ..core.schema import RawEvent


def attack_stream() -> list[RawEvent]:
    """Build the adversarial event battery (injection payloads in DATA fields only)."""
    ...  # TODO
