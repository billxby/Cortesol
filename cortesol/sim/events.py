"""Event generators — the 7 classes (area B).

genuine, noisy, hyped, fraudulent, contradictory, out_of_scope, injection
(schema.EventClass). Each class produces the peptide-result artifacts that make
its gold op self-evident: fraudulent -> fails GRIM/statcheck; hyped -> big claimed
effect, tiny n, marginal p; genuine replication -> fresh lab, consistent effect;
injection -> payload only in DATA fields.

Reference: Fine-Tuning Plan §Stage 0, Fraud and Hype Signals §sim-hooks,
Prompt Injection Defense §attacks. Emits RawEvent with a populated sim_meta.
"""

from __future__ import annotations

from ..core.schema import RawEvent


def emit_stream(seed: int, length: int) -> list[RawEvent]:
    """Generate a balanced-class event stream against a seeded world."""
    ...  # TODO
