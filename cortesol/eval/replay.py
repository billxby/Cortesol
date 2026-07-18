"""Replay harness (area C). Run streams through the pipeline for each system and
print the eval table. Entry point for `make eval`.

Reference: Fine-Tuning Plan §eval-protocol. Held-out streams from unseen seeds +
a shifted event-class mix (OOD test), plus the red-team battery.
"""

from __future__ import annotations


def main() -> None:
    """Replay held-out streams for every baseline + tuned system; print the table."""
    ...  # TODO


if __name__ == "__main__":
    main()
