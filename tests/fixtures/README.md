# Fixtures

`toy_peptide_stream.jsonl` — six hand-written events, one per interesting class
(`genuine`, `hyped`, `fraudulent`, `injection`, `out_of_scope`, `contradictory`).
It doubles as documentation of the `RawEvent` shape and the peptide evidence
fields. Each line has public (untrusted) fields plus a `sim_meta` sidecar holding
the ground truth and the gold ops — the thing the extractor/engine must never see
(strip it with `RawEvent.untrusted_view()`); only the eval harness reads it.

Use it as the shared substrate all three areas can test against before the real
simulator exists.
