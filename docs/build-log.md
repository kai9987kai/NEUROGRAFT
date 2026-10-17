# Local build record

## Implemented

- Immutable multi-source schemas, sparse graph statistics, source and topology hashes, readiness gates and typed importers.
- Independent recurrent graph encoders, relation-wise aggregation, sparse artificial routing, external state/reset and ablations.
- Matched seeded synthetic task training, validation-only selection, per-epoch JSONL, inference checkpoints and restricted replay.
- Frozen Qwen embedding adapters, zero-graft baseline, padding-mask intersection, cache-free generation, and the actual circuit-to-language wrapper.
- CLI commands, revision-pinned worm acquisition, source receipts, examples, HTML reporting, launcher and installable wheel.

## Verification

- Full suite: `python -m pytest -q` — 122 passed in 43.70 seconds.
- Final replay mismatch / HTML regressions: 2 passed in 10.85 seconds.
- Compilation: `python -m compileall -q neurograft` — exit 0.
- Actual tiny Qwen3 and hybrid Qwen3.5 implementations: zero-gate equality, graph-language gradients and unchanged frozen backbone verified. These are random initializations, not pretrained weights.
- Final fixture experiment: five seeds, four variants, 30 epochs, all 20 runs complete; 384.48 seconds.
- Authentic-worm mixed smoke: one seed, four variants, two epochs, all four runs complete; 11.72 seconds.
- All 24 selected checkpoints exactly reproduce their saved test logits on this CPU runtime.
- Independent review: four material findings fixed and reviewed again; no new material regression found.
- A local environment `pyreadline3` console destructor can print an ignored invalid-handle message after pytest exits. Verified test runs exit 0; this is recorded rather than changing the user's global packages.

## Review-driven corrections

CLI input hashing originally preceded the CSV-size guard. Importers now own hashing after enforcing bounds. An explicit language injection mask now intersects the attention mask. Replay compares actual logits with the saved hash and returns a failing status on mismatch. Topology digests canonicalize undirected orientation and record order while preserving typed observations; ordinary bundle digests remain serialization-sensitive.

An incomplete preliminary experiment was stopped and marked aborted after the hashing issue. The final experiment uses the same task sizes, model width, optimization settings, epochs and seeds under corrected hashing. No test-result-driven hyperparameter change was made.

## Scope and remaining gates

The acquired worm derivative retains 302 declared neurons and 4,719 observations. Twelve inconsistent electrical pairs are excluded explicitly; both original values and zero-padding aliases are recorded. Raw sources remain intact. Connectivity-data redistribution rights remain unresolved.

Female fly, male fly and human inputs remain fixtures. Authentic imports need bounded source exports and an authorized human matrix. A pretrained Qwen runtime, language-adapter training and physiological validation remain separate unfinished milestones. There is no established biological or language-performance improvement, external publication, remote repository, or empirical replication.
