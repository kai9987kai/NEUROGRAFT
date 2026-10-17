# Typed graphs and ingestion build report

17 October 2026. Task 1 of the shared implementation plan.

## Implemented files

- `neurograft/graphs.py`: frozen typed manifests, nodes, edges, source graphs and bundles; canonical source ordering; validation; SHA-256 integrity; sparse connectivity/readiness statistics; relation-aware degree controls.
- `neurograft/ingest.py`: local typed edge CSV and labelled symmetric human matrix imports, with checksums calculated from the actual input bytes and processing receipts.
- `neurograft/fixtures.py`: seeded synthetic graphs for all four canonical modules.
- `tests/test_graphs.py`, `tests/test_ingest.py`: integrity, provenance, semantics, immutability, connectivity, importer and degree-preservation tests.

No other implementation files were changed by the graph task. No package installation, GPU use, download, source publication or commit was performed by this task.

## Shared API and storage

The exact interfaces from `docs/implementation-plan.md` are implemented. Graphs expose frozen tuple nodes/edges; bundles expose frozen tuple graphs in `SOURCE_KEYS` order. Bundles accept one to four distinct sources, allowing a standalone authentic source to be inspected and saved before four-source assembly. Node IDs equal `source_id:original_id`; original IDs remain separately available.

Imported manifests require a valid hexadecimal SHA-256 plus nonempty species, release/version, URL, citation, licence and retrieval timestamp. Source scale, node kind, endpoint resolution, relation semantics, finite weights, confidence range and duplicate observations are checked. Chemical observations are directed. Electrical and human macro observations are stored as one undirected record. Source autapses remain permitted and separately counted.

Functional observations retain signed correlation values in [-1,1]. Chemical/electrical/structural weights are nonnegative; none are converted into excitation/inhibition labels. Structural and functional human relations remain distinct.

`GraphBundle.to_dict()` includes schema version and a checksum of the full normalized payload. `save/load` use UTF-8 JSON; changed payloads, absent hashes, nonfinite JSON values and duplicate object keys are rejected. These checks establish content integrity relative to the saved checksum, not source authenticity against an independent trusted archive.

`digest()` includes provenance. Additional `topology_digest()` methods on graphs and bundles hash normalized nodes and weighted observations while excluding manifests and processing receipts. Evaluation can therefore distinguish unchanged edges from a receipt-only provenance change.

## Connectivity and readiness

Statistics use sparse O(N+E) traversal. They report actual node/observation counts, relation counts, weak components, isolates, largest component, self-loops and computational arc counts. Undirected observations contribute two computational arcs except for self-loops, which contribute one. Density is unique off-diagonal computational pairs divided by N(N-1), so multiple relations do not inflate pair density.

Top-level bundle readiness reports missing authentic sources and the authentic four-source gate. Fixture inputs never satisfy this gate. Degree-rewired imported controls also fail it. Access authorization is reported separately as `authorized`, `rights_ready` and `rights_four_source_ready`: an imported technically valid graph can still have unresolved rights. Metadata declarations do not establish the scientific representativeness of a sample.

## Local import receipts and limits

Edge CSV input requires exactly the declared node and edge headers; optional confidence values may be empty. Duplicate IDs/edges, unresolved endpoints, incorrect field counts, unknown directed flags and nonfinite values fail validation. Original local IDs are namespaced without conflating source modules.

Each edge CSV manifest hash is SHA-256 of a canonical JSON map containing the actual node-file and edge-file SHA-256 values. Both individual file checksums and retained row counts appear in processing receipts. Human matrix manifests use the raw matrix file checksum directly. Importing a manifest already labelled fixture preserves fixture status.

Human matrices require unique matching row/column labels, square shape, finite values and symmetry (relative tolerance 1e-9, absolute tolerance 1e-12). Functional correlations validate the signed range, including diagonal entries. Validated diagonals are excluded, and nonzero upper-triangle values with absolute weight at least the declared threshold are retained. Thresholding and diagonal exclusion are recorded; retained associations are not claimed to be causal transmission.

Every local input file has a 64 MiB first-build cap, checked before reading and again on the bounded read. Large whole-connectome imports and scalable streaming formats are outside this first local importer.

## Rewired controls

Bounded seeded double-edge swaps operate within each relation/direction partition. Directed in/out degrees and undirected degrees are preserved independently for every relation. Source nodes and original graph objects remain unchanged. Existing self-loops stay fixed; proposals creating new loops, duplicate pairs or unchanged endpoint sets are rejected. Per-relation weight/confidence multisets are preserved by keeping them with edge records; node weighted strengths are not promised to be preserved.

Receipts record seed, requested swaps per eligible edge, actual attempted and successful swaps, per-partition attempt/success counts, fixed-loop count, parent graph digest and weight policy. Dense or small partitions can validly complete with zero successful swaps. Downstream reports should inspect topology hashes and success counts rather than assume every control changed topology.

## Verification evidence

1. Initial test-first run before implementation: `python -m pytest tests/test_graphs.py tests/test_ingest.py -q` exited 1, with 39 failures caused by the missing typed graph/import/fixture implementation. The explicit existence assertion reported `typed graph layer is missing`.
2. First implementation run: the same focused command passed 39 tests.
3. Added edge-case tests exposed a missing observation-only hash and lack of pre-read file bounds: 2 failures and 42 passes. After implementing both, the focused command passed **44 tests** in 1.62 seconds, exit 0.
4. Full current project suite: `python -m pytest -q` passed **109 tests** in **45.38 seconds**, exit 0. No failures or warnings were printed in this full-suite run.
5. `python -m compileall -q neurograft/graphs.py neurograft/ingest.py neurograft/fixtures.py` exited 0.
6. An isolated interpreter imported graph, ingestion and fixture modules and asserted `torch` was absent from `sys.modules`; it printed `Graph imports verified without torch` and exited 0.

The earlier focused pytest processes printed a local `pyreadline3` console-destructor invalid-handle message after successful test summaries. This did not change their exit status; the final full-suite run did not print that message. No source workaround was introduced for that external console cleanup behavior.

The graph-layer tests establish local software behavior and synthetic control integrity. They do not establish authentic four-source completion, model benefit, physiology or biological validity.
