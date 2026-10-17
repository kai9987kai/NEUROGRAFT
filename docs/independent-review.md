# Independent NEUROGRAFT code review

17 October 2026. This is a review of the first-build implementation as inspected during integration, before the root integrator's follow-up repairs. There is no Git checkout or immutable review commit. Line references below identify the inspected source; subsequent edits may move them.

**Current status after focused re-review:** All four findings below are resolved in the inspected production code. The uncertainty wording is also corrected. No new material regression was identified in the repairs or the new circuit-to-language wrapper. The original findings remain below as an audit trail; the focused follow-up at the end supersedes their open status.

## Scope and method

Read the first-build design, implementation plan, all production modules in `neurograft`, the focused tests, and the graph, dynamics and evaluation build reports. Reviewed source/data integrity, sparse computation, control matching, held-out selection, checkpoint replay, input bounds, and language integration.

No Torch runtime, model loading, training, expensive tests, downloads or source-data edits were performed by this reviewer. A root-owned five-seed benchmark was already running. The only implementation-side artifact written by this reviewer is this review document. Pure-Python graph/import inspection and a bounded pure-Python rewiring search were used to confirm a topology-reporting defect. Previously reported test outcomes remain integrator evidence rather than independently rerun evidence.

## Prioritized actionable findings

### P2: CLI imports read the full source before the 64 MiB importer guard

**Evidence:** `neurograft/cli.py:30-35`, particularly line 33; import call sites at lines 125 and 127. The importer itself checks file size before reading in `neurograft/ingest.py:18-25`.

`_manifest` computes `hashlib.sha256(Path(input_path).read_bytes())` while preparing the manifest, before the bounded importer is called. Python evaluates this `setdefault` argument even if the manifest already declares a SHA-256. Consequently the CLI can read a multi-gigabyte input into memory before the importer rejects it. This defeats the intended first-build input allocation bound and unnecessarily reads every valid input twice. Human matrices and normalized edge CSVs are both affected.

**Suggested reproducer:** Create an otherwise complete manifest that already contains a valid SHA-256. Monkeypatch `Path.read_bytes` for the source input to raise if touched, and make the source `stat().st_size` exceed 64 MiB. Run `main(['import-human', ...])` or the normalized CSV equivalent. The current CLI reaches `read_bytes` before the importer can return its size-limit error. The regression should require the size-limit error with no full source read.

**Suggested repair:** Do not pre-hash source bytes in `_manifest`. Supply a valid placeholder when needed because both importers independently replace the claimed hash with checksums of their actual bounded reads. Alternatively share a bounded read path rather than using unrestricted `read_bytes`.

### P2: Topology hashes can report a changed graph when only undirected endpoint orientation changed

**Evidence:** `neurograft/graphs.py:211-214` hashes the raw node/edge arrays. `neurograft/evaluation.py:403-404` uses that hash for `topology_changed`. Rewiring can represent an undirected pair in either orientation at `neurograft/graphs.py:363-366`.

The topology digest is sensitive to record order and to the orientation of an undirected edge. Those are serialization differences, not differences in the weighted observation graph. This can turn an effectively unchanged rewired control into a reported changed topology, weakening the control's diagnostic receipt.

**Confirmed reproducer, without Torch:**

```python
from neurograft.fixtures import make_fixture_bundle
from neurograft.graphs import rewire_graph

g = make_fixture_bundle(seed=0, nodes_per_source=4).graphs[0]
c = rewire_graph(g, seed=6)

def canonical(graph):
    return sorted(
        (e.relation, e.directed,
         e.src if e.directed else min(e.src, e.dst),
         e.dst if e.directed else max(e.src, e.dst),
         e.weight, e.confidence)
        for e in graph.edges
    )

assert canonical(g) == canonical(c)
assert g.topology_digest() != c.topology_digest()  # Current defect
```

The only final record difference is the electrical pair `worm:0 -- worm:1` becoming `worm:1 -- worm:0`; connectivity, direction semantics, weights and confidence remain equal. The inspected hashes were `939705c3e21fd3dee1a78e83d37e3f9e4b7de4fd14b9dbefe0a0acae39f752b1` and `e3df3672cb1ee7a5417c9442155f2086bcefbc50ffde923aa96fe509f3ddaab4` respectively. The swap receipt records eight successful intermediate swaps, which does not imply a different final topology.

**Suggested repair:** Canonicalize undirected endpoints and sort node/edge records for `topology_digest`, while retaining the existing serialization-sensitive full `digest`. Cover reversed undirected endpoints, edge/node permutation, actual directed-edge changes, and weight changes. A provenance-only rewire receipt should continue to change the full digest while leaving topology unchanged.

### P2: CLI replay never checks the regenerated logits against the saved replay receipt

**Evidence:** `neurograft/cli.py:144-157`, especially lines 155-157. Checkpoint creation records `test_logits_hash` and `data_hashes` in `neurograft/evaluation.py:394`; `load_checkpoint` verifies weights, buffers, graph and configuration but returns the replay hash as metadata.

The replay command recomputes and prints a new hash, then returns success regardless of whether it matches `metadata['test_logits_hash']`. A runtime/task change or corrupted replay receipt can therefore produce a successful command despite failed deterministic replay. The saved checkpoint's integrity and the reproducibility of its held-out output are separate checks.

**Suggested reproducer:** Change only `test_logits_hash` in a small valid checkpoint to 64 zeroes, keeping all weights, graph and their checksums intact. Run the replay command. It currently prints a different hash and exits zero. The regression should assert an explicit mismatch and a nonzero exit code. The good-checkpoint path should expose an explicit successful comparison, and regenerating/validating the saved test-data hash would strengthen the task replay contract.

**Suggested repair:** Compare the regenerated hash to the recorded hash and fail clearly on mismatch. The root integrator confirmed this finding and had already started a covering regression when this review was written.

### P2: An explicit injection mask can enable residuals at attention-masked padding positions

**Evidence:** `neurograft/llm.py:46-50`. When `injection_mask` is provided, `attention_mask` no longer constrains residual injection.

For a padded prompt, a caller-supplied injection mask can mark a padded token as eligible and the adapter then changes its embedding even though `attention_mask` declares it invalid. The effect depends on the selected backbone's padding and recurrence implementation. The wrapper's masked-injection contract should consistently preserve padding; the original tiny Qwen3.5 test exercised full attention only and did not resolve hybrid-attention behavior.

**Suggested reproducer:** Use left-padded token IDs `[[0, 1, 2]]`, attention mask `[[0, 1, 1]]`, injection mask `[[1, 1, 1]]`, a nonzero gate and nonzero graph context. Capture `inputs_embeds` at the frozen backbone. The current wrapper adds a residual at the first token. Require the embedding at that position to remain equal to its base embedding, and require nonzero allowed-position residuals. Also cover the path with no explicit injection mask.

**Suggested repair:** Intersect an explicit injection mask with the attention mask, validating both shapes before use. Validate binary/finite mask values if that is the public API contract. The root integrator agreed to add a regression and intersection logic, plus an actual hybrid Qwen3.5 integration test.

## Scientific wording correction

`neurograft/evaluation.py:278-279` states that intervals reflect initialization variation only. That is accurate for the fixed-topology variants with the fixed task splits, but the rewired control also draws a distinct graph from the model seed at lines 244-252. A grafted-minus-rewired interval includes graph-rewiring variation as well as training-initialization variation. Use broader run-seed wording or a comparison-specific caveat. This is a scope-description correction rather than an error in the bootstrap calculation.

## Positive findings and spec-compliance judgment

The first-build architecture is substantially implemented. Source records are frozen, typed and separately hashed; constructors reject endpoint, source, scale, direction, duplicate-observation and nonfinite-value errors. Computational expansion of undirected observations preserves reported observation counts. Imported raw checksums and processing receipts are explicit. Rewiring is bounded, partitioned by relation/direction, preserves per-node directed/undirected degrees and per-relation weight/confidence multisets, fixes existing loops, and records unsuccessful attempts. It does not claim to preserve weighted strengths.

Graph recurrence actually aggregates source-edge messages, uses separate relation transforms and GRUs, and keeps the artificial port overlay separate from source observations. States are external, episodes reset, routing masks precede normalization, and a null route exists. Parameter allocation and initialization are matched across benchmark variants; fixed examples and epoch shuffles are shared; validation selects the checkpoint; test evaluation occurs afterward. The report correctly distinguishes inactive gradients/compute from equal parameter allocation, and warns that the task is engineered to require source communication and memory. No unsupported biological-connectome advantage was found in the inspected production/report scope statements.

The independent pure-Python inspection loaded the saved worm bundle and verified the edge-list SHA-256 against both its manifest and acquisition receipt, and the network SHA-256 against the receipt. Current retained counts are 302 declared neurons, 4,719 observations (3,638 chemical and 1,081 electrical), 5,787 computational arcs, 47 self-loops, three weak components and two isolated nodes. The largest component contains 300 nodes. Raw input files were not changed. These checks confirm internal provenance/integrity, not independent source licensing or scientific representativeness.

The authentic four-source milestone remains incomplete: only worm is imported, female fly, male fly and human are missing, and worm rights remain unresolved (`authorized=False`). This is correctly represented by readiness metadata. Fixture completion must remain separate from authentic input completion. No pretrained Qwen runtime or biological/LLM benefit is established by the tiny randomly initialized model tests. The optional language adapter has a plausible differentiable frozen-backbone path; root integration is extending the test to connect the actual graph circuit to language loss and exercise hybrid Qwen behavior.

Subject to repairing and verifying the findings above, the implementation meets the core graph-only first-build intent. The integration acceptance gate still requires the root-owned completed benchmark, final suite, replay verification, and review of the final delivery documentation. This review does not declare those pending steps complete and does not count the authentic-data or pretrained-model milestones as achieved. No P0/P1 defect was established in this bounded read-only review.

## Focused follow-up after repairs

17 October 2026. Re-read the changed production code and related regression tests without running Torch, tests, model loading, training or benchmark workloads. The root integrator reports **122 tests passed in 43.70 seconds**, including a tiny real Qwen3.5 implementation with both linear and full attention, and actual graph-to-language gradient/frozen-backbone checks. This outcome is integrator-provided evidence, not an independently rerun suite. The final corrected five-seed fixture experiment was still active at the time of this follow-up.

| Original finding | Status | Current code evidence |
|---|---|---|
| Unbounded CLI input pre-hash | Resolved | `neurograft/cli.py:30-38` only reads the manifest and supplies a transient valid hash when absent; actual source reads and hashes remain inside bounded importers. The oversized-input regression verifies that the CLI reaches the 64 MiB guard before a full source read. |
| Representation-sensitive topology hash | Resolved | `neurograft/graphs.py:211-228` sorts nodes, canonicalizes only undirected endpoints, retains complete records and sorts typed edges. Regression tests cover representation invariance, actual directed/weighted changes and the exact unchanged seed-6 control found by this review. |
| Replay did not compare saved predictions | Resolved | `neurograft/cli.py:158-166` compares the regenerated hash with the checkpoint hash, exposes both hashes and `matches_checkpoint`, and returns 2 with an error on mismatch. |
| Injection mask overrode padding | Resolved | `neurograft/llm.py:46-54` validates both masks as finite binary matching shapes and multiplies both into the residual. An explicit injection mask can now only narrow valid positions. The regression captures actual backbone input embeddings. |

The topology repair fully preserves parallel typed observations allowed by this schema. For example, chemical and electrical observations between the same neurons, or structural and functional observations between the same regions, remain separate complete records in the canonical hash. There is no set conversion or deduplication. The sort key includes relation and directedness, as well as endpoints; graph validation already guarantees uniqueness within that typed key. Directed orientation remains meaningful. Weights, confidence, source IDs and all node records remain in the hashed payload. Thus record permutation and undirected representation changes become invariant without merging different scientific observation types or losing their values. The full `digest()` deliberately continues to retain serialization order and provenance.

The bootstrap caveat now explicitly includes run-seed variation in initialization/training and, for rewired controls, topology rewiring, while excluding data-split uncertainty (`neurograft/evaluation.py:278-279`). This resolves the scientific wording correction.

The new `NeurograftLanguageModel` at `neurograft/llm.py:86-110` calls the real circuit to obtain context, then passes it directly through the differentiable language adapter. The circuit's normal `forward` starts a fresh graph episode; generation computes graph context once for the explicit cache-free full-prefix reference. The disabled path skips circuit computation and reaches the existing direct backbone bypass with `context=None`. No `detach` or training-time `no_grad` was added to the enabled forward path. Parent module training mode still reaches `FrozenLanguageGraft.train`, which keeps the frozen backbone in evaluation mode. The inspected end-to-end regression backpropagates a real language loss into relation parameters, updates trainable parameters, and checks that frozen backbone weights remain unchanged. No new material correctness regression was identified in this wrapper.

One optional coverage improvement remains: the CLI replay regression currently asserts the matching-hash path. A second case modifying only saved `test_logits_hash` would directly exercise the mismatch/nonzero-exit branch. The branch itself is implemented correctly on inspection, so this is not an open production defect.

**Follow-up judgment:** No open material findings remain from this review. The core first-build software is acceptable on the inspected repaired code and reported focused/full-suite evidence. Completion of the final experiment, delivery-document checks and deterministic replay remain root-owned integration work. Authentic four-source acquisition, rights clearance and pretrained-model verification remain separate unachieved milestones, as before.
