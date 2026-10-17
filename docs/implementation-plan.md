# NEUROGRAFT Implementation Plan

**Goal:** Deliver a CPU-local sparse multi-source circuit research package, with honest source readiness, a trainable graft, controlled memory benchmark, optional Qwen integration, and reproducible reports.

**Architecture:** Immutable typed source graphs feed four independent recurrent graph encoders. A sparse artificial routing layer exchanges pooled states. A policy reads the human module; an optional gated residual projects graph context into a frozen language model.

**Tech stack:** Python 3.12, PyTorch CPU, standard library CLI/storage, optional Transformers and PyArrow. No package installation or GPU assumption.

**Spec:** `NEUROGRAFT_First_Build_Design.md`. User explicitly requested immediate implementation after reviewing the design.

## Global constraints

- Test fixtures must never count as authentic biological inputs.
- Raw observations remain immutable; overlays and degree-rewired controls remain separate.
- CPU-small defaults, explicit state reset, seeded splits, parameter-matched controls; no claim of biological benefit from synthetic tasks.
- No automatic acceptance of account/data terms, token disclosure, or large downloads.
- Build in the supplied new workspace. Use independent file ownership for parallel work; no Git repository or external service is required to run.

## Shared interfaces

`neurograft.graphs`: `SOURCE_KEYS = ('worm','fly_f','fly_m','human')`; frozen `SourceManifest(source_id, species, version, url, citation, license, scale, status, sha256, retrieved_at, processing=(), authorized=False)`; `Node(id, source_id, kind, original_id)`; `Edge(src,dst,relation,weight,source_id,directed=True,confidence=None)`; `SourceGraph(manifest,nodes,edges)`; `GraphBundle(graphs)` exposing `.digest()`, `.stats()`, `.save(path)` and `.load(path)`. Constructors validate; graph and bundle have `.to_dict()/.from_dict()` and `.digest()`. Graph nodes/edges and bundle graphs use tuples. `rewire_graph(graph, seed, swaps_per_edge=5)` returns a degree-preserving control. Status is `fixture` or `imported`; scale is `single_neuron` or `brain_region`; observed relations are `chemical`, `electrical`, `macro_structural`, `macro_functional`.

`neurograft.fixtures.make_fixture_bundle(seed=0, nodes_per_source=8) -> GraphBundle` creates explicitly synthetic test graphs. `neurograft.ingest.import_edge_csv(nodes_path,edges_path,manifest) -> SourceGraph` accepts node headers `id,kind` and edge headers `src,dst,relation,weight,directed,confidence`; IDs are original local identifiers and namespaced in import. `import_human_matrix(path,manifest,relation='macro_functional',threshold=0.0)` imports a labelled symmetric CSV matrix. Importers calculate input checksums and processing receipts rather than trusting a user-provided hash.

`neurograft.dynamics`: `DynamicsConfig(input_dim=4,hidden_dim=12,output_dim=2,top_k=2)`; `GraphCircuit(bundle,config)`; `.initial_state(batch_size)`; `.step(observations,state=None,graft_enabled=True,recurrence=True)` returning `CircuitOutput(logits,state,context,routes)` with observations `[B,4,I]`, state tuple of `[B,N,H]`, context `[B,4H]`, routes `[B,4,5]` (receiver,sender including null). `.forward(sequence, graft_enabled=True, recurrence=True)` accepts `[B,T,4,I]`, returns final output, resets at the start. `.context_dim=4H`. Only human-module pooled state feeds the policy head; all-module context is exposed for later language adapters. All parameters exist in all control variants.

`neurograft.evaluation`: `BenchmarkConfig(seeds=(0,1,2),epochs=20,train_size=128,val_size=64,test_size=128,batch_size=32,delay=4,hidden_dim=12,learning_rate=0.01,data_seed=1729,threads=2)`; `run_benchmark(bundle,config,output_dir,variants=('grafted','no_graft','no_recurrence','rewired'))->dict`. Task signals are split between source inputs; no shortcut to the human policy input. Fixed disjoint data seeds, paired examples for variants; every run saves JSONL metrics and checkpoint with graph hash/config/weights. Report includes seed-level accuracy, parameter counts, timing, memory, paired uncertainty, topology hashes, controls and evidence scope. No resume training is claimed.

## Review focus

- Corrupt graph hashes, graph/source ID collisions, nonfinite weights and mismatched scales are rejected.
- Undirected edge expansion must not inflate biological counts; signed human correlations retain semantics.
- Source sampling and rewiring never mutate the original graph.
- Training loss must pass through actual graph recurrence and sparse routing; variants share initialized parameters and data.
- Zero-graft LM bypass preserves the base call and frozen parameters while allowing graph gradients when enabled.

## Tasks

### 1. Typed graphs and ingestion

Files: `neurograft/graphs.py`, `neurograft/ingest.py`, `neurograft/fixtures.py`, `tests/test_graphs.py`, `tests/test_ingest.py`.

- [x] Write integrity/round-trip/import/degree tests; verify missing implementation fails.
- [x] Implement immutable schemas, validation, hashes, connectivity/readiness statistics, CSV import and human matrix import.
- [x] Build small synthetic fixtures and degree-preserving rewiring that records attempted/successful swaps.
- [x] Run focused tests and record evidence.

### 2. Trainable dynamics

Files: `neurograft/dynamics.py`, `tests/test_dynamics.py`.

- [x] Write gradient, deterministic reset, routing-mask, source-immutability and ablation tests first.
- [x] Implement typed sparse edge aggregation with per-source GRUs, pooling, top-k sender routing plus null, and explicit state.
- [x] Verify finite nonzero gradients, parameter update, recurrence and no-graft behavior on bounded examples.

### 3. Evaluation and checkpoints

Files: `neurograft/evaluation.py`, `tests/test_evaluation.py`.

- [x] Write paired-split/reproducibility/control-parameter tests first.
- [x] Implement synthetic delayed XOR cue task, seeded train/validation/test, four matched variants, JSONL and checkpoint output.
- [x] Implement paired intervals and record enough caveats to distinguish synthetic evidence from topology benefit.
- [x] Run a small benchmark integration test and record evidence.

### 4. CLI, acquisition, model interface

Files: package metadata/entrypoints, `neurograft/cli.py`, `neurograft/acquire.py`, `neurograft/llm.py`, root-owned tests.

- [x] Add fixture, inspect, import-csv, import-human, fetch-worm, benchmark, doctor, report and model-probe commands.
- [x] Fetch a small authentic worm edge source at a pinned revision, save raw bytes and hash, preserve explicit source-data terms status.
- [x] Add a frozen model residual adapter with disabled bypass and full-prefix cache-free generation reference.
- [x] Verify real tiny initialized Qwen backend interface if available; never describe it as pretrained.
- [x] Provide memory-guarded explicit pretrained loading; don't silently substitute a cached different model.

### 5. Integration and delivery

Files: README, source/license notes, report, launch scripts and local HTML report.

- [x] Run full test suite, compile checks, CLI flows and a five-seed bounded experiment.
- [x] Obtain an independent whole-project review and fix material findings with covering tests.
- [x] Save the measured report, verify deterministic replay and checkpoint integrity, open the report for review.
- [x] State actual authentic-data/model readiness and remaining external inputs precisely.

## Progress

Implementation started after the user's “build it now” instruction. No remote publication or Git history mutation is part of this build.

First-build verification: 122 tests; final targeted regressions 2; 24 exact checkpoint replays; five-seed fixture study and authentic-worm mixed smoke saved. Authentic fly/HCP inputs and pretrained-model execution remain separately incomplete under the design gates.
