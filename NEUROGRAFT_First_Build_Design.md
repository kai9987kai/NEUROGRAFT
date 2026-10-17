# NEUROGRAFT: first-build design for review

17 October 2026

**Status:** Design reference, approved for implementation by the user's subsequent build instruction. Current measured implementation status is recorded in `README.md` and `reports/`; the design below describes intended gates, not proof that all gates have passed.

## Intended outcome

Build the first milestone described in the pasted request: preserve separate worm, female fly, male fly, and human region graphs; connect their encoded states through explicitly synthetic, trainable bridges; and test a recurrent circuit adapter before attaching a compact pretrained language model.

The supplied research and engineering plan is the design reference. Its 16-week roadmap describes the larger research programme, not work already completed. This document narrows the first implementation to a testable local instrument. Noncommercial local research is the proposed starting context; it is not an assertion that every input or derivative has cleared distribution rights.

## Local constraints observed

- The current workspace is empty and has no existing application or Git checkout.
- Windows reports a Snapdragon X Plus, eight logical processors, and approximately 15.61 GiB physical RAM. Only 1.80 GiB was available at the time of inspection.
- The system drive had approximately 21.6 GiB free.
- The selected Python is 3.12.10 and reports AMD64, despite the ARM-based processor. Native ARM performance must not be assumed.
- PyTorch **2.11.0+cpu** and Transformers **5.12.1** successfully import; PyTorch reports CUDA unavailable. This confirms imports, not a Qwen forward pass or model-architecture compatibility.
- Package discovery also finds NumPy, SciPy, pandas, PyArrow, NetworkX, psutil, and pytest. Discovery alone does not establish that every package or compiled operator works.
- A later sample showed 2.02 GiB RAM available. No working pretrained-model runtime has been established.

These are a dated snapshot. Resource and runtime checks must be repeated before training. Keep initial graph experiments small and bounded; avoid whole-fly dense matrices, raw MRI downloads, environment duplication, and speculative multi-gigabyte model downloads.

## Approaches considered

| Approach | Benefit | Limitation |
|---|---|---|
| **Staged source-backed sparse prototype — recommended** | Establishes graph integrity, trainability, and controlled evaluation within local limits | Authentic four-source completion depends on obtaining each source with provenance |
| Immediate full-connectome and Qwen integration | Reaches the intended architecture directly | Combines unresolved acquisition, memory, runtime, and evaluation problems |
| Fixture-only demonstration | Reproducible offline development and useful contract tests | Cannot establish that biological connectomes were imported or help performance |

Use small, clearly labelled fixtures for tests, while maintaining a separate authentic-data readiness gate. Missing biological inputs remain missing; fixtures cannot silently satisfy that gate.

## Proposed first-build architecture

```text
Versioned source files + provenance manifests
                    |
          Typed sparse graph import
                    |
     Immutable graphs: WORM / FLY_F / FLY_M / HUMAN
                    |
       Separate graph encoders and recurrent states
                    |
       Synthetic sparse bridge overlay + reset
                    |
           Trainable circuit adapter
                    |
   Synthetic memory / sensory-integration benchmark
                    |
      Matched controls + reproducible run report

Later integration:
Circuit adapter -> gated embedding residual -> frozen compact Qwen
```

### 1. Source registry and graph storage

Each imported source has a manifest containing dataset identifier, exact release or commit, retrieval URL, acquisition timestamp, file SHA-256, citation, declared licence, node scale, and explicit processing history. Source status is recorded as unavailable, fixture, or imported-and-validated. Rights/access metadata are separate from technical validation.

Preserve source files and normalized observations separately from learned parameters. Node IDs retain their source identifiers with collision-free namespaces. Edge records retain direction, relation, weight semantics, optional confidence, and provenance. Chemical counts, electrical contacts, structural tract estimates, and functional correlations remain distinct relations. Unknown connections are not automatically biological zero; absent confidence remains unknown.

Use sparse edge arrays for computation. A normalized export can use Parquet when the installed runtime is verified. Initial imports should accept bounded local CSV edge tables and labelled human matrices, with explicit endpoint, shape, numeric-value, and semantics validation. Import recipes must record how upstream formats were converted.

For human functional matrices, preserve signed correlations as associations; do not treat them as synapse counts or infer causal transmission. Electrical and undirected macro edges need a documented bidirectional computational representation without duplicating reported biological counts. Neurotransmitter labels alone do not determine excitation or inhibition.

### 2. Authentic input acquisition

| Source | Verified access point and finding | First-build requirement |
|---|---|---|
| Worm | OpenWorm CElegansNeuroML exposes a hermaphrodite edge list and points to c302 for newer models | Pin a repository revision, preserve chemical/electrical distinctions, and report actual imported counts. The CSV includes muscle targets: explicitly identify neuron-only rows rather than assuming 302 imported nodes |
| Female fly | FlyWire public v783 is an October 2023 snapshot under CC BY-NC 4.0; newer annotations may accompany it | Obtain a bounded Codex circuit export or documented token-based static download with source neuron IDs. Record filters: default FAFB threshold is five synapses per pair, and neuropil rows can repeat neuron pairs |
| Male fly | MaleCNS v1.0 was released **8 June 2026**; **3 September 2026** is the paper-publication date. Official site states CC BY 4.0 | Prefer a bounded neuPrint query with the user's account/token, such as the documented DNge104 adjacency example. No-account bulk alternatives list 13 MB annotations and 1.1 GB weights; do not fetch them speculatively |
| Human | HCP-YA 2025 was released **11 August 2025** and warns against mixing its processing with S1200 2017 | Obtain an authorized parcel matrix with parcellation, modality, preprocessing, source cohort, and terms recorded; do not imply that a ready-made matrix has already been found |

The worm software licence does not by itself settle the licence of every embedded connectivity table. Do not accept account agreements on the user's behalf or claim access that has not been established. A user-supplied authorized human matrix is an acceptable import path. HCP terms continue to govern redistributed source and derived data, and recipients must register and accept the terms; a derived adjacency matrix is not automatically freely licensed project output.

### 3. Trainable circuit computation

Begin with small hidden states and short recurrent unrolls, configured explicitly and recorded in runs. Use separate relation transforms and per-source normalization so large fly degrees do not automatically dominate. A compact PyTorch sparse edge-aggregation implementation avoids making PyG compiled extensions a prerequisite for this first build.

Pool each source into a small number of latent ports. Synthetic bridges connect these ports and carry explicit source/target module IDs, trainable gates, and their artificial provenance. Apply routing masks before normalization. Permit a null route and a complete graft bypass. Keep state external to immutable observations and provide explicit reset at episode boundaries.

Report both module-level port connectivity and connectivity of the selected computational graph. A connected module diagram does not prove that every retained biological node is connected. Report isolated components and sampling exclusions explicitly.

The first trainable output is a small task head or circuit adapter. A forward/backward check must demonstrate finite gradients and a real parameter update. This is engineered recurrent computation, not calibrated physiological dynamics.

### 4. Model interface after the graph gate

The requested Qwen3.5-0.8B exists and is Apache-2.0. Its language component has width 1024 and 24 layers with hybrid linear/full attention. Qwen3-0.6B is a simpler text-only alternative. Selection remains conditional on a successful local runtime and memory probe; neither is downloaded by this design review.

Prefer an additive gated residual on existing text embeddings for initial integration, with an explicit bypass delegating to the original model when disabled. This avoids changing sequence length in the zero-graft baseline. Freeze backbone parameters while retaining autograd through the backbone during adapter training. Wrapping its forward pass in `no_grad` would prevent graph-adapter learning.

Define how graph state is injected on prompt processing and how it behaves during cached decoding. Test masks, positions, batch padding, cache resets, and episode boundaries against the pinned model implementation. Do not claim exact equivalence merely because a residual starts at zero; compare actual logits and generation on a fixed deterministic configuration. Avoid initializing both a multiplicative gate and its projection to zero, which can block useful initial gradients.

## Evaluation design

The first benchmark is a small synthetic delayed-cue or partially observed sensory task with train/validation/test seeds declared before training. Task observations must actually travel through the graph computation. Synthetic labels establish only synthetic-task behaviour.

Compare grafted, graft-disabled, recurrence-disabled, and degree-preserving rewired variants. Rewiring must stay within the intended source and relation partitions, preserve directed in/out degree, and record handling of loops, duplicate edges, and weights. Keep matched variants identical in trainable parameter allocation and optimization steps; disclose any compute difference rather than claiming exact matching.

Use the same examples and paired seeds across variants. Report every seed, held-out score, parameter count, runtime, memory observations, and a paired uncertainty estimate when enough runs are available. A smoke test is not a powered ablation study. Do not tune on the final test split or report improvement in advance.

## Acceptance gates

1. **Provenance:** Every imported node and observed edge resolves to a declared source. Actual counts and filtering receipts are exported. Fixture and unavailable inputs are unmistakable.
2. **Graph integrity:** Invalid endpoints, relation/scale mismatches, nonfinite values, and conflicting IDs are rejected. Source hashes and normalized observation hashes do not change after training or overlay changes.
3. **Serialization:** Typed semantics, weights, IDs, and provenance survive round-trip export/import.
4. **Grafting:** Artificial edges are stored separately; all active source modules can exchange information through the intended port routes, and connectivity limitations are reported.
5. **Computation:** Deterministic seeded forward runs, finite gradients, parameter updates, explicit episode reset, and null-graft equivalence pass focused tests.
6. **Controls:** A bounded held-out benchmark runs with matched variants and saves complete configurations and results, including null or negative findings.
7. **Authentic four-source milestone:** All four sources pass ingestion gates. A mixed real/fixture run cannot satisfy this gate.
8. **LLM milestone, separately:** A pinned real pretrained backbone runs through the graph adapter, preserves the explicit bypass baseline, and supports adapter gradients. This gate is additional to graph-only completion.

## Deliverables after implementation approval

A Python package and CLI; typed graph and manifest schemas; bounded import recipes; separate labelled test fixtures; sparse recurrent encoders and graft overlay; a trainable adapter; a reproducible benchmark command; focused integrity and gradient tests; and a README describing current source status and measured results.

The first implementation should expose local commands and saved reports. A 3D viewer, FlyGym, detailed ion-channel simulation, LoRA, a Supermix connection, and a production service belong to later milestones. The full requested architecture remains the destination, with graph, data, LLM, and scientific-validation completion tracked separately.

## Primary sources checked

- [OpenWorm CElegansNeuroML](https://github.com/openworm/CElegansNeuroML)
- [OpenWorm c302](https://github.com/openworm/c302)
- [FlyWire public-release and citation guidelines](https://join.flywire.ai/guidelines)
- [FlyWire Codex](https://codex.flywire.ai/)
- [Codex download and filtering FAQ](https://codex.flywire.ai/faq)
- [MaleCNS release notes](https://male-cns.janelia.org/release/)
- [MaleCNS project news and access](https://male-cns.janelia.org/)
- [MaleCNS downloads](https://male-cns.janelia.org/download/)
- [HCP-YA 2025 release](https://humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release)
- [HCP access announcement](https://icb.humanconnectome.org/study/hcp-young-adult/article/updated-hcp-young-adult-data-released-connectomedb-powered-balsa)
- [HCP data-use terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms)
- [Official Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Official Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)

Source verification establishes availability and documented properties, not successful local import, runtime compatibility, or an empirical advantage.
