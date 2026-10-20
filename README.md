---
library_name: pytorch
tags:
  - connectome
  - graph-neural-networks
  - neuroscience
  - continual-learning
  - 3d-environment
  - qwen
---

# NEUROGRAFT

A local research instrument for connecting separate, provenance-preserving neural graphs through an artificial trainable routing layer.

The working package includes typed sparse graphs, CSV and human-matrix importers, recurrent graph encoders, sparse synthetic grafts, a delayed-memory benchmark with matched controls, validated inference checkpoints, a persistent live 3D navigation and continuous-learning workspace (`python -m neurograft live`), and a differentiable interface to frozen Qwen backbones. The pretrained model is an optional dependency and is not downloaded automatically.

**Current evidence:** the circuit implementation runs and trains locally. The OpenWorm source has been acquired and normalized with explicit filtering receipts. Female fly, male fly, and human inputs in the included demonstration are **synthetic fixtures**. Qwen interface tests use tiny **randomly initialized** models from the actual Transformers implementations. A pretrained Qwen checkpoint has not been loaded or trained. No biological-connectome advantage is established.

## Learned navigation experiments (v0.10)

The optional `context_residual` readout gives the policy a learned residual over all four encoded sources. It starts at zero, preserving the original common weights and initial logits. Existing checkpoints load with their original `human` readout. The live Navigation panel identifies the checkpoint's readout; loading an experimental candidate is an explicit separate step.

The curriculum adds `rotation_augmentation`, `collection_policy: "expert"` and `include_readout_controls`. Clockwise quarter turns transform coordinates, headings, wall directions and action labels together while retaining zero padding. Expert-only collection permits matched readout/augmentation comparisons; the existing mixed teacher/learner collection remains the default. A new `no_context` frozen intervention bypasses the residual. In a residual model, `no_graft` still permits direct all-source readout, so it is not a human-source-only control.

These are testable engineering hypotheses informed by [navigation symmetry research](https://arxiv.org/abs/2309.13043v2), [practical symmetry representations](https://arxiv.org/abs/2505.13431) and [limits of privileged-teacher distillation](https://arxiv.org/html/2412.00985v3). Augmentation does not guarantee architectural equivariance, and a richer readout cannot reveal unseen walls. Full-map fixed-tie labels, short observation windows and expert-only distribution shift remain limits. See the [research and experiment design](docs/v10-learning-design.md).

Run the four-arm, three-seed study in a **new** directory. It declares every configuration before training, retains all arms and compares common-weight, original-data and base-minibatch hashes. Each seed changes both initialization and training-world offset; comparisons are matched within that seed. The residual has additional parameters, so this is not a parameter-matched topology comparison.

```powershell
python tools/study_navigation_readouts.py --bundle reports/runs/navigation-v08-smoke-20261019/experiment/source.bundle.json --checkpoint reports/runs/navigation-v08-smoke-20261019/experiment/candidate.pt --out reports/runs/new-v10-study
python tools/study_navigation_readouts.py --verify reports/runs/new-v10-study --replay
```

Plain verification checks sealed child archives, declared schedules/budgets, matched exposure receipts and derived aggregates. `--replay` additionally reruns the frozen evaluations; it does not repeat training. Candidate exports remain paused, with learning disabled and no automatic promotion.

Measured results, validation receipts and experimental limits are recorded in the [v0.10 build report](docs/v10-build-report.md).

## Observation-only navigation (v0.9)

The live workspace now offers three explicit action-selection modes: the original **raw policy** with its configured guidance, **policy with legal-action masking**, and **observed map with policy tie-breaking**. Choose a mode in the Navigation controller panel. Mask/map modes bypass teacher actions; online imitation labels can still train the model if learning is enabled. The chat commands `navigation` and `why` identify the active controller.

The map planner sees public arena size, current position, known goal and adjacent-wall sensors. It remembers observed cells, commits to an unexplored frontier and routes only through known free cells. The browser shows that partial map separately from the full 3D arena. Unknown cells stay dark. This is symbolic engineering with noiseless grid observations, not learned visual SLAM or evidence of stronger raw neural navigation.

Compare eight frozen controls on matched worlds: raw policy, no grafts, no recurrence, random, privileged full-map oracle, legal masking, observed map plus policy prior, and observed map without that prior:

```powershell
python -m neurograft evaluate-live --bundle data/imported/mixed.bundle.json --checkpoint data/live/session.pt --seeds 41000,41001 --size 11 --max-steps 96 --include-controllers --out reports/new-controller-trials.json
python -m neurograft study-navigation --bundle data/imported/mixed.bundle.json --checkpoint data/live/session.pt --out reports/runs/new-controller-study
python -m neurograft verify-navigation-study reports/runs/new-controller-study --replay
```

The study defaults to sizes 7/11, all six topologies and two declared seeds (24 scenarios), with a 96-step horizon. It freezes input bytes, declares the protocol before evaluating, and exports each condition, aggregate JSON/HTML and a file manifest. It trains and promotes nothing. Proposed-wall and override rates reveal what legal filtering changes; the map-only control isolates the learned prior. These correlated synthetic scenarios are exploratory.

New v3 live checkpoints preserve controller mode and exact observed-map state. v1/v2 checkpoints remain readable in raw-policy mode with empty navigation memory; existing v2 model/observer state is retained. Default legacy five-way evaluation configuration, hashes and recorded outputs remain unchanged, and v0.8 archives still replay exactly. See the [v0.9 design and verification scope](docs/v09-navigation-design.md).

The local fixed-weight study reached **24/24 goals with observed-map navigation**, versus **18/24 with masking** and **0/24 with raw policy**. The map-only control also reached 24/24, so these results do not establish a learned-policy gain. See the [build report](docs/v09-build-report.md) and [complete comparison](reports/runs/navigation-v09-controller-study-20261019/report.html).

## Auditable navigation curriculum (v0.8)

Training now cycles through the Cartesian product of configured world sizes and topologies. A finite per-action reservoir retains expert-labelled visits across rounds instead of discarding all but the newest samples. Per-round receipts record scenario coverage, teacher/learner actions, retained replay samples and imitation loss. Configuration checks bound collection, optimizer updates and evaluation seeds before a run starts. This finite-memory dataset aggregation approach is informed by [Ross et al.](https://proceedings.mlr.press/v15/ross11a.html); it is not a replication of their full-data algorithm or a guarantee of improved navigation.

Each new experiment archives the exact baseline bytes and source bundle, `config.json`, `progress.json`, `report.json`, a standalone HTML report, candidate checkpoint and a SHA-256 manifest. Verify it without needing the original source paths:

```powershell
python -m neurograft verify-navigation reports/runs/my-new-run
python -m neurograft verify-navigation reports/runs/my-new-run --replay
```

The second command reruns both frozen test evaluations in the current runtime. Hash checks detect changed files; they are not authenticity signatures. Old runs without the complete archive are retained as legacy evidence and cannot pass this new verifier. Candidate files are paused inference exports with fresh live optimizer state, not resumable curriculum checkpoints. Test performance does not trigger automatic promotion.

Live startup now binds its port before starting learning, preserves restored pause/learning settings, and releases resources without saving on startup failure. Environmental overlays (substrates, plume, antennae, optic clearance, gate permeability and circadian lighting) are deterministic diagnostic approximations. They do not alter collision physics or enter the policy's four-by-four observation.

See the [v0.8 build report](docs/v08-build-report.md) and [complete smoke experiment](reports/runs/navigation-v08-smoke-20261019/experiment/report.html). That small candidate still reached **0/4** test goals; the upgrade verifies broader sampling and auditable execution, not useful autonomous navigation.

An isolated curriculum training protocol (`python -m neurograft train-navigation`) designed to train and benchmark candidate navigation policies without mutating the live session or overwriting captured baselines.

- **Dataset aggregation & progressive imitation**: Gathers shortest-path labels across diverse world sizes (`5×5` to `31×31`), topologies, and disjoint seeds. Rollouts progressively shift from expert teacher to learner-controlled exploration.
- **Action-balanced finite replay & counter augmentation**: Replay buffers cap memory storage and balance gradient batches equally across action classes to eliminate directional collapse. Lifetime goal progress counters are randomized during training rollouts to prevent distribution shift.
- **Cross-source routing warmup**: Allows all three cross-source senders (`top_k=3`) during an explicit routing warmup before returning to standard `top_k=2` for validation and exports.
- **Validation-only checkpoint selection**: Best models are chosen strictly by validation success rate, then SPL, then lowest collision rate (preserving earlier ties). Test seeds are evaluated once only on the selected model.
- **Standalone offline HTML reports**: Generates rich self-contained HTML reports (`report.html`) containing side-by-side trial tables, validation curves, and full sha256 checksum receipts.

```powershell
python -m neurograft train-navigation --bundle data/imported/mixed.bundle.json --checkpoint reports/runs/navigation-v05-20261018/frozen-input.pt --out reports/runs/navigation-v06-candidate --rounds 4 --worlds 16 --updates 32
```

## Reproducible live laboratory (v0.5)

The live workspace now includes **Navigation trials**: evaluate a frozen copy of
the current policy on matched seeded arenas while the original session continues.
Compare full policy, disabled grafts, disabled recurrence, seeded random actions,
and a full-map shortest-path oracle. Export JSON with graph/model/protocol hashes,
per-seed action traces, success rate, SPL path efficiency, collisions, and attempted
steps. These inference interventions reuse identical weights; they are not
separately trained controls. Declared evaluation seeds are not proof of unseen
training worlds. Small runs are exploratory.

Evaluate an existing checkpoint without changing it:

```powershell
python -m neurograft evaluate-live --bundle data/imported/mixed.bundle.json --checkpoint data/live/session.pt --seeds 10000,10001,10002 --size 11 --preset labyrinth --max-steps 48 --out reports/my-navigation-trials.json
```

The output must be a new path. Defaults are three worlds and 48 steps, with a cap
of 16 seeds and 128 steps. SPL counts successful cell moves relative to shortest
distance; failed trials score zero. Collision rate and attempted steps capture
blocked actions separately. See [Anderson et al.](https://arxiv.org/abs/1807.06757)
for the metric's basis. This grid instrument does not establish robot transfer.

New v2 live checkpoints preserve the synced anchor, auxiliary module/optimizer
state, memory, compass, phase history, STP, bioelectric state and topology preset,
in addition to the policy and replay state. Legacy v1 files remain readable, with
a warning that omitted auxiliary state must be reinitialized. Their missing state
cannot be reconstructed. `session.checkpoint_compatibility` reports that boundary.
Use a new named checkpoint when retaining an upgraded copy.

Diagnostic reads no longer advance phase history or global retrieval randomness.
Lesion probes compare the same current weights and copied input window. These
are logit sensitivities, not evidence of biological causal mechanisms.

| Component | Role in the live experiment |
|---|---|
| Sparse GraphCircuit encoders, GRUs, grafts and navigation head | Produces actions and receives supervised imitation gradients |
| Frozen anchor | Supplies the KL regularizer; `anchor sync` updates its reference |
| Forward self-model | Trains a separate predictive observer; does not choose navigation actions |
| Memory, compass, phase plane, STP and bioelectric substrate | Diagnostic observers; their lesions/gains do not alter the navigation policy |
| Consolidation, byte bridge, Dale audit and spectral probes | Standalone or untrained observer calculations; no demonstrated policy benefit |

The browser startup syntax failure, inaccessible Concentric/Hierarchical presets,
checkpoint omissions and stale-weight lesion comparison are repaired in this
version. Health reports include worker status and degraded service errors.

**Measured captured-policy result:** eight seeded 11×11 labyrinth trials with a
48-step horizon yielded 0/8 full-policy successes (SPL 0), 1/8 seeded-random
successes, and 8/8 oracle successes (SPL 1). This checkpoint has not demonstrated
useful unassisted navigation on these trials. Its input was a legacy v1 capture;
missing observer state is recorded, and those observers do not drive this policy.
See [the raw report](reports/runs/navigation-v05-20261018/report.json) and
[capture provenance](reports/runs/navigation-v05-20261018/capture.json).
Implementation details and validation are in the [v0.5 build report](docs/v05-build-report.md).

## Live 3D environment, continuous learning, and saved state (v0.4)

Start the persistent local live session and open `http://127.0.0.1:8768/`:

```powershell
python -m neurograft live --bundle data/imported/mixed.bundle.json --state data/live/session.pt --port 8768
# or use the helper launcher:
.\Run-NEUROGRAFT.ps1 -Mode Live
```

What the live session provides:

- **Scalable Multi-Chamber 3D Environment (`15×15` default, scalable `5×5` to `31×31`)**: Supports live architectural presets (`labyrinth`, `chambers`, `corridors`, `pillars`, `concentric`, `hierarchical`) with guaranteed full free-cell connectivity, 8-ray LiDAR / optic-flow clearance beams (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`), topological choke-point doorway rings, 2D advection-diffusion chemoattractant gradient plumes, and tortuosity/detour index metrics.
- **Continuous online adaptation (`supervised-shortest-path-imitation`)**: On each step in the multi-chamber navigation world, source-specific sensory features feed the worm (beacon bearing/distance), female-fly (local wall/hazard clearances), male-fly (visit recency and heading velocity), and human (active goal beacon selector and step urgency) graph modules. When **Learn** is enabled, the 4-way navigation head (`north`, `east`, `south`, `west`), sparse cross-source grafts, and recurrent encoders update online via Adam (`lr=8e-3`, window length `4`, batch size `8`) with gradient clipping (`1.0`) against the deterministic shortest-path teacher action, plus an interactive **⚡ Boost ×16** training burst trigger.
- **2024–2026 Connectome & Cognitive Architecture Grounding (`session.telemetry()`)**:
  - **Bioelectric Morphogenetic Substrate (BMS)** (*Morpheus & Memory Carrier Observatory*): Computes persistent non-spiking membrane potentials $V_{\text{mem}} \in [-85\text{mV}, -20\text{mV}]$ diffusing across gap junctions: $\frac{\mathrm{d}V}{\mathrm{d}t} = \frac{1}{C_m} [ g_{\text{gap}} \Delta V - g_{\text{leak}} (V - V_{\text{rest}}) + \gamma_{\text{pump}} \cdot \text{Act} ]$. This simulated voltage is an observer signal and does not modulate the navigation policy.
  - **Connectome Forward Self-Model & Agency Stabilization Audit** (*GhostInTheMachine & QuantumBot*): Introspectively forecasts the connectome's next latent state $\hat{s}_{t+1} = f_\theta(s_t, a_t)$ and contrasts against a counterfactual null action $\hat{s}_{t+1}^{\text{null}} = f_\theta(s_t, \vec{0})$. Computes a descriptive action-versus-null prediction score $\Delta_{\text{agency}} = \frac{E_{\text{null}} - E_{\text{pred}}}{\max(\epsilon, E_{\text{null}})}$, reality drift EMA, and self-alignment without modifying the main policy weights.
  - **Adaptive Memory Archaeology & Tissue Resection Assays** (*Mnemorph, MOLT, TESSERA & Archimedes*): Assesses connectome structural resilience, cross-species opcode/motif conservation, and Memory Trace Preservation Ratios (MTPR) under targeted virtual lesions and bioelectric rescue: classifies recovery into `engram-preserved`, `behaviorally-compensated`, or `irreversible-loss`.
  - **Tsodyks-Markram Short-Term Synaptic Plasticity (STP)** (*Tsodyks, Pawelzik & Markram PNAS 1998*): Implements frequency-dependent synaptic depression and facilitation with finite release probability $u \in (0, 1]$ and vesicle availability $R \in (0, 1]$, relaxing to resting identity efficacy $E = 1.000\times$ at quiescence.
  - **Chebyshev Spectral Graph Diffusion** (*Hammond, Vandergheynst & Gribonval 2011*): Computes multi-scale wavelet graph diffusion energies (local $s=0.5$, meso $s=2.0$, macro $s=6.0$) in $\mathcal{O}(K \cdot |\mathcal{E}|)$ via truncated Chebyshev orthogonal polynomials without matrix inversion.
  - **Byte-Level Semantic Guidance Bridge** (*NexusFlow & Byte-Level Attractors*): Untrained projection mapping high-dimensional connectome latent context into ASCII byte distributions and heuristic display labels (`foraging-chemotaxis`, `choke-point-maneuver`, `goal-exploitation`, `arousal-exploration`).
  - **Connectome Latent Consolidation Substrate (CLCS)** (*Supermix-Expanse v2 Latent MoE Consolidation*): Consolidates heterogeneous cross-species representations (worm, fly female, fly male, human) into a shared normalized latent space using Top-2 SwiGLU sparse latent experts, 16 learned associative memory slots with query-key attention, and a zero-initialized highway residual output gate (`out_gate = 0.0`) providing identity preservation at initialization in its standalone tests. This block is not an optimized input to the live navigation head.
  - **Dale's Principle / Law & Synaptic Input Fraction Scaling** (*Janelia Male-CNS Connectome & Collatz-Wielandt contraction*): Enforces biological neurotransmitter polarity on chemical synapses (cholinergic/monoaminergic $+1$, GABAergic/glutamatergic $-1$), normalized by measured incoming input fractions $f_{\text{in}}(post, pre) = \frac{W[post, pre]}{\sum_k W[post, k]}$. Computes shifted power iteration and Collatz-Wielandt upper bounds guaranteeing contractive rate dynamics ($\rho(|W|) \le 0.90$).
  - **Anti-Forgetting Soft-KL Anchor Distillation** (*Supermix-Expanse Anchor Checkpoint*): Online imitation is regularized against the frozen reference distribution; forgetting prevention has not been established. The objective uses KL distillation against the frozen reference policy: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{imitation}} + 0.15 \cdot D_{\text{KL}}(\pi_{\text{anchor}} \parallel \pi_{\text{student}})$.
  - **Central Complex (CX) 16-Wedge Ring Attractor Compass** (*FlyWire Drosophila Connectome & Central Complex E-PG/P-EN Models*): Couples insect modules (`fly_female`, `fly_male`) to a biologically authentic 16-wedge recurrent ring attractor with Mexican-hat cosine connectivity and angular velocity integration. Computes Population Vector Average (PVA) heading $\theta_{\text{PVA}}$, PVA coherence $R \in [0, 1]$, and triggers an optomotor counter-torque stabilization reflex if turbulence or wall proximity collapses coherence below $R_{\text{crit}} = 0.35$.
  - **Active Inference & Expected Free Energy (EFE) Decision Arbiter** (*Friston, Parr, Pezzulo Free Energy Principle*): Evaluates policies by decomposing action values into pragmatic homeostatic value ($-\|x - x_{\text{goal}}\|$), epistemic curiosity (information gain from LiDAR open vistas over unvisited cells), and empowerment (degrees of navigable escape freedom), dynamically balanced by Dopamine (exploitation) and Octopamine (arousal).
  - **Tri-Tier Cognitive Memory Hierarchy with Matched Lesion Assays** (*Supermix-Poseidon Mnemosyne architecture*): Maintains Tier 1 (surprise-driven recurrent working memory), Tier 2 (value-gated episodic store indexed by reward, surprise, and recency), and Tier 3 (topological cognitive spatial map tracking visit frequency, cell clearance, and landmark confidence), with 5 matched lesion modes (`intact`, `erased`, `shifted`, `random_retrieval`, `memory_free`) selectable live via the UI.
  - **Lyapunov Separatrix & Phase-Plane Stability Engine** (*Nonlinear Dynamical Systems & Separatrix Geometry*): Tracks the 4D homeostatic state $[E, S, D, V]$ (energy efficiency, goal proximity, barrier clearance, heading alignment), computing Lyapunov energy $L(\mathbf{x})$, orbital drift $\dot{L}$ (contracting attractor basin vs divergent saddle), and Euclidean distance to the critical obstacle collision separatrix.
  - **Virtual Connectome Lesion & Logit Sensitivity** (*Lappalainen et al., Nature Sep 2024; Shiu et al., Nature Oct 2024*): Counterfactual *in silico* silencing (`ablate_source`) computes per-source $\ell_2$ logit shift using the same weights and input window when silencing `worm`, `fly_female`, `fly_male`, or `human` in real time.
  - **Liquid Time-Constant (LTC) Continuous-Time Integration** (*Hasani et al., Nature Machine Intelligence 2022*): Standalone circuit integration accepts liquid retention factors. The live display estimates time constants $\tau_s(x,h)$ and retention $\alpha_s(t) = \exp(-\Delta t / \tau_s)$ for inspection; the live policy does not receive these estimated retention factors.
  - **Endogenous 3-Factor Monoaminergic Neuromodulation** (*Bargmann / Flavell C. elegans foraging & FlyWire neuromodulatory motifs*): Reward prediction error (Dopamine), goal proximity / exploited shelter (Serotonin), and novelty / boundary hazard arousal (Octopamine) produce a displayed plasticity gain estimate ($0.5\times\text{–}1.9\times$) and classify the active behavioral regime (`choke-point-maneuver`, `goal-exploitation`, `arousal-exploration`, `foraging-chemotaxis`).
  - **Rich-Club Hub & Reciprocal Motif Telemetry** (*Towlson et al. C. elegans Rich-Club*): Identifies high-degree Rich-Club command hubs (`AVAL`, `AVAR`, `AVBL`, `AVBR`, `PVCL`, `PVCR`), reciprocal chemical dyads, and electrical gap-junction edges alongside the top-4 firing authentic *C. elegans* neurons.
- **Persistent state and named checkpoints**: Versioned atomic `weights_only=True` checkpoints (`data/live/session.pt` plus named saves under `data/live/checkpoints/<name>.pt`) capture model weights, navigation head weights, Adam optimizer state, recurrent hidden states across all four source graphs, world state and private deterministic RNG state, auxiliary state, loss/reward history, and chat/fact memory. Initialization isolates global PyTorch randomness; ticks use private RNG state. Checkpoints verify bundle SHA-256 digests, ZIP entry CRCs, local headers, and tensor finiteness on load.
- **Local command, introspection & memory chat (`local-memory-and-control`)**: The chat drawer explicitly reports its backend as `local-memory-and-control` (not a pretrained LLM). It supports natural commands (`status`, `pause`, `resume`, `step 5`, `evolve 16` / `boost 16`, `world 19 labyrinth`, `preset chambers`, `dale` / `dales law`, `consolidation` / `clcs`, `anchor sync`, `compass` / `pva`, `efe` / `active inference`, `memory`, `lesion memory <mode>`, `phase` / `lyapunov`, `lesion` / `ablation`, `neuromodulators` / `liquid`, `motifs` / `rich club`, `top neurons` / `firing`, `routes` / `bridges`, `why` / `explain`, `learn on/off`, `go to cyan|amber|rose`, `move north|east|south|west`, `save <name>`, `load <name>`, `remember key = value`, `recall key`, `forget key`, `facts`, `reset world`, `help`, `bioelectric`, `agency`, `archaeology`).
- **Interactive 3D Three.js workspace**: Displays the scalable multi-chamber arena (`11²`, `15²`, `19²`, `25²`), Central Complex (CX) 16-wedge circular compass dial, Active Inference EFE probability bars, 1-click memory lesion buttons, Lyapunov phase plane status, 8-ray 3D LiDAR laser fan, glowing choke-point doorway portals, agent pose and trail, BFS shortest-path teacher ribbon, floating 4-source **3D Connectome Constellation Halo** with 12 live quadratic Bezier cross-species bridge arcs, beacons, hazards, live 4×5 cross-source route matrix, neuromodulation & virtual lesion bars, rolling loss/accuracy sparkline, keyboard navigation (`WASD` / arrow keys / `Space`), and deterministic automation hooks (`window.render_game_to_text()`, `window.advanceTime(ms)`).


## Run from this folder

Python 3.11+ and PyTorch are required. The supplied Windows machine already has Python 3.12 and PyTorch CPU; run these commands directly without installing another environment:

```powershell
python -m neurograft doctor
python -m neurograft live --port 8768
python -m neurograft inspect data/fixtures.bundle.json
python -m neurograft benchmark --bundle data/fixtures.bundle.json --out reports/runs/my-experiment --seeds 0,1,2,3,4 --epochs 30
python -m neurograft report reports/runs/my-experiment/report.json --out reports/my-experiment.html
```

Open the generated HTML in a browser. The included `reports/index.html` is the saved measured build report once validation finishes (`.\Run-NEUROGRAFT.ps1 -Mode Report` serves it on `http://127.0.0.1:8767/index.html`). Output bundles and benchmark directories must be new: existing observations and experiments are not silently overwritten. A failed run retains its running marker and partial receipts; choose a new directory after diagnosing it. Benchmark training resume is not implemented; live sessions support full state save/restore via `data/live/session.pt` and named checkpoints.

For another machine, install the package with `python -m pip install -e .` or `python -m pip install -e '.[llm,test]'`. This build did not install or upgrade global packages. Run `python -m neurograft --help` for all commands.

## What computes

Each source graph has its own node states, relation transforms and GRU. Messages aggregate along the actual sparse edges, normalized by absolute incoming weight. Undirected observations are stored once and expanded into computational arcs. Node structural features prevent an entirely identical-state broadcast from making graph structure irrelevant by construction.

Four pooled representations feed a batch-dependent router. It masks self routes, chooses top-k cross-source senders, and retains a null route. The learned bridges exist outside observed graph records. State is passed explicitly between `step` calls; `forward` starts a fresh episode. The task head reads only the human module, so the synthetic split-cue task requires cross-source communication.

```python
import torch
from neurograft.fixtures import make_fixture_bundle
from neurograft.dynamics import DynamicsConfig, GraphCircuit

torch.set_num_threads(2)
torch.manual_seed(0)
bundle = make_fixture_bundle()
circuit = GraphCircuit(bundle, DynamicsConfig())
observations = torch.zeros(2, 6, 4, 4)  # batch, time, source, features
result = circuit(observations)
print(result.logits.shape, result.context.shape, result.routes.shape)
# [2,2], [2,48], [2,4,5]
```

The human source slot describes macro-scale graph resolution. It does not supply human cognitive abilities. Its engineered state update is not a physiological model, and functional correlations are used as signed associations rather than claimed synapses.

## Authentic data and provenance

Every node retains its source identifier. Source manifests record the release, URL, citation, licence declaration, raw checksums, time and processing receipts. Save/load checks the normalized bundle digest. Technical authenticity and rights readiness are reported separately.

Acquire the pinned small worm source:

```powershell
python -m neurograft fetch-worm --out data/raw/new-openworm --electrical-conflicts exclude
python -m neurograft inspect data/raw/new-openworm/worm.bundle.json
python -m neurograft assemble --base data/fixtures.bundle.json --source data/raw/new-openworm/worm.bundle.json --out data/imported/mixed.bundle.json
```

The included acquisition is in `data/raw/openworm-pinned/`. It preserves the original CSV and `CellsOnly.net.nml`. The latter declares 302 neuron populations; the generated-cell directory also includes a muscle model, so filenames alone are not used as a neuron list. A recorded alias map aligns motor names such as `VA1` in the model to `VA01` in the edge table, preserving the edge table's original IDs. Neurons with no surviving observations remain present.

The source has inconsistent counts in some mirrored electrical pairs. Default acquisition rejects them. The explicit `exclude` policy removes those ambiguous pairs from the working graph, records both source values and excluded-row counts, and leaves the raw source unchanged. Equal mirrored electrical contacts are stored once. The normalized graph is therefore a documented filtered derivative, not a claim that the original table is internally consistent or complete.

To use fly data, export a bounded source-ID-preserving circuit and normalize it to the CSV schema. Codex and neuPrint account/token steps belong to the data owner; tokens are not stored by this package. To use human data, supply an authorized labelled region matrix and its manifest. See [CITATIONS.md](CITATIONS.md) and [LICENSE_MATRIX.md](LICENSE_MATRIX.md).

Normalized neuron CSV headers:

```text
nodes: id,kind
edges: src,dst,relation,weight,directed,confidence
```

Use local original IDs, `kind=neuron` or `region`, and relations `chemical`, `electrical`, `macro_structural`, `macro_functional`. Chemical records are directed; electrical and macro records are undirected and stored once. Confidence may be blank. Imports reject inconsistent scales, duplicate observations, unresolved endpoints and nonfinite values. Local CSV files are capped at 64 MiB each.

```powershell
python -m neurograft import-csv --nodes examples/worm-nodes.csv --edges examples/worm-edges.csv --manifest examples/worm-manifest.json --out data/example-worm.bundle.json
python -m neurograft import-human --matrix examples/human-matrix.csv --manifest examples/human-manifest.json --out data/example-human.bundle.json
```

The examples are fixtures and remain fixtures after import. For authentic data, explicitly declare `status=imported`, a real source version, URL, citation, licence, and access status in the manifest. The importer computes file checksums. `authorized` records the caller's rights declaration, not a legal determination or an automated verification of provenance.

## Benchmark and controls

At time zero, worm and female-fly modules receive separate signed bits. After a delay, the human module receives a common query indicator and predicts XOR. Its input never contains the answer or either cue. The male-fly module initially receives no direct task signal. This is a deliberately small synthetic communication/memory task with only four underlying cue patterns; distinct PRNG seeds create separate nuisance draws, not novel biological behaviours or a language-generalization test.

Four variants use identical allocated trainable parameters, initial weights, examples, minibatch orders and optimizer steps:

- `grafted`: sparse routing and recurrence.
- `no_graft`: routing disabled.
- `no_recurrence`: recurrent history disabled.
- `rewired`: relation-wise degree-preserving graph controls, with actual swap receipts.

Rewiring preserves observed self-loops, directed in/out degrees, undirected degrees, and relation-wise weight multisets. It does not guarantee preserved motifs, connectivity, weighted degrees or physiology. Reported topology hashes reveal whether a control actually changed. These controls have different active gradients and effective compute; equal parameter counts do not establish exact compute matching.

Select checkpoints using validation accuracy and loss, then evaluate test once. Reports retain every seed, paired bootstrap intervals, parameters, timing, sampled process RSS, source status, graph hashes and all epoch logs. Five-seed intervals are exploratory. Read gains over `no_graft` in light of the task's intentional requirement for communication; a topology claim requires the rewired control and stronger independent tasks.

Replay an inference checkpoint using its original input bundle:

```powershell
python -m neurograft replay --bundle data/fixtures.bundle.json --checkpoint reports/runs/verified-five-seed/runs/seed_0/grafted/checkpoint.pt
```

Checkpoints include configurations, selected weights, graph and weight hashes, task version, split seeds and topology. Rewired checkpoints reconstruct their own control graph and verify it against the original input. Exact replay is scoped to the recorded CPU runtime. Checkpoints use restricted `weights_only=True` loading with an input-size guard; only load files from sources you trust.

## Frozen Qwen interface

`FrozenLanguageGraft` projects circuit context into the model's existing token embeddings through a trainable zero-initialized gate. It preserves sequence length and has an explicit original-model bypass. Backbone parameters stay frozen, while autograd remains enabled through the backbone so a language loss can train the graph and projection once the gate opens. Projection weights start nonzero to let the gate learn; encoders receive no language gradient while the gate is exactly zero.

The cache-free greedy reference recomputes the full prefix and does not advance circuit state on every generated token. Cached hybrid DeltaNet decoding and LoRA are future work. Do not interpret interface tests with random tiny weights as pretrained capability.

`NeurograftLanguageModel(circuit, language_adapter)` wires a `GraphCircuit` directly to `FrozenLanguageGraft`. Its forward accepts `input_ids`, `graph_sequence`, optional masks and language labels. The tested pathway carries language-loss gradients into actual graph relation transforms while the backbone remains unchanged. Supply only graph observations available at prediction time; deriving graph inputs from answer labels would leak supervision.

For an explicit real model runtime probe, provide the full immutable Hugging Face revision:

```powershell
python -m neurograft model-probe --model-id Qwen/Qwen3.5-0.8B --revision FULL_40_CHARACTER_COMMIT_SHA --download
```

Loading defaults to local files only unless `--download` is supplied, and requires at least 6 GiB available RAM for the FP32 probe. This PC currently has substantially less free RAM. Qwen3-0.6B is supported only when explicitly selected; a different cached model is never substituted. This command checks the pretrained backend and zero gate, not a trained graph-language adapter.

## Validation and project files

The saved five-seed fixture experiment used 10,606 allocated trainable parameters, 30 epochs per seed/variant, fixed 128/64/256 train/validation/test draws, and four matched variants. It completed in approximately 384 seconds on the recorded CPU runtime.

| Variant | Mean held-out accuracy |
|---|---:|
| Grafted recurrent circuit | 60.0% |
| Graft disabled | 50.0% |
| Recurrence disabled | 49.84% |
| Degree-preserving rewired graphs | 54.14% |

One grafted seed reached 100%; four remained at 50%. The paired graft-minus-rewired difference was +5.86 percentage points, with a descriptive bootstrap interval from -12.42 to +30.00 points. This does **not** establish a reliable topology advantage. These graphs are synthetic fixtures and the task has only four cue patterns.

The separate authentic-worm/mixed-fixture smoke experiment ran all four variants for two epochs with one seed. All scored 50%; its purpose was to verify actual sparse source ingestion, training and replay, not claim learned biological capability. All **24** saved checkpoints across the two experiments reproduced their recorded held-out logits exactly on the current CPU runtime.

Read `reports/runs/verified-five-seed/report.json`, `reports/runs/worm-mixed-smoke/report.json`, `reports/checkpoint-replay.json`, and `reports/circuit-trace.json` for complete receipts. The earlier `reports/runs/fixture-five-seed/` directory contains an interrupted preliminary run, explicitly marked `ABORTED.json` after independent review found a topology-hash representation issue. It is not used for the reported final scores. Hyperparameters were retained for the corrected final run.

```powershell
python -m pytest -q
python -m compileall -q neurograft
```

`neurograft/` contains the package; `tests/` covers integrity, gradients, masking, controls, deterministic replay and the real tiny Qwen interfaces. `docs/implementation-plan.md` records scope; component reports and independent review record verification evidence. `reports/` holds measured local artifacts. Raw imported data and checkpoints are excluded by `.gitignore`; no remote repository was created or published.

Future milestones are authentic female/male fly and authorized human imports, resource-verified pretrained Qwen execution, language-adapter training, physiological calibration where supported, and broader controlled tasks. Each has a separate completion gate.
