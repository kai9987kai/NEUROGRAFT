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

## Live 3D environment, continuous learning, and saved state (v0.2)

Start the persistent local live session and open `http://127.0.0.1:8768/`:

```powershell
python -m neurograft live --bundle data/imported/mixed.bundle.json --state data/live/session.pt --port 8768
# or use the helper launcher:
.\Run-NEUROGRAFT.ps1 -Mode Live
```

What the live session provides:

- **Scalable Multi-Chamber 3D Environment (`15×15` default, scalable `5×5` to `31×31`)**: Supports live architectural presets (`labyrinth`, `chambers`, `corridors`, `pillars`) with guaranteed full free-cell connectivity, 8-ray LiDAR / optic-flow clearance beams (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`), topological choke-point doorway rings, exponential chemoattractant gradient plumes, and tortuosity/detour index metrics.
- **Continuous online adaptation (`supervised-shortest-path-imitation`)**: On each step in the multi-chamber navigation world, source-specific sensory features feed the worm (beacon bearing/distance), female-fly (local wall/hazard clearances), male-fly (visit recency and heading velocity), and human (active goal beacon selector and step urgency) graph modules. When **Learn** is enabled, the 4-way navigation head (`north`, `east`, `south`, `west`), sparse cross-source grafts, and recurrent encoders update online via AdamW (`lr=8e-3`, window length `4`, batch size `8`) with gradient clipping (`1.0`) against the deterministic shortest-path teacher action, plus an interactive **⚡ Boost ×16** evolution burst trigger.
- **2024–2026 Connectome & Cognitive Architecture Grounding (`session.telemetry()`)**:
  - **Connectome Latent Consolidation Substrate (CLCS)** (*Supermix-Expanse v2 Latent MoE Consolidation*): Consolidates heterogeneous cross-species representations (worm, fly female, fly male, human) into a shared normalized latent space using Top-2 SwiGLU sparse latent experts, 16 learned associative memory slots with query-key attention, and a zero-initialized highway residual output gate (`out_gate = 0.0`) guaranteeing exact identity preservation at initialization.
  - **Dale's Principle / Law & Synaptic Input Fraction Scaling** (*Janelia Male-CNS Connectome & Collatz-Wielandt contraction*): Enforces biological neurotransmitter polarity on chemical synapses (cholinergic/monoaminergic $+1$, GABAergic/glutamatergic $-1$), normalized by measured incoming input fractions $f_{\text{in}}(post, pre) = \frac{W[post, pre]}{\sum_k W[post, k]}$. Computes shifted power iteration and Collatz-Wielandt upper bounds guaranteeing contractive rate dynamics ($\rho(|W|) \le 0.90$).
  - **Anti-Forgetting Soft-KL Anchor Distillation** (*Supermix-Expanse Anchor Checkpoint*): Online imitation is continuously anchored against representational collapse and catastrophic forgetting via soft-KL divergence teacher distillation against a frozen initial anchor policy: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{imitation}} + 0.15 \cdot D_{\text{KL}}(\pi_{\text{student}} \parallel \pi_{\text{anchor}}) + 0.05 \cdot \mathcal{L}_{\text{balance}}$.
  - **Central Complex (CX) 16-Wedge Ring Attractor Compass** (*FlyWire Drosophila Connectome & Central Complex E-PG/P-EN Models*): Couples insect modules (`fly_female`, `fly_male`) to a biologically authentic 16-wedge recurrent ring attractor with Mexican-hat cosine connectivity and angular velocity integration. Computes Population Vector Average (PVA) heading $\theta_{\text{PVA}}$, PVA coherence $R \in [0, 1]$, and triggers an optomotor counter-torque stabilization reflex if turbulence or wall proximity collapses coherence below $R_{\text{crit}} = 0.35$.
  - **Active Inference & Expected Free Energy (EFE) Decision Arbiter** (*Friston, Parr, Pezzulo Free Energy Principle*): Evaluates policies by decomposing action values into pragmatic homeostatic value ($-\|x - x_{\text{goal}}\|$), epistemic curiosity (information gain from LiDAR open vistas over unvisited cells), and empowerment (degrees of navigable escape freedom), dynamically balanced by Dopamine (exploitation) and Octopamine (arousal).
  - **Tri-Tier Cognitive Memory Hierarchy with Matched Lesion Assays** (*Supermix-Poseidon Mnemosyne architecture*): Maintains Tier 1 (surprise-driven recurrent working memory), Tier 2 (value-gated episodic store indexed by reward, surprise, and recency), and Tier 3 (topological cognitive spatial map tracking visit frequency, cell clearance, and landmark confidence), with 5 matched lesion modes (`intact`, `erased`, `shifted`, `random_retrieval`, `memory_free`) selectable live via the UI.
  - **Lyapunov Separatrix & Phase-Plane Stability Engine** (*Nonlinear Dynamical Systems & Separatrix Geometry*): Tracks the 4D homeostatic state $[E, S, D, V]$ (energy efficiency, goal proximity, barrier clearance, heading alignment), computing Lyapunov energy $L(\mathbf{x})$, orbital drift $\dot{L}$ (contracting attractor basin vs divergent saddle), and Euclidean distance to the critical obstacle collision separatrix.
  - **Virtual Connectome Lesion & Causal Attribution** (*Lappalainen et al., Nature Sep 2024; Shiu et al., Nature Oct 2024*): Counterfactual *in silico* silencing (`ablate_source`) computes the exact per-source policy KL divergence $D_{\mathrm{KL}}(\pi_{\text{intact}} \parallel \pi_{-s})$ and $\ell_2$ logit shift when silencing `worm`, `fly_female`, `fly_male`, or `human` in real time.
  - **Liquid Time-Constant (LTC) Continuous-Time Integration** (*Hasani et al., Nature Machine Intelligence 2022*): Input-dependent liquid time constants $\tau_s(x,h)$ and retention $\alpha_s(t) = \exp(-\Delta t / \tau_s)$ dynamically modulate recurrent state decay across sensory, obstacle-avoidance, path-integration, and executive sub-graphs.
  - **Endogenous 3-Factor Monoaminergic Neuromodulation** (*Bargmann / Flavell C. elegans foraging & FlyWire neuromodulatory motifs*): Reward prediction error (Dopamine), goal proximity / exploited shelter (Serotonin), and novelty / boundary hazard arousal (Octopamine) drive a live plasticity gain multiplier ($0.5\times\text{–}1.9\times$) and classify the active behavioral regime (`choke-point-maneuver`, `goal-exploitation`, `arousal-exploration`, `foraging-chemotaxis`).
  - **Rich-Club Hub & Reciprocal Motif Telemetry** (*Towlson et al. C. elegans Rich-Club*): Identifies high-degree Rich-Club command hubs (`AVAL`, `AVAR`, `AVBL`, `AVBR`, `PVCL`, `PVCR`), reciprocal chemical dyads, and electrical gap-junction edges alongside the top-4 firing authentic *C. elegans* neurons.
- **Persistent state and named checkpoints**: Atomic `weights_only=True` checkpoints (`data/live/session.pt` plus named saves under `data/live/checkpoints/<name>.pt`) capture model weights, navigation head weights, AdamW optimizer state, recurrent hidden states across all four source graphs, world state and deterministic RNG state, Python/PyTorch RNG states, loss/reward history, and chat/fact memory. Checkpoints verify bundle SHA-256 digests, ZIP entry CRCs, local headers, and tensor finiteness on load.
- **Local command, introspection & memory chat (`local-memory-and-control`)**: The chat drawer explicitly reports its backend as `local-memory-and-control` (not a pretrained LLM). It supports natural commands (`status`, `pause`, `resume`, `step 5`, `evolve 16` / `boost 16`, `world 19 labyrinth`, `preset chambers`, `dale` / `dales law`, `consolidation` / `clcs`, `anchor sync`, `compass` / `pva`, `efe` / `active inference`, `memory`, `lesion memory <mode>`, `phase` / `lyapunov`, `lesion` / `ablation`, `neuromodulators` / `liquid`, `motifs` / `rich club`, `top neurons` / `firing`, `routes` / `bridges`, `why` / `explain`, `learn on/off`, `go to cyan|amber|rose`, `move north|east|south|west`, `save <name>`, `load <name>`, `remember key = value`, `recall key`, `forget key`, `facts`, `reset world`, `help`).
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
