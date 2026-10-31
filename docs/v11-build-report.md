# NEUROGRAFT v0.11: Connectome Predictive World Models & Central Complex Steering

The v0.11 implementation, architectural extensions, and full 613-test regression suite are locally verified. This report records the verification boundary as of 21 October 2026; it documents the implemented mechanisms, test receipts, and empirical scope boundaries truthfully, without unearned claims of biological equivalence or unverified checkpoint promotion.

v0.10 explored matched learned-navigation studies comparing human vs. all-source residual readouts and quarter-turn augmentation. v0.11 advances the architecture by incorporating foundational principles from recent 2024–2026 neurobiology and predictive representation research:
1. **Drosophila Central Complex (CX) PFL Steering Engine**: Vector-based heading and goal integration modeled after connectome PFL1, PFL2, and PFL3 neural circuits.
2. **Connectome Predictive World Model Head**: An internal forward dynamics head predicting sensory observation transitions $\hat{o}_{t+1} \sim p(o_{t+1} \mid c_t, a_t)$ with zero-initialization invariance.
3. **Recovery-Augmented DAgger Curriculum**: Off-policy exploration with expert recovery labeling to address compounding errors under distribution drift.
4. **Topological Cognitive Memory Graph**: Online directed transition graph $G = (V, E)$ in Tier 3 memory tracking empirical transition probabilities, BFS geodesics, and reachability.

---

## Implemented Mechanisms

### 1. Drosophila Central Complex (CX) Vector Steering
Located in [`neurograft/compass.py`](../neurograft/compass.py) and integrated into [`neurograft/navigation_controller.py`](../neurograft/navigation_controller.py):
- **PFL1 Alignment**: Calculates cosine alignment between cardinal action angles and goal vector $\cos(\theta_a - \theta_g)$.
- **PFL2 Gain Modulation**: Computes dynamic steering gain scaled by angular heading-goal error:
  $$\text{gain}_{\text{PFL2}} = 1.0 + \gamma \frac{|\theta_g - \theta_h|}{\pi}$$
- **PFL3 Asymmetric Turn Drive**: Computes turning bias relative to current body heading $\cos(\theta_a - \theta_h)$.
- **Controller Action**: The new `'cx_steer'` mode biases raw circuit logits with weighted CX steering scores while applying deterministic collision masking against detected obstacles.
- **Backward Compatibility**: `MODES` tuple retains `('policy', 'masked_policy', 'observed_map')` for exact backward compatibility with v0.9/v0.10 test suites; `ALL_MODES` exposes `('policy', 'masked_policy', 'observed_map', 'cx_steer')` to `live.py` and `service.py`.

### 2. Predictive World Model Forward Head
Located in [`neurograft/dynamics.py`](../neurograft/dynamics.py):
- **Architecture**: A two-layer MLP mapping concatenated context representation $c_t$ and one-hot action $a_t$ to predicted sensory observation $\hat{o}_{t+1} \in \mathbb{R}^{4 \times D}$.
- **Zero-Initialization Invariance**: Final linear weights and biases are initialized to exact zeros ($\mathbf{W} = \mathbf{0}, \mathbf{b} = \mathbf{0}$). Fresh models with `predictive_head=True` yield bit-identical initial logits and baseline action distributions.
- **Multi-Task Objective**: During training in [`neurograft/navigation_training.py`](../neurograft/navigation_training.py), optional loss weighting $\lambda_{\text{pred}}$ penalizes mean squared prediction error:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \lambda_{\text{pred}} \frac{1}{4D} \sum_{i=1}^4 \sum_{j=1}^D (\hat{o}_{t+1, ij} - o_{t+1, ij})^2$$

### 3. Recovery-Augmented Curriculum
Located in [`neurograft/navigation_training.py`](../neurograft/navigation_training.py):
- Added `collection_policy='recovery'`: Interleaves policy exploration actions with expert demonstrations to sample off-trajectory recovery states.
- Mitigates covariate shift without modifying legacy sample dictionary schemas or hash keys for `mixed` or `expert` collection modes.

### 4. Topological Cognitive Memory Graph
Located in [`neurograft/memory.py`](../neurograft/memory.py):
- Implements `TopologicalMemoryGraph` within Tier 3 spatial memory.
- Accumulates transition counts $N(u, a, v)$ across visited $(x, z)$ coordinates.
- Computes empirical transition distributions $P(v \mid u, a)$, unweighted BFS geodesics between known nodes, and reachability ratios.
- Kept isolated from `TriTierMemory.to_dict()` top-level schema to prevent checkpoint deserialization key validation errors in legacy session states.

---

## Verification Receipts

### 1. Test Suite Verification
The complete test suite passed **732 tests in ~9 minutes** with **0 failures**:
- `tests/test_predictive_world_model.py`: passed (zero-init invariance, gradient flow, state dict roundtrips, context rejection).
- `tests/test_cx_pfl_steering.py` & `tests/test_steering_controls.py`: passed (directional preferences, gain modulation, wall masking, controller integration).
- `tests/test_topological_memory.py` & `tests/test_topological_checkpoint.py`: passed (transition counts, BFS shortest path, reachability ratios, checkpoint continuation, migration from v2/v3).
- `tests/test_recovery_curriculum.py` & `tests/test_predictive_training_integrity.py`: passed (smoke training runs with recovery policy and predictive multi-task loss).
- `tests/test_v11_compatibility.py` & `tests/test_predictive_recovery_study.py`: passed (study pipeline, exact hash invariance, backward compatibility).
- `tests/test_server.py`: 38 passed (stabilized against Windows loopback connection resets).
- Legacy suites: All regression tests across `dynamics`, `live`, `memory`, `navigation`, `stp`, `bioelectric`, and `server` passed with zero regressions.

### 2. Wheel Package Parity
- **Wheel**: `dist/neurograft-0.11.0-py3-none-any.whl`
- **SHA-256**: `d8a3accb33616429f808ebd833429dd75591a3c6d7af9354ce1f0963f436d61c`
- **Package Files**: 48 package files, matching source repository exactly.
- **Verification Receipt**: [`reports/v11-package-verification.json`](../reports/v11-package-verification.json).

### 3. Predeclared Learning Study (`navigation-v11-predictive-recovery-20261021`)
A 4-arm matched learning study across 2 initialization seeds (0, 1000) evaluated the empirical impact of the forward predictive head and recovery-augmented curriculum across 16 frozen test episodes per arm:

| Arm | Collection Policy | Predictive Loss ($\lambda$) | Success Rate | Mean SPL | Collision Rate | Prediction MSE (Probe) |
|---|---|---|---:|---:|---:|---:|
| `expert_aux0` | expert | 0.0 | 7/16 (43.8%) | 0.4375 | 84.6% | 0.3251 |
| `expert_aux05` | expert | 0.5 | 7/16 (43.8%) | 0.4271 | 84.1% | **0.1227** |
| `recovery_aux0` | recovery | 0.0 | 3/16 (18.8%) | 0.1750 | 90.7% | 0.3251 |
| `recovery_aux05` | recovery | 0.5 | 3/16 (18.8%) | 0.1750 | 85.7% | **0.1024** |

Key findings:
- **Sensory Transition Prediction**: The recovery predictor's mean observation MSE was 0.1024, compared with 0.3251 for the matched loss-off head, whose zero-initialized output remained untrained. The approximately 68% reduction is relative to that zero predictor. On the same frozen probes, copy-current scored 0.1915 and explicit local kinematics scored **0.05884**, outperforming the learned predictor overall. Learned recovery prediction did improve wall-channel MSE to approximately **0.1877**, versus **0.23535** for the kinematic baseline's copied walls. These results distinguish learning unknown wall features from relearning deterministic position and heading updates; they do not establish an overall predictive advantage over kinematics. Raw probe values and baseline definitions are retained in [`prediction-probes.json`](../reports/runs/navigation-v11-predictive-recovery-20261021/prediction-probes.json).
- **Behavioral Result**: Adding prediction loss did not improve raw goal success: expert collection reached 7/16 with either loss setting, and recovery collection reached 3/16 with either setting. Expert SPL declined slightly; recovery SPL stayed unchanged. Recovery collisions declined from 90.7% to 85.7%, a descriptive secondary change in this small pilot. The pilot provides no demonstrated navigation benefit from the auxiliary objective.

**Archive verification boundary (rechecked 31 October 2026):** the existing v11 study fails plain verification with `study artifact hash mismatch: report.html`. The JSON/probe numbers above can be inspected, but the archive as a whole cannot currently be described as passing its seal verification. Its original files and manifest remain unchanged; this narrative correction does not repair or reseal the historical archive.

### 4. Live Session State Preservation
- `data/live/session.pt` was checked and verified: 0 modifications to weights or schema.
- Deserialization across legacy formats (v1, v2, v3) is verified and clean.

---

## Empirical Scope Boundaries

1. **Synthetic Fixture Grounding**: The underlying circuits and evaluations run on synthetic grid worlds. While inspired by Drosophila connectomic CX architecture and mammalian cognitive maps, these mechanisms do not claim biological equivalence or live organism behavior.
2. **Computational Constraints**: All algorithms operate strictly in CPU memory with $O(N + E)$ time and space bounds, requiring no external GPU acceleration.
3. **No Unearned Checkpoint Promotion**: The candidate architectures are provided as opt-in tools and controllers; the saved live state remains untouched.
