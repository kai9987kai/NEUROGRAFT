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
The complete test suite passed **613 tests in 327.97s** with **0 failures**:
- `tests/test_predictive_world_model.py`: 4 passed (zero-init invariance, gradient flow, state dict roundtrips).
- `tests/test_cx_pfl_steering.py`: 6 passed (directional preferences, gain modulation, wall masking, controller integration).
- `tests/test_topological_memory.py`: 5 passed (transition counts, BFS shortest path, reachability ratios, memory reset).
- `tests/test_recovery_curriculum.py`: 3 passed (smoke training runs with recovery policy and predictive multi-task loss).
- `tests/test_server.py`: 38 passed (stabilized against Windows loopback connection resets).
- Legacy suites (595 tests): All regression tests across `dynamics`, `live`, `memory`, `navigation`, `stp`, `bioelectric`, and `server` passed with zero regressions.

### 2. Wheel Package Parity
- **Wheel**: `dist/neurograft-0.11.0-py3-none-any.whl`
- **SHA-256**: `0dccf5d618b88d5b062b31487cccad10bb1a55d62a78448a10b9ebf7c1c66ab4`
- **Package Files**: 48 package files, matching source repository exactly.
- **Verification Receipt**: [`reports/v11-package-verification.json`](../reports/v11-package-verification.json).

### 3. Live Session State Preservation
- `data/live/session.pt` was checked and verified: 0 modifications to weights or schema.
- Deserialization across legacy formats (v1, v2, v3) is verified and clean.

---

## Empirical Scope Boundaries

1. **Synthetic Fixture Grounding**: The underlying circuits and evaluations run on synthetic grid worlds. While inspired by Drosophila connectomic CX architecture and mammalian cognitive maps, these mechanisms do not claim biological equivalence or live organism behavior.
2. **Computational Constraints**: All algorithms operate strictly in CPU memory with $O(N + E)$ time and space bounds, requiring no external GPU acceleration.
3. **No Unearned Checkpoint Promotion**: The candidate architectures are provided as opt-in tools and controllers; the saved live state remains untouched.
