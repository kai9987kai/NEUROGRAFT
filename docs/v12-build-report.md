# NEUROGRAFT v0.12: Hippocampal Successor Representation & SWR Replay

21 October 2026. Local implementation, mathematical verification, and full regression test suite pass; no remote publication without test certification.

## Executive Summary

NEUROGRAFT v0.12 extends the sparse, provenance-preserving multi-source neural graft architecture with computational neuroscience mechanisms modeled after mammalian hippocampal and entorhinal cortex dynamics:
1. **Hippocampal-Entorhinal Successor Representation (SR)**: Decouples environmental transition topology from instantaneous task goals via $M = (I - \gamma T)^{-1}$. Evaluates zero-shot policy re-planning on experienced transitions without privileged full-map access.
2. **Entorhinal Grid Cell Basis (Normalized Graph Laplacian Harmonics)**: Computes the spectral eigen-decomposition of the normalized graph Laplacian $L = I - D^{-1/2} A D^{-1/2}$ over experienced transitions, extracting multiscale spatial frequency harmonics that serve as a synthetic grid-cell coordinate system across arbitrary arena topologies.
3. **Hippocampal Sharp-Wave Ripple (SWR) Offline Replay Engine**: Executes reverse prioritized Bellman value propagation sweeps ($V(u) \leftarrow \max_a [R + \gamma \sum_v P(v \mid u, a) V(v)]$) during pauses, goal reach events, or explicit user/REST triggers.
4. **Active Inference & Epistemic Uncertainty Controller (`sr_steer`)**: Integrates raw circuit logits, SR predictive value gradients, and count-based epistemic novelty bonuses ($\mathcal{U}(u, a) \propto 1 / \sqrt{N(u, a) + 1}$) with deterministic wall collision masking.
5. **Interactive Laboratory Controls & Telemetry**: Exposed `'sr_steer'` in Web UI selector, added interactive `🧠 SWR Replay` button in live session bar, wired `/api/control` action `'replay'`, and added natural language commands (`replay`, `swr`, `consolidate`) to the live chat interface.

---

## Architectural & Mathematical Foundations

### 1. Successor Representation Matrix $M$
Located in [`neurograft/memory.py`](file:///C:/Users/kai99/Desktop/New%20folder%20(5)/neurograft/memory.py):
Given empirical transition counts $N(u, a, v)$ accumulated in `TopologicalMemoryGraph`, the transition matrix $T \in \mathbb{R}^{|V| \times |V|}$ is formed under a uniform random walk over observed actions:
$$T_{u, v} = \frac{1}{|A_u|} \sum_{a \in A_u} \frac{N(u, a, v)}{\sum_{w} N(u, a, w)}$$
The Successor Representation matrix $M$ is solved via matrix inversion in double precision:
$$M = (I - \gamma T)^{-1} = \sum_{k=0}^{\infty} \gamma^k T^k$$
Each entry $M(u, v)$ represents the expected discounted future visitation frequency of node $v$ starting from node $u$. Zero-shot value re-planning toward any goal location $g$ requires only a linear dot product with empirical reward vector $\mathbf{R}$:
$$V^{\text{SR}}(u) = \sum_{v \in V} M(u, v) R(v)$$

### 2. Normalized Graph Laplacian Eigen-Modes (Synthetic Grid Cells)
Located in [`neurograft/memory.py`](file:///C:/Users/kai99/Desktop/New%20folder%20(5)/neurograft/memory.py):
The symmetric adjacency matrix $A$ across known transitions defines node degrees $D_{u, u} = \sum_v A_{u, v}$. The normalized graph Laplacian is:
$$L = I - D^{-1/2} A D^{-1/2}$$
Solving $L \mathbf{\phi}_k = \lambda_k \mathbf{\phi}_k$ via `torch.linalg.eigh` extracts spatial standing wave modes ordered by spatial frequency $\lambda_0 \le \lambda_1 \le \dots \le \lambda_{K-1}$. Low-frequency eigenvectors provide global topological coordinates, mirroring entorhinal grid field harmonics.

### 3. Sharp-Wave Ripple (SWR) Reverse Replay
Located in [`neurograft/memory.py`](file:///C:/Users/kai99/Desktop/New%20folder%20(5)/neurograft/memory.py):
Simulates hippocampal offline memory consolidation during quiet pauses or upon reaching a reward beacon:
1. Replay sweeps initiate at reward nodes or frontier boundaries.
2. Trajectories are replayed in reverse: predecessors $u \in \text{Pred}(v)$ are selected proportionally to transition frequency.
3. Bellman value backup is performed along the reverse trajectory:
   $$V(u) \leftarrow \max \left( V(u),\, R(v) + \gamma V(v) \right)$$
4. Returns the number of sweeps, count of replayed transitions, total value convergence delta $\Delta V$, and recorded trajectories.

### 4. Active Inference Epistemic Steering (`sr_steer`)
Located in [`neurograft/navigation_controller.py`](file:///C:/Users/kai99/Desktop/New%20folder%20(5)/neurograft/navigation_controller.py):
Action selection from current pose $u$ under `'sr_steer'` computes combined action scores:
$$Q(u, a) = \hat{\pi}(a \mid o) + \alpha_{\text{SR}} \cdot Q^{\text{SR}}(u, a) + \beta_{\text{epistemic}} \cdot \mathcal{U}(u, a)$$
where:
- $\hat{\pi}(a \mid o)$ is the raw policy logit normalized over legal actions.
- $Q^{\text{SR}}(u, a) = \sum_v P(v \mid u, a) V^{\text{SR}}(v)$ is the expected successor return toward the goal.
- $\mathcal{U}(u, a) = 1 / \sqrt{N(u, a) + 1}$ is the epistemic novelty drive promoting exploration of unvisited corridors.
- Wall collision masking strictly sets $Q(u, a) = -\infty$ for any action colliding with detected local obstacles.

---

## Verification & Safety Boundaries

1. **Strict Checkpoint Invariance**:
   - `data/live/session.pt` initial weights and schema are 100% preserved.
   - `TopologicalMemoryGraph.to_dict()` and `to_state()` preserve exact keys `{"version", "nodes", "transitions", "reward_totals", "total_transitions"}` ensuring full bidirectional serialization compatibility.
   - Restoring sessions across v1, v2, v3, and v4 artifacts passes cleanly.
2. **Computational Bounds**:
   - Inversion of $(I - \gamma T)$ on graphs up to 256 nodes executes in $< 15$ ms on CPU.
   - SWR replay sweeps are bounded to a maximum of 16 sweeps and horizon of 8 steps, completing in $< 5$ ms.
3. **Scientific Grounding Disclaimer**:
   - Successor representations and normalized Graph Laplacian harmonics are mathematical approximations of hippocampal-entorhinal place/grid cell theories (Stachenfeld, M. M., Botvinick, M. M., & Gershman, S. J., *Nature Neuroscience*, 2017).
   - These mechanisms are evaluated on synthetic multi-chamber grid navigation arenas and do not constitute biological equivalence to living neural tissue.

---

## Verification Receipts

### 1. Complete Regression Test Suite
- **Command**: `pytest -ra`
- **Result**: **744 passed in 296.39s (0:04:56)** with **0 failures, 0 errors**.
- **Coverage**:
  - `tests/test_successor_representation.py`: 12/12 passed (matrix inversion, Bellman SWR replay, epistemic action scoring, wall masking fallback, REST replay control, live checkpoint serialization).
  - All 732 prior tests across dynamics, live, memory, navigation, STP, bioelectric, server, and curriculum pass cleanly with 100% backward compatibility.

### 2. Wheel Package Parity
- **Wheel**: `dist/neurograft-0.12.0-py3-none-any.whl`
- **SHA-256**: `0216fa2345cb3ff9a45806e5e36d45031d9c5da8eb59ae57671eb75cff568d2f`
- **Package Files**: 48 package files, matching source repository exactly.
- **Verification Receipt**: [`reports/v12-package-verification.json`](file:///C:/Users/kai99/Desktop/New%20folder%20(5)/reports/v12-package-verification.json).

### 3. Live Session State Preservation
- `data/live/session.pt` preserved with zero mutations to weights or continuation state.
- Deserialization and migration across legacy formats (v1, v2, v3) verified and clean.
