# v0.12: Hippocampal-Entorhinal Successor Representations, Sharp-Wave Ripple Replay & Active Inference

21 October 2026. Architectural specification and empirical design.

## Research reviewed (2024–2026)

| Primary source | Biological / Algorithmic Principle | NEUROGRAFT Integration & Scope Boundary |
| --- | --- | --- |
| Stachenfeld, Botvinick, & Gershman (*Nature Neuroscience*, 2017); Dayan (1993) | The hippocampal predictive map as a Successor Representation: $M = (I - \gamma T)^{-1}$ represents expected future discounted state occupancy, separating environment transition structure from immediate reward values. | Implemented `compute_successor_representation()` and `successor_action_values()` in `TopologicalMemoryGraph`. Operates directly on the agent's experienced topological transitions without privileged map knowledge. |
| Whittington et al. (*Cell*, 2020; *Trends Cogn. Sci.*, 2022); George et al. (*Science*, 2021) | Tolman-Eichenbaum Machine (TEM): Entorhinal grid cells represent eigenvectors of the structural transition graph (multiscale spatial harmonics), decomposing complex topologies into low-dimensional coordinate bases. | Added `graph_laplacian_modes()` computing eigenvalues and eigenvectors of the normalized graph Laplacian $L = I - D^{-1/2} A D^{-1/2}$, exposing grid-like spatial wave modes in telemetry. |
| Buzsáki (*Nat. Rev. Neurosci.*, 2015); Ólafsdóttir et al. (*Neuron*, 2018); Liu et al. (*Cell*, 2021) | Sharp-Wave Ripples (SWR, 150–250 Hz) during pauses and post-reward quiescence replay compressed spatial sequences reverse and forward, performing offline value propagation and memory consolidation. | Implemented `trigger_swr_replay()` in `TopologicalMemoryGraph` and `TriTierMemory`. Propagates value along directed topological edges during pauses, goal reach events, or explicit user/chat consolidation requests. |
| Friston et al. (2024–2026 Active Inference); Schmidhuber (Curiosity, 1991–2020) | Free Energy Principle & Epistemic Value: Action selection balances pragmatic exploitation (maximizing reward) with epistemic exploration (reducing transition entropy / visiting unobserved states). | Integrated epistemic uncertainty bonuses $\mathcal{U}(u, a) \propto 1 / \sqrt{N(u, a) + 1}$ into `sr_steering_action`, preventing dead-end entrapment and driving frontier discovery. |

---

## Mathematical Formulation

### 1. Successor Representation (SR)
Given the experienced topological graph $G = (V, E)$ with nodes $V = [v_1, \dots, v_n]$ and transition counts $C(u, a, v)$:
1. For each node $u \in V$ with active actions $\mathcal{A}(u)$:
   $$T_{u, v} = \frac{1}{|\mathcal{A}(u)|} \sum_{a \in \mathcal{A}(u)} \frac{C(u, a, v)}{\sum_{v'} C(u, a, v')}$$
   If node $u$ has no outgoing transitions, $T_{u, u} = 1.0$ (absorbing state).
2. The Successor Representation matrix $M \in \mathbb{R}^{n \times n}$ with temporal discount $\gamma \in [0, 0.99]$ satisfies:
   $$M = (I - \gamma T)^{-1} = \sum_{k=0}^\infty \gamma^k T^k$$
   where $M(u, v)$ denotes the expected discounted occupancy of node $v$ starting from node $u$.

### 2. Action Values & Epistemic Uncertainty
When the agent at location $u$ seeks goal $g$, the Successor Representation action value for action $a$ is:
$$Q_{\text{SR}}(u, a, g) = \sum_{v \in V} P(v \mid u, a) M(v, g)$$
The epistemic uncertainty bonus is:
$$\mathcal{U}(u, a) = \frac{1}{\sqrt{N(u, a) + 1}}$$
For candidate unvisited actions leading to frontier cells, an optimistic prior bonus is applied.

The controller selects:
$$\text{score}(a) = \text{logit}(a) + w_{\text{SR}} \cdot Q_{\text{SR}}(u, a, g) + w_{\text{epistemic}} \cdot \mathcal{U}(u, a)$$
subject to deterministic adjacent wall masking ($a \in \text{legal}$).

### 3. Graph Laplacian Harmonics (Grid-Cell Basis)
The structural graph possesses symmetric adjacency matrix $A$ and degree matrix $D$. The normalized Laplacian is:
$$L = I - D^{-1/2} A D^{-1/2}$$
Eigen-decomposition $L \mathbf{\phi}_k = \lambda_k \mathbf{\phi}_k$ yields orthogonal spatial harmonics:
- $\lambda_1 \approx 0$: Trivial uniform mode.
- $\lambda_2$: Fiedler vector (primary spatial axis).
- $\lambda_{k > 2}$: Higher-frequency harmonic standing waves across chambers and corridors.

### 4. Sharp-Wave Ripple (SWR) Replay
During pauses, goal triggers, or user requests, prioritized reverse sweeps propagate values:
$$V(u) \leftarrow \max_a \left[ R(u, a) + \gamma \sum_{v} P(v \mid u, a) V(v) \right]$$
logging convergence delta $\Delta V$ and replayed edge sequences into telemetry.

---

## Architectural Commitments & Invariants

1. **Exact Backward Compatibility**:
   - `MODES = ('policy', 'masked_policy', 'observed_map')` remains invariant.
   - `ALL_MODES = ('policy', 'masked_policy', 'observed_map', 'cx_steer', 'sr_steer')` extends controller dispatch cleanly.
   - Checkpoint state dictionaries for schemas v1, v2, v3, and v4 load without data loss or key errors.
2. **Deterministic CPU Bounds**:
   - Successor matrix inversion and Laplacian eigen-decomposition run in $< 20$ ms on CPU for all valid arena sizes ($N \le 961$).
   - Matrix operations use PyTorch `float64` for numerical stability and zero external non-core dependencies.
3. **Truthful Scope**:
   - The cognitive map operates strictly on experienced transitions, without privileged access to unseen labyrinth geometry.
