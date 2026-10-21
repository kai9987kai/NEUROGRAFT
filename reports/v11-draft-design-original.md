# v0.11: Connectome Predictive World Models, Central Complex PFL Steering & Topological Cognitive Memory

21 October 2026. Architectural design and empirical specification.

## Research reviewed (2024–2026)

| Primary source | Biological / Algorithmic Principle | NEUROGRAFT Integration & Scope Boundary |
| --- | --- | --- |
| Currier et al., *Drosophila Connectome PFL Steering Circuits*, Nature 2024; Matheson et al., 2022 | Central Complex (CX) PFL1/PFL2/PFL3 neurons transform allocentric heading (from E-PG ring attractors) and fan-shaped body goal vectors into egocentric turning commands and forward drive. | Implemented `CXPFLSteering` in `neurograft/compass.py` and `cx_steering_action` in `neurograft/navigation_controller.py`. Operates strictly on local observations and ring-attractor headings without privileged global maps. |
| Hafner et al., *Mastering Diverse Domains through World Models (DreamerV3)*, 2024; Ha & Schmidhuber 2018 | Internal forward models predicting sensory transitions $o_{t+1} \sim p(o_{t+1} \mid s_t, a_t)$ provide grounded counterfactual simulation and predictive representations. | Added zero-initialized `predictor` in `GraphCircuit` mapping $(c_t, a_t) \to \hat{o}_{t+1}$. Integrated into multi-task training loss $\mathcal{L} = \mathcal{L}_{\text{policy}} + \lambda_{\text{pred}} \text{MSE}(\hat{o}_{t+1}, o_{t+1})$. |
| Laskey et al., *DART: Noise Injection for Robust Imitation Learning*, CoRL 2017; Ross et al., *DAgger*, AISTATS 2011 | Supervised policies suffer from compounding errors when deviating off-distribution. Deliberate off-policy perturbation with expert correction trains recovery behaviors. | Added `collection_policy='recovery'` to `CurriculumConfig` and `navigation_training.py`. Interleaves policy proposals and teacher recovery labels, logging explicit learner vs. teacher transition steps. |
| Tolman, *Cognitive Maps in Rats and Men*, 1948; Dayan, *Improving Generalization with Successor Representations*, 1993 | Spatial cognition constructs an episodic topological transition graph $G = (V, E)$ rather than Euclidean coordinate grids alone. | Implemented `TopologicalMemoryGraph` in `neurograft/memory.py` as Tier 3 cognitive map, tracking transition probabilities $P(v \mid u, a)$, BFS topological geodesics, and reachability ratios. |

---

## Architectural Mechanisms

### 1. Drosophila Central Complex (CX) Vector Steering Engine
The insect central complex executes vector-based spatial steering through three distinct PFL cell classes:
1. **PFL1 Neurons**: Compute heading-goal vector alignment $\cos(\theta_{\text{action}} - \theta_{\text{goal}})$.
2. **PFL2 Neurons**: Modulate global steering gain based on angular deviation between heading and goal:
   $$\text{gain}_{\text{PFL2}} = 1 + \gamma \frac{|\theta_{\text{goal}} - \theta_{\text{heading}}|}{\pi}$$
3. **PFL3 Neurons**: Drive asymmetric turning commands by comparing action direction with current heading $\cos(\theta_{\text{action}} - \theta_{\text{heading}})$.

Together with deterministic local wall masking, `cx_steering_action` computes:
$$\text{score}(a) = \text{logit}(a) + w_{\text{steer}} \cdot \text{gain}_{\text{PFL2}} \cdot \left[0.65 \cos(\theta_a - \theta_g) + 0.35 \cos(\theta_a - \theta_h)\right]$$
for each non-blocked cardinal action $a \in \{0, 1, 2, 3\}$.

### 2. Connectome Predictive World Model
The `GraphCircuit` now incorporates an optional transition predictor:
$$\hat{o}_{t+1} = \text{Predictor}(c_t, \text{one\_hot}(a_t))$$
where $c_t$ is the concatenated 4-source pooled context vector.
- **Zero-Initialization Invariance**: The final projection layer of `self.predictor` is initialized to exact zeros ($\mathbf{W} = \mathbf{0}, \mathbf{b} = \mathbf{0}$), ensuring that baseline logits, initial forward dynamics, and common model hashes remain bit-identical when predictive features are introduced.
- **Multi-Task Objective**: During training with `predictive_head=True` and $\lambda_{\text{pred}} > 0$:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{a}_t, a_t^*) + \lambda_{\text{pred}} \frac{1}{4 D} \sum_{i=1}^4 \sum_{j=1}^D (\hat{o}_{t+1, ij} - o_{t+1, ij})^2$$

### 3. Topological Cognitive Memory Graph
Tier 3 spatial memory is augmented with an online directed graph $G = (V, E)$ where nodes represent visited coordinate tuples $(x, z)$ and edges represent experienced transitions:
- **Empirical Transition Probabilities**:
  $$P(v \mid u, a) = \frac{N(u, a, v)}{\sum_{v'} N(u, a, v')}$$
- **Topological Shortest Paths**: Breadth-First Search (BFS) computes geodesic node sequences without requiring a full or privileged obstacle map.
- **Reachability Ratio**: Computes the fraction of known nodes reachable from the current position, detecting spatial entrapment.

### 4. Recovery-Augmented DAgger Curriculum
In `collection_policy='recovery'`, trajectories are sampled under a mixed policy regime:
- The learner executes its current policy with probability 0.5, allowing realistic drift and dead-end entry.
- The oracle step is executed with probability 0.5, demonstrating immediate recovery actions from displaced positions.
- All visited steps record the optimal expert action $a^*$ as the training target, directly mitigating the covariate shift that previously led to 99.2% collision rates on raw policies.

---

## Compatibility & Integrity Commitments

1. **State Preservation**: Existing live session `data/live/session.pt` (schema v1, v2, v3) continues to deserialize and execute identically without mutation.
2. **Sample Hash Invariance**: For legacy training configurations (`predictive_head=False`, `collection_policy in ('mixed', 'expert')`), sample dictionaries do not include auxiliary keys, maintaining bit-for-bit SHA-256 hash compatibility with all historic verification receipts.
3. **Deterministic CPU Bounds**: All computations operate in strictly bounded $O(N + E)$ time without GPU requirements or dynamic memory growth.
