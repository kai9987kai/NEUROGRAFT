# Goal-conditioned replay learning laboratory

28 October 2026. Direction chosen by Codex under the request to advance the whole project; the user prioritizes learning and experiments. Keep the existing Python/PyTorch/browser architecture.

## Problem and approach

The v0.12 replay routine writes values that navigation never reads. Its dense successor controller rebuilds every matrix entry on every decision. The new experiment must connect empirical experience, offline value updates, and observable decisions, with matched controls and reproducible artifacts.

Alternatives considered: more neural heads would add another unconfirmed mechanism; optimizing the dense SR alone would improve latency without testing replay; bounded sparse goal-conditioned planning directly closes the behavior/evidence gap. Choose the latter and retain existing controllers as comparison surfaces.

## Mechanism

Add `ReplayPlanner` in `neurograft/replay.py`. It consumes only the empirical `TopologicalMemoryGraph`, a public goal coordinate, and a backup budget. It keeps a single goal-conditioned state-value vector. For observed `(u,a)` use expected backup `Q(u,a)=sum_v P(v|u,a)[1(v=g)+gamma*1(v!=g)*V(v)]`. The goal is terminal with V(g)=0. Unobserved actions have value zero, without invented transitions. V(u)=max_a Q(u,a). Goal switches clear the vector; graph changes permit both increases and decreases. Reconstruct residual priorities each replay invocation, with deterministic predecessor propagation and coordinate tie breaks. Alternative schedules are cyclic uniform replay and disabled replay. Count state backups and report the remaining Bellman residual. A fixed budget is a bound, not proof of convergence.

Expose an opt-in `replay_steer` controller. Each decision performs at most 32 state backups, then ranks legal actions by value with raw logits as ties. If all values are zero it falls back to the masked policy. Learning remains empirical model-based tabular value learning; neural weights still use the existing supervised imitation objective. Do not label this an implementation of biological ripples or Bayesian exploration.

## Persistence and user controls

Persist planner goal, values, uniform cursor, and backup counter in an optional navigation checkpoint field; older checkpoints load with an empty planner. All new state is strictly bounded and validated before installation. Read-only snapshots never replay. World resets and disabling memory clear or bypass planning. Replay controls in the new mode update the same planner that actions read. Existing v0.12 replay remains available and is explicitly identified as a diagnostic. Add browser controller selection and compact value-learning diagnostics.

## Experiments

Add a local CPU replay study with independent seeded synthetic graph instances, shared experience across schedules, budget curves, goal relocation, branching and stochastic collision outcomes. Compare prioritized, cyclic uniform and disabled replay; compute an independent converged value-iteration reference for value error and decision agreement. Include directed graphs, unknown-goal controls, and deterministic traces. Hash canonical report content and independently recompute on verification. Timings are descriptive and excluded from deterministic content. No live checkpoint is read or overwritten by studies; do not promote weights. Report null results and schedule costs honestly.

## Research and scope

- Antonov and Dayan, *Exploring replay*, Nature Communications, 15 February 2025: https://www.nature.com/articles/s41467-025-56731-y. Motivates investigating replay during learning and uncertainty; our count model is not their Bayesian belief-space method.
- Bakermans et al., *Constructing future behavior in the hippocampal formation through composition and replay*, Nature Neuroscience, 2025: https://www.nature.com/articles/s41593-025-01908-3. Motivates behavior and structural/reward transfer assays; we do not reproduce compositional building blocks.
- *Hippocampal theta sweeps indicate goal direction during navigation*, Nature Neuroscience, 2026: https://www.nature.com/articles/s41593-026-02365-2. Motivates separating goal-directed decisions from generic replay telemetry; no biological equivalence claimed.
- Mattar and Daw, 2018: https://www.nature.com/articles/s41593-018-0232-z. Need-times-gain prioritization is distinct from our engineering residual-priority queue.

## Validation and boundaries

Test analytical chains, branches, stochastic expectations, goal relocation, downward corrections, strict malformed-state rejection, deterministic continuation, budget bounds, and graph immutability. Run full pytest, real browser controls and responsive rendering, wheel/source parity, strict study replay and git diff checks. Preserve the initial dirty `data/live/session.pt` SHA256 `8a3630eaebd3242ee43bca3e4bd0375c330592645fcd87bbce4413f8d813f5f2` and existing `research-review/` contents. Historical v11 archive currently fails its report.html hash check; retain it untouched and disclose this evidence limitation.
