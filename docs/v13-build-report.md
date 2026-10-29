# NEUROGRAFT v0.13: empirical replay learning and matched experiments

28 October 2026. Local development on `codex/replay-learning-lab`; no checkpoint promotion or remote publication performed by this upgrade.

## What changed and why

The v0.12 replay path wrote diagnostic values that no navigation controller consumed. Its SR controller rebuilt a dense all-pairs matrix on every action. The new opt-in `replay_steer` controller consumes a persistent goal-conditioned value vector learned from executed transition counts. Each decision permits 32 state backups. An explicit Replay memory action applies another bounded batch, without stepping the world or changing neural weights. Legal actions are ranked by empirical value; raw neural logits break ties. Unknown goals use the masked neural proposal.

The implementation uses deterministic residual-prioritized Bellman backups with predecessor propagation, terminal arrival reward and stochastic outcome expectations. It permits downward corrections after revised transition evidence. It never manufactures reverse or neighboring edges. Goal changes invalidate the old value vector on the next replay; read-only diagnostics mark old goal/model summaries stale. Values, goal, cursor and backup counter survive checkpoint restoration; legacy navigation checkpoints initialize an empty planner. World resizing clears the planner. Memory-free and erased modes bypass this controller's memory use.

This is empirical model-based tabular learning. Neural weights continue to use the existing online supervised imitation objective; this upgrade does not establish stronger neural generalization. The old SR and SWR-inspired diagnostics remain available for comparison, with their role described explicitly.

## Research used

The direction draws on [He, Daigle and Bashivan's ICLR 2026 episodic spatial world model](https://proceedings.iclr.cc/paper_files/paper/2026/hash/23c2c77c921111ec8902ac482d92b289-Abstract-Conference.html), [Yu et al.'s 2026 goal-directed theta sweeps](https://www.nature.com/articles/s41593-026-02365-2), [Antonov and Dayan's 2025 exploration/replay model](https://www.nature.com/articles/s41467-025-56731-y), and [Bakermans et al.'s 2025 constructive replay](https://www.nature.com/articles/s41593-025-01908-3). These motivate separate experience, replay and goal-change assays. Our implementation does not predict unseen transitions, implement Bayesian belief-space exploration, reproduce compositional neural representations, or establish biological equivalence. Residual prioritization is an established engineering approach; it is distinct from [Mattar and Daw's need-times-gain theory](https://www.nature.com/articles/s41593-018-0232-z).

## Reproducible experiment

The [standalone study report](../reports/runs/replay-learning-20261028/report.html) and [full machine-readable evidence](../reports/runs/replay-learning-20261028/report.json) contain eight independent seeded synthetic grid/branch instances. All schedules share each graph's executed transition history, including stochastic collision self-loops and directed support. A 35% shuffled history prefix and full history are evaluated separately, at backup caps 0, 4, 16 and 64. Fresh-goal and relocated-goal phases begin from shared cold or warm continuation states. Curves use a separate clone at each cap, rather than cumulative budget consumption.

The oracle is an independent synchronous value-iteration implementation on the same observed count model. Agreement is averaged per graph over states with a unique best reference action; it is not a physical goal-reaching rate. Reference residuals are below 5.91e-14.

| Full exposure, 64-backup cap | Mean value MAE | Mean decisive-action agreement |
|---|---:|---:|
| Fresh goal, prioritized | 0.087229 | 97.92% |
| Fresh goal, cyclic uniform | 0.340128 | 63.91% |
| Fresh goal, replay disabled | 0.568212 | 17.81% |
| Relocated goal, prioritized | 0.070235 | 98.44% |
| Relocated goal, cyclic uniform | 0.121769 | 94.17% |

At these full-exposure caps, both active schedules actually execute 64 backups. The partial relocated prioritized condition averages 56.5 because one instance converges early. Zero-budget schedules match; disabled replay remains unchanged across caps; unknown-goal controls remain zero. Directly observed arrival at the goal can have nonzero action value even with replay disabled: this comes from the shared immediate transition model.

The partial prefixes happen to include every location, with fewer observed state-actions/counts. They do not establish unseen-location transfer. Each exposure has a different empirical oracle, so lower error under partial experience does not establish an exposure benefit. Several finite-budget cases remain unconverged, and one partial relocated instance ties cyclic replay. Coordinate-ordered cyclic replay is sensitive to goal location and ordering. Eight selected graph instances are a local exploratory pilot, not a confirmatory generalization study.

Hashes check corruption, not authorship. Plain verification checks hashes, the declared protocol, independently derived metrics and HTML rendering. `--replay` additionally recomputes every deterministic schedule and continuation result. The report SHA256 is `33b1de2f9c0fe9cb48958edd178570ac31e3592b715e8ad01c1ff60840a97aa5`.

```powershell
python -m neurograft study-replay --out reports/runs/replay-learning-new
python -m neurograft verify-replay-study reports/runs/replay-learning-new --replay
```

The command refuses to overwrite an existing directory and never opens a live checkpoint. The existing neural/frozen-navigation evaluation remains a separate experiment; it does not silently include this empirical replay assay.

## End-to-end controller costs

[Timing receipt](../reports/v13-replay-latency.json), generated with [the reproducible benchmark](../tools/benchmark_replay.py), includes graph validation/copies, priority construction, residual scans and action selection. These costs are in addition to the counted state backups.

| Experienced nodes | Legacy SR controller, raw range (ms) | Replay32 + action, raw range (ms) |
|---|---:|---:|
| 64 | 43.65–57.70 (3 samples) | 3.53–5.63 (3 samples) |
| 256 | 674.89–688.19 (3 samples) | 17.27–18.62 (3 samples) |
| 961 | 10,383.79 (1 sample) | 65.42–142.93 (3 samples) |

These are small descriptive measurements on one CPU runtime, not equal-quality speedup estimates. The larger replay cases have not propagated the distant goal to the start after 96 backups and use the legal-logit fallback. Neural GraphCircuit inference and full live tick cost are excluded. Equal backup caps do not imply equal wall time. Historical v0.12 inversion timing claims do not bound the complete controller call.

## Verification and preserved state

- Initial baseline: 744 tests passed in 436.49 seconds.
- New planner: 90 targeted analytical, correction, persistence and malformed-input tests passed.
- Matched study: 14 targeted tests passed, including coordinated resealing/tampering rejection under strict replay.
- Live integration tests cover behavioral action changes, legal masking, read purity, memory bypass, legacy loading, invalid-state atomicity, exact next-step continuation and stale diagnostics.
- The final browser regression passed 26 checks with no page errors, Three.js/WebGL rendering, desktop/mobile layout, replay controls, zero teacher mixing, unchanged neural weights/steps during manual replay and named-checkpoint restoration. Screenshots and receipt: `reports/browser-v13/`.
- Independent review checked 20 fresh/revised empirical models against a separate synchronous solver; maximum value error was 5.03e-12 and graph state was unchanged. The review's stale-diagnostics finding is recorded in the implementation ledger.
- Final full regression: **855 passed in 549.40 seconds**; [saved output](../reports/v13-tests.txt). No failures or skips were reported.
- Final wheel: `dist/neurograft-0.13.0-py3-none-any.whl`, SHA256 `cf350ead8a4edcf260759aa429fa83c148b19cbebc91d2c8d5b5a5e7eb2e432c`. All 44 package Python/web files match the source byte for byte, with no missing files; isolated extracted-wheel import reports version 0.13.0. [Package receipt](../reports/v13-package-verification.json).
- Initial package verification caught cached CRLF/LF byte differences in existing build output. `python -c "from setuptools import setup; setup(script_args=['build_py','--force'])"` forced a fresh source copy before rebuilding; final parity passes.
- Independent final review approved the stale-diagnostics fix, including paused goal changes, post-action new experience, and pure snapshot reads. JavaScript syntax and `git diff --check` pass.
- [Combined validation receipt](../reports/v13-verification.json) ties together current source hashes, test output, browser checks, package and study seals.

The real existing checkpoint loaded read-only with 145,988 steps, 145,981 updates, `sr_steer` mode and restored auxiliary state. The new planner begins empty on that legacy checkpoint. No ticks or save were performed. Its before/after SHA256 is `8a3630eaebd3242ee43bca3e4bd0375c330592645fcd87bbce4413f8d813f5f2`; [receipt](../reports/v13-legacy-load.json). Existing untracked `research-review/` content is preserved.

The historical v11 predictive-recovery archive currently fails its existing verifier with `study artifact hash mismatch: report.html`. Those files were retained untouched. This upgrade does not reseal that archive or treat its displayed aggregate as newly verified evidence.
