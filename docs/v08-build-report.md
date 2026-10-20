# NEUROGRAFT v0.8 local build report

This upgrade improves the existing curriculum, evidence exports, diagnostics and live-service lifecycle. It does not promote a candidate into the saved live workspace.

## Changes

- Curriculum scenarios traverse the Cartesian size/topology product. The prior correlated modulo schedule visited only 12 of the default 24 combinations and omitted its 11-by-11 labyrinth evaluation condition.
- Per-action Algorithm R reservoirs keep a finite sample of historical expert-labelled visits. Protected class quotas and shuffled balanced minibatches replace whole-round tail truncation. Missing action classes leave reserved slots unused; no guarantee is made that every world remains represented.
- Round receipts expose collected conditions, actions, teacher/learner decisions, lifetime/retained samples and cross-entropy loss. Configuration validation now agrees with the 16-seed evaluator and places explicit limits on collection and optimizer work.
- Runs archive exact baseline checkpoint bytes before loading, the source bundle, configuration, progress, raw report, selected inference checkpoint, standalone HTML and file manifest. `verify-navigation RUN --replay` validates identities and reruns the recorded frozen trials without the original input paths. The verifier derives protocol hashes from archived configuration and checkpoint unroll instead of trusting matching strings. Hashes are integrity checks, not authenticity signatures or proof that training happened as declared.
- Diagnostics capture replay and model together under the session lock, then run on a frozen copy outside that lock. Tests reproduce nested replay and weight changes immediately after unlocking.
- Service startup binds before learning, preserves restored run/learning flags and rolls back failures without saving an existing checkpoint. Optional mirror mutations return 503 until readiness commits. Normal disconnected-client response writes no longer produce server tracebacks; unrelated I/O errors still propagate.
- The UI identifies synthetic environmental observers, distinguishes fixture/source labels, and explains what actually drives policy actions. Environmental overlays remain visual/diagnostic approximations, without policy inputs or movement-physics effects.

The replay design is informed by [Ross, Gordon and Bagnell's dataset aggregation work](https://proceedings.mlr.press/v15/ross11a.html). Finite stratified replay is an engineering adaptation, not a replication of the original full-data algorithm. Frozen navigation metrics follow the project's existing [navigation evaluation reference](https://arxiv.org/abs/1807.06757).

## Measured smoke experiment

`reports/runs/navigation-v08-smoke-20261019/experiment/` contains a complete fixture-only experiment: four nodes per source, hidden size 8, unroll 4, two rounds of 24 worlds, 16 optimizer steps per round, 128 replay slots, four frozen test worlds and a 24-step horizon. Validation selected round 1; round 2 tied and retained the earlier checkpoint. Test results did not select the checkpoint.

| Measure | Captured baseline | Selected candidate |
|---|---:|---:|
| Test goals reached | 0/4 | 0/4 |
| SPL | 0 | 0 |
| Collision rate | 93.75% | 91.67% |
| Mean terminal distance | 4.0 | 2.5 |

The full-map oracle reached 4/4 goals with SPL 1. Each training round covered all 24 size/topology conditions. At the end, replay retained 128 of 362 collected visits, balanced at 32 per action, representing 44 worlds and both rounds. Baseline bytes were unchanged. Both archived frozen evaluations replayed exactly locally.

These small results verify the training/export pipeline and do not demonstrate useful autonomous navigation, reliable performance gains or a biological-connectome advantage. Candidate files are paused inference exports with fresh live optimizer state; curriculum resume is not implemented.

## Validation artifacts

The final integrated suite passed **389 tests in 158.74 seconds**. Desktop and mobile checks passed with zero page errors. All 40 packaged Python/static files match the source, README metadata matches, and the extracted wheel replayed both frozen evaluations exactly. Wheel SHA-256: `65e7970d534d2258da444681a1e63fb3143db26a60eaff34fe8e425c854e89a1`.

- `reports/v08-tests.txt`: final complete Python regression output.
- `output/playwright/v08-final/browser-verification.json`: isolated desktop/mobile startup, stepping, learning toggle, topology controls, evaluation, export, stale-weight notice, layout, WebGL and browser-error checks.
- `output/playwright/v08/game-client/`: bundled browser action-loop states and visually inspected canvas screenshots.
- `output/playwright/v08/curriculum-report.png`: visually inspected standalone HTML report.
- `reports/v08-package-verification.json`: wheel/source equality, package hash and exact frozen-trial replay from an isolated extraction of the wheel.
- `reports/runs/navigation-v08-smoke-20261019/verification.json`: smoke experiment metrics, retention and replay receipt.

Independent review found a protocol identity gap and a delayed mirror-start race. Both were reproduced with failing tests, repaired and reviewed again with no remaining important findings. Source compilation, browser JavaScript parsing and whitespace checks also passed.

All work is local. No commit, remote push, model promotion or publication was performed. The pre-existing dirty live checkpoint and v0.7 wheel were preserved.
