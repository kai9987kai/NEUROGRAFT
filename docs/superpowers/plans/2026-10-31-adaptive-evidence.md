# Adaptive evidence implementation plan

> **For agentic workers:** Use focused parallel implementers for the independent study and UI, with root owning engine/integration and fresh whole-change review. Steps use checkbox syntax for tracking.

**Goal:** Add inspectable causal adaptive memory, a controlled streaming study, and backward-compatible interactive tools.

**Architecture:** Keep v1 engine/studies byte-compatible. Add a detached integer-count adaptive selector and v2 workbench subclass. Integrate into the existing local HTTP/CLI/package paths.

**Tech Stack:** Python 3.11+, existing PyTorch dependency, pytest, existing offline HTML/JS and Playwright verification.

**Spec:** `docs/superpowers/specs/2026-10-31-adaptive-evidence-design.md`

## Global constraints

- No new runtime dependencies; no checkpoint mutation/promotion or remote publication.
- Integer observed transition counts, 256 nodes / 2048 events / windows 4,8,16.
- Fixed prequential rule: decay .95, threshold log(4), minimum four fresh outcomes, Beta(.5,.5) scoring only.
- v1 receipt and historical strict replay semantics unchanged.

## Review focus

- Batched observations must score one event at a time before updates.
- Imported legacy receipts must restore exactly and remain usable in the current UI.
- Unknown actions and boundary collisions must not fabricate planning edges.
- Hidden truth and phase labels must never enter adaptation; identical prefixes imply identical behavior.
- Resealed metric tampering must fail strict deterministic replay.

### Task 1: Adaptive engine and v2 workbench (root)

Files: add `neurograft/adaptive_memory.py`, `neurograft/adaptive_workbench.py`, `tests/test_adaptive_memory.py`, `tests/test_adaptive_workbench.py`; modify `neurograft/workbench_service.py`.

Interfaces: `AdaptiveTransitionMemory(graph, window=8)`, `.graph`, `predict(source,action)`, `observe(source,action,destination)`, `diagnostics()`; `AdaptiveReplayWorkbench` extends `ReplayWorkbench` with VERSION=2 and models cumulative/recent/adaptive.

- [x] Write analytical pre-update scoring, local selection, invalid-input atomicity, cap, matched-prefix and reversal tests; run and confirm missing feature failure.
- [x] Implement fixed rule with detached diagnostics and integer graph counts.
- [x] Write v2 receipt/batch/continuation and v1 service-import tests; run red, implement subclass and dispatch, run green.

### Task 2: Controlled study (parallel implementer)

Files: add `neurograft/adaptation_study.py`, `tests/test_adaptation_study.py`; root integrates CLI.

Consumes Task 1 interfaces. Produces `run_adaptation_study(out, seeds=DEFAULT_SEEDS, window=8, budget=64)` and `verify_adaptation_study(out,replay=False)`.

- [x] Add failing study/protocol/causality/replay/tampering tests, then implement declared stream generation, scoring, independent reference metrics, complete seed-level evidence, HTML and seals.
- [x] Verify tests and provide exact commands, schema and measured limitations.

### Task 3: UI and evidence narrative (parallel implementer)

Files: update `neurograft/web/workbench.{html,css,js}`, `tools/verify_workbench_ui.cjs`, and bounded v11 narrative correction in `docs/v11-build-report.md`.

Consumes v2 snapshot `adaptive_evidence` and models; legacy snapshot lacks them.

- [x] Extend existing browser verification with adaptive selection/diagnostics and v1 import; confirm red before UI implementation.
- [x] Render three policies, causal score/selection explanations, 9-arm wording, legacy control disablement, desktop/mobile behavior; retain existing controls.
- [x] Correct v11 comparison with kinematic baseline using existing archived numbers; do not edit sealed artifacts.

### Task 4: Integration, research evidence and verification (root + fresh reviewer)

Files: CLI, version, README, `docs/v15-build-report.md`, reports/verification receipts.

- [x] Add CLI test red, wire study and verify commands, bump to 0.15.0 consistently.
- [x] Run eight-seed pilot and strict verification; record every result without tuning.
- [x] Request fresh review of algorithm, causal study, compatibility and integration; address actionable findings with regressions.
- [x] Run full suite, browser tools, old strict replay, wheel/source checks and checkpoint hash comparison. Summarize outcome and limitations in build report and final answer.

## Execution ledger

- Initial state: branch `codex/replay-learning-lab`, HEAD `9bd1fc8`; existing dirty checkpoint and untracked historical research/study artifacts preserved.
- User opted out of design questions and authorized choosing and implementing the direction. Reuse current feature checkout; no worktree or dependency setup needed.
- Task 1: engine and v2 integration implemented; 74 focused adaptive/legacy workbench/API tests passed. Added count diagnostics and maximum-counter numerical regression afterward; final regression pending.
- Ruling: a reversal test initially assumed six collisions must lead to a cumulative selection during a run of successes. Direct score tracing disproved that expectation: the recent expert can remain predictively better throughout that stream. Replaced the fixture with four collisions, whose declared rule analytically reverses on the next success. Algorithm parameters are unchanged; sticky selection remains a measured limitation.
- Task 2: study implementer added explicit pre-execution protocol declaration and test. No pilot results have been used to adjust parameters.
- Task 2 complete: 19 study/CLI tests passed. Eight-seed pilot and strict replay passed; report SHA256 `19ce05976ce4abaeeac4bcf8ffc408284db638f88cb411b020e43fd66820a06f`. Algorithm parameters unchanged. Negative sparse/adverse and guarded-recent comparisons are retained.
- Task 3 complete: final capture-capable workbench browser run passed 26 checks plus the separate 26-check live browser regression, with unchanged source-session digest and no unexpected page errors. Legacy receipts, exact score continuation, averages, mobile layouts and causal controls were exercised.
- Final independent review found no material defects; added pair count visibility and an explicit maximum-count finite log-loss regression. Corrected README wording to distinguish proper forecast losses from the heuristic adaptive selection score.
- Final packaging includes 53 exact source files and isolated v1/v2 imports. Refreshed the main live UI version label to 0.15; no behavior changed after the full-suite run began.
- Task 4 complete: full regression 969 passed in 589.04s; 52 browser checks; new and historical strict replays; source/wheel/README metadata parity; original checkpoint hash unchanged. Final receipt `reports/v15-verification.json`; wheel SHA256 `5bb4f73c94ace7f34671959561a9e46012db9c23edb8f4ba63da3384fb4de373`. Isolated fixture workbench running on port 8770, PID 8400. No commit/push/promotion performed.
