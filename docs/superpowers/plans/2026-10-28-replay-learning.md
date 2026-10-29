# Replay Learning Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task by task. Track completion below.

**Goal:** Connect bounded empirical replay learning to navigation and reproducible matched experiments.

**Architecture:** A standalone sparse goal-conditioned value planner consumes experienced transitions. Live integration persists its continuation state; a separate study compares replay schedules against independently converged reference values.

**Tech Stack:** Existing Python 3.11+, PyTorch, pytest, vanilla JS and Three.js; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-28-replay-learning-design.md`

## Global Constraints

- Preserve live checkpoint and existing study artifacts; no promotion or publication.
- At most 32 backups per live decision, 4096 per explicit planner call; finite built-in numeric inputs and bounded state.
- Keep legacy MODES contract; add replay_steer to ALL_MODES only.
- State reads never learn, consume RNG, or update the planner.
- Existing dirty checkout reused on codex/replay-learning-lab to preserve local user context.

## Review Focus

- A collision outcome must contribute its measured probability, not a fabricated neighboring edge.
- Goal relocation and weaker transition evidence must remove stale values.
- Disabled and zero-budget controls must perform no state backups.
- Save/load must reproduce the next action and values exactly, including uniform cursor.
- Study verification must reject content tampering and reproduce all deterministic results.

### Task 1: Planner

- [x] Write failing analytical and bounded-input tests in tests/test_replay.py.
- [x] Implement ReplayPlanner in neurograft/replay.py: replay, action_values, to_state/from_state, snapshot.
- [x] Verify targeted tests and review the diff.

### Task 2: Matched experiment and CLI

- [x] Write failing tests in tests/test_replay_study.py.
- [x] Implement replay_study.py with fixed seeded instances, independent reference, schedules, budget curve, goal relocation, report and strict recomputation.
- [x] Expose CLI study-replay and verify-replay-study. Run a local study and exact verification.

### Task 3: Live integration and interface

- [x] Write failing tests for behavior, persistence, lesions and service controls.
- [x] Integrate replay_steer, bounded manual replay and optional checkpoint state.
- [x] Add controller option, diagnostics and real browser assertions.

### Task 4: Validation and documentation

- [x] Independent review and fixes, full pytest, browser checks, wheel/source parity and checkpoint hash check.
- [x] Refresh version, README and citations; write research/results report with limitations and review ledger.

## Decision and verification ledger

- Initial inspection: main checkout contains a modified live checkpoint and untracked research-review/; no Python training process found. Reuse checkout on a new codex branch.
- Ruling: proceed with a concrete research-backed implementation under the user's broad authorization and explicit learning/experiments priority. Routine design choices remain reversible and recorded here.
- Baseline pytest started before implementation (session 21337).

- Task 1: planner90 targeted checks passed; independent20-model review worst error5.03e-12; experienced graph remains immutable.
- Task 2: study14 checks passed; eight-seed pilot report and strict deterministic replay verified; results and null/control limitations in docs/v13-build-report.md.
- Task 3: initial integration6 checks passed. Final review found stale goal/model diagnostics; regression failed before fix. One fix wave added pure stale flags and historical-summary text; integration7 checks pass and scoped re-review approved.
- Browser final:26 checks pass, zero page errors, WebGL renderer, learned values actually backed up on known goal, pure manual replay, exact named-checkpoint restoration and mobile/desktop layout.
- Packaging: initial byte parity check detected cached CRLF/LF differences. Forced build_py recopy corrected all44 package files; final wheel SHA256 cf350ead8a4edcf260759aa429fa83c148b19cbebc91d2c8d5b5a5e7eb2e432c, isolated import0.13.0, byte parity clean.
- Real existing checkpoint loads read-only at145988 steps; before/after digest8a3630eaebd3242ee43bca3e4bd0375c330592645fcd87bbce4413f8d813f5f2. No live continuation or promotion performed.
- Baseline744 tests passed; pre-final-diagnostic-fix854 tests passed. Final fresh full suite including the stale-diagnostic regression:855 passed in549.40s, saved to reports/v13-tests.txt.
- Leave the implementation local on codex/replay-learning-lab; remote publication is outside this request. Existing dirty checkpoint and research-review/ are retained.

- Final whole-change review and scoped fix re-review approved; final source/wheel parity44files and original checkpoint hash rechecked. All implementation-plan tasks complete locally.
