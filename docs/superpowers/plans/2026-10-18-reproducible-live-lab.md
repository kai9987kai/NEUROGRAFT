# Reproducible Live Lab Implementation Plan

> Agent execution: independent implementation scopes, followed by integration and fresh review.

**Goal:** Make live adaptation repeatable and add measured frozen navigation trials.

**Architecture:** One pure evaluation module clones the existing GraphCircuit.
CLI and HTTP share its validation and reporting. Existing live state gets a
versioned auxiliary checkpoint extension, while diagnostics become observational.

**Tech stack:** Python, PyTorch CPU, standard-library HTTP, existing Three.js.

**Spec:** ../specs/2026-10-18-reproducible-live-lab-design.md

## Constraints and review focus

- Preserve data/live/session.pt, acquired graphs, previous reports and dist files.
- No new dependencies, external service writes, publication, merge or process termination.
- Distinguish inference lesions from trained baselines and live accuracy from frozen trials.
- Reject invalid or oversized protocols before cloning models or allocating worlds.
- Restore synced anchors and every diagnostic's state; observe without changing future state.
- Handle concurrent evaluation and continued training with an immutable report identity.

## Tasks

- [x] Add tests for evaluator bounds, oracle ceiling, deterministic matched episodes,
  state/RNG preservation, and exclusive CLI exports; observe missing-feature failures.
- [x] Implement navigation_evaluation.py, CLI evaluate-live, preset persistence and
  targeted regression tests. Keep model-copy time inside the lock and rollout outside.
- [x] Repair and test complete checkpoint continuation and pure same-weight diagnostics
  in live.py/live_state.py (learning agent exclusive scope).
- [x] Repair web syntax and topology controls; add HTTP evaluation, health state and
  browser table/export, tested in test_server.py (product agent exclusive scope).
- [x] Integrate, run the full suite and browser smoke, review changes independently,
  repair material findings, generate a bounded receipt, build/inspect version 0.5.
- [x] Update README and build report with verified evidence and limitations.

Execution proceeds under the user's broad improvement authorization. The current
checkout is reused so its existing source data and local session remain accessible.
