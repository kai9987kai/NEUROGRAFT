# NEUROGRAFT 0.6: navigation curriculum & candidate evaluation

18 October 2026. Local implementation, validation, and candidate curriculum benchmarking.

## Summary

Following the measured limitations in the v0.5 captured policy (0/8 unassisted goals, 99.22% collisions), NEUROGRAFT v0.6 introduces an isolated navigation curriculum training instrument, non-mutating replay/routing diagnostics, and self-contained HTML candidate evidence reporting.

The live session, captured baseline (`reports/runs/navigation-v05-20261018/frozen-input.pt`), and global RNGs remain untouched. Candidate policies are trained in separate experiment directories, selected solely by held-out validation performance, and evaluated on frozen test seeds without teacher mixing or planner assistance.

## Implemented architecture

1. **Pure replay & routing diagnostics (`neurograft/navigation_diagnostics.py`)**:
   - Computes replay action distribution, goal-progress feature bounds, and deterministic 32-sample window interpolation without advancing RNGs or mutating live sessions.
   - Evaluates cross-source human module routing mass and policy logit distributions under `torch.no_grad()`.

2. **Curriculum training & candidate export (`neurograft/navigation_training.py`)**:
   - `CurriculumConfig` with complete parameter validation: bounds on rounds, worlds, updates, sizes (5–31), presets, replay capacity, learning rate, and disjoint train/validation/test seed spaces.
   - Bounded progressive imitation rollouts with expert shortest-path teacher mixing across sizes and presets.
   - Action-balanced mini-batch sampling from finite replay buffers, eliminating directional collapse.
   - Randomization of lifetime goal progress counters in training rollouts to eliminate evaluation distribution shift.
   - Cross-source routing warmup allowing 3 cross-source senders (`top_k=3`) during warmup before restoring standard `top_k=2` for validation and exports.
   - Validation-only selection key `(success_rate, spl, -collision_rate)` preserving earlier ties, testing held-out test seeds strictly once on the chosen candidate.
   - Export of loadable candidate live sessions with synchronized anchor models, clean steps/updates counters, and strict `top_k=2` routing.

3. **Standalone offline reporting (`neurograft/navigation_report.py`)**:
   - Generates standalone zero-external-dependency HTML reports (`report.html`) containing side-by-side variant comparisons, validation progression curves, model/graph/protocol hashes, and embedded raw JSON download receipts.

4. **CLI integration (`neurograft/cli.py`)**:
   - Exposed `python -m neurograft train-navigation` with flexible command-line flags and JSON config loading.

## Verification evidence

- **Full test suite**: 354 passed in 119.65s (0 failures, 0 regressions across all modules).
- **Navigation test suite**: 44 passed across diagnostics, evaluation, reporting, and curriculum training.
- **Protocol isolation**: Evaluated candidate run in `reports/runs/navigation-v06-candidate/`:
  - Output directory creation refusal verified when destination exists.
  - Baseline checkpoint SHA-256 verified byte-identical before and after training (`3E143F8EB514EB59D4FB0E6F1D681C33D39CEF4D769F47A35991B941AA6057AF`).
  - Python and PyTorch RNG states verified bit-for-bit identical before and after candidate training.
  - Manifest and status files recorded with sha256 hashes of all generated artifacts.
- **Packaged runtime**: Built `neurograft-0.6.0.tar.gz` and `neurograft-0.6.0-py3-none-any.whl`.
