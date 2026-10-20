# NEUROGRAFT v0.8 upgrade

User intent: advance the existing local research instrument with useful, tested improvements. Keep the active checkpoint and prior experiment evidence intact.

This is a bounded upgrade to existing curriculum, experiment export and service startup flows. The main mechanism is deterministic action-stratified reservoir replay: a finite training memory that retains earlier expert-labelled learner visits instead of retaining only the tail of the latest round. Cartesian scenario scheduling fixes correlated size/topology selection. This is inspired by dataset aggregation, with finite replay rather than the original full-data algorithm: https://proceedings.mlr.press/v15/ross11a.html.

- [x] Curriculum: first reproduce missing Cartesian coverage, loss of older replay samples and mismatched evaluation limits; implement private-RNG stratified reservoirs, explicit compute bounds and per-round receipts; test reproducibility and checkpoint isolation.
- [x] Evidence: archive the exact baseline and graph used, configuration, progress and raw results beside the selected candidate. Seal completed exports with file hashes. Add `verify-navigation RUN [--replay]` for integrity/identity validation and optional exact frozen trial replay. An integrity receipt is not an authenticity signature or proof of training provenance.
- [x] Startup: bind before starting the background worker; preserve restored run/learning flags; clean up on failures without saving or advancing the existing checkpoint. Test occupied port and startup failures.
- [x] Presentation: expose coverage, replay and loss in the standalone experiment report; clarify environmental observers in the live UI and chat. The current plume, substrate, gates and circadian calculations are deterministic visual/diagnostic approximations. They do not change collision physics or feed the navigation policy.
- [x] Verification: run focused tests, the complete Python suite, isolated desktop/mobile browser checks with screenshot inspection, a small reproducible curriculum experiment, evidence verification/replay and wheel/source parity. Request an independent review of the final diff.

No candidate promotion is part of this upgrade. Candidate checkpoints remain paused inference exports with fresh live optimizer state, not resumable curriculum runs. Test seeds are evaluated after validation selection. Small synthetic results support implementation verification, not a biological or general navigation capability claim.
