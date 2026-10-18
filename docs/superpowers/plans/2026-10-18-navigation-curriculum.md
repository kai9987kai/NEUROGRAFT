# Navigation curriculum v0.6

The v0.5 captured policy failed all eight navigation trials and collided on
99.22% of attempted moves. This change targets that measured failure without
changing the live session or claiming a biological advantage.

## Design

- Keep GraphCircuit and its four feature rows unchanged. Train an isolated fresh
  candidate by default; optionally initialize weights from the captured baseline.
- Aggregate shortest-path labels on expert and progressively learner-controlled
  trajectories across sizes, topologies and disjoint world seeds. Balance replay
  sampling by action, cap storage, and vary lifetime-progress counters in training.
- Allow all three cross-source senders during an explicit routing warmup. Every
  validation/test rollout and exported model uses the original top-two routing.
- Freeze training/validation/test seeds and budget before any training. Select
  the best round by validation success, then SPL, then fewer collisions. Test only
  the selected model. Include initial untrained validation in selection.
- Keep baseline bytes, protocol, progress, diagnostic and evaluation reports,
  selected training weights/optimizer, and a loadable candidate live checkpoint.
  Export starts a fresh paused session with a fresh online optimizer and synchronized
  anchor; offline update counts stay in the experiment report.
- Output to an exclusively created run directory. Mark RUNNING/FAILED/COMPLETE,
  refuse overwrite, hash all finished artifacts, and never promote automatically.
- Add a CLI command and a standalone HTML comparison for convenient review.

## Method and limits

Dataset aggregation follows the motivation of Ross, Gordon and Bagnell (2011),
https://arxiv.org/abs/1011.0686. This bounded implementation uses a mixed rollout
schedule, balanced finite replay, and a small partially observed graph policy;
it does not inherit a general performance guarantee. The full-map teacher knows
obstacles outside the policy's local observation window. Some teacher decisions
are therefore not identifiable from policy observations. No collision mask,
planner fallback or teacher assistance is used in reported policy evaluation.

## Validation

Regression tests cover config bounds and disjoint seeds, deterministic sampling,
source immutability, validation-only selection, matched evaluation worlds,
normal inference routing, exported checkpoint loading, output overwrite refusal,
failure receipts and global RNG preservation. Execute a bounded CPU run on the
captured mixed graph and report the measured outcome, including failures. Run
the full suite, render the standalone report, and build/check the new package.
