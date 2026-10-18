# Reproducible live laboratory upgrade

The user requested a whole-project advancement. The existing project is a local,
CPU-only synthetic neural-circuit research instrument. This upgrade makes the
existing live learning experiment inspectable, repeatable and usable. It does
not add another unvalidated biological mechanism or download a model.

## Design

1. Restore the browser application and route all six existing topology presets
   through the API and chat. Report service failures truthfully.
2. Add a frozen navigation evaluation using a copied model and independent seeded
   worlds. Disable teacher assistance and optimizer updates. Compare policy,
   no-graft, no-recurrence, seeded random and shortest-path oracle on matched
   initial worlds. These are inference-time controls, not independently trained
   baselines. Record per-episode actions, success, moved path length, collisions,
   teacher agreement and success-weighted inverse path length (SPL).
3. Expose the same bounded evaluator through CLI, HTTP and a browser result table
   with JSON export. Results identify graph, weights, checkpoint training counters,
   protocol configuration and scope. The last report remains identified as a
   frozen result even as the live model continues changing.
4. Save all continuation-relevant auxiliary state in a validated v2 checkpoint.
   Load legacy files with explicit compatibility limits; leave the user's original
   session file untouched during development. Preserve topology preset metadata.
5. Make telemetry observational: no phase-history or random-retrieval side effects,
   same-weight intact/lesion comparisons, measured consolidation statistics.

## Bounds and interfaces

No new runtime dependencies. NavigationEvalConfig accepts 1-16 distinct integer
seeds, sizes 5-31, all NavigationWorld.PRESETS and 1-128 steps per episode. The
default is three seeds (10000-10002), size 11, labyrinth, 48 steps. Evaluation
copies model state under the session lock and releases it before rollout. HTTP
allows one evaluation at a time and returns a conflict for concurrent requests.
Frozen reports are JSON-safe and deterministic for a fixed CPU runtime, weights
and protocol. Exports refuse to overwrite an existing result.

SPL = mean(success * shortest_distance / max(shortest_distance, moved_path_length)).
Collisions do not increase moved path length; collision rate and attempted steps
are reported separately. Success is arrival at the initial goal, with automatic
termination on first arrival. The oracle has full map access and is a diagnostic
ceiling. Evaluation seeds are declared, not proof of unseen training worlds.

## Verification

Regression tests cover exact continuation after anchor synchronization, auxiliary
state restoration, legacy loading, telemetry purity, matched worlds, deterministic
replay, global RNG preservation, malformed/bounded inputs, oracle correctness,
checkpoint/output non-overwrite, all presets, HTTP errors and JS syntax. Complete
Python suite plus a real-browser smoke verifies the integrated workflow. Save a
small measured receipt and inspect built wheel contents. No publication or merge.

## References

Anderson et al., On Evaluation of Embodied Navigation Agents (2018),
https://arxiv.org/abs/1807.06757 . SPL is adapted to the existing discrete grid;
these synthetic results establish neither biological advantage nor robot transfer.
