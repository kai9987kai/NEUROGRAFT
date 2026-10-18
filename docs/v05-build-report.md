# NEUROGRAFT 0.5: reproducible live laboratory

18 October 2026. Local implementation and validation; no remote publication.

## Result

The live browser application starts again. Navigation trials now measure a frozen
policy against matched inference interventions, random actions and a shortest-path
oracle. New checkpoints retain the full continuation state introduced since the
original live implementation. Observing diagnostics no longer changes simulation
or global RNG state, and current-weight lesion comparisons no longer include an
optimizer step in the measured effect.

Implemented repairs include all six topology presets in API/chat, persisted world
presets, executed-turn compass integration, correct previous-action alignment in
the forward observer, phase-history resets after goal/world discontinuities,
truthful health state, and finite-magnitude validation of auxiliary tensors.

## Measured policy capture

The captured legacy input is retained in
`reports/runs/navigation-v05-20261018/frozen-input.pt`; `capture.json` records its
SHA-256, model/graph hashes, compatibility warning and source path. The original
live checkpoint was read, not written. The captured policy had 106646 live steps
and 106639 updates; these counters alone are not evidence of useful navigation.

Protocol: seeds 10000-10007, size 11, labyrinth, 48 attempted actions per trial,
four-observation inference window, no policy learning or teacher mixing.

| Variant | Goal successes | Mean SPL | Collision rate |
|---|---:|---:|---:|
| Full policy | 0/8 | 0 | 99.22% |
| Grafts disabled | 0/8 | 0 | 99.22% |
| Recurrence disabled | 0/8 | 0 | 99.22% |
| Seeded random | 1/8 | 0.0714 | 35.14% |
| Full-map shortest-path oracle | 8/8 | 1 | 0% |

The policy failed to demonstrate useful unassisted navigation in this small
captured run. Inference interventions reuse weights; they are not separately
trained ablations. The seeds are explicitly declared but not certified unseen.
SPL counts moved distance, while collisions and attempted steps are separate.
Results are exploratory and do not establish biological effects or robot transfer.
The reference for SPL is [Anderson et al. (2018)](https://arxiv.org/abs/1807.06757).

## Reproducibility and compatibility

Checkpoint payload v2 stores the anchor, observer module weights/optimizer,
memory, compass, phase history, STP, bioelectric state, prior latent/action,
diagnostic summaries, explicit assay report and initial parameter reference.
Integrity, shapes, primitive schemas, finite numeric bounds and optimizer
moments are checked before installing a candidate. V1 files load with a warning:
their omitted auxiliary history cannot be reconstructed. This limitation does not
affect the frozen policy evaluation, which only uses the saved GraphCircuit.

The self-model trains as a separate observer. Memory, compass, phase, bioelectric,
STP, byte projection and consolidation do not supply navigation policy inputs.
The consolidation telemetry now uses measured routes rather than constant display
values. The byte bridge is untrained; display labels are not language capability.

## Verification evidence

- Final full Python suite: 327 passed in 63.26 seconds after state, browser and API repairs.
- New evaluator tests cover matched worlds, deterministic traces, global RNG and
  session preservation, bounded malformed inputs, oracle ceiling and output safety.
- New continuation tests cover synchronized anchors, exact future updates, full
  auxiliary restoration, legacy warnings and malformed/overflowing auxiliary state.
- Server regressions cover all presets, evaluation conflicts and failure health;
  JavaScript syntax is checked with Node.
- Built wheel smoke loads the extracted package, resumes a synchronized anchor,
  evaluates a frozen policy and loads the packaged web assets.
- Chromium desktop and mobile checks passed with Three.js WebGL rendering, no
  console/page errors, matched trial results and JSON export, a stale-weight
  notice and no page-level horizontal overflow. Browser checks use fixtures,
  separately from the captured eight-world policy evaluation above.

Browser and final package receipts are retained in `output/playwright/v0.5/` and
`reports/v05-package-validation.json`. The reusable browser check is
`tools/verify_live_ui.py`; it uses an isolated temporary fixture session and an
already installed Playwright/Chromium, without opening an existing live state.

## Next research step

Train a separate candidate on a balanced multi-world curriculum, retain an
unchanged captured baseline, and freeze a navigation protocol before selection.
Measure collisions, success and path efficiency before promoting candidate
weights. This upgrade provides the evaluation and continuation tools needed for
that experiment; it does not promote the captured failing policy.
