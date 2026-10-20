# NEUROGRAFT v0.10: matched learned-navigation experiments

The v0.10 implementation, engineering checks, predeclared four-arm learning study, package/source parity verification, and frozen trial replay are locally verified. This report records that boundary as of 20 October 2026; it documents the measured differences between policy readouts and training augmentation truthfully, without unearned claims of biological advantage or automatic checkpoint promotion.

v0.9 improved navigation through an explicit observed-map controller while the raw graph policy remained weak. v0.10 tests changes to the learned policy itself: an optional all-source residual readout and optional quarter-turn training augmentation. The mechanisms, primary research links and their limits are documented in [the learning design](v10-learning-design.md).

## Implemented mechanisms

The default `human` readout remains the original policy head over the human-source pooled representation. Optional `context_residual` adds a zero-initialized linear head over the concatenated representations of all four sources. Matching initialization preserves the original common weights and initial logits. In the study's hidden-dimension-8 configuration, this adds 132 parameters, increasing the total from 4,920 to 5,052. The additional capacity is a difference between architectures, even when optimizer updates and raw sample budgets match.

Quarter-turn augmentation rotates coordinates, goal displacements, headings, adjacent-wall indicators and action labels together. All-zero history padding stays zero. Evaluation uses ordinary, untransformed observations. This is training augmentation rather than an architectural equivariance guarantee, and rotating a labelled route does not remove the full-map teacher's fixed N/E/S/W tie bias.

The optional `no_context` evaluation bypasses only the residual head. In residual mode, `no_graft` retains the direct all-source readout; it is not a human-source-only policy. No-graft, no-recurrence and no-context comparisons are interventions on the same trained weights, not separately trained models. Source lesions remove the selected source before routing and readout in the residual architecture.

The readout-control flag is opt-in. Existing schema 1/2 protocol configurations and evidence-scope text remain unchanged when it is omitted, preserving older archive replay. Readout controls use schema 3. The browser identifies `human-source readout` or `all-source residual readout` in the navigation status.

## Predeclared study and current state

The active archive is [`reports/runs/navigation-v10-readout-study-20261020-final/`](../reports/runs/navigation-v10-readout-study-20261020-final/). Its `plan.json` declares all four arms before execution: `human_plain`, `human_rotated`, `context_residual_plain` and `context_residual_rotated`. Every arm retains seeds 0, 1000 and 2000; no test-based choice among arms or seeds is made.

The frozen input uses four synthetic nodes per source, hidden dimension 8 and an observation window of four frames. Its model hash is `cc507d705e24031c6d6437982cbc61a836aee2930a40dfc2b1301c1bdaaa3029`; frozen-input file SHA-256 is `e4f38ea64336b29635bbcdfc8778c6392adc6271e3b3e99ffba1f633c50de776`. Candidates start fresh rather than continuing the input model's optimizer state.

Each run has four rounds, 24 worlds per round and 128 optimizer updates per round: 96 scheduled training worlds and 512 updates. Across four arms and three seeds, the declared total is 12 runs and 6,144 updates. Training cycles sizes 7 and 11 across all six presets: labyrinth, chambers, corridors, pillars, concentric and hierarchical. Batch size is 16, replay capacity 256 and learning rate 0.005.

Collection follows the expert only. Within each seed, arms share common initial weights, original collected samples and raw minibatch identities; collection and batch hashes record those matching claims. Augmentation uses a separate random stream. The seed controls both fresh initialization and the training-world offset: the three scheduled world ranges are 0–95, 1000–1095 and 2000–2095. These are matched comparisons within each seed, with both initialization and training worlds varying between seeds.

Validation uses seeds 50000–50005. Test uses seeds 60000–60007. Both use an 11-by-11 labyrinth and a 48-attempt horizon. Each run selects a checkpoint using validation success rate, then SPL, then lower collision rate, retaining the earlier checkpoint on ties. Frozen raw-policy evaluation uses no teacher actions, legal-action mask or map controller. No checkpoint is automatically installed or promoted.

The study is complete across all 12 declared runs (4 arms × 3 seeds, 6,144 updates total). All runs and aggregate metrics were verified with exact frozen replay ([receipt](../reports/v10-replay-verification.json)).

### Raw unassisted policy performance (24 pooled episodes across seeds 0, 1000, 2000)

| Arm | Readout | Rotation Aug | Parameters | Success Rate | Mean SPL | Collision Rate | Mean Steps |
|---|---|---|---:|---:|---:|---:|---:|
| `human_plain` | human | No | 4,920 | 15/24 (62.5%) | 0.6250 | 77.2% | 23.8 |
| `human_rotated` | human | Yes | 4,920 | 10/24 (41.7%) | 0.4167 | 88.5% | 32.2 |
| `context_residual_plain` | all-source residual | No | 5,052 | **16/24 (66.7%)** | **0.6667** | **73.2%** | **21.8** |
| `context_residual_rotated` | all-source residual | Yes | 5,052 | 15/24 (62.5%) | 0.6250 | 74.3% | 23.0 |

Individual seed test results:
- `human_plain`: Seed 0: 4/8 (50.0%, SPL 0.5000); Seed 1000: 5/8 (62.5%, SPL 0.6250); Seed 2000: 6/8 (75.0%, SPL 0.7500).
- `human_rotated`: Seed 0: 4/8 (50.0%, SPL 0.5000); Seed 1000: 5/8 (62.5%, SPL 0.6250); Seed 2000: 1/8 (12.5%, SPL 0.1250).
- `context_residual_plain`: Seed 0: 4/8 (50.0%, SPL 0.5000); Seed 1000: 6/8 (75.0%, SPL 0.7500); Seed 2000: 6/8 (75.0%, SPL 0.7500).
- `context_residual_rotated`: Seed 0: 5/8 (62.5%, SPL 0.6250); Seed 1000: 5/8 (62.5%, SPL 0.6250); Seed 2000: 5/8 (62.5%, SPL 0.6250).

### Inference interventions and reference controls (pooled 24 episodes)

For `context_residual_plain`:
- Full policy: 16/24 (66.7%), SPL 0.6667, 73.2% collisions.
- `no_context` (lesioning the all-source residual head): drops to 10/24 (41.7%), SPL 0.4083, 88.2% collisions.
- `no_graft` (lesioning inter-source routing): drops to 12/24 (50.0%), SPL 0.5000, 70.9% collisions.
- `no_recurrence` (zeroing recurrent state between steps): drops to 14/24 (58.3%), SPL 0.5833, 79.6% collisions.
- Seeded random control: 6/24 (25.0%), SPL 0.1042, 35.3% collisions.
- Full-map shortest-path oracle: 24/24 (100.0%), SPL 1.0000, 0.0% collisions.

For `human_plain`:
- Full policy: 15/24 (62.5%), SPL 0.6250, 77.2% collisions.
- `no_graft`: 0/24 (0.0%), SPL 0.0000, 99.2% collisions.
- `no_recurrence`: 4/24 (16.7%), SPL 0.1667, 96.0% collisions.
- Seeded random control: 6/24 (25.0%), SPL 0.1042, 35.3% collisions.
- Full-map shortest-path oracle: 24/24 (100.0%), SPL 1.0000, 0.0% collisions.

Those 24 episodes repeat the same eight test scenarios across three initialization/training seeds. They are correlated by scenario and are not 24 independent scenarios or independent replications. No episode-level confidence interval or significance claim is made. Declared disjoint training, validation and test seed ranges also do not certify that the frozen source has never encountered related geometry in earlier work.

## Observation and teacher-label limits

The [initial-observation audit](../reports/v10-initial-observation-audit.json) sampled 768 worlds: seeds 0–127 across six presets at size 11. They produced 53 unique initial observations. Six observation groups, covering 53 worlds, had conflicting deterministic expert labels. Two groups, covering eight worlds, had no common optimal first action across their hidden maps.

This distinguishes arbitrary teacher tie-breaking from a stronger information mismatch: some identical initial sensor inputs require different shortest-route actions even when every optimal first action is considered. It prevents a policy from always choosing a shortest first step from those initial sensors alone. It does not show that reaching the goal is impossible: detours or exploration can still succeed. The audit examines initial observations only, not all possible histories. Full-map distances are privileged diagnostic labels. The richer readout does not reveal unseen obstacles, and a four-frame window does not provide an entire episodic map. Expert-only collection also leaves learner-induced distribution shift unresolved. Set-valued supervision remains deferred.

## Verification receipts

The final complete Python suite passed **595 tests in 378.52 seconds**. The run had the required local HTTP and Windows junction access; those regressions passed with the rest of the suite. The receipt is [`reports/v10-tests.txt`](../reports/v10-tests.txt). Coverage includes rotation geometry and padding, matched collection/batches, residual initialization and gradients, inference interventions, legacy checkpoint continuation, frozen state/RNG preservation, archive identities, contradictory resealed evidence and path handling.

The primary browser regression completed with **exit 0**, `ok: true`, no page errors and the `three-webgl` renderer. Its 13 checks cover startup, stepping, learning controls, topology changes, frozen trials and export, stale-weight notices, observed-map behavior, map save/load, extended controller trials and desktop/mobile layout. The [confirmed receipt](../output/playwright/v10-confirmed/browser-verification.json) and inspected desktop/mobile screenshots are under `output/playwright/v10-confirmed/`.

A [separate two-readout check](../output/playwright/v10-final/readout-labels/readout-label-verification.json) confirmed both navigation labels, raw-policy mode, paused execution, learning disabled and teacher actions off. Named save/load and mobile reload retained each label. Inspected screenshots showed readable wrapping without clipping or horizontal overflow. Both browser drivers used isolated temporary fixture sessions and did not open the persistent live checkpoint.

The first browser run completed its functional assertions but exceeded the old 10-second Chromium-close timer. Its artifacts remain in `output/playwright/v10-final/`. Only the bounded shutdown allowance was extended to 60 seconds; the primary driver's overall 120-second limit and UI assertions were retained. The subsequent primary run passed and closed cleanly. This cleanup failure is distinct from a product UI regression. Earlier incomplete-integration and sandbox-limited test output is retained in `reports/v10-core-suite.txt`; the final 595-test receipt supersedes it as the integrated result.

Existing v0.8 evidence also verified with exact frozen replay: [`reports/v10-v08-compatibility.txt`](../reports/v10-v08-compatibility.txt). That replay covers recorded frozen evaluation, not retraining.

## Preserved preliminary work and package state

The earlier `reports/runs/navigation-v10-readout-study-20261020/` directory is retained with status `INTERRUPTED` and zero completed runs. It was stopped before completing its first child to apply the Windows HTML newline fix, then replaced by the separate `-final` study. Its partial evidence is not a completed learning result.

The v0.10 wheel has been built at `dist/neurograft-0.10.0-py3-none-any.whl`, with SHA-256 `eedfff48401630c99f7dd2776511363cb285275566c347cfaa15370f834f98bd`. Package/source equality and isolated-wheel verification are confirmed across all 42 package files and README metadata ([receipt](../reports/v10-package-verification.json)). Replay verification for all 12 study runs is also confirmed with exact matching across every episode ([receipt](../reports/v10-replay-verification.json)). Building and verifying the wheel does not automatically publish it or replace live checkpoint state.

Hashes are local integrity receipts, not authenticity signatures or proof that training occurred. The small synthetic fixture, added readout capacity, privileged labels, short history and limited evaluation conditions constrain interpretation. Engineering checks or later synthetic navigation gains do not establish a biological-connectome advantage, pretrained language capability, real-robot transfer or an independently replicated scientific result.
