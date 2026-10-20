# NEUROGRAFT v0.9: observed-map navigation

The new optional navigation controller reaches goals using remembered local observations while the underlying raw policy remains weak. This release adds a working, inspectable engineering mechanism and matched controls to explain its contribution. It does not retrain or promote a model.

## Mechanism and user flow

Choose **Raw policy**, **Policy with legal-action mask**, or **Observed map + policy tie-break** in the live Navigation controller panel. The default remains raw policy with the checkpoint's configured guidance. The mask/map modes bypass teacher actions; expert labels can still support imitation training when learning is enabled. Chat `navigation` and `why` identify the mode, effective guidance and map decision.

The map receives the existing 4-by-4 observation and public grid size. It decodes current/goal coordinates, records the current cell and adjacent free/blocked cells, commits to a not-yet-visited frontier, and runs breadth-first search over known free cells. A known route to the goal takes priority. Policy logits break equal-cost choices. The browser renders only this observed map, route, known goal and current agent; unknown cells remain dark. It also identifies the last executed action source, including manual overrides.

The controller has no world object, hidden obstacle map, environmental telemetry or teacher-action input. It uses noiseless synthetic grid sensors and exact coordinates. This is symbolic episodic planning rather than learned perception, visual SLAM or a biological mechanism. It is related to modular mapping/planning architectures such as [Active Neural SLAM](https://arxiv.org/abs/2004.05155), without implementing that paper's learned SLAM module. Legal filtering is adjacent to [action-masking research](https://arxiv.org/abs/2006.14171), but this is deterministic selection for an imitation policy, not a replication of those policy-gradient experiments.

## Measured fixed-weight study

The input is the exact v0.8 fixture candidate archived at `reports/runs/navigation-v08-smoke-20261019/experiment/candidate.pt`. Model SHA-256 is `cc507d705e24031c6d6437982cbc61a836aee2930a40dfc2b1301c1bdaaa3029`. All four source graphs are synthetic four-node fixtures; no authentic-connectome or pretrained-language improvement is tested.

The declared study uses sizes 7 and 11, all six topologies, seeds 41000/41001 and a 96-step horizon: 24 scenarios, each evaluated with eight controls and fresh empty navigation memory. No parameter updates occur. The source checkpoint, declaration, per-condition JSON, aggregate JSON/HTML and manifest are in `reports/runs/navigation-v09-controller-study-20261019/`.

| Action selection | Goals reached | Mean SPL | Collision rate | Mean attempted steps |
|---|---:|---:|---:|---:|
| Raw policy | 0/24 | 0.0000 | 96.57% | 96.00 |
| Policy + legal mask | 18/24 | 0.6250 | 0% | 29.00 |
| Observed map + policy prior | 24/24 | 0.9642 | 0% | 7.00 |
| Observed map without policy prior | 24/24 | 0.9395 | 0% | 7.25 |
| Privileged full-map shortest path | 24/24 | 1.0000 | 0% | 6.58 |

The random, no-graft and no-recurrence controls also reached 0/24. The mask removes invalid moves by construction, so zero collisions are not evidence of learned collision avoidance. Observed-map decisions override 62.5% of raw proposals; those raw proposals still point at walls 48.81% of the time on its trajectories.

The map-only control reaches every goal too. The small descriptive SPL difference is not a reliable learned-prior benefit: scenarios share seeds across conditions and small arenas may share geometry across topology labels. This is an exploratory local synthetic result, not independent replication, robot transfer, a topology advantage or stronger raw neural capability.

Reproduce with a new output directory:

```powershell
python -m neurograft study-navigation --bundle reports/runs/navigation-v08-smoke-20261019/experiment/source.bundle.json --checkpoint reports/runs/navigation-v08-smoke-20261019/experiment/candidate.pt --out reports/runs/new-v09-study
python -m neurograft verify-navigation-study reports/runs/new-v09-study --replay
```

## State and evidence integrity

v3 live checkpoints retain the mode and exact bounded map state, including committed frontier and policy proposal metadata. Validation checks the map against world occupancy, step/visit budgets and the actual post-decision pose, allowing manual moves and collisions. v1/v2 checkpoints read in raw-policy mode with fresh map state; v2 observer and policy history remain intact. Existing v0.8 archived evaluations still replay exactly.

Default study verification checks file hashes, declaration/configuration, model/graph identity, aggregates and action geometry without neural inference. It rejects inconsistent SPL, rewards/distances, poses, action traces, proposal/override counts and prior labels. `--replay` additionally reruns the captured model and every controller. Hashes detect changed files and are not authenticity signatures or proof of training provenance.

Independent review exposed and reproduced an invalid pose/map restore and coherently resealed contradictory trial metrics. Both were repaired with failing-first regressions. Source graphs, prior evidence, the live checkpoint and existing wheels remain preserved. No commit or publication was performed.

A final HTTP regression exposed an early startup rejection with an unread JSON body. The handler now consumes the existing bounded body before its readiness check, retaining mutation rejection while avoiding this [TCP reset risk described in RFC 9112](https://www.rfc-editor.org/rfc/rfc9112.html#section-9.6). A deterministic regression checks body consumption before the rejection response.

## Verification receipts

The final complete Python suite passed **527 tests in 222.24 seconds** after the startup-response fix. Desktop/mobile checks passed with zero page errors, including the observed route display and exact map save/load. Source compilation, browser syntax and whitespace checks passed. The saved live checkpoint retained SHA-256 `4806191a4393ff83a25d1e74df0a16115e2289d8ff7a93f2f8402bbe1c43419f`.

The final v0.9 wheel matches all 42 packaged Python/static files and README metadata. An isolated extraction reproduced all 12 study conditions and their 192 variant/scenario episodes exactly, with the aggregate unchanged. Wheel SHA-256: `a72929ab325a6fe74819964aeb5158cf8cf7c81092f26ff1480796a040a70fd5`.

- `reports/v09-tests.txt`: integrated Python regression results.
- `reports/v09-tests-before-startup-drain-fix.txt`: retained failed validation that led to the HTTP repair.
- `output/playwright/v09-final/browser-verification.json`: raw/extended trials, observed-only map/route, exact map save/load, desktop/mobile layout, WebGL and zero page errors.
- `output/playwright/v09/game-client/`: bundled browser action loop with manual overrides in map mode.
- `reports/v09-package-verification.json`: packaged source/static/README equality and exact study replay from an isolated wheel extraction.
- Study `manifest.json`, `protocol.json` and `report.json`: exact frozen input, conditions and action traces.
