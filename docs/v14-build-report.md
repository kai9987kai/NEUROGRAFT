# NEUROGRAFT v0.14: live transition-revision workbench

29 October 2026. Implemented locally on `codex/replay-learning-lab`, building on the existing v0.13 work. This report records local validation before Git publication. No checkpoint promotion.

## Interactive tool

The [live workbench](http://127.0.0.1:8770/workbench) is running as an isolated fixture service. It never opens a neural session or checkpoint. Launch it again with `python -m neurograft workbench --port 8770`, or `Run-Workbench.ps1`.

The new page provides a clickable experienced-transition graph, value heatmaps, coordinate/keyboard inspection, goal changes, explicit collision/move observations, single-backup stepping, bounded play/pause, two transition-memory policies and three replay schedules. Schedule cards expose actual backups, stale values, empirical value error, Bellman residual and decisive-action agreement. Mobile comparison cards stack at readable sizes. The ordered log and JSON receipt preserve an experiment for exact import/replay.

On a normal `neurograft live` server, the main header links to the same page. **Capture live graph** copies empirical counts, public goal, step count and source identities while holding the session lock, then releases it before workbench planning. It does not tick, train or save that source session. The standalone fixture service explains that live capture requires the main server.

Try this:

1. Inspect `(3,1)` in the default fixture. Keep the goal at `(5,1)`.
2. Supply an eastward collision outcome eight times, then run a replay batch.
3. Switch between cumulative and recent evidence and between replay schedules. Export a receipt, reset, and import it to reconstruct the exact state.

## What the mechanism does

Six independent value planners receive the same supplied evidence and requested backup cap: cumulative versus recent transition counts, each crossed with residual-prioritized, cyclic uniform and disabled replay. A changed model retains its old values until replay corrects them. A changed goal clears values with zero backups. Read-only snapshots never replay.

The recent model retains a frozen action's baseline counts until its **first new supplied observation**. That state/action then uses only its last W new outcomes, with W in {4,8,16}. Other state/actions retain their baseline counts. This is an explicit local reset/window rule. Captured aggregate counts have no chronology; the implementation never fabricates a historical outcome sequence from those counts.

The workbench is an empirical planning instrument. Supplied events are operator experiments, not automatically collected observations of the live agent. Neither memory policy changes neural weights. Empirical oracle agreement and a small residual mean the supplied model is well solved; they do not establish that the model matches the current world.

## Research basis and limits

- [He, Daigle and Bashivan, ESWM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/23c2c77c921111ec8902ac482d92b289-Abstract-Conference.html): independently updateable episodic memories motivate controlled memory revision. This tool does not predict unseen transitions or reproduce ESWM.
- [Duan et al., DEER, preprint 18 September 2025](https://arxiv.org/abs/2509.15032): environmental changes should be separated from replay scheduling. The window estimator is a simpler engineering baseline, not DEER.
- [Palafox and Fridovich-Keil, CLAW, preprint revised 21 September 2026](https://arxiv.org/abs/2609.12278v2): fresh transition evidence can adapt a model at test time. No neural adapters or hypernetwork are implemented here.
- [Zhang et al., World-In-World, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html): decision utility should be evaluated separately from model proxies. This study uses finite analytic policy evaluation rather than that embodied benchmark.

The revision rule is not claimed as a new algorithm, biological replay model, or demonstrated connectome advantage.

## Controlled study

The [final HTML report](../reports/runs/revision-learning-20261029-final/report.html), [machine-readable evidence](../reports/runs/revision-learning-20261029-final/report.json), plan and manifest preserve eight seeded two-route instances with geometry, outcome probabilities and acquisition-count variation. Identical ordered supplied observations feed all six arms. Independent current dynamics are withheld from the workbench and evaluated by a separate solver.

Conditions include baseline, an unobserved closure, observed closure, unobserved reopening, observed reopening, goal relocation, an unbiased stationary sample and a conditional stationary adverse-run stress case. Changing the held-out environment supplies no knowledge to the planners until an explicit observation command. The existing navigation environment's visual gates are diagnostic overlays and are not used as physical barriers in this assay.

At a 64-backup post-change cap, with prioritized replay:

| Condition | Cumulative current-dynamics action agreement | Recent current-dynamics action agreement | Cumulative target Brier risk | Recent target Brier risk |
|---|---:|---:|---:|---:|
| Closure after supplied observations | 70.77% | 100.00% | 1.39734 | 0.00000 |
| Reopening after supplied observations | 98.21% | 98.61% | 0.02907 | 0.00750 |
| Unbiased stationary sample | 98.86% | 100.00% | 0.00295 | 0.03148 |
| Conditional stationary adverse run | 98.61% | 66.86% | 0.02065 | 1.38031 |

These are means across eight seeded finite instances. Agreement uses states with a unique current-dynamics best action, not sampled goal-reaching success. The Brier quantity is excess distribution risk, not individual-outcome loss. Frozen deterministic policies are also evaluated analytically for discounted start-value regret, with no evaluation feedback. Closure start regret was 0.43604 for cumulative evidence and zero for recent evidence; the stationary adverse stress reverses the tradeoff (approximately zero versus 0.30793).

The adverse-run case is a deliberately conditional stress intervention, not evidence that such a sequence is common. The unbiased stationary sample already shows a narrower tradeoff: recent counts have worse probability estimation even when action choices remain good. Reopening can penalize retaining a too-recent closure until new success observations arrive. These results do not establish a universal recency benefit or neural navigation improvement. Equal backup caps are not equal computation. Baseline warm-up requests 256 backups, and per-stage requested/actual/cumulative work is retained separately.

An accounting regression caught that unobserved stages initially displayed the previous call's backup count. The final report records actual counter deltas per stage; all 12 unobserved-stage aggregate rows now show zero new backups. The preliminary `revision-learning-20261029` directory is preserved locally as superseded development output; use the `-final` archive for current evidence.

```powershell
python -m neurograft study-revision --out reports/runs/revision-study-new
python -m neurograft verify-revision-study reports/runs/revision-study-new --replay
```

Final report SHA256: `37300fc774c3093c447a4179e8e176d9a2a9959a7d78b3cf755da08eec6899fe`. Strict deterministic replay passed. Plain verification checks seals, graph/value structure, derived metrics, stage backup accounting and rendering; strict replay also regenerates seeded worlds, observation commands and schedules. Hashes provide consistency checks rather than authenticity signatures.

## Validation and preservation

- 64 focused engine/API/study tests passed, including the new accounting regression. The engine alone has 40 analytical and strict-state tests.
- The workbench browser suite passed 21 checks, plus an explicit readable-mobile-label assertion: actual controls, six-arm comparison, play/pause, supplied observations, export, invalid-import rejection, exact restoration, goal changes, desktop/mobile layout and frozen capture. Its complete live-session digest remained unchanged. One expected HTTP 400 is the deliberate tampered-import test; there were no page errors or unexpected console errors. Receipt and screenshots: `reports/browser-v14/workbench/`.
- The existing live laboratory's 26 desktop/mobile checks passed, including WebGL rendering, learning controls, legacy evaluation, replay action source and checkpoint continuation. Receipt: `reports/browser-v14/live/`.
- Independent review and scoped re-review approved chronology, oracle separation, limits, capture isolation, atomic replacement, receipts and stage accounting. Review fixes restrict the 2MiB body allowance to receipt import and correctly name `source_bundle_sha256`.
- The wheel contains 50 package Python/web files, all matching source bytes, with no missing assets. An isolated extracted-wheel import creates all six workbench arms and reports version 0.14.0. Wheel SHA256: `7bb4a07efaf8c13f91f2540e6359ca7d307db71433fcb5ed297f8de33bbddae6`. [Package receipt](../reports/v14-package-verification.json).
- Full final regression: **919 passed in 412.34 seconds** with no failures or skips. [Saved test output](../reports/v14-tests.txt). [Combined verification receipt](../reports/v14-verification.json) records the current source hashes, wheel, browser checks, study seal and preserved checkpoint.

Interactive limits are 256 nodes, 128 commands, 2,048 new observations and 4,096 requested backups per arm between resets. Each command permits at most 256 backups or 16 observations. Capture/import validation constructs a detached candidate before replacing the workbench. The source session has a separate lock and ownership path. Unsupported or oversized captures fail without truncating evidence. Loopback Host/origin checks and existing 64KiB limits remain in force; only receipt import permits a 2MiB body.

The original saved session SHA256 remains `8a3630eaebd3242ee43bca3e4bd0375c330592645fcd87bbce4413f8d813f5f2`. Existing v0.13 results, live state, untracked research notes and earlier archives are retained. The pre-existing v11 HTML hash mismatch remains documented in the v0.13 report; it was not resealed or represented as verified here.

## Git publication preparation

Verified payload bytes are preserved with scoped Git attributes. Successful browser receipts and screenshots are included under reports/browser-v13/ and reports/browser-v14/. A pre-existing whitespace-only blank line in compass.py was cleaned before publication; the wheel was rebuilt and the package/source receipts refreshed. No executable code changed in this preparation. Source bundle state and unrelated research notes are excluded from the commit.

Publication preparation passed79 focused tests and strict replay of both studies from an isolated export of the staged Git snapshot. All50 staged source blobs match the rebuilt wheel and refreshed verification hashes. See reports/v14-publication-verification.json and reports/v14-publication-tests.txt.
