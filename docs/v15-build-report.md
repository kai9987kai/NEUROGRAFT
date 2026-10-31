# NEUROGRAFT v0.15: causal adaptive transition evidence

31 October 2026. Upgrade on `codex/replay-learning-lab`. This report records local validation before the subsequently requested Git publication; no checkpoint promotion.

## What changed

The live replay workbench now compares cumulative, recent and adaptive transition evidence across the existing prioritized, cyclic uniform and disabled replay schedules. The new adaptive rule selects evidence separately for each state/action using a discounted pre-update predictive log score. It keeps actual positive integer counts and can return to cumulative evidence. Four fresh local outcomes and an evidence score greater than log(4) are required before selecting recent counts. The decay is 0.95 per local observation; the recent window remains 4, 8 or 16 new outcomes. These parameters were fixed before the pilot and were not tuned on its results.

Each event is forecast and scored before either expert incorporates it, including repeated observations in a single command. Binary move/collision forecasts use Beta(0.5,0.5) smoothing for Brier and natural log loss. The planning graph uses unsmoothed observed counts: smoothing never creates an experienced edge. The score is an engineering heuristic, not a posterior, confidence level, certified detector or new biological replay mechanism.

The UI exposes average pre-update losses, local evidence, choices, fresh-outcome counts and switches. It preserves graph inspection, replay budgets, play/pause, receipt export/import and frozen live capture. New fixtures/captures use nine independent planners. Historical v1 receipts reconstruct with the original six-arm engine and disable the unavailable adaptive control; existing v13/v14 study code and archive semantics are unchanged.

Launch the isolated tool:

```powershell
python -m neurograft workbench --port 8770
```

## Primary research and implementation choice

[Tao, Wen and Han, ICML 2026](https://proceedings.mlr.press/v306/tao26c.html) distinguish policy-relevant model error from a global prediction proxy. That motivates evaluating both proper forecasts and actual frozen-policy value; this is not their neural PMER method. [DARLING, revised May 2026 preprint](https://arxiv.org/abs/2604.16684v2), motivates explicit change/stationary assays and highlights separability/reachability assumptions absent from our heuristic. [DEER, September 2025 preprint](https://arxiv.org/abs/2509.15032), motivates separating environment revision from value replay. Neither method is reproduced here, and their reported performance or guarantees are not claimed for NEUROGRAFT.

Adaptive windows and run-length inference were alternatives. The selected predictive-expert baseline fits the existing integer transition contract and exposes a small fixed rule that can fail visibly. Autonomous reacquisition and persistent neural context require distinct interaction/training protocols and remain proposed work in the [research roadmap](research-roadmap.md). See the [design](superpowers/specs/2026-10-31-adaptive-evidence-design.md) and [implementation plan](superpowers/plans/2026-10-31-adaptive-evidence.md).

## Fixed eight-seed pilot

The [standalone report](../reports/runs/adaptive-evidence-20261031/report.html) and [complete evidence](../reports/runs/adaptive-evidence-20261031/report.json) retain eight seeded two-route instances (15000..15007). `plan.json` was written before execution. Five cases cover unconditional stationary samples, four closure/reopening blocks, gradual probability drift, conditional adverse observations followed by unbiased recovery, and sparse revisits. Four forecast models share every ordered observation; a guarded-recent ablation isolates the four-event warmup. Nine planning arms share requested backup caps at declared checkpoints. Their baseline warmup permits 256 backups; later observed checkpoints permit 64; unobserved changes request and spend zero new backups. Actual counts are retained per arm. Equal backup caps do not imply equal total compute.

Mean pre-update binary Brier loss (lower is better; equal weight per seed):

| Supplied stream | Cumulative | Recent | Guarded recent | Adaptive |
|---|---:|---:|---:|---:|
| Stationary | 0.147784 | 0.161328 | 0.160925 | 0.149814 |
| Repeated reversals | 0.316120 | 0.182036 | 0.209476 | 0.210845 |
| Gradual drift | 0.315832 | 0.234654 | 0.236546 | 0.256160 |
| Conditional adverse + recovery | 0.264920 | 0.210234 | 0.265113 | 0.259827 |
| Sparse revisits | 0.238289 | 0.128714 | 0.154055 | 0.144588 |

Adaptive memory reduces reversal Brier loss relative to cumulative by about 33%, while remaining closer to cumulative than recent under stationarity. Recent is better on all changing-stream full-case Brier means in this pilot. Adaptive is slightly worse than guarded recent on reversals and materially worse on gradual drift, so the score selector's benefit is not established uniformly over the warmup guard. On conditional adverse + recovery, adaptive has lower Brier than cumulative but worse log loss (0.754328 versus 0.749120); the two proper scores need not rank the models identically. This full-case mean includes recovery and must not be read as protection against the adverse prefix.

Independent current-dynamics frozen-policy start regret after prioritized replay (discount 0.9; these are analytic values, not sampled navigation success):

| Checkpoint | Cumulative | Recent | Adaptive |
|---|---:|---:|---:|
| First observed closure | 0.403276 | 0.000000 | 0.000000 |
| Second observed closure | 0.403276 | 0.000000 | 0.000000 |
| Forced stationary adverse prefix | 0.000000 | 0.299438 | 0.299438 |
| Closed route after only two sparse revisits | 0.403276 | 0.000000 | 0.403276 |
| Midpoint of gradual drift | 0.066626 | 0.045327 | 0.045327 |

The adaptive guard delays useful revision when evidence is sparse and cannot reject a misleading adverse prefix that is observationally identical to closure. An explicit matched-prefix control reconstructs identical forecasts, selected graphs and diagnostics under different evaluation-only truths. Stationary adaptive selections also occur: eight post-event recent choices across 768 events, concentrated in three seeds, with eight switches. These are descriptive false-switch proxies, not a calibrated error rate. Stage summaries distinguish recent-at-entry from a new switch and count first-switch delays in local visits and relative stream events.

The evaluator solves a distinct held-out transition model and each frozen policy; it never returns outcomes to the memory. Small empirical Bellman residual means the empirical count model is well solved, not that current dynamics are correct. Gradual-drift planning checkpoints use endpoint truth; individual forecast errors use the truth at each event. Pair samples are supplied, rather than autonomously acquired trajectories. Cases/stages within a seed are correlated; seed-wise differences and ranges are retained without significance tests or a universal-dominance/promotion gate. This is an exploratory synthetic pilot, not evidence of stronger neural navigation, biological benefit or deployment reliability.

Reproduce in a new directory:

```powershell
python -m neurograft study-adaptation --out reports/runs/adaptation-study-new
python -m neurograft verify-adaptation-study reports/runs/adaptation-study-new --replay
```

Strict replay regenerated all streams, prequential scores and planning states. Plain verification rederives causal scores and independent metrics, checks derived aggregates and HTML, and verifies artifact hashes. Coordinated resealing of altered seeded outcomes can remain internally consistent; strict replay rejects it. SHA256 checks consistency, not authenticity. Report SHA256: `19ce05976ce4abaeeac4bcf8ffc408284db638f88cb411b020e43fd66820a06f`.

## Neural evidence correction

The v11 narrative now explicitly compares its learned next-observation predictor against the existing strong kinematic baseline: overall learned recovery MSE approximately 0.1024 versus kinematics 0.05884. The learned wall channel improves (approximately 0.1877 versus 0.23535), but no raw goal-success gain was measured. The previously emphasized reduction was relative to an untrained zero predictor. Archived study bytes are preserved; the known v11 `report.html` seal mismatch remains unrepaired and is not presented as verified by this upgrade.

## Verification

The fresh full regression passed **969 tests in 589.04 seconds**, with no failures or skips ([test output](../reports/v15-tests.txt)). The eight-seed strict study replay, old v13/v14 strict replays, **26 workbench browser checks** and **26 existing live browser checks** also passed. Both browser suites report no unexpected page errors; the intentional tampered-receipt test produces one expected HTTP 400. Frozen live capture leaves the source-session digest unchanged. Workbench checks include v1 continuation, batch/single score equivalence, read/replay/goal purity, actual controls and desktop/mobile layout. Existing live checks include WebGL, neural controls, frozen evaluations and checkpoint/replay continuation. The [combined verification receipt](../reports/v15-verification.json) records current source hashes, package, study, browser results and checkpoint preservation.

The wheel contains 53 package Python/web files, all matching source bytes, with no missing or extra assets. An isolated extracted-wheel import reports 0.15.0, creates nine arms, restores a v2 receipt and imports a v1 receipt. Wheel SHA256: `5bb4f73c94ace7f34671959561a9e46012db9c23edb8f4ba63da3384fb4de373`. Package receipt: [v15-package-verification.json](../reports/v15-package-verification.json).

Scoped Git attributes preserve the new study, browser and verification artifact bytes across checkouts, and mark the wheel as binary. Package README metadata also matches the current source text after decoding UTF-8 and normalizing line endings.

The original checkpoint SHA256 remains the preservation baseline `8a3630eaebd3242ee43bca3e4bd0375c330592645fcd87bbce4413f8d813f5f2`. Existing dirty artifacts and research notes are retained. No new dependency, pretrained-model download, checkpoint promotion or remote publication was performed.

The isolated v15 fixture workbench is running at [127.0.0.1:8770/workbench](http://127.0.0.1:8770/workbench), PID 8400. Its health endpoint confirms `live_session_opened: false`; launch details are in [the runtime receipt](../reports/v15-runtime.json). This running process is separate from the user's saved live session.

## Git publication preparation

The user subsequently requested committing the v15 upgrade to GitHub. The publication includes code, tests, research documentation, the sealed eight-seed archive, browser evidence and the verified wheel. It excludes the pre-existing modified live checkpoint, superseded revision output, v14 command receipt and unrelated research-review notes. Source hashes are checked against the completed full-suite receipt; an isolated export of the staged Git tree verifies package parity, study replay and focused compatibility tests. Publication verification is recorded in `reports/v15-publication-verification.json` and `reports/v15-publication-tests.txt`.
