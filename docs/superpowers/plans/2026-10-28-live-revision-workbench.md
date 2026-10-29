# Live Revision Workbench Implementation Plan

> Use subagent-driven-development and verification-before-completion. Keep user-authorized reversible work moving without additional approval checkpoints.

**Goal:** A live interactive empirical model/replay experiment tool with controlled evidence revision, reproducible receipts and current-dynamics tests.

**Architecture:** Isolated workbench engine, offline SVG frontend, loopback routes shared with LiveApp, and fixture-only CLI server. Independent agents own engine and frontend; root owns integration/experiment runner/docs.

**Spec:** `docs/superpowers/specs/2026-10-28-live-revision-workbench-design.md`

**Global constraints:** Preserve original checkpoint and existing dirty v0.13 work. No new dependencies. Explicit supplied observations; no fabricated history.256 nodes,128 commands,2048 new events,4096 requested backups per arm; import body cap2MiB. No automatic live training/promotion/publication.

## Tasks

- [x] Engine: test-first `neurograft/workbench.py`, `tests/test_workbench.py`; six arms, evidence policies, pure metrics, strict receipts and atomic commands.
- [x] UI: `web/workbench.html`, `.css`, `.js`; responsive graph, controls, comparison, play/pause, provenance, export/import and useful errors.
- [x] Integration: server/service routes, main workspace link, fixture-only workbench CLI/launcher; API isolation tests.
- [x] Evidence: seeded close/reopen/stationary study, empirical versus current-dynamics metrics, strict replay and scientific limitations.
- [x] Verification: independent review, full pytest, browser interactions, package parity, launch responsive tool and preserve original checkpoint.
- [x] Documentation/version/receipts and concise final report.

## Review focus

- Recent windows start only from new explicitly ordered events; captured counts have no temporal order.
- Planner consistency and current-dynamics quality remain distinct.
- Observation/import failure cannot partially mutate state or the live session.
- Browser play cannot queue overlapping requests or continue after an explicit pause.
- Receipts reproduce values and metrics exactly; no hidden clock/RNG/session dependencies.

## Ledger

- v0.13 remains dirty and uncommitted on codex/replay-learning-lab. Baseline checkpoint hash confirmed unchanged; no Python training process found.
- Direction selected under broad user authorization: interactive revision workbench rather than more disconnected mechanisms. All changes remain local and reviewable.

- Engine40 analytical/strict-state tests passed. API/study/engine focused total64 passed after root completed interrupted verifier hardening; the full suite includes all of them.
- Review fixes: import-only2MiB allowance; source_bundle_sha256 provenance name; strict duplicate node/value checks and canonical empirical digest. Failing regressions observed before fixes, then passing focused checks.
- Study stage backup counter initially repeated the previous call in unobserved stages. Added failing no-new-backups regression and actual counter-delta accounting, then independent scoped review approved. All12 unobserved aggregate rows now report0 new backups. Superseded preliminary output retained; final archive revision-learning-20261029-final.
- UI browser21 checks passed plus readable mobile-label assertion; original live browser26 checks passed. No page errors or unexpected console errors; one intentional400 for tampered receipt. Capture leaves full session digest unchanged.
- Final study8 independent seeds and strict deterministic replay verified. Conditional recency adaptation benefit and stationary adverse-run cost both reported; no universal or neural-navigation gain claimed.
- Final wheel0.14.0 source parity50 files, isolated import creates6 arms. SHA2567bb4a07efaf8c13f91f2540e6359ca7d307db71433fcb5ed297f8de33bbddae6.
- Final full regression:919 passed in412.34s. git diff --check and JavaScript syntax checks clean.
- Launched isolated workbench at http://127.0.0.1:8770/workbench, PID19732, health responds, no live checkpoint opened. Seed17, one32-backup starter batch; left running for user interaction.
- Original checkpoint SHA256 remains8a3630eaebd3242ee43bca3e4bd0375c330592645fcd87bbce4413f8d813f5f2. Existing v0.13 work and research notes retained. No commits, push, publication, merge or promotion performed.
