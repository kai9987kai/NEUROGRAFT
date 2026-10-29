# Live replay and model-revision workbench (v0.14)

The user requests another cohesive research-led upgrade, explicitly including a live interactive tool. Build on the validated v0.13 code in the current checkout; preserve all prior dirty work and the real checkpoint hash `8a3630eaebd3242ee43bca3e4bd0375c330592645fcd87bbce4413f8d813f5f2`. The user previously prioritized learning and experiments.

## Direction and alternatives

More offline tables would leave the mechanism hard to explore. A cosmetic dashboard would not address model staleness. Choose a live, isolated workbench that crosses transition-memory policy with replay schedule, supports actual interventions and verifiable receipts, and exposes where a converged planner is using stale evidence. Keep Python/PyTorch, loopback HTTP, vanilla JS and offline assets; no added dependencies.

## Mechanism and evidence boundary

Capture the empirical live graph by copying its canonical state under the session lock. Never step, train, save, or change that live session when using the workbench. Compare six planners: cumulative versus recent transition evidence, each with prioritized, cyclic uniform and disabled replay. The recent model uses the original aggregate counts until the first *new* observation for a particular state/action; thereafter that action uses only its last W new outcomes. Other actions retain the frozen baseline counts. W is 4, 8 or 16. Aggregate live counts contain no chronology: never manufacture an ordered pre-capture history from them.

New outcome events are explicitly supplied experimental observations. They are not observations automatically collected by the real agent. A collision can revise a previously successful action; repeated successes can restore it. Model changes retain old values until replay corrects them, while goal changes clear values with zero backups. Snapshot reads are pure. A low empirical Bellman residual establishes planning consistency with a model, not that model's accuracy about current dynamics.

## Interactive product

Serve `/workbench` with a responsive SVG graph, clickable inspected cell/goal, arrows and counts, cumulative/recent toggle, three schedule cards, value heatmap, actual backup/residual/error indicators, one-step and bounded play/pause, observation controls, fixture reset, frozen live capture, JSON export/import, and a visible operation history. Label supplied observations, unknown goals, stale values, no-policy/no-neural-training scope, and errors clearly. Include keyboard-accessible alternatives to SVG clicks. Add a visible link from the main live workspace. Automatic play runs bounded batches through the same commands, with one request in flight and an explicit stop.

Provide `python -m neurograft workbench --port ...` to launch a fixture-only server without a live checkpoint. A live server also exposes the workbench and optional capture. Launch the fixture tool for this user and verify it responds. Persist experiment receipts by user export; do not auto-write the real session.

## API and limits

`ReplayWorkbench` owns independent graph/model/planner state. Commands are strict dictionaries: goal coordinates, replay budget0..256, observation source/action/destination/count1..16, and reset. Workbench control/replacement validates a complete candidate before installation. At most256 nodes,128 operations,2048 new events and4096 total requested backups per arm; max receipt body2MiB on workbench import only. Existing API body caps remain unchanged. Read snapshots and receipt verification never mutate the source graph/session. Receipts contain an initial graph/goal/window/provenance, ordered operations and SHA256 of deterministic final state; import reruns all operations and rejects changed results. Hashes are consistency checks, not authorship signatures.

`GET /api/workbench` returns the snapshot; `POST /api/workbench` dispatches commands plus fixture/capture/import; `GET /api/workbench/receipt` exports. Frozen capture locks only while copying. A separate workbench lock serializes mutations without blocking ordinary live controls during planning.

Snapshot contract: version1, goal[x,z], window, provenance, events, operations count, models{cumulative,recent} each containing nodes,edges(source/action/destination/count),total_transitions,sha256; arms{model}{schedule} each containing values[{position,value}],total_backups,summary,stale,metrics{value_mae,bellman_residual,action_agreement,decisive_states}; history; evidence_scope. Additional fields must be documented.

## Experiments and research

Run eight seeded synthetic cases with identical ordered observations across all six arms. Include barrier closure, reopening and stationary stochastic controls. Separately measure planning error against each empirical oracle and decisions against held-out current dynamics. Report nulls/tradeoffs, actual backup counts and current-dynamics evidence separately; do not choose a winner solely from empirical residual. The existing environment's visually changing gates are diagnostic cues and are not used as physical barrier interventions.

Current primary motivation includes He et al., ESWM (ICLR2026), independently updateable episodic memories; Antonov and Dayan, Exploring replay (Nature Communications2025), replay under changing uncertainty; and World-In-World (ICLR2026), evaluating task utility rather than proxy quality. Specific sources and their limits go in the build report. The recent-window rule is an explicit engineering baseline, not a reproduction of these methods or a novelty claim.

## Validation

Analytical revision/recovery cases, cumulative-vs-recent stream identity, chronology boundaries, downward corrections, pure reads, deterministic receipt replay/tamper rejection, atomic failed import, original live-state hash, HTTP limits and loopback origins, real desktop/mobile interaction and download/import, complete pytest, fresh wheel/source parity and independent review. Preserve the v0.13 archive and its receipts as historical evidence.
