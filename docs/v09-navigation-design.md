# v0.9 observation-only navigation

Intent: make navigation more useful while measuring where its behavior comes from. Preserve all v0.8 artifacts, raw policy metrics and the saved live checkpoint.

The selected design adds optional legal-action filtering and an episodic observed-map planner to the existing four-action navigation flow. The planner receives only the public grid size, current normalized position/goal, four adjacent-wall sensors and policy logits. It never receives the obstacle map, world object, teacher action or shortest path. It commits to an observed-free frontier and plans over observed-free cells; a known route to the goal takes priority. Logits rank equal candidates. A map-only control removes that prior.

This is an engineering hybrid, not a new neural learning algorithm. Its mapping is exact symbolic bookkeeping from noiseless grid sensors, not learned visual SLAM. Related modular mapping/planning work: https://arxiv.org/abs/2004.05155. Legal filtering has adjacent research in action masking: https://arxiv.org/abs/2006.14171; our deterministic imitation-policy selector is not a replication of those policy-gradient experiments.

Implementation and checks:

- Pure validated controller and canonical exact state, committed frontier, deterministic tie-breaking, finite map and malformed-state rejection.
- Optional live `policy`, `masked_policy`, `observed_map` modes. Defaults retain current behavior. Mask/map modes bypass teacher actions; manual control remains explicit. Observation memory resets on goal/world/episode changes. v3 checkpoints preserve mode/map; v1/v2 remain readable with default raw-policy mode.
- Optional extended frozen trial suite compares the five legacy variants plus `masked_policy`, `observed_map`, `map_without_policy`, using identical seeded worlds/horizon. Default legacy config, schema, protocol hashes and replay outputs remain identical. Controller runs add action traces, proposed-wall/override counts, map coverage and decision reasons. Report zero collisions as a consequence of filtering rather than a learned capability.
- Expose modes and extended trials through API, CLI and browser. Show observed-map counts/reasons and teacher bypass clearly. Do not render unknown cells as explored.
- Run focused/full tests, actual matched trials on a captured fixture checkpoint, legacy archive replay, desktop/mobile browser actions and screenshots, independent review, and wheel/source/replay checks.

No automatic promotion or change to the live checkpoint. Better assisted navigation must not be presented as stronger raw neural navigation, source-topology advantage, robot transfer or biological validation. The full-map shortest-path oracle remains a separately labelled privileged control.
