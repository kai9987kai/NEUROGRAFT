# v0.11: predictive recovery learning and experienced-transition memory

21 October 2026. This design advances the existing draft while preserving the v0.10 study, source graphs and live checkpoint. The question is whether a small action-conditioned prediction objective helps raw learned navigation, particularly when collection includes learner mistakes. Every mechanism remains a synthetic experiment.

## Primary research reviewed

| Source | Transferable principle | Boundary |
| --- | --- | --- |
| He, Daigle and Bashivan, [Building spatial world models from sparse transitional episodic memories](https://proceedings.iclr.cc/paper_files/paper/2026/hash/23c2c77c921111ec8902ac482d92b289-Abstract-Conference.html), ICLR 2026 | Evaluate transition knowledge obtained from experience and its spatial utility. | This coordinate/count graph stores experienced edges; it does not infer unseen transitions or reproduce the learned ESWM. |
| Jain et al., [A Smooth Sea Never Made a Skilled SAILOR](https://papers.nips.cc/paper_files/paper/2025/hash/5a966a8d451fae43806ac3c22d13c3ad-Abstract-Conference.html), NeurIPS 2025 | Recovery beyond demonstrations needs explicit evaluation; prediction should be assessed through behavioral utility. | This auxiliary predictor and imitation policy have no learned reward model or test-time search and do not reproduce SAILOR. |
| Hafner et al., [Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2), Nature, 2 April 2025 | Action-conditioned prediction can support representations and behavior. | One-step observation prediction supplies neither imagined rollouts nor DreamerV3 behavior learning. The original [preprint](https://arxiv.org/abs/2301.04104) was submitted in 2023. |
| Ross et al., [DAgger](https://proceedings.mlr.press/v15/ross11a.html), 2011 | Label learner-visited states to address induced distribution shift. | Recovery here is a fixed 50% teacher/50% learner mixture. [DART](https://proceedings.mlr.press/v78/laskey17a.html) optimizes demonstration noise; this is not DART. |
| Westeinde et al., [Transforming a head direction signal into a goal-oriented steering command](https://www.nature.com/articles/s41586-024-07039-2), and Mussells Pires et al., [Converting an allocentric goal into an egocentric steering signal](https://www.nature.com/articles/s41586-023-07006-3), Nature, 7 February 2024 | Heading and goal signals contribute to steering, including opponent PFL3 signals and gain modulation. | The retained CXPFLSteering name describes an engineered cosine prior. Its 0.65/0.35 mixture is a design choice, not recovered biology or demonstrated PFL1 computation. It reads heading directly from observations. |

These sources were selected for relevance and inspectable mechanisms, rather than a claim of universally best research.

## Prediction and recovery

An optional head maps the four-source pooled context and **executed action** to the next 4-by-4 observation. Its zero-initialized output layer preserves common initial parameters, RNG state and policy logits. Prediction does not directly select an inference action.

The auxiliary MSE and expert-action cross entropy are logged separately, together with their weighted total. Executed actions remain distinct from teacher labels. Quarter-turn augmentation transforms observation history, teacher action, executed action and next observation together, retaining zero padding. Goal completion changes the goal; timeout can teleport the agent. Those boundary targets are excluded. An all-boundary minibatch records no prediction MSE rather than artificial zero error.

Recovery collects teacher labels at states reached by a fixed mixture of learner and teacher actions. Existing mixed/expert collection retains its historical RNG draw and data/minibatch identities. Checkpoint initialization requires matching readout and predictor architecture. Candidate exports remain paused and do not resume the curriculum optimizer or automatically replace live state.

## Controlled pilot

Four predeclared arms compare expert/recovery collection and auxiliary weight 0/0.5. All include the same predictor architecture, providing a parameter-matched objective control. One collection round precedes optimization, so trajectories, replay and raw minibatches match across auxiliary weights within each collection-policy/seed pair. Expert and recovery arms intentionally visit different states and report teacher/learner exposure separately.

The bounded pilot uses two initialization/training seeds, 24 worlds spanning sizes 7 and 11 and six presets, and 256 optimizer updates per run. Validation-only selection compares round zero and one; every arm and seed is retained. Eight frozen test scenarios receive zero teacher actions and no controller for the raw-policy comparison. Pooling across initialization seeds is descriptive, not independent replication.

Separate held-out random-action probes report prediction MSE by position, heading, goal, walls and reward/clock channels. Copy-current and explicit local kinematics baselines expose gains attributable to easily copied or deterministic features. Terminal observations are excluded. These probes do not select checkpoints, loss weights or a winning test arm.

## Memory and steering assistance

The directed transition graph records explicit pre-action position, executed action and post-action position, including collisions. Timeout teleports and unrelated arenas never become edges. Coordinates, action geometry, counts and rewards are bounded and validated. Canonical graph state is optional checkpoint auxiliary state; older checkpoints restore with an empty graph. It remains an observer and does not feed policy logits. BFS over experienced edges is not a learned successor representation.

The optional cx_steer mode combines a synthetic goal/heading prior with neural logits and legal local actions, bypassing teacher actions. The frozen cx_without_policy control removes learned logits. These measure engineered assistance separately from raw learned success. Steering comparisons are opt-in; old schema 1/2/3 protocols retain exact defaults.

## Verification and limits

Failing-first regressions cover strict predictor inputs, gradients, legacy RNG streams, rotated targets, terminal masking, mode save/load, bounded graph continuation and archive compatibility. Final checks include the complete suite, desktop/mobile browser flows, the declared pilot, prediction baselines, frozen replay and source/package parity. Measured results are recorded after execution in the build report.

Fixture graphs, short histories, privileged labels, a small test set and limited updates constrain interpretation. Better one-step prediction need not improve navigation. Masking, steering and mapping gains belong to their engineered mechanisms. These experiments establish no biological-connectome advantage, emergent cognitive map, pretrained language capability, robot transfer or independent replication.
