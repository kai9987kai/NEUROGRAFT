# v0.10: matched learned-navigation experiments

The requested outcome is a research-informed improvement to NEUROGRAFT that runs on this PC and can be measured. v0.9 improved navigation through an explicit observed-map planner, while its raw circuit policy remained weak. This iteration targets the learner and preserves the live checkpoint, source graphs, prior artifacts and default architecture.

## Research reviewed on 20 October 2026

| Primary source | Relevant idea | Application and boundary |
| --- | --- | --- |
| Zhao et al., [E(2)-Equivariant Graph Planning for Navigation](https://arxiv.org/abs/2309.13043v2), RA-L 2024 | Exploit geometric symmetry to share learning across reference frames. | Rotate training observations, headings, walls and action labels together. This augmentation does not enforce the paper's architectural equivariance. |
| Wang et al., [A Practical Guide for Incorporating Symmetry in Diffusion Policy](https://arxiv.org/abs/2505.13431), NeurIPS 2025 | Practical representation and feature choices can incorporate symmetry without a complete policy redesign. | Use a small quarter-turn transformation and test its effect separately. NEUROGRAFT does not implement diffusion policies or reproduce their experiments. |
| Cai et al., [Provable Partially Observable Reinforcement Learning with Privileged Information](https://arxiv.org/html/2412.00985v3), NeurIPS 2024, revised February 2025 | A privileged teacher can require information the student cannot infer. Expert distillation needs additional conditions for its guarantees. | Keep hidden-map supervision explicit. A richer readout does not reveal unobserved obstacles or recover omitted history. |
| Li et al., [From Action Labels to Sets](https://arxiv.org/html/2502.07645v3), April 2026 revision | Set-valued action supervision can avoid treating one action as the only acceptable target. | Deferred: the current fixed-order shortest-path labels retain tie bias. A discrete shortest-action-set loss would be our adaptation, not a reproduction of this continuous corrective-feedback method. |
| Ross et al., [DAgger](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011 | Learner-visited data addresses induced distribution shift. | Existing mixed collection remains available. Expert-only collection is added for controlled comparisons; its results cannot establish robustness to learner-induced distribution shift. |

These sources were selected for relevance and inspectable mechanisms, including recent work, rather than a claim that one method is universally best.

## Selected change

1. An optional `context_residual` readout adds a zero-initialized linear head over all four pooled source representations to the original human-source logits. Default `human` initialization, parameter keys and behavior remain compatible. This is an engineering hypothesis motivated by the code's restricted cue access, not a mechanism validated by the papers above.
2. Optional clockwise quarter-turn augmentation transforms coordinates, displacements, headings, walls and labels. All-zero left-padding frames remain zero. Evaluation remains on untransformed observations. The transform preserves a labelled route; fixed N/E/S/W oracle tie-breaking need not commute with rotation.
3. Expert-only collection supports four predeclared arms: each readout with and without augmentation. Within each initialization replicate they share initial common weights, original collected samples, base minibatches, optimizer count and evaluation seeds. Extra residual parameters are recorded as a capacity difference. Independent augmentation RNG never changes sampling or collection RNG.
4. A `no_context` inference intervention disables the extra head. `no_graft` still leaves all-source readout access intact, so it must not be interpreted as a human-source-only policy in the residual architecture. Source lesions in the residual architecture remove that source before routing and readout.
5. Every arm exports a paused candidate chosen only by its validation scores. Test results are reported for all arms and seeds; no test-based winner is promoted. Existing archive verification and optional exact frozen replay are reused. Data/batch hashes establish recorded matching, not authenticity or proof that training occurred.

## Validation

Failing-first tests cover geometry, padding, matched sample exposure, zero-initialized equivalence, gradients, intervention semantics, legacy checkpoints, exact continuation, frozen state/RNG preservation, archived identities and aggregate verification. The final checks include the full suite, an actual four-arm multi-seed training experiment, archived replay, browser/report inspection and wheel/source parity. Results and limitations belong in `v10-build-report.md` after execution.

No inference controller, privileged oracle or legal-action mask is permitted in the raw learned-policy comparison. Fixture topology and short windows remain explicit limitations. Synthetic navigation gains would not establish biological topology advantage, pretrained language capability or robot transfer.
