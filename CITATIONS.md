# Sources and attribution

The data-source list is not a statement that every dataset has been acquired.

| Source | Pinned or proposed scope | Primary reference |
|---|---|---|
| Worm | Acquired CElegansNeuroML commit `1ca08b14f4e795a973ce447565e0c97b897ee4dc`; original edge CSV and CellsOnly population network retained | [OpenWorm CElegansNeuroML](https://github.com/openworm/CElegansNeuroML), [c302](https://github.com/openworm/c302) |
| Female fly | Proposed FAFB FlyWire v783, October 2023 snapshot; no authentic import yet | [FlyWire guidelines](https://join.flywire.ai/guidelines), [Codex FAQ](https://codex.flywire.ai/faq), [female-brain publication](https://www.nature.com/articles/s41586-024-07558-y) |
| Male fly | Proposed `male-cns:v1.0`, released 8 June 2026; no authentic import yet | [MaleCNS release notes](https://male-cns.janelia.org/release/), [downloads](https://male-cns.janelia.org/download/) |
| Human | Proposed authorized HCP-YA 2025 parcel network; no authentic import yet | [HCP-YA 2025 release](https://humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release), [data-use terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms) |
| Qwen | Requested pretrained Qwen3.5-0.8B; interface validated with small random configurations, no pretrained execution | [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Transformers implementation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) |
| Qwen alternative | Explicitly selectable text-only Qwen3-0.6B; small random interface tested | [model card](https://huggingface.co/Qwen/Qwen3-0.6B), [Transformers implementation](https://huggingface.co/docs/transformers/model_doc/qwen3) |

Related motivation: [connectome-constrained fly visual networks](https://www.nature.com/articles/s41586-024-07939-3) and [NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y). These papers do not validate NEUROGRAFT's synthetic cross-species mapping. Dataset specimens, counts, processing and granularity differ; preserve their own definitions.

## Replay learning research reviewed on 28 October 2026

| Primary source | What it informs here | Boundary |
|---|---|---|
| He, Daigle and Bashivan, [Building spatial world models from sparse transitional episodic memories](https://proceedings.iclr.cc/paper_files/paper/2026/hash/23c2c77c921111ec8902ac482d92b289-Abstract-Conference.html), ICLR 2026 | Separate empirical memories from their planning consequences; compare partial and full exposure | ESWM predicts unseen transitions; our replay planner uses executed transitions only |
| Yu, Ji, Ormond, O'Keefe and Burgess, [Hippocampal theta sweeps indicate goal direction during navigation](https://www.nature.com/articles/s41593-026-02365-2), 1 July 2026 | Goal-conditioned decision and goal-relocation assays | Correlations in recorded neural sequences do not validate our tabular planner or biological equivalence |
| Antonov and Dayan, [Exploring replay](https://www.nature.com/articles/s41467-025-56731-y), 15 February 2025 | Replay during learning; corrections after unexpected evidence | Our residual queue does not implement their Bayesian belief-space planning |
| Bakermans et al., [Constructing future behavior through composition and replay](https://www.nature.com/articles/s41593-025-01908-3), 10 March 2025 | Distinguish behavioral consequences from diagnostic replay | We do not reproduce compositional hippocampal representations |
| Mattar and Daw, [Prioritized memory access explains planning and hippocampal replay](https://www.nature.com/articles/s41593-018-0232-z), 22 October 2018 | Compare replay schedules with behavioral endpoints | Residual-only priority is established engineering, distinct from need-times-gain prioritization |


## Transition revision and live experiments (v0.14)

- Duan et al., [Sample Efficient Experience Replay in Non-stationary Environments](https://arxiv.org/abs/2509.15032), preprint, 18 September 2025. Motivates separating environmental change from replay scheduling; the recent-count estimator does not implement DEER.
- Palafox and Fridovich-Keil, [Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning](https://arxiv.org/abs/2609.12278v2), preprint revised 21 September 2026. Motivates explicit fresh evidence and model revision; no hypernetwork or neural adaptation method is reproduced here.
- Zhang et al., [World-In-World: World Models in a Closed-Loop World](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html), ICLR 2026. Motivates evaluating decision utility separately from model/planning proxy scores. Our finite analytic policy assay is not this benchmark.
- [ESWM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/23c2c77c921111ec8902ac482d92b289-Abstract-Conference.html) supplies episodic-memory motivation; captured aggregate counts cannot reconstruct temporal episodes or unseen spatial transitions.
