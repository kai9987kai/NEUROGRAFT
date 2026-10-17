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
