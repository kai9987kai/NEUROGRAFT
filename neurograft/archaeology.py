"""Adaptive Memory Archaeology & Tissue Resection Assays.

Inspired by Mnemorph, MOLT, TESSERA & Archimedes:
Assesses the structural resilience, engram preservation, and homeostatic recovery
of multi-species connectomes under acute virtual tissue resection and synaptic
ablation. Computes cross-species opcode conservation indices and Memory Trace
Preservation Ratios (MTPR) under bioelectric rescue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .graphs import GraphBundle, SourceGraph


@dataclass(frozen=True)
class ResectionSpec:
    """Specification of virtual surgical lesion or tissue resection."""
    target_species: str = "fly_f"  # 'worm', 'fly_f', 'fly_m', 'human', or 'all'
    mode: str = "rich_club_hubs"  # 'rich_club_hubs', 'random_nodes', 'bridge_edges'
    fraction: float = 0.30


@dataclass
class ArchaeologyReport:
    """Quantitative diagnostics from an archaeological tissue resection assay."""
    target_species: str
    resection_mode: str
    intact_logits_entropy: float
    lesioned_fidelity: float
    bioelectric_rescued_fidelity: float
    mtpr: float  # Memory Trace Preservation Ratio in [0.0, 1.0]
    rescue_delta: float
    regime: str  # 'engram-preserved', 'behaviorally-compensated', 'irreversible-loss'
    species_resilience: dict[str, float]
    cross_species_motif_conservation: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_species": self.target_species,
            "resection_mode": self.resection_mode,
            "intact_entropy": round(self.intact_logits_entropy, 4),
            "lesioned_fidelity": round(self.lesioned_fidelity, 4),
            "rescued_fidelity": round(self.bioelectric_rescued_fidelity, 4),
            "mtpr": round(self.mtpr, 4),
            "rescue_delta": round(self.rescue_delta, 4),
            "regime": self.regime,
            "species_resilience": {k: round(v, 4) for k, v in self.species_resilience.items()},
            "cross_species_motif_conservation": {
                k: round(v, 4) for k, v in self.cross_species_motif_conservation.items()
            },
        }


class MemoryArchaeologyAssay:
    """Runs virtual resection assays and evaluates cross-species opcode conservation."""

    def __init__(self, bundle: GraphBundle) -> None:
        self.bundle = bundle
        self.source_keys = [g.manifest.source_id for g in bundle.graphs]
        self._motif_conservation = self._index_motif_conservation()
        self._last_report: ArchaeologyReport | None = None

    def _index_motif_conservation(self) -> dict[str, float]:
        """Compute structural opcode / motif conservation across species."""
        conservation: dict[str, float] = {}
        for graph in self.bundle.graphs:
            src = graph.manifest.source_id
            num_nodes = max(1, len(graph.nodes))
            num_edges = len(graph.edges)
            density = num_edges / max(1.0, float(num_nodes * (num_nodes - 1)))
            
            # Reciprocity / bidirectional edge count
            edge_pairs = set()
            reciprocal = 0
            for e in graph.edges:
                pair = (e.src, e.dst)
                rev = (e.dst, e.src)
                if rev in edge_pairs:
                    reciprocal += 1
                edge_pairs.add(pair)
            reciprocity = (reciprocal * 2) / max(1.0, float(num_edges))

            # Electrical gap junction fraction (opcode coordination)
            gap_count = sum(1 for e in graph.edges if e.relation == "electrical")
            gap_fraction = gap_count / max(1.0, float(num_edges))

            # Composite conservation index
            score = 0.40 * min(1.0, density * 50.0) + 0.35 * reciprocity + 0.25 * gap_fraction
            conservation[src] = round(float(score), 4)

        return conservation

    def run_resection_assay(
        self,
        model: nn.Module,
        inputs: Tensor,
        spec: ResectionSpec | None = None,
        bioelectric_gain: float = 1.05,
    ) -> ArchaeologyReport:
        """Execute a tissue resection assay comparing intact, lesioned, and rescued states."""
        if spec is None:
            spec = ResectionSpec()

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(2).expand(-1, -1, 4, -1)

        model.eval()
        with torch.no_grad():
            # 1. Intact baseline forward pass
            intact_out = model(inputs)
            intact_logits = intact_out.logits.detach()
            intact_probs = F.softmax(intact_logits, dim=-1)
            intact_entropy = float(
                -(intact_probs * torch.log(intact_probs.clamp(min=1e-9))).sum(dim=-1).mean().item()
            )

            # 2. Resected forward pass (virtual ablation)
            source_idx = None
            if spec.target_species in self.source_keys:
                source_idx = self.source_keys.index(spec.target_species)
            
            lesioned_out = model(inputs, ablate_source=source_idx)
            lesioned_logits = lesioned_out.logits.detach()
            lesioned_probs = F.softmax(lesioned_logits, dim=-1)

            # Compute fidelity as cosine similarity of probability distributions
            cos_sim_lesioned = float(
                F.cosine_similarity(intact_probs, lesioned_probs, dim=-1).mean().item()
            )
            lesioned_fidelity = max(0.0, min(1.0, cos_sim_lesioned))

            # 3. Bioelectrically rescued policy
            # Depolarizing bioelectric gain sharpens rescued logits toward attractor state
            rescued_logits = lesioned_logits * bioelectric_gain
            rescued_probs = F.softmax(rescued_logits, dim=-1)
            cos_sim_rescued = float(
                F.cosine_similarity(intact_probs, rescued_probs, dim=-1).mean().item()
            )
            rescued_fidelity = max(0.0, min(1.0, cos_sim_rescued))

            # Memory Trace Preservation Ratio (MTPR)
            rescue_delta = rescued_fidelity - lesioned_fidelity
            mtpr = (rescued_fidelity + lesioned_fidelity) / 2.0

            # Species resilience scores
            species_resilience: dict[str, float] = {}
            for idx, src in enumerate(self.source_keys):
                ablated = model(inputs, ablate_source=idx).logits.detach()
                sim = float(
                    F.cosine_similarity(intact_probs, F.softmax(ablated, dim=-1), dim=-1).mean().item()
                )
                species_resilience[src] = max(0.0, min(1.0, sim))

        # Classify archaeological regime
        if mtpr >= 0.85:
            regime = "engram-preserved"
        elif mtpr >= 0.65 or rescue_delta > 0.01:
            regime = "behaviorally-compensated"
        else:
            regime = "irreversible-loss"

        report = ArchaeologyReport(
            target_species=spec.target_species,
            resection_mode=spec.mode,
            intact_logits_entropy=intact_entropy,
            lesioned_fidelity=lesioned_fidelity,
            bioelectric_rescued_fidelity=rescued_fidelity,
            mtpr=mtpr,
            rescue_delta=rescue_delta,
            regime=regime,
            species_resilience=species_resilience,
            cross_species_motif_conservation=self._motif_conservation,
        )
        self._last_report = report
        return report

    def latest_summary(self) -> dict[str, Any]:
        """Return dict summary for telemetry."""
        if self._last_report is not None:
            return self._last_report.to_dict()
        return {
            "target_species": "none",
            "resection_mode": "none",
            "intact_entropy": 1.386,
            "lesioned_fidelity": 1.0,
            "rescued_fidelity": 1.0,
            "mtpr": 1.0,
            "rescue_delta": 0.0,
            "regime": "engram-preserved",
            "species_resilience": {k: 1.0 for k in self.source_keys},
            "cross_species_motif_conservation": self._motif_conservation,
        }
