"""Bioelectric Morphogenetic Substrate (BMS).

Inspired by Morpheus & Memory Carrier Observatory: computes persistent non-spiking
membrane potentials V_mem in [-85mV, -20mV] diffusing across gap junctions.
Acts as a slow morphogenetic memory attractor that coordinates neural excitability
and guides homeostatic recovery after synaptic lesion or perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .graphs import GraphBundle, SourceGraph


@dataclass(frozen=True)
class BioelectricConfig:
    """Parameters for bioelectric membrane potential dynamics."""
    dt: float = 0.20  # Integration time step (seconds)
    v_rest: float = -65.0  # Resting potential (mV)
    v_min: float = -85.0  # Hyperpolarization bound (mV)
    v_max: float = -20.0  # Depolarization bound (mV)
    c_m: float = 1.0  # Membrane capacitance (uF/cm^2)
    g_leak: float = 0.30  # Leak conductance (mS/cm^2)
    g_gap: float = 0.45  # Gap junction coupling conductance
    pump_gain: float = 25.0  # Depolarization pump per unit neural activity

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or not math.isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if not (self.v_min < self.v_rest < self.v_max):
            raise ValueError("must satisfy v_min < v_rest < v_max")
        if self.g_leak <= 0.0 or self.g_gap <= 0.0:
            raise ValueError("conductances must be positive")


class SparseGapJunctionDiffuser(nn.Module):
    """Computes Laplacian diffusion of voltage across gap junctions."""

    def __init__(self, num_nodes: int, src: Sequence[int], dst: Sequence[int], weights: Sequence[float]) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        src_t = torch.tensor(src, dtype=torch.long)
        dst_t = torch.tensor(dst, dtype=torch.long)
        w_t = torch.tensor(weights, dtype=torch.float32).abs()

        self.register_buffer("src", src_t)
        self.register_buffer("dst", dst_t)
        self.register_buffer("weight", w_t)

        deg = torch.zeros(num_nodes, dtype=torch.float32)
        if src_t.numel():
            deg.index_add_(0, dst_t, w_t)
        self.register_buffer("gap_degree", deg)

    def forward(self, v: Tensor) -> Tensor:
        """Compute sum_j g_ij (V_j - V_i) = (A @ V - D * V)."""
        # v has shape [batch, N]
        batch_size = v.shape[0]
        if not self.src.numel():
            return torch.zeros_like(v)

        diffused = v.new_zeros(batch_size, self.num_nodes)
        messages = v.index_select(1, self.src) * self.weight[None, :]
        diffused.index_add_(1, self.dst, messages)
        # Net flux = incoming voltage sum - gap_degree * V_self
        return diffused - self.gap_degree[None, :] * v


class BioelectricMorphogeneticSubstrate(nn.Module):
    """Slow bioelectric potential field across multi-species connectomes."""

    def __init__(self, bundle: GraphBundle, config: BioelectricConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else BioelectricConfig()
        self.source_keys = [graph.manifest.source_id for graph in bundle.graphs]
        self.node_counts = {graph.manifest.source_id: len(graph.nodes) for graph in bundle.graphs}

        # Build sparse gap junction diffusers for each species
        diffusers = {}
        for graph in bundle.graphs:
            src_id = graph.manifest.source_id
            pos = {node.id: i for i, node in enumerate(graph.nodes)}
            src_list, dst_list, w_list = [], [], []
            for edge in graph.edges:
                if edge.relation == "electrical":
                    s_idx = pos[edge.src]
                    d_idx = pos[edge.dst]
                    w = float(edge.weight)
                    src_list.append(s_idx)
                    dst_list.append(d_idx)
                    w_list.append(w)
                    if not edge.directed and s_idx != d_idx:
                        src_list.append(d_idx)
                        dst_list.append(s_idx)
                        w_list.append(w)
            diffusers[src_id] = SparseGapJunctionDiffuser(len(graph.nodes), src_list, dst_list, w_list)
        self.diffusers = nn.ModuleDict(diffusers)

    def initial_voltage(self, batch_size: int = 1, device: torch.device | None = None) -> dict[str, Tensor]:
        """Initialize resting membrane potentials V_rest (-65 mV)."""
        return {
            src: torch.full((batch_size, count), self.config.v_rest, dtype=torch.float32, device=device)
            for src, count in self.node_counts.items()
        }

    def step(
        self,
        voltages: dict[str, Tensor],
        neural_activities: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], dict[str, object]]:
        """Advance bioelectric field by one integration time step.

        dV/dt = (1/C_m) * [ g_gap * Diff(V) - g_leak * (V - V_rest) + pump_gain * Act ]
        """
        new_v = {}
        telemetry = {}
        dt = self.config.dt
        cm = self.config.c_m
        gleak = self.config.g_leak
        ggap = self.config.g_gap
        vrest = self.config.v_rest
        pump = self.config.pump_gain

        all_means = []

        for src in self.source_keys:
            v_curr = voltages[src]
            act = neural_activities[src]
            if act.ndim == 3:
                act = act.abs().mean(dim=-1)

            diffuser = self.diffusers[src]
            gap_flux = diffuser(v_curr) * ggap
            leak_flux = -gleak * (v_curr - vrest)
            pump_flux = pump * act.clamp(0.0, 1.0)

            dv_dt = (gap_flux + leak_flux + pump_flux) / cm
            v_next = (v_curr + dv_dt * dt).clamp(self.config.v_min, self.config.v_max)
            new_v[src] = v_next

            mean_v = float(v_next.mean().item())
            all_means.append(mean_v)
            var_v = float(v_next.var().item()) if v_next.shape[-1] > 1 else 0.0

            telemetry[src] = {
                "mean_v_mv": round(mean_v, 2),
                "var_v": round(var_v, 4),
                "min_v_mv": round(float(v_next.min().item()), 2),
                "max_v_mv": round(float(v_next.max().item()), 2),
            }

        # Excitability modulation gain: depolarized nodes fire more easily
        global_mean_v = sum(all_means) / max(1, len(all_means))
        depolarization_shift = max(0.0, global_mean_v - vrest)
        excitability_gain = round(1.0 + 0.05 * depolarization_shift, 3)

        summary = {
            "species_potentials": telemetry,
            "global_mean_mv": round(global_mean_v, 2),
            "resting_baseline_mv": vrest,
            "excitability_gain": excitability_gain,
            "depolarized_ratio": round(sum(1 for m in all_means if m > vrest + 2.0) / max(1, len(all_means)), 2),
        }

        return new_v, summary
