"""Short-term synaptic plasticity (STP) using Tsodyks-Markram dynamics.

Biological chemical synapses in C. elegans and Drosophila undergo activity-dependent
depression (vesicle depletion, recovery tau_rec) and facilitation (residual calcium,
decay tau_fac). This module implements a vector-sparse, deterministic Tsodyks-Markram
state substrate over biological node populations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class STPConfig:
    """Parameters for Tsodyks-Markram short-term plasticity."""
    dt: float = 0.2  # Step interval in seconds
    u_baseline: float = 0.35  # Baseline release probability U
    tau_rec: float = 0.80  # Vesicle recovery time constant (seconds)
    tau_fac: float = 0.25  # Facilitation decay time constant (seconds)
    facilitation_weight: float = 0.40  # Calcium influx per presynaptic event
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or not math.isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if not 0.0 < self.u_baseline <= 1.0:
            raise ValueError("u_baseline must be in (0, 1]")
        if self.tau_rec <= 0.0 or self.tau_fac <= 0.0:
            raise ValueError("tau_rec and tau_fac must be positive")


@dataclass(frozen=True)
class STPState:
    """Per-source Tsodyks-Markram synaptic state."""
    u: tuple[Tensor, ...]  # Utilization / release probability per node
    r: tuple[Tensor, ...]  # Readily releasable pool fraction per node


class TsodyksMarkramSubstrate(nn.Module):
    """Short-term synaptic depression and facilitation across source connectomes.

    Operates in O(N) node-level state space, where each node's presynaptic output
    is scaled by effective synaptic efficacy E = (u * R) / U_baseline.
    When idle or at low frequency, u -> U_baseline and R -> 1.0, yielding E -> 1.0,
    which exactly preserves baseline network transmission.
    """

    def __init__(
        self,
        node_counts: Sequence[int],
        config: STPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else STPConfig()
        if not node_counts or any(c <= 0 for c in node_counts):
            raise ValueError("node_counts must be a non-empty sequence of positive integers")
        self.node_counts = tuple(node_counts)

        # Decay factors per step
        self.alpha_rec = math.exp(-self.config.dt / self.config.tau_rec)
        self.alpha_fac = math.exp(-self.config.dt / self.config.tau_fac)

    def initial_state(self, batch_size: int = 1, device: torch.device | None = None) -> STPState:
        """Create initial resting state: u = U_baseline, R = 1.0."""
        u_tensors = []
        r_tensors = []
        for count in self.node_counts:
            u_init = torch.full((batch_size, count), self.config.u_baseline, dtype=torch.float32, device=device)
            r_init = torch.ones((batch_size, count), dtype=torch.float32, device=device)
            u_tensors.append(u_init)
            r_tensors.append(r_init)
        return STPState(u=tuple(u_tensors), r=tuple(r_tensors))

    def step(
        self,
        node_activities: Sequence[Tensor],
        state: STPState | None = None,
    ) -> tuple[tuple[Tensor, ...], STPState, dict[str, float]]:
        """Advance Tsodyks-Markram dynamics by one step.

        Parameters
        ----------
        node_activities:
            Tuple of [batch, N_i] or [batch, N_i, hidden] tensors containing node activation magnitudes.
        state:
            Previous STPState. If None, initialized to resting state.

        Returns
        -------
        efficacies:
            Tuple of [batch, N_i, 1] gain factors to scale outgoing messages.
        new_state:
            Updated STPState.
        telemetry:
            Summary metrics (mean availability, mean release prob, mean efficacy).
        """
        if len(node_activities) != len(self.node_counts):
            raise ValueError(f"expected {len(self.node_counts)} source activity tensors")

        batch_size = node_activities[0].shape[0]
        device = node_activities[0].device
        current_state = state if state is not None else self.initial_state(batch_size, device=device)

        new_u = []
        new_r = []
        efficacies = []
        all_u = []
        all_r = []
        all_eff = []

        u_base = self.config.u_baseline
        f_weight = self.config.facilitation_weight

        for idx, (act_tensor, prev_u, prev_r) in enumerate(zip(node_activities, current_state.u, current_state.r)):
            # If node activity has hidden dimension [batch, N, hidden], take norm/abs mean
            if act_tensor.ndim == 3:
                act = act_tensor.abs().mean(dim=-1).clamp(0.0, 1.0)
            elif act_tensor.ndim == 2:
                act = act_tensor.abs().clamp(0.0, 1.0)
            else:
                raise ValueError("node_activities must have shape [batch, N] or [batch, N, hidden]")

            if not self.config.enabled:
                # Identity passthrough
                eff = torch.ones_like(act)[:, :, None]
                efficacies.append(eff)
                new_u.append(prev_u)
                new_r.append(prev_r)
                continue

            # Tsodyks-Markram updates:
            # In the absence of presynaptic spikes, u relaxes toward u_base with time constant tau_fac:
            u_relaxed = u_base + (prev_u - u_base) * self.alpha_fac
            u_next = (u_relaxed + f_weight * (1.0 - u_relaxed) * act).clamp(0.01, 1.0)

            # R_t relaxes toward 1.0 with recovery time constant tau_rec, minus consumed resources:
            used_resources = (u_next * prev_r * act).clamp(0.0, 1.0)
            r_next = (1.0 + (prev_r - used_resources - 1.0) * self.alpha_rec).clamp(0.0, 1.0)

            # Synaptic efficacy E = (u * R) / U_baseline
            eff_scalar = (u_next * r_next) / max(1e-4, u_base)
            # Safe clamping to prevent explosive positive feedback
            eff_clamped = eff_scalar.clamp(0.05, 3.0)

            new_u.append(u_next)
            new_r.append(r_next)
            efficacies.append(eff_clamped[:, :, None])

            all_u.append(float(u_next.mean().item()))
            all_r.append(float(r_next.mean().item()))
            all_eff.append(float(eff_clamped.mean().item()))

        summary = {
            "mean_release_prob": sum(all_u) / max(1, len(all_u)) if all_u else u_base,
            "mean_vesicle_pool": sum(all_r) / max(1, len(all_r)) if all_r else 1.0,
            "mean_synaptic_efficacy": sum(all_eff) / max(1, len(all_eff)) if all_eff else 1.0,
            "depression_fraction": sum(1 for r in all_r if r < 0.90) / max(1, len(all_r)) if all_r else 0.0,
            "facilitation_fraction": sum(1 for u in all_u if u > u_base * 1.1) / max(1, len(all_u)) if all_u else 0.0,
        }

        return tuple(efficacies), STPState(u=tuple(new_u), r=tuple(new_r)), summary
