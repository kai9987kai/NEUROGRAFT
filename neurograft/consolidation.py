"""Connectome Latent Consolidation Substrate (CLCS).

Inspired by Supermix-Expanse's cross-source latent consolidation architecture,
this module consolidates heterogeneous multi-species neural representations
(worm, fly hemibrain, fly male-CNS, human cortical) into a shared latent space
using sparse Top-2 SwiGLU experts, learned associative memory, and zero-initialized
highway residual gates that guarantee an exact identity at initialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DEFAULT_SOURCES = ("worm", "fly_f", "fly_m", "human")


def _rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMS normalization along the final dimension."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for the Connectome Latent Consolidation Substrate."""
    hidden_size: int = 8
    shared_dim: int = 32
    latent_experts: int = 4
    latent_rank: int = 16
    latent_top_k: int = 2
    memory_slots: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.shared_dim <= 0:
            raise ValueError("hidden_size and shared_dim must be positive")
        if self.latent_experts <= 0:
            raise ValueError("latent_experts must be positive")
        if not 1 <= self.latent_top_k <= self.latent_experts:
            raise ValueError("latent_top_k must be between 1 and latent_experts")
        if self.latent_rank <= 0 or self.memory_slots < 0:
            raise ValueError("latent_rank must be positive and memory_slots nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SwiGLULatentExpert(nn.Module):
    """Compact shared-space expert with SwiGLU gating."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.up = nn.Linear(dim, rank * 2, bias=False)
        self.down = nn.Linear(rank, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class LatentAssociativeMemory(nn.Module):
    """Learned persistent associative memory slots with query-key attention."""

    def __init__(self, slots: int, dim: int) -> None:
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.memory = nn.Parameter(torch.empty(slots, dim))
        nn.init.normal_(self.memory, std=1.0 / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.scale = 1.0 / math.sqrt(dim)
        self.gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid ~= 0.119 at start

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        # z: (..., dim)
        q = self.query_proj(z)
        # q: (..., dim), memory: (slots, dim) -> attn: (..., slots)
        scores = torch.matmul(q, self.memory.t()) * self.scale
        attn = torch.softmax(scores, dim=-1)
        values = self.value_proj(self.memory)  # (slots, dim)
        mem = torch.matmul(attn, values)
        gated = z + torch.sigmoid(self.gate) * mem
        return gated, attn


class NativeConsolidationBlock(nn.Module):
    """Per-vector shared-latent MoE + learned memory with zero-gate identity bypass.

    At initialization, ``out_gate`` is all zeros, making this block an exact identity.
    Training gradients flow through ``out_gate`` to smoothly engage consolidated capacity.
    """

    def __init__(self, config: ConsolidationConfig) -> None:
        super().__init__()
        self.config = config
        h, d = config.hidden_size, config.shared_dim
        self.to_shared = nn.Linear(h, d, bias=False)
        self.router = nn.Linear(d, config.latent_experts, bias=True)
        self.experts = nn.ModuleList([
            SwiGLULatentExpert(d, config.latent_rank) for _ in range(config.latent_experts)
        ])
        self.expert_scale = nn.Parameter(torch.tensor(0.25))
        self.dropout = nn.Dropout(config.dropout)

        if config.memory_slots > 0:
            self.memory = LatentAssociativeMemory(config.memory_slots, d)
        else:
            self.memory = None

        self.to_hidden = nn.Linear(d, h, bias=False)
        self.out_gate = nn.Parameter(torch.zeros(h))

    def _route(self, z: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.router(z)
        k = self.config.latent_top_k
        if k < logits.shape[-1]:
            topv, topi = logits.topk(k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, topi, topv)
            probs = torch.softmax(masked, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)

        expert_outs = torch.stack([expert(z) for expert in self.experts], dim=-2)
        mixed = torch.sum(probs.unsqueeze(-1) * expert_outs, dim=-2)
        return mixed, probs

    def forward(self, hidden: Tensor) -> tuple[Tensor, dict[str, Any]]:
        # hidden: (batch, hidden_size) or (batch, seq, hidden_size)
        z0 = self.to_shared(_rms_norm(hidden))
        z = _rms_norm(z0)
        mixed_expert, route_probs = self._route(z)
        z = z0 + self.expert_scale.tanh() * self.dropout(mixed_expert)

        memory_attn = None
        if self.memory is not None:
            z, memory_attn = self.memory(z)

        latent = _rms_norm(z)

        # Zero-gate identity check
        gate_zero = bool((self.out_gate.detach() == 0).all())
        needs_grad = self.training and torch.is_grad_enabled() and self.out_gate.requires_grad
        if gate_zero and not needs_grad:
            out = hidden
        else:
            write = self.to_hidden(latent)
            out = hidden + self.out_gate * write

        # Routing diagnostics
        route_mean = route_probs.reshape(-1, route_probs.shape[-1]).mean(dim=0)
        balance = float(self.config.latent_experts) * float(route_mean.pow(2).sum().item())
        entropy = -float((route_probs.clamp_min(1e-9).log() * route_probs).sum(dim=-1).mean().item())

        aux: dict[str, Any] = {
            "latent": latent,
            "route_probs": route_probs,
            "router_balance": balance,
            "router_entropy": round(entropy, 4),
            "gate_abs_mean": float(self.out_gate.detach().abs().mean().item()),
        }
        if memory_attn is not None:
            mem_ent = -float((memory_attn.clamp_min(1e-9).log() * memory_attn).sum(dim=-1).mean().item())
            aux["memory_attn"] = memory_attn
            aux["memory_entropy"] = round(mem_ent, 4)

        return out, aux


class ConnectomeConsolidationStack(nn.Module):
    """Consolidation substrate bridging all 4 source connectomes into unified latent capacity."""

    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ConsolidationConfig()
        self.block = NativeConsolidationBlock(self.config)
        self.context_projection = nn.Linear(4 * self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(self, pooled_sources: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Consolidate pooled representations from 4 source graphs.

        Args:
            pooled_sources: Tensor of shape (batch, 4, hidden_size) or flattened (batch, 4 * hidden_size).

        Returns:
            Tuple of (consolidated_vector of shape (batch, hidden_size), aux telemetry dict).
        """
        if pooled_sources.ndim == 3 and pooled_sources.shape[1] == 4:
            flattened = pooled_sources.flatten(start_dim=1)
        elif pooled_sources.ndim == 2 and pooled_sources.shape[1] == 4 * self.config.hidden_size:
            flattened = pooled_sources
        else:
            raise ValueError(f"expected (batch, 4, {self.config.hidden_size}) or (batch, {4 * self.config.hidden_size})")

        fused_in = self.context_projection(flattened)
        return self.block(fused_in)

    def aux_loss(self, aux: dict[str, Any]) -> Tensor:
        """Compute router load-balancing auxiliary loss."""
        balance = aux.get("router_balance", 1.0)
        p = next(self.parameters())
        # Ideal balance is 1.0 when perfectly uniform
        return p.new_tensor(max(0.0, balance - 1.0))


class TeacherConsolidationFusion(nn.Module):
    """Training-time multi-view InfoNCE alignment and teacher distillation target."""

    def __init__(self, dim: int = 8, shared_dim: int = 32) -> None:
        super().__init__()
        self.source_projectors = nn.ModuleList([
            nn.Linear(dim, shared_dim, bias=False) for _ in range(4)
        ])
        self.query = nn.Linear(shared_dim, shared_dim, bias=False)
        self.key = nn.Linear(shared_dim, shared_dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, sources: Sequence[Tensor], student_latent: Tensor) -> dict[str, Any]:
        """Align heterogeneous sources and compute consensus target.

        Args:
            sources: 4 tensors of shape (batch, dim).
            student_latent: Tensor of shape (batch, shared_dim).

        Returns:
            Dict containing consensus target and alignment loss.
        """
        projected = [
            _rms_norm(proj(src)) for proj, src in zip(self.source_projectors, sources)
        ]
        stack = torch.stack(projected, dim=1)  # (batch, 4, shared_dim)
        student_norm = _rms_norm(student_latent)

        q = self.query(student_norm).unsqueeze(1)  # (batch, 1, shared_dim)
        k = self.key(stack)                        # (batch, 4, shared_dim)
        temp = self.temperature.exp().clamp(0.1, 2.0)
        scores = (q * k).sum(dim=-1) / (math.sqrt(stack.shape[-1]) * temp)
        weights = torch.softmax(scores, dim=-1)    # (batch, 4)
        fused = torch.sum(weights.unsqueeze(-1) * stack, dim=1)  # (batch, shared_dim)

        # Multi-view InfoNCE alignment loss between projected sources
        align_losses = []
        for i in range(4):
            for j in range(i + 1, 4):
                sim = (projected[i] * projected[j]).sum(dim=-1)
                align_losses.append(1.0 - sim.mean())
        align_loss = torch.stack(align_losses).mean() if align_losses else student_latent.new_zeros(())

        return {
            "fused_consensus": _rms_norm(fused),
            "source_weights": weights.detach().cpu().tolist(),
            "alignment_loss": align_loss,
        }
