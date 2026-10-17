"""Connectome Byte-Level Semantic Guidance Bridge (Supermix-Expanse transfer).

Bridges the 4-source biological connectome pooled latent representation
into 256-channel byte likelihood distributions and semantic attractor tokens,
enabling direct neural state decoding without external vocabulary mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn


SEMANTIC_ANCHOR_PHRASES = [
    "foraging-chemotaxis",
    "choke-point-maneuver",
    "compass-heading-locked",
    "goal-exploitation",
    "arousal-exploration",
    "separatrix-recovery",
    "associative-recall",
    "synaptic-depression-active",
]


@dataclass(frozen=True)
class ByteBridgeConfig:
    """Configuration for connectome byte-level semantic bridge."""
    context_dim: int = 48  # 4 sources * 12 hidden_dim
    byte_vocab_size: int = 256  # 256 ASCII / UTF-8 byte tokens
    temperature: float = 0.80

    def __post_init__(self) -> None:
        if self.context_dim <= 0 or self.byte_vocab_size <= 0:
            raise ValueError("context_dim and byte_vocab_size must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


class ConnectomeByteBridge(nn.Module):
    """Linear-bottleneck projection from connectome context to byte channels."""

    def __init__(self, config: ByteBridgeConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ByteBridgeConfig()
        mid_dim = max(32, self.config.context_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.config.context_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, self.config.byte_vocab_size, bias=False),
        )

        # Pre-encode semantic anchor phrase byte frequencies for similarity classification
        anchor_tensors = []
        for phrase in SEMANTIC_ANCHOR_PHRASES:
            counts = torch.zeros(self.config.byte_vocab_size, dtype=torch.float32)
            raw_bytes = phrase.encode("utf-8")
            for b in raw_bytes:
                counts[b] += 1.0
            norm = counts.norm().clamp_min(1e-6)
            anchor_tensors.append(counts / norm)
        self.register_buffer("anchor_profiles", torch.stack(anchor_tensors, dim=0))

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Compute byte logits and byte probabilities from pooled context.

        Parameters
        ----------
        context:
            Tensor of shape [batch, context_dim].

        Returns
        -------
        logits:
            [batch, 256] byte logits.
        probs:
            [batch, 256] byte distribution.
        """
        if context.ndim != 2:
            raise ValueError("context must be a 2D tensor [batch, dims]")
        if context.shape[-1] != self.config.context_dim:
            if context.shape[-1] < self.config.context_dim:
                pad_size = self.config.context_dim - context.shape[-1]
                context = torch.cat((context, context.new_zeros(context.shape[0], pad_size)), dim=-1)
            else:
                context = context[:, :self.config.context_dim]
        logits = self.encoder(context)
        probs = torch.softmax(logits / self.config.temperature, dim=-1)
        return logits, probs

    def decode_semantic_intent(self, context: Tensor) -> dict[str, object]:
        """Identify the closest semantic biological attractor and compute byte entropy."""
        with torch.no_grad():
            _, probs = self.forward(context)
            mean_probs = probs.mean(dim=0)
            entropy = float(-torch.sum(mean_probs * torch.log(mean_probs.clamp_min(1e-9))).item())

            # Dot-product similarity against semantic anchor byte profiles
            profile_sims = torch.matmul(self.anchor_profiles, mean_probs)
            best_idx = int(torch.argmax(profile_sims).item())
            best_phrase = SEMANTIC_ANCHOR_PHRASES[best_idx]
            match_score = float(profile_sims[best_idx].item())

            # Top active byte characters
            top_bytes = torch.topk(mean_probs, k=6)
            top_chars = [
                chr(int(idx)) if 32 <= int(idx) <= 126 else f"\\x{int(idx):02x}"
                for idx in top_bytes.indices
            ]

            return {
                "semantic_intent": best_phrase,
                "confidence": round(match_score, 4),
                "byte_entropy": round(entropy, 4),
                "top_byte_chars": top_chars,
                "vocab_spread": round(float((mean_probs > (1.0 / 256.0)).sum().item()) / 256.0, 3),
            }
