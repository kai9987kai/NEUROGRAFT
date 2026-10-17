"""Chebyshev Polynomial Spectral Graph Diffusion Substrate.

Implements multi-scale spectral graph wavelets and heat diffusion over
biological connectome graphs via Chebyshev polynomial expansion of the
normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}. Runs in O(K * |E|)
sparse operations without dense matrix inversion or eigen-decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .graphs import SourceGraph


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for Chebyshev spectral graph diffusion."""
    order: int = 3  # Chebyshev polynomial order K
    scales: tuple[float, ...] = (0.5, 2.0, 6.0)  # Multi-scale diffusion parameters (local, meso, macro)

    def __post_init__(self) -> None:
        if self.order < 1 or self.order > 8:
            raise ValueError("order must be between 1 and 8")
        if not self.scales or any(s <= 0.0 for s in self.scales):
            raise ValueError("scales must be positive finite values")


class SparseNormalizedLaplacian(nn.Module):
    """Sparse normalized graph Laplacian L_sym = I - D^{-1/2} A D^{-1/2}."""

    def __init__(self, num_nodes: int, src: Sequence[int], dst: Sequence[int], weights: Sequence[float]) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        src_t = torch.tensor(src, dtype=torch.long)
        dst_t = torch.tensor(dst, dtype=torch.long)
        w_t = torch.tensor(weights, dtype=torch.float32).abs()

        # Compute degree D_i = sum_j |A_ij|
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        if src_t.numel():
            deg.index_add_(0, dst_t, w_t)
        deg_inv_sqrt = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))

        # Normalized edge weights: A_norm[dst, src] = w / (sqrt(deg[dst]) * sqrt(deg[src]))
        if src_t.numel():
            norm_w = w_t * deg_inv_sqrt[dst_t] * deg_inv_sqrt[src_t]
        else:
            norm_w = w_t

        self.register_buffer("src", src_t)
        self.register_buffer("dst", dst_t)
        self.register_buffer("norm_weight", norm_w)
        self.register_buffer("has_degree", deg > 0)

    def multiply_laplacian(self, x: Tensor) -> Tensor:
        """Compute L_sym @ x = (I - D^{-1/2} A D^{-1/2}) @ x."""
        # x has shape [batch, N, hidden] or [batch, N]
        batch_size = x.shape[0]
        if not self.src.numel():
            return torch.zeros_like(x)

        if x.ndim == 3:
            hidden = x.shape[-1]
            ax = x.new_zeros(batch_size, self.num_nodes, hidden)
            messages = x.index_select(1, self.src) * self.norm_weight[None, :, None]
            ax.index_add_(1, self.dst, messages)
            # L_sym @ x = x - A_norm @ x (for nodes with connections)
            mask = self.has_degree[None, :, None].to(x.dtype)
            return x * mask - ax
        else:
            ax = x.new_zeros(batch_size, self.num_nodes)
            messages = x.index_select(1, self.src) * self.norm_weight[None, :]
            ax.index_add_(1, self.dst, messages)
            mask = self.has_degree[None, :].to(x.dtype)
            return x * mask - ax

    def multiply_scaled_laplacian(self, x: Tensor) -> Tensor:
        """Compute rescaled operator L_tilde @ x = (L_sym - I) @ x.

        Assuming lambda_max ~ 2.0, rescaled L_tilde in [-I, I].
        """
        # L_tilde = L_sym - I
        return self.multiply_laplacian(x) - x


def _chebyshev_coefficients(scale: float, order: int) -> list[float]:
    """Compute Chebyshev expansion coefficients for heat kernel exp(-s * (theta + 1)).

    Using Gauss-Chebyshev numerical quadrature.
    """
    quad_points = 32
    coeffs = []
    for k in range(order + 1):
        total = 0.0
        for p in range(quad_points):
            # theta in (-1, 1)
            theta_p = math.cos(math.pi * (p + 0.5) / quad_points)
            # lambda = theta + 1 in [0, 2]
            heat = math.exp(-scale * (theta_p + 1.0))
            t_k = math.cos(k * math.acos(theta_p))
            total += heat * t_k
        c_k = (2.0 / quad_points) * total
        if k == 0:
            c_k *= 0.5
        coeffs.append(c_k)
    return coeffs


class ChebyshevSpectralDiffusion(nn.Module):
    """Multi-scale Chebyshev polynomial spectral diffusion filter."""

    def __init__(self, graph: SourceGraph, config: SpectralConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else SpectralConfig()
        num_nodes = len(graph.nodes)
        node_pos = {node.id: i for i, node in enumerate(graph.nodes)}
        src, dst, weights = [], [], []

        for edge in graph.edges:
            s_idx = node_pos[edge.src]
            d_idx = node_pos[edge.dst]
            w = float(edge.weight)
            src.append(s_idx)
            dst.append(d_idx)
            weights.append(w)
            if not edge.directed and s_idx != d_idx:
                src.append(d_idx)
                dst.append(s_idx)
                weights.append(w)

        self.laplacian = SparseNormalizedLaplacian(num_nodes, src, dst, weights)
        self.num_nodes = num_nodes

        # Precompute Chebyshev coefficients for each scale
        self.coeffs = {
            scale: _chebyshev_coefficients(scale, self.config.order)
            for scale in self.config.scales
        }

    def diffuse(self, x: Tensor, scale: float = 1.0) -> Tensor:
        """Apply heat diffusion exp(-s * L_sym) to node signal x via Chebyshev polynomials.

        Parameters
        ----------
        x:
            Node activity tensor [batch, N, hidden] or [batch, N].
        scale:
            Diffusion scale s > 0.
        """
        coeffs = _chebyshev_coefficients(scale, self.config.order)

        # T_0(L_tilde) @ x = x
        t_0 = x
        result = coeffs[0] * t_0

        if self.config.order >= 1:
            # T_1(L_tilde) @ x = L_tilde @ x
            t_1 = self.laplacian.multiply_scaled_laplacian(x)
            result = result + coeffs[1] * t_1

            # Recurrence: T_k = 2 * L_tilde @ T_{k-1} - T_{k-2}
            t_prev2 = t_0
            t_prev1 = t_1
            for k in range(2, self.config.order + 1):
                t_k = 2.0 * self.laplacian.multiply_scaled_laplacian(t_prev1) - t_prev2
                result = result + coeffs[k] * t_k
                t_prev2 = t_prev1
                t_prev1 = t_k

        return result

    def multi_scale_energies(self, x: Tensor) -> dict[str, float]:
        """Compute diffusion energy ratios across configured scales (local, meso, macro)."""
        energies = {}
        base_norm = float(x.detach().norm().item()) + 1e-6
        for scale in self.config.scales:
            diffused = self.diffuse(x, scale=scale)
            diff_norm = float(diffused.detach().norm().item())
            energies[f"scale_{scale:.1f}"] = round(diff_norm / base_norm, 4)
        return energies
