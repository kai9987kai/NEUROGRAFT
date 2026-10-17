"""Tests for Chebyshev Polynomial Spectral Graph Wavelet and Diffusion Substrate."""

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle
from neurograft.spectral_diffusion import ChebyshevSpectralDiffusion, SpectralConfig


def test_spectral_config_validation():
    with pytest.raises(ValueError, match="order"):
        SpectralConfig(order=0)
    with pytest.raises(ValueError, match="order"):
        SpectralConfig(order=9)
    with pytest.raises(ValueError, match="scales"):
        SpectralConfig(scales=(-1.0,))


def test_chebyshev_diffusion_energy_contraction():
    bundle = make_fixture_bundle(seed=11, nodes_per_source=8)
    graph = bundle.graphs[0]
    config = SpectralConfig(order=3, scales=(0.5, 2.0, 6.0))
    diffuser = ChebyshevSpectralDiffusion(graph, config)

    # Unit impulse on single node
    batch_size = 2
    num_nodes = len(graph.nodes)
    hidden_dim = 6
    x = torch.zeros(batch_size, num_nodes, hidden_dim)
    x[:, 0, :] = 1.0

    energies = diffuser.multi_scale_energies(x)
    assert "scale_0.5" in energies
    assert "scale_2.0" in energies
    assert "scale_6.0" in energies

    # Heat diffusion on graph is contractive
    assert energies["scale_0.5"] <= 1.5
    assert energies["scale_6.0"] >= 0.0

    # Test 2D shape [batch, N]
    x_2d = torch.zeros(batch_size, num_nodes)
    x_2d[:, 0] = 1.0
    energies_2d = diffuser.multi_scale_energies(x_2d)
    assert len(energies_2d) == 3
