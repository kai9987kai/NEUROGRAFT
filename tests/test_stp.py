"""Tests for Tsodyks-Markram Short-Term Synaptic Plasticity Substrate."""

import pytest
import torch

from neurograft.stp import STPConfig, STPState, TsodyksMarkramSubstrate


def test_stp_config_validation():
    with pytest.raises(ValueError, match="dt"):
        STPConfig(dt=-0.1)
    with pytest.raises(ValueError, match="u_baseline"):
        STPConfig(u_baseline=0.0)
    with pytest.raises(ValueError, match="u_baseline"):
        STPConfig(u_baseline=1.2)
    with pytest.raises(ValueError, match="tau"):
        STPConfig(tau_rec=-0.5)


def test_stp_initial_state_and_identity_preservation():
    node_counts = [10, 20, 30, 40]
    config = STPConfig(dt=0.20, u_baseline=0.35, tau_rec=0.8, tau_fac=0.25)
    substrate = TsodyksMarkramSubstrate(node_counts, config)

    state = substrate.initial_state(batch_size=2)
    assert len(state.u) == 4 and len(state.r) == 4
    for count, u_t, r_t in zip(node_counts, state.u, state.r):
        assert u_t.shape == (2, count)
        assert r_t.shape == (2, count)
        assert torch.allclose(u_t, torch.tensor(0.35))
        assert torch.allclose(r_t, torch.tensor(1.00))

    # At zero activation, efficacy remains close to resting identity
    zero_acts = [torch.zeros(2, count) for count in node_counts]
    efficacies, next_state, summary = substrate.step(zero_acts, state)
    for eff in efficacies:
        assert eff.shape[-1] == 1
        assert torch.allclose(eff, torch.ones_like(eff), atol=1e-2)
    assert summary["mean_vesicle_pool"] >= 0.99


def test_stp_depression_under_sustained_activation():
    node_counts = [16]
    # Fast depression regime
    config = STPConfig(dt=0.20, u_baseline=0.50, tau_rec=1.5, tau_fac=0.05, facilitation_weight=0.1)
    substrate = TsodyksMarkramSubstrate(node_counts, config)

    state = substrate.initial_state(batch_size=1)
    high_act = [torch.ones(1, 16)]

    # Run for multiple steps
    for _ in range(25):
        _, state, summary = substrate.step(high_act, state)

    # Vesicle pool should be significantly depleted
    assert summary["mean_vesicle_pool"] < 0.60
    assert summary["depression_fraction"] > 0.0


def test_stp_facilitation_under_low_baseline():
    node_counts = [16]
    # Facilitation regime: low U baseline, slow facilitation decay, fast recovery
    config = STPConfig(dt=0.20, u_baseline=0.15, tau_rec=0.1, tau_fac=1.0, facilitation_weight=0.8)
    substrate = TsodyksMarkramSubstrate(node_counts, config)

    state = substrate.initial_state(batch_size=1)
    act = [torch.ones(1, 16)]

    _, state, summary = substrate.step(act, state)
    # Release probability should rise above baseline
    assert summary["mean_release_prob"] > 0.15
    assert summary["facilitation_fraction"] > 0.0
