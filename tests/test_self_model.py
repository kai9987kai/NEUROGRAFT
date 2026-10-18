"""Tests for Connectome Forward Self-Model & Agency Audit."""

import pytest
import torch

from neurograft.self_model import ConnectomeForwardSelfModel, SelfModelConfig


def test_self_model_shapes_and_adaptation():
    config = SelfModelConfig(latent_dim=32, action_dim=4, hidden_dim=64)
    model = ConnectomeForwardSelfModel(config)

    # Test exact dimension input
    exact_latent = torch.randn(1, 32)
    next_state = model.forecast(exact_latent, action=1)
    assert next_state.shape == (1, 32)

    null_state = model.counterfactual_null(exact_latent)
    assert null_state.shape == (1, 32)

    # Test smaller dimension input (padded)
    small_latent = torch.randn(1, 16)
    pred_small = model.forecast(small_latent, action=2)
    assert pred_small.shape == (1, 32)

    # Test larger dimension input (sliced)
    large_latent = torch.randn(1, 48)
    pred_large = model.forecast(large_latent, action=0)
    assert pred_large.shape == (1, 32)


def test_self_model_step_audit_and_agency():
    config = SelfModelConfig(latent_dim=16, action_dim=4, learning_rate=0.01)
    model = ConnectomeForwardSelfModel(config)

    s_t = torch.randn(1, 16)
    action = 1
    # Create actual next state close to forecast
    with torch.no_grad():
        s_next = model.forecast(s_t, action) + 0.01 * torch.randn(1, 16)

    audit = model.step_audit(s_t, action, s_next, train=False)
    assert "agency_score" in audit
    assert "prediction_error" in audit
    assert "reality_drift" in audit
    assert "agency_regime" in audit
    assert "self_alignment" in audit

    # Since next state was generated near the forecast for action 1, agency should be positive
    assert audit["agency_score"] > 0.0
    assert audit["agency_regime"] in ("autonomous-agency", "passive-concordance")
    assert 0.0 <= audit["self_alignment"] <= 1.0


def test_self_model_online_learning_decreases_error():
    config = SelfModelConfig(latent_dim=8, action_dim=4, learning_rate=0.05)
    model = ConnectomeForwardSelfModel(config)
    model.train()

    s_t = torch.ones(1, 8)
    action = 2
    s_target = torch.full((1, 8), 2.5)

    initial_audit = model.step_audit(s_t, action, s_target, train=False)
    initial_err = initial_audit["prediction_error"]

    # Train repeatedly on transition
    for _ in range(40):
        model.step_audit(s_t, action, s_target, train=True)

    final_audit = model.step_audit(s_t, action, s_target, train=False)
    assert final_audit["prediction_error"] < initial_err


def test_self_model_explain():
    model = ConnectomeForwardSelfModel()
    s = torch.randn(1, 32)
    model.step_audit(s, 0, s)
    summary = model.explain()
    assert "Connectome Forward Self-Model" in summary
    assert "Agency Gain" in summary
