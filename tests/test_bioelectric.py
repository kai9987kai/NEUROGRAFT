"""Tests for Bioelectric Morphogenetic Substrate (BMS)."""

import pytest
import torch

from neurograft.bioelectric import (
    BioelectricConfig,
    BioelectricMorphogeneticSubstrate,
    SparseGapJunctionDiffuser,
)
from neurograft.fixtures import make_fixture_bundle


def test_bioelectric_config_validation():
    with pytest.raises(ValueError, match="dt"):
        BioelectricConfig(dt=-0.1)
    with pytest.raises(ValueError, match="v_min < v_rest < v_max"):
        BioelectricConfig(v_rest=-90.0, v_min=-85.0)
    with pytest.raises(ValueError, match="conductances"):
        BioelectricConfig(g_leak=0.0)


def test_sparse_gap_junction_diffuser():
    # 3 nodes: 0 -- 1 -- 2
    diffuser = SparseGapJunctionDiffuser(
        num_nodes=3,
        src=[0, 1, 1, 2],
        dst=[1, 0, 2, 1],
        weights=[1.0, 1.0, 1.0, 1.0],
    )
    v = torch.tensor([[ -60.0, -70.0, -60.0 ]])
    flux = diffuser(v)
    assert flux.shape == (1, 3)
    # Node 1 is at -70, neighbors are at -60. Incoming flux should be positive (depolarizing)
    assert flux[0, 1].item() > 0.0
    # Net flux across closed symmetrical system sums to zero
    assert abs(float(flux.sum().item())) < 1e-4


def test_bioelectric_initial_voltage_and_relaxation():
    bundle = make_fixture_bundle(seed=1, nodes_per_source=3)
    config = BioelectricConfig(dt=0.20, v_rest=-65.0)
    substrate = BioelectricMorphogeneticSubstrate(bundle, config)

    voltages = substrate.initial_voltage(batch_size=1)
    assert "worm" in voltages
    assert voltages["worm"].shape == (1, 3)
    assert torch.allclose(voltages["worm"], torch.tensor(-65.0))

    # At zero neural activity, voltages stay at resting potential
    zero_act = {src: torch.zeros(1, 3) for src in voltages}
    new_v, summary = substrate.step(voltages, zero_act)
    assert torch.allclose(new_v["worm"], torch.tensor(-65.0), atol=1e-2)
    assert summary["global_mean_mv"] == pytest.approx(-65.0, abs=0.1)
    assert summary["excitability_gain"] == 1.0


def test_bioelectric_depolarization_and_gain():
    bundle = make_fixture_bundle(seed=2, nodes_per_source=4)
    substrate = BioelectricMorphogeneticSubstrate(bundle)

    voltages = substrate.initial_voltage(batch_size=1)
    high_act = {src: torch.ones(1, 4) for src in voltages}

    for _ in range(20):
        voltages, summary = substrate.step(voltages, high_act)

    # Sustained activity pumps voltage higher (depolarization)
    assert summary["global_mean_mv"] > -65.0
    assert summary["excitability_gain"] > 1.0
    # Values never exceed v_max
    for src in voltages:
        assert voltages[src].max().item() <= substrate.config.v_max
