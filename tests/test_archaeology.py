"""Tests for Memory Archaeology & Tissue Resection Assays."""

import pytest
import torch

from neurograft.archaeology import ArchaeologyReport, MemoryArchaeologyAssay, ResectionSpec
from neurograft.dynamics import DynamicsConfig, GraphCircuit
from neurograft.fixtures import make_fixture_bundle


def test_archaeology_motif_conservation_index():
    bundle = make_fixture_bundle(seed=3, nodes_per_source=4)
    assay = MemoryArchaeologyAssay(bundle)
    motifs = assay._motif_conservation
    assert "worm" in motifs and "fly_f" in motifs
    assert motifs["worm"] >= 0.0
    assert motifs["fly_f"] >= 0.0


def test_archaeology_resection_assay():
    bundle = make_fixture_bundle(seed=4, nodes_per_source=4)
    assay = MemoryArchaeologyAssay(bundle)
    circuit = GraphCircuit(bundle, DynamicsConfig(hidden_dim=4, output_dim=4))

    # Test inputs: [batch=1, seq=4, num_sources=4, feature=4]
    inputs = torch.randn(1, 4, 4, 4)

    spec = ResectionSpec(target_species="worm", mode="rich_club_hubs")
    report = assay.run_resection_assay(circuit, inputs, spec=spec, bioelectric_gain=1.05)

    assert isinstance(report, ArchaeologyReport)
    assert report.target_species == "worm"
    assert 0.0 <= report.mtpr <= 1.0
    assert 0.0 <= report.lesioned_fidelity <= 1.0
    assert 0.0 <= report.bioelectric_rescued_fidelity <= 1.0
    assert report.regime in ("engram-preserved", "behaviorally-compensated", "irreversible-loss")
    assert "worm" in report.species_resilience
    assert "fly_f" in report.species_resilience

    summary = assay.latest_summary()
    assert summary["target_species"] == "worm"
    assert summary["mtpr"] == round(report.mtpr, 4)
