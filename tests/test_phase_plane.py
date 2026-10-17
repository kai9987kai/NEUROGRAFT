"""Unit tests for Nonlinear Dynamical Systems Phase-Plane & Lyapunov Separatrix Stability Engine."""
import pytest
from neurograft.phase_plane import PhasePlaneEngine


def test_phase_plane_audit_and_contracting_attractor():
    engine = PhasePlaneEngine(dt=0.20)
    # First step: start away from target
    first = engine.audit(energy_rate=0.4, goal_proximity=0.3, clearance_margin=0.4, heading_alignment=0.5)
    assert first["lyapunov_energy"] > 0.0
    assert first["lyapunov_drift"] == 0.0
    assert not first["in_critical_basin"]

    # Second step: move closer to target fixed point [0.85, 0.90, 0.80, 1.00]
    second = engine.audit(energy_rate=0.8, goal_proximity=0.85, clearance_margin=0.75, heading_alignment=0.95)
    # Energy should decrease, meaning drift < 0 (contracting attractor)
    assert second["lyapunov_energy"] < first["lyapunov_energy"]
    assert second["lyapunov_drift"] < 0.0
    assert second["stability_regime"] in ("contracting_attractor", "transient_orbit")


def test_phase_plane_critical_extinction_basin():
    engine = PhasePlaneEngine(dt=0.20)
    # Near obstacle (clearance margin < 0.15)
    critical = engine.audit(energy_rate=0.2, goal_proximity=0.1, clearance_margin=0.08, heading_alignment=0.1)
    assert critical["in_critical_basin"]
    assert critical["stability_regime"] == "extinction_hazard"


def test_phase_plane_serialization_roundtrip():
    engine = PhasePlaneEngine(dt=0.25)
    engine.audit(energy_rate=0.7, goal_proximity=0.8, clearance_margin=0.6, heading_alignment=0.9)
    data = engine.to_dict()
    assert data["dt"] == 0.25
    restored = PhasePlaneEngine.from_dict(data)
    assert restored.dt == 0.25
    assert restored.last_lyapunov == pytest.approx(engine.last_lyapunov)
    assert restored.last_state == pytest.approx(engine.last_state)
