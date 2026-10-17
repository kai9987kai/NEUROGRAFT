"""Unit tests for Central Complex (CX) 16-wedge ring attractor compass."""
import math
import pytest
from neurograft.compass import CXRingCompass


def test_cx_ring_initialization_and_bump_profile():
    compass = CXRingCompass(num_wedges=16, initial_heading=0.0)
    assert compass.num_wedges == 16
    assert len(compass.angles) == 16
    assert len(compass.state) == 16
    assert abs(sum(compass.state) - 1.0) < 1e-6
    # Wedge 0 should have the highest activation when initial_heading=0
    assert compass.state[0] == max(compass.state)
    pva, coh = compass.readout()
    assert abs(pva - 0.0) < 0.1 or abs(pva - 2 * math.pi) < 0.1
    assert coh > 0.60
    assert not compass.is_disoriented()


def test_cx_ring_velocity_integration_and_shift():
    compass = CXRingCompass(num_wedges=16, initial_heading=0.0)
    # Apply positive angular velocity (counter-clockwise turn)
    for _ in range(5):
        compass.step(angular_velocity=1.0, dt=0.20)
    pva, coh = compass.readout()
    assert coh > 0.35
    # Heading should have shifted positive
    assert pva > 0.0


def test_cx_ring_disorientation_and_optomotor_reflex():
    compass = CXRingCompass(num_wedges=16)
    # Force flattened uniform state (zero coherence)
    compass.state = [1.0 / 16.0] * 16
    pva, coh = compass.readout()
    assert coh < 0.10
    assert compass.is_disoriented()
    # Optomotor reflex should guide action towards target heading
    stabilized = compass.optomotor_stabilization_action(current_action=0, target_heading=math.pi / 2.0)
    assert stabilized in (0, 1, 2, 3)


def test_cx_ring_serialization_roundtrip():
    compass = CXRingCompass(num_wedges=16, initial_heading=1.2)
    compass.step(angular_velocity=0.5, dt=0.2)
    data = compass.to_dict()
    assert "num_wedges" in data and "pva_heading" in data and "pva_coherence" in data
    restored = CXRingCompass.from_dict(data)
    assert restored.num_wedges == compass.num_wedges
    assert abs(restored.readout()[0] - compass.readout()[0]) < 1e-4
    assert abs(restored.readout()[1] - compass.readout()[1]) < 1e-4


@pytest.mark.parametrize("wedges", [2, 100, -1])
def test_cx_ring_invalid_wedges(wedges):
    with pytest.raises(ValueError):
        CXRingCompass(num_wedges=wedges)
