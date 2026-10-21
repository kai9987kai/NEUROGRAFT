"""Tests for Drosophila Central Complex (CX) PFL1/PFL2/PFL3 vector steering."""
import math
import pytest
from neurograft.compass import CXPFLSteering
from neurograft.navigation_controller import cx_steering_action, ALL_MODES


def test_cx_pfl_steering_initialization():
    steer = CXPFLSteering(steering_weight=2.0, gain_scale=1.5)
    assert steer.steering_weight == 2.0
    assert steer.gain_scale == 1.5


def test_cx_pfl_steering_prior_zero_displacement():
    steer = CXPFLSteering()
    prior = steer.compute_steering_prior(heading=0.0, goal_dx=0.0, goal_dz=0.0)
    assert prior == [0.0, 0.0, 0.0, 0.0]


def test_cx_pfl_steering_prior_directional_preference():
    steer = CXPFLSteering()
    # Heading East (0.0), Goal East (+X: dx=1.0, dz=0.0)
    # Action directions: North(0) = -pi/2, East(1) = 0, South(2) = pi/2, West(3) = pi
    prior_east = steer.compute_steering_prior(heading=0.0, goal_dx=1.0, goal_dz=0.0)
    # East (action 1) should have highest prior
    assert max(range(4), key=lambda i: prior_east[i]) == 1

    # Goal South (+Z: dx=0.0, dz=1.0)
    prior_south = steer.compute_steering_prior(heading=0.0, goal_dx=0.0, goal_dz=1.0)
    assert max(range(4), key=lambda i: prior_south[i]) == 2


def test_cx_pfl_steering_wall_masking():
    steer = CXPFLSteering()
    # Goal East (1), but East is blocked by wall
    walls = [0, 1, 0, 0]  # East blocked
    prior = steer.compute_steering_prior(heading=0.0, goal_dx=1.0, goal_dz=0.0, walls=walls)
    assert prior[1] == -1e6

    # steer() should choose best non-blocked action
    logits = [0.0, 10.0, 0.0, 0.0]  # Logits strongly prefer East
    action = steer.steer(heading=0.0, goal_dx=1.0, goal_dz=0.0, walls=walls, policy_logits=logits)
    assert action != 1  # Never blocked East
    assert action in (0, 2, 3)


def test_cx_steering_action_integration():
    # Observation: agent at (0.5, 0.5), heading East (dx=1, dz=0), goal at East (+X)
    obs = [
        [0.5, 0.5, 1.0, 0.0],  # Agent pos + heading vector (cos=1, sin=0 -> 0.0 rad)
        [0.8, 0.5, 0.3, 0.0],  # Goal pos + goal displacement (+X)
        [0, 0, 0, 0],          # Walls
        [0.0, 0.0, 0.0, 1.0],  # Auxiliary
    ]
    logits = [0.0, 0.0, 0.0, 0.0]
    action = cx_steering_action(obs, logits)
    assert action == 1  # East

    # Block East
    obs[2] = [0, 1, 0, 0]
    action_blocked = cx_steering_action(obs, logits)
    assert action_blocked != 1


def test_all_modes_contains_cx_steer():
    assert "cx_steer" in ALL_MODES
