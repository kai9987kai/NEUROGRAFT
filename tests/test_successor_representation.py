"""Tests for Hippocampal Successor Representation, SWR Replay, and Active Inference Navigation."""

import math
import pytest
import torch

from neurograft.memory import TopologicalMemoryGraph, TriTierMemory
from neurograft.navigation_controller import (
    ALL_MODES,
    MODES,
    sr_steering_action,
    legal_action,
)
from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveSession, LiveConfig
from neurograft.service import LiveApp


def test_modes_contract_preserved():
    """Legacy unit tests depend on MODES == ('policy', 'masked_policy', 'observed_map')."""
    assert MODES == ("policy", "masked_policy", "observed_map")
    assert "sr_steer" in ALL_MODES
    assert "cx_steer" in ALL_MODES


def test_sr_computation_empty_graph():
    graph = TopologicalMemoryGraph()
    nodes, sr_matrix = graph.compute_successor_representation()
    assert nodes == []
    assert sr_matrix == {}


def test_sr_computation_linear_chain():
    graph = TopologicalMemoryGraph()
    # Actions: 0=(0, -1) N, 1=(1, 0) E, 2=(0, 1) S, 3=(-1, 0) W
    # Linear chain East: (0, 0) -1-> (1, 0) -1-> (2, 0)
    gamma = 0.9
    graph.record_transition((0, 0), 1, (1, 0), 0.0)
    graph.record_transition((1, 0), 1, (2, 0), 1.0)

    nodes, sr_dict = graph.compute_successor_representation(discount=gamma)
    assert (0, 0) in nodes
    assert (1, 0) in nodes
    assert (2, 0) in nodes

    # Check discounting along the chain
    assert sr_dict[(0, 0)][(1, 0)] > 0.0
    assert sr_dict[(0, 0)][(2, 0)] > 0.0
    assert abs(sr_dict[(0, 0)][(2, 0)] - gamma * sr_dict[(1, 0)][(2, 0)]) < 1e-4


def test_sr_computation_cycle():
    graph = TopologicalMemoryGraph()
    # Cycle: (5, 5) -1-> (6, 5) -3-> (5, 5)
    gamma = 0.8
    graph.record_transition((5, 5), 1, (6, 5), 0.0)
    graph.record_transition((6, 5), 3, (5, 5), 0.0)

    nodes, sr_dict = graph.compute_successor_representation(discount=gamma)
    assert (5, 5) in nodes
    assert (6, 5) in nodes

    expected_diag = 1.0 / (1.0 - gamma**2)
    expected_offdiag = gamma / (1.0 - gamma**2)

    assert abs(sr_dict[(5, 5)][(5, 5)] - expected_diag) < 1e-3
    assert abs(sr_dict[(6, 5)][(6, 5)] - expected_diag) < 1e-3
    assert abs(sr_dict[(5, 5)][(6, 5)] - expected_offdiag) < 1e-3


def test_graph_laplacian_modes():
    graph = TopologicalMemoryGraph()
    # Cycle of 3 nodes: (5, 5) -1-> (6, 5) -2-> (6, 6) -0-> (6, 5) ...
    graph.record_transition((5, 5), 1, (6, 5), 0.0)
    graph.record_transition((6, 5), 2, (6, 6), 0.0)
    graph.record_transition((6, 6), 0, (6, 5), 0.0)

    modes = graph.graph_laplacian_modes(k=3)
    assert modes["num_nodes"] == 3
    assert len(modes["eigenvalues"]) == 3
    assert abs(modes["eigenvalues"][0]) < 1e-4
    assert "mode_0" in modes["modes"]


def test_swr_replay_value_propagation():
    graph = TopologicalMemoryGraph()
    # (5, 5) -1-> (6, 5) -1-> (7, 5) with reward 10.0 at (7, 5)
    graph.record_transition((5, 5), 1, (6, 5), 0.0)
    graph.record_transition((6, 5), 1, (7, 5), 10.0)

    swr = graph.trigger_swr_replay(max_sweeps=4, discount=0.9)
    assert swr["sweeps"] >= 1
    assert swr["replayed_edges"] > 0
    assert swr["value_delta"] > 0.0

    # Values should have propagated backwards
    v_after_6 = graph.estimated_values.get((6, 5), 0.0)
    v_after_5 = graph.estimated_values.get((5, 5), 0.0)
    assert v_after_6 > 0.0
    assert v_after_5 > 0.0
    assert v_after_6 >= v_after_5


def test_tritier_memory_swr_replay():
    mem = TriTierMemory()
    mem.record_step(
        step=1,
        agent_pos=(5, 5),
        action=1,
        reward=1.0,
        state_features=[0.0] * 16,
        surprise=0.1,
        transition_start=(4, 5),
    )
    swr = mem.trigger_swr_replay(max_sweeps=2)
    assert "sweeps" in swr
    assert "value_delta" in swr
    assert "replayed_edges" in swr


def test_successor_action_values():
    graph = TopologicalMemoryGraph()
    # (5, 5) with action 0 (N) -> (5, 4) (goal with reward)
    # (5, 5) with action 2 (S) -> (5, 6) (empty)
    graph.record_transition((5, 5), 0, (5, 4), 10.0)
    graph.record_transition((5, 5), 2, (5, 6), 0.0)
    graph.trigger_swr_replay(max_sweeps=2)

    q_vals = graph.successor_action_values((5, 5), goal_pos=(5, 4), discount=0.9, epistemic_weight=0.1)
    # Action 0 (North) must have higher expected return than action 2 (South)
    assert q_vals[0] > q_vals[2]


def test_sr_steering_action_wall_avoidance():
    # observation: 4x4 list of lists, where row 2 is [N, E, S, W] walls (0=free, 1=wall)
    # North is wall (1), others free (0)
    obs = [
        [0.5, 0.5, 0.0, -1.0],
        [0.5, 0.4, 0.0, -0.1],
        [1.0, 0.0, 0.0, 0.0],  # N=1 (wall), E=0, S=0, W=0
        [0.0, 0.0, 0.0, 1.0],
    ]
    logits = [10.0, 1.0, 1.0, 1.0]  # Raw policy wants North

    graph = TopologicalMemoryGraph()
    action = sr_steering_action(
        observation=obs,
        logits=logits,
        topological_graph=graph,
        current_pos=(5, 5),
        goal_pos=(5, 4),
    )
    # Action 0 (North) is a wall, so it MUST NOT be chosen
    assert action != 0
    assert action in (1, 2, 3)


def test_sr_steering_fallback():
    # Only West is free
    obs = [
        [0.5, 0.5, 0.0, -1.0],
        [0.8, 0.8, 0.3, 0.3],
        [1.0, 1.0, 1.0, 0.0],  # N=1, E=1, S=1, W=0
        [0.0, 0.0, 0.0, 1.0],
    ]
    logits = [5.0, 4.0, 3.0, 1.0]
    action = sr_steering_action(
        observation=obs,
        logits=logits,
        topological_graph=None,
        current_pos=(2, 2),
        goal_pos=(5, 5),
    )
    assert action == 3  # West is the only legal action


def test_live_session_sr_steer_integration(tmp_path):
    bundle = make_fixture_bundle(nodes_per_source=3)
    session = LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2))
    assert session.navigation_mode == "policy"

    # Switch to sr_steer
    snap = session.set_navigation_mode("sr_steer")
    assert snap["navigation"]["mode"] == "sr_steer"
    assert "Successor Representation" in snap["navigation"]["reason"]

    # Step in sr_steer mode
    session.tick(steps=5)
    assert session.steps == 5

    # Trigger chat replay
    chat_res = session.chat("replay")
    assert "Sharp-Wave Ripple" in chat_res["reply"]
    assert "sweeps" in chat_res["reply"]

    # Save and reload
    state_file = tmp_path / "test_session.pt"
    session.save(state_file)
    restored = LiveSession.load(state_file, bundle)
    assert restored.navigation_mode == "sr_steer"
    assert restored.steps == 5


def test_service_replay_control(tmp_path):
    bundle = make_fixture_bundle(nodes_per_source=3)
    state_file = tmp_path / "service_session.pt"

    app = LiveApp(bundle=bundle, state_path=state_file, background=False)
    try:
        # Pause before manual steps
        app.control({"action": "pause"})

        # Switch to sr_steer
        state = app.control({"action": "navigation", "mode": "sr_steer"})
        assert state["navigation"]["mode"] == "sr_steer"

        # Advance some steps
        state = app.control({"action": "step", "steps": 3})
        assert state["steps"] >= 3

        # Trigger SWR replay via control
        reply = app.control({"action": "replay", "sweeps": 3})
        assert "replay" in reply
        assert 1 <= reply["replay"]["sweeps"] <= 3
        assert "value_delta" in reply["replay"]
    finally:
        app.close(save=False)
