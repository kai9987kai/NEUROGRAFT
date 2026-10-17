"""Unit tests for Active Inference Expected Free Energy (EFE) Decision Arbiter."""
import pytest
from neurograft.active_inference import ActiveInferenceArbiter


def test_active_inference_evaluation_structure():
    arbiter = ActiveInferenceArbiter()
    result = arbiter.evaluate_actions(
        agent_pos=(5, 5),
        goal_pos=(5, 1),
        obstacles={(5, 4)},  # obstacle directly North
        world_size=11,
        lidar_8=[0.5] * 8,
        visit_counts={(5, 6): 5},
        dopamine=0.6,
        octopamine=0.4,
    )
    assert len(result["efe_values"]) == 4
    assert len(result["efe_probs"]) == 4
    assert abs(sum(result["efe_probs"]) - 1.0) < 1e-4
    assert len(result["pragmatic"]) == 4
    assert len(result["epistemic"]) == 4
    assert len(result["empowerment"]) == 4
    assert result["best_action"] in (0, 1, 2, 3)
    # North action (0) hits obstacle (5, 4), so pragmatic score should be heavily penalized (-5.0)
    assert result["pragmatic"][0] == -5.0
    # Best action should not be the blocked North action
    assert result["best_action"] != 0


def test_active_inference_neuromodulatory_epistemic_scaling():
    arbiter = ActiveInferenceArbiter(base_epistemic_weight=0.4)
    # High octopamine (arousal/novelty) increases epistemic weight
    res_high_oct = arbiter.evaluate_actions(
        agent_pos=(3, 3), goal_pos=(7, 7), obstacles=set(), world_size=11,
        lidar_8=[0.5] * 8, dopamine=0.2, octopamine=0.9,
    )
    # Low octopamine, high dopamine decreases epistemic weight
    res_low_oct = arbiter.evaluate_actions(
        agent_pos=(3, 3), goal_pos=(7, 7), obstacles=set(), world_size=11,
        lidar_8=[0.5] * 8, dopamine=0.9, octopamine=0.1,
    )
    assert res_high_oct["epistemic_weight"] > res_low_oct["epistemic_weight"]
