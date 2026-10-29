"""Analytical and hostile-input checks for empirical goal-conditioned replay."""
import copy
import json

import pytest

from neurograft.memory import TopologicalMemoryGraph


def planner(**kwargs):
    # Import here so the initial missing implementation is a failing test.
    from neurograft.replay import ReplayPlanner

    return ReplayPlanner(**kwargs)


def restore(state):
    from neurograft.replay import ReplayPlanner

    return ReplayPlanner.from_state(state)


def chain():
    graph = TopologicalMemoryGraph()
    for x in range(3):
        graph.record_transition((x, 0), 1, (x + 1, 0))
    return graph


def test_prioritized_chain_uses_one_backup_then_predecessor_propagation():
    graph, learner = chain(), planner()
    first = learner.replay(graph, (3, 0), budget=1)
    assert first["backups"] == 1
    assert first["residual_before"] == 1.0
    assert first["residual_after"] == 0.9
    assert first["value_delta"] == 1.0
    assert not first["converged"]
    assert learner.values == {(0, 0): 0.0, (1, 0): 0.0, (2, 0): 1.0, (3, 0): 0.0}
    second = learner.replay(graph, (3, 0), budget=2)
    assert second["backups"] == 2
    assert second["converged"]
    assert learner.values[(0, 0)] == pytest.approx(0.81)
    assert learner.values[(1, 0)] == 0.9
    assert learner.values[(3, 0)] == 0.0
    assert learner.action_values(graph, (0, 0), (3, 0)) == {0: 0.0, 1: 0.81, 2: 0.0, 3: 0.0}


def test_observed_branch_actions_are_ranked_by_goal_value_not_reward_or_distance():
    graph, learner = TopologicalMemoryGraph(), planner()
    graph.record_transition((1, 1), 0, (1, 0), reward=1000.0)
    graph.record_transition((1, 1), 1, (2, 1))
    graph.record_transition((2, 1), 0, (2, 0))
    learner.replay(graph, (2, 0), budget=20)
    assert learner.action_values(graph, (1, 1), (2, 0)) == {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.0}
    assert learner.action_values(graph, (2, 0), (2, 0)) == dict.fromkeys(range(4), 0.0)


def test_collision_probability_contributes_expected_discounted_value():
    graph, learner = TopologicalMemoryGraph(), planner()
    graph.record_transition((0, 0), 1, (1, 0))
    for _ in range(3):
        graph.record_transition((0, 0), 1, (0, 0))
    result = learner.replay(graph, (1, 0), budget=4096)
    # V = .25 + .75 * .9 * V, hence V = 10 / 13.
    assert result["converged"]
    assert learner.values[(0, 0)] == pytest.approx(10 / 13, abs=4e-12)
    assert learner.action_values(graph, (0, 0), (1, 0))[1] == pytest.approx(10 / 13, abs=4e-12)
    assert learner.values[(1, 0)] == 0.0


def test_new_collision_evidence_lowers_values_for_same_goal():
    graph, learner = TopologicalMemoryGraph(), planner()
    graph.record_transition((0, 0), 1, (1, 0))
    learner.replay(graph, (1, 0), budget=8)
    assert learner.values[(0, 0)] == 1.0
    graph.record_transition((0, 0), 1, (0, 0))
    result = learner.replay(graph, (1, 0), budget=4096)
    # V = .5 + .5 * .9 * V, hence V = 10 / 11.
    assert result["residual_before"] == pytest.approx(0.05)
    assert learner.values[(0, 0)] == pytest.approx(10 / 11, abs=4e-12)


def test_removed_edges_and_nodes_remove_stale_goal_values():
    graph, learner = chain(), planner()
    learner.replay(graph, (3, 0), budget=20)
    del graph.transitions[((1, 0), 1)]
    learner.replay(graph, (3, 0), budget=20)
    assert learner.values[(0, 0)] == learner.values[(1, 0)] == 0.0
    smaller = TopologicalMemoryGraph()
    smaller.add_node((0, 0))
    smaller.add_node((3, 0))
    learner.replay(smaller, (3, 0), budget=0)
    assert learner.values == {(0, 0): 0.0, (3, 0): 0.0}


def test_goal_switch_clears_previous_values_even_with_disabled_replay():
    graph, learner = chain(), planner()
    learner.replay(graph, (3, 0), budget=20)
    result = learner.replay(graph, (0, 0), budget=32, strategy="none")
    assert result["backups"] == 0
    assert learner.goal == (0, 0)
    assert all(value == 0.0 for value in learner.values.values())
    assert learner.action_values(graph, (1, 0), (0, 0)) == dict.fromkeys(range(4), 0.0)


@pytest.mark.parametrize("strategy,budget", [("none", 32), ("prioritized", 0), ("uniform", 0)])
def test_disabled_or_zero_budget_performs_no_backups_but_observed_goal_outcomes_have_q(strategy, budget):
    graph, learner = chain(), planner()
    result = learner.replay(graph, (3, 0), budget=budget, strategy=strategy)
    assert result["backups"] == 0
    assert learner.to_state()["total_backups"] == 0
    assert all(value == 0.0 for value in learner.values.values())
    assert learner.action_values(graph, (2, 0), (3, 0))[1] == 1.0
    assert learner.action_values(graph, (1, 0), (3, 0))[1] == 0.0
    assert result["residual_after"] == result["residual_before"] == 1.0


def test_action_value_reads_do_not_switch_goal_or_change_state():
    graph, learner = chain(), planner()
    learner.replay(graph, (3, 0), budget=2)
    before = learner.to_state()
    assert learner.action_values(graph, (2, 0), (3, 0))[1] == 1.0
    assert learner.action_values(graph, (0, 0), (2, 0))[1] == 0.0
    assert learner.action_values(graph, (1, 0), (2, 0))[1] == 1.0
    assert learner.to_state() == before


def test_unknown_goal_has_no_invented_values_or_nearest_goal_substitution():
    graph, learner = chain(), planner()
    learner.replay(graph, (3, 0), budget=20)
    result = learner.replay(graph, (30, 30), budget=32)
    assert not result["goal_known"]
    assert not result["converged"]
    assert result["backups"] == 0
    assert learner.values == {}
    assert learner.action_values(graph, (2, 0), (30, 30)) == dict.fromkeys(range(4), 0.0)


def test_uniform_schedule_cycles_sorted_nonterminal_states_and_persists_cursor():
    graph, learner = chain(), planner()
    learner.replay(graph, (3, 0), budget=2, strategy="uniform")
    assert learner.values[(0, 0)] == learner.values[(1, 0)] == learner.values[(2, 0)] == 0.0
    restored = restore(json.loads(json.dumps(learner.to_state())))
    assert restored.snapshot() == learner.snapshot()
    one = learner.replay(graph, (3, 0), budget=1, strategy="uniform")
    two = restored.replay(graph, (3, 0), budget=1, strategy="uniform")
    assert one == two
    assert learner.values[(2, 0)] == 1.0
    assert restored.to_state() == learner.to_state()
    learner.replay(graph, (3, 0), budget=3, strategy="uniform")
    assert learner.values[(1, 0)] == 0.9
    assert learner.values[(0, 0)] == 0.0


def test_prioritized_tie_break_and_continuation_are_recording_order_independent():
    edges = [((1, 1), 1, (2, 1)), ((2, 0), 2, (2, 1)), ((0, 1), 1, (1, 1))]
    graphs = [TopologicalMemoryGraph(), TopologicalMemoryGraph()]
    for edge in edges:
        graphs[0].record_transition(*edge)
    for edge in reversed(edges):
        graphs[1].record_transition(*edge)
    learners = [planner(), planner()]
    for graph, learner in zip(graphs, learners):
        learner.replay(graph, (2, 1), budget=1)
    assert learners[0].values[(1, 1)] == 1.0
    assert learners[0].values[(2, 0)] == 0.0
    assert learners[0].to_state() == learners[1].to_state()
    restored = restore(learners[0].to_state())
    assert restored.replay(graphs[0], (2, 1), budget=4) == learners[0].replay(graphs[0], (2, 1), budget=4)
    assert restored.snapshot() == learners[0].snapshot()


def test_replay_and_inspection_never_modify_graph_or_return_live_state():
    graph, learner = chain(), planner()
    before = graph.to_state()
    result = learner.replay(graph, [3, 0], budget=8)
    learner.action_values(graph, [0, 0], [3, 0])
    snapshot, state = learner.snapshot(), learner.to_state()
    json.dumps(snapshot, allow_nan=False)
    assert snapshot == learner.snapshot()
    assert graph.to_state() == before
    result["backups"] = 900
    state["values"][0]["value"] = 999
    snapshot["total_backups"] = -1
    assert learner.to_state()["total_backups"] == 3
    assert learner.values[(0, 0)] == pytest.approx(0.81)


@pytest.mark.parametrize("discount", [True, None, "0.9", -0.01, 1.0, 1.1, float("nan"), float("inf"), 10**1000])
def test_invalid_discounts_are_rejected(discount):
    with pytest.raises(ValueError):
        planner(discount=discount)


@pytest.mark.parametrize("budget", [True, False, None, "1", 1.0, -1, 4097, float("inf")])
def test_invalid_budgets_are_rejected_atomically(budget):
    graph, learner = chain(), planner()
    learner.replay(graph, (3, 0), budget=1)
    before = learner.to_state()
    with pytest.raises(ValueError):
        learner.replay(graph, (2, 0), budget=budget)
    assert learner.to_state() == before


@pytest.mark.parametrize("position", [None, "1,2", (1,), (1, 2, 3), (-1, 0), (31, 0), (True, 0), (1.0, 0)])
def test_invalid_goal_and_current_coordinates_are_rejected_atomically(position):
    graph, learner = chain(), planner()
    before = learner.to_state()
    with pytest.raises(ValueError):
        learner.replay(graph, position)
    with pytest.raises(ValueError):
        learner.action_values(graph, position, (3, 0))
    with pytest.raises(ValueError):
        learner.action_values(graph, (0, 0), position)
    assert learner.to_state() == before


@pytest.mark.parametrize("strategy", ["random", "", None, True, [], {}])
def test_invalid_strategies_are_rejected_atomically(strategy):
    graph, learner = chain(), planner()
    before = learner.to_state()
    with pytest.raises(ValueError):
        learner.replay(graph, (3, 0), strategy=strategy)
    assert learner.to_state() == before


def malicious_states():
    valid = {
        "version": 1, "discount": 0.9, "goal": [1, 0],
        "values": [{"position": [0, 0], "value": 0.5}, {"position": [1, 0], "value": 0.0}],
        "uniform_cursor": 1, "total_backups": 1, "last_summary": None,
    }
    cases = []
    for key, value in [("version", True), ("version", 2), ("extra", 1), ("discount", True),
                       ("discount", 1), ("goal", [31, 0]), ("goal", [True, 0]),
                       ("uniform_cursor", -1), ("uniform_cursor", True), ("uniform_cursor", 962),
                       ("total_backups", -1), ("total_backups", True), ("total_backups", 2**53),
                       ("values", {}), ("values", [{}] * 962), ("last_summary", {}),
                       ("last_summary", {"unknown": "field"})]:
        state = copy.deepcopy(valid)
        state[key] = value
        cases.append(state)
    for value in [True, -0.1, 1.1, float("nan"), float("inf"), "0.5"]:
        state = copy.deepcopy(valid)
        state["values"][0]["value"] = value
        cases.append(state)
    duplicate = copy.deepcopy(valid)
    duplicate["values"].append(duplicate["values"][0].copy())
    cases.append(duplicate)
    terminal = copy.deepcopy(valid)
    terminal["values"][1]["value"] = 0.1
    cases.append(terminal)
    no_goal = copy.deepcopy(valid)
    no_goal["goal"] = None
    cases.append(no_goal)
    unknown_value_field = copy.deepcopy(valid)
    unknown_value_field["values"][0]["extra"] = 1
    cases.append(unknown_value_field)
    return cases


@pytest.mark.parametrize("state", malicious_states())
def test_malformed_checkpoint_state_is_rejected_without_changing_input(state):
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        restore(state)
    # NaN is deliberately hostile; compare canonical text rather than NaN equality.
    assert json.dumps(state, sort_keys=True) == json.dumps(before, sort_keys=True)


def test_summary_checkpoint_fields_are_strict_and_finite():
    learner = planner()
    learner.replay(chain(), (3, 0), budget=2)
    valid = learner.to_state()
    for key, value in [("backups", True), ("budget", 4097), ("residual_after", float("nan")),
                       ("residual_before", -1), ("value_delta", 1.1), ("converged", 1),
                       ("goal_known", "yes"), ("model_nodes", 962),
                       ("model_transitions", 2**53), ("strategy", "random"), ("extra", 1)]:
        state = copy.deepcopy(valid)
        state["last_summary"][key] = value
        with pytest.raises(ValueError):
            restore(state)


@pytest.mark.parametrize("trace", [
    [{"position": [0, 0], "before": 0.0, "after": 1.0, "extra": 1}],
    [{"position": [31, 0], "before": 0.0, "after": 1.0}],
    [{"position": [0, 0], "before": True, "after": 1.0}],
    [{"position": [0, 0], "before": 0.0, "after": float("inf")}],
    [{"position": [0, 0], "before": 0.0, "after": 1.0}] * 33,
])
def test_checkpoint_trace_cannot_hide_unbounded_or_unknown_data(trace):
    learner = planner()
    learner.replay(chain(), (3, 0), budget=2)
    state = learner.to_state()
    state["last_summary"]["trace"] = trace
    with pytest.raises(ValueError):
        restore(state)


@pytest.mark.parametrize("source,action,destination,count", [
    ((0, 0), True, (1, 0), 1),
    ((0, 0), 1, (1, 0), True),
    ((0, 0), 1, (1, 0), 0),
    ((0, 0), 1, (1, 0), float("inf")),
    ((0, 0), 1, (1, 0), 2**53),
    ((0, 0), 1, (2, 0), 1),
    ((0, 0), 1, (31, 0), 1),
    ((0, 0), 1, (True, 0), 1),
    ((2, 0), 1, (3, 0), 1),
])
def test_malformed_empirical_model_is_rejected_before_installing_new_goal(source, action, destination, count):
    learner, graph = planner(), TopologicalMemoryGraph()
    graph.add_node((0, 0))
    graph.add_node((1, 0))
    graph.transitions[(source, action)] = {destination: count}
    before = learner.to_state()
    with pytest.raises(ValueError):
        learner.replay(graph, (1, 0))
    assert learner.to_state() == before


def test_long_replay_trace_is_bounded_and_roundtrips_exactly():
    learner, graph = planner(), TopologicalMemoryGraph()
    graph.record_transition((0, 0), 1, (1, 0))
    graph.record_transition((0, 0), 1, (0, 0))
    result = learner.replay(graph, (1, 0), budget=64, strategy="uniform")
    assert result["backups"] == 64
    assert len(result["trace"]) == 32
    assert restore(learner.to_state()).snapshot() == learner.snapshot()


def test_zero_discount_only_immediate_observed_goal_outcomes_have_value():
    learner, graph = planner(discount=0), chain()
    learner.replay(graph, (3, 0), budget=32)
    assert learner.values == {(0, 0): 0.0, (1, 0): 0.0, (2, 0): 1.0, (3, 0): 0.0}


def test_default_empty_state_has_detached_json_roundtrip():
    learner = planner()
    assert learner.goal is None
    assert learner.values == {}
    assert restore(json.loads(json.dumps(learner.to_state()))).snapshot() == learner.snapshot()
