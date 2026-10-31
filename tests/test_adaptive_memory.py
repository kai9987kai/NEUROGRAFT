"""Analytical causal-scoring and evidence-selection checks."""
import copy
import math

import pytest

from neurograft.memory import TopologicalMemoryGraph


def memory(graph=None, window=8):
    from neurograft.adaptive_memory import AdaptiveTransitionMemory
    return AdaptiveTransitionMemory(graph or baseline(), window=window)


def baseline():
    graph = TopologicalMemoryGraph()
    for _ in range(9):
        graph.record_transition((0, 0), 1, (1, 0))
    graph.record_transition((0, 0), 1, (0, 0))
    graph.record_transition((1, 0), 1, (2, 0))
    return graph


def test_forecasts_are_scored_before_either_expert_observes_event():
    bank = memory()
    before = bank.predict((0, 0), 1)
    assert before == dict.fromkeys(('cumulative', 'recent', 'guarded_recent', 'adaptive'), 9.5 / 11)
    result = bank.observe((0, 0), 1, (0, 0))
    assert result['predictions'] == before
    assert result['brier']['adaptive'] == pytest.approx((9.5 / 11) ** 2)
    assert result['log_loss']['adaptive'] == pytest.approx(-math.log(1.5 / 11))
    assert result['evidence'] == 0
    assert bank.predict((0, 0), 1)['recent'] == .25
    # Event two compares the genuinely different pre-update experts.
    second = bank.observe((0, 0), 1, (0, 0))
    assert second['predictions']['cumulative'] == 9.5 / 12
    assert second['predictions']['recent'] == .25
    assert second['evidence'] == pytest.approx(math.log(.75 / (2.5 / 12)))


def test_ordinary_single_collision_keeps_baseline_but_repeated_closure_can_select_recent():
    bank = memory()
    result = bank.observe((0, 0), 1, (0, 0))
    assert result['selected_after'] == 'cumulative'
    assert bank.graph.transitions[((0, 0), 1)] == {(1, 0): 9, (0, 0): 2}
    for _ in range(3):
        result = bank.observe((0, 0), 1, (0, 0))
    assert result['fresh_count'] == 4
    assert result['selected_after'] == 'recent'
    assert bank.graph.transitions[((0, 0), 1)] == {(0, 0): 4}
    assert bank.diagnostics()['pairs'][0]['cumulative_count'] == 14
    assert bank.diagnostics()['pairs'][0]['recent_count'] == 4
    assert bank.diagnostics()['pairs'][0]['retained_count'] == 4
    assert bank.graph.transitions[((1, 0), 1)] == {(2, 0): 1}


def test_selection_can_reverse_and_only_genuinely_observed_edges_enter_graph():
    bank = memory(window=4)
    for _ in range(4):
        bank.observe((0, 0), 1, (0, 0))
    assert bank.diagnostics()['pairs'][0]['selected'] == 'recent'
    choices = []
    for _ in range(30):
        choices.append(bank.observe((0, 0), 1, (1, 0))['selected_after'])
    assert 'cumulative' in choices
    assert all(destination in (source, (source[0] + 1, source[1]))
               for (source, _), counts in bank.graph.transitions.items()
               for destination in counts)
    assert all(type(count) is int and count > 0 for counts in bank.graph.transitions.values()
               for count in counts.values())


def test_identical_observed_prefixes_have_identical_predictions_and_state():
    # Hidden labels 'closure' and 'stationary unlucky run' never enter the API.
    closure, unlucky = memory(), memory()
    for destination in [(0, 0)] * 8 + [(1, 0)] * 3:
        assert closure.observe((0, 0), 1, destination) == unlucky.observe((0, 0), 1, destination)
    assert closure.diagnostics() == unlucky.diagnostics()
    assert closure.graph.to_state() == unlucky.graph.to_state()


def test_pair_evidence_is_local_and_reads_are_detached():
    source = baseline()
    initial = copy.deepcopy(source.to_state())
    bank = memory(source)
    for _ in range(4):
        bank.observe((0, 0), 1, (0, 0))
    unchanged = bank.predict((1, 0), 1)
    assert len(bank.diagnostics()['pairs']) == 1
    snapshot = bank.diagnostics()
    snapshot['pairs'][0]['evidence'] = 999
    snapshot['losses']['adaptive']['brier'] = 999
    graph = bank.graph
    graph.nodes.clear()
    assert bank.diagnostics()['pairs'][0]['evidence'] != 999
    assert bank.graph.nodes
    assert bank.predict((1, 0), 1) == unchanged
    assert source.to_state() == initial


@pytest.mark.parametrize('source,action,destination', [
    ((0, 0), True, (0, 0)), ((0, 0), 4, (0, 0)), ((0, 0), 1, (2, 0)),
    ((-1, 0), 1, (0, 0)), ((0, 0), 0, (0, -1)), ((0., 0), 1, (1, 0)),
])
def test_invalid_observation_is_atomic(source, action, destination):
    bank = memory()
    before = bank.diagnostics(), bank.graph.to_state()
    with pytest.raises(ValueError):
        bank.observe(source, action, destination)
    assert (bank.diagnostics(), bank.graph.to_state()) == before


@pytest.mark.parametrize('window', [True, 0, 3, 32, 8.0])
def test_window_validation(window):
    with pytest.raises(ValueError):
        memory(window=window)


def test_unobserved_action_prediction_does_not_add_edges_and_boundary_collision_is_valid():
    bank = memory()
    before = bank.graph.to_state()
    assert bank.predict((0, 0), 0) == dict.fromkeys(('cumulative', 'recent', 'guarded_recent', 'adaptive'), .5)
    assert bank.graph.to_state() == before
    bank.observe((0, 0), 0, (0, 0))
    assert bank.graph.transitions[((0, 0), 0)] == {(0, 0): 1}
    assert bank.predict((0, 0), 0)['adaptive'] == .25


def test_event_cap_is_enforced_before_mutation_and_losses_are_exact_totals():
    bank = memory()
    losses = {key: {'brier': 0., 'log_loss': 0.} for key in bank.predict((0, 0), 1)}
    for index in range(2048):
        result = bank.observe((0, 0), 1, (index % 2, 0))
        for key in losses:
            for metric in losses[key]:
                losses[key][metric] += result[metric][key]
    before = bank.diagnostics()
    assert before['events'] == 2048
    assert before['losses'] == losses
    with pytest.raises(ValueError, match='event'):
        bank.observe((0, 0), 1, (0, 0))
    assert bank.diagnostics() == before


def test_oversize_initial_graph_rejected():
    graph = TopologicalMemoryGraph()
    for x in range(17):
        for z in range(16):
            graph.add_node((x, z))
    with pytest.raises(ValueError, match='node'):
        memory(graph)


def test_near_maximum_count_log_loss_remains_finite_on_unlikely_outcome():
    graph = baseline()
    key = ((0, 0), 1)
    graph.transitions[key] = {(1, 0): TopologicalMemoryGraph.MAX_TRANSITIONS - 2}
    graph.total_transitions = TopologicalMemoryGraph.MAX_TRANSITIONS - 1
    graph.adjacency[(0, 0)] = {(1, 0)}
    bank = memory(graph)
    result = bank.observe((0, 0), 1, (0, 0))
    assert all(math.isfinite(value) for value in result['log_loss'].values())
    assert result['log_loss']['adaptive'] > 30
    before = bank.diagnostics()
    with pytest.raises(ValueError, match='count'):
        bank.observe((0, 0), 1, (0, 0))
    assert bank.diagnostics() == before
