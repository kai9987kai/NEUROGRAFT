"""Tests for TopologicalMemoryGraph and Tri-Tier Cognitive Spatial Memory."""
import pytest
from neurograft.memory import TopologicalMemoryGraph, TriTierMemory


def test_topological_memory_node_and_edge_addition():
    graph = TopologicalMemoryGraph()
    assert len(graph.nodes) == 0

    # Record transition from (1, 1) to (1, 2) via action 2 (South)
    graph.record_transition((1, 1), 2, (1, 2), reward=0.5)
    assert len(graph.nodes) == 2
    assert (1, 1) in graph.nodes
    assert (1, 2) in graph.nodes
    assert graph.total_transitions == 1
    assert (1, 2) in graph.adjacency[(1, 1)]
    assert graph.reward_totals[(1, 2)] == 0.5


def test_topological_memory_transition_probabilities():
    graph = TopologicalMemoryGraph()
    # From (2, 2) action 1 leads to (3, 2) 3 times, and to (2, 2) (failed move/wall) 1 time
    graph.record_transition((2, 2), 1, (3, 2))
    graph.record_transition((2, 2), 1, (3, 2))
    graph.record_transition((2, 2), 1, (3, 2))
    graph.record_transition((2, 2), 1, (2, 2))

    prob_success = graph.transition_probability((2, 2), 1, (3, 2))
    prob_fail = graph.transition_probability((2, 2), 1, (2, 2))
    prob_other = graph.transition_probability((2, 2), 1, (0, 0))

    assert prob_success == 0.75
    assert prob_fail == 0.25
    assert prob_other == 0.0


def test_topological_memory_shortest_path():
    graph = TopologicalMemoryGraph()
    # Path: (0, 0) -> (1, 0) -> (2, 0) -> (2, 1)
    graph.record_transition((0, 0), 1, (1, 0))
    graph.record_transition((1, 0), 1, (2, 0))
    graph.record_transition((2, 0), 2, (2, 1))

    # Same node
    path_same = graph.shortest_topological_path((0, 0), (0, 0))
    assert path_same == [(0, 0)]

    # Shortest path
    path = graph.shortest_topological_path((0, 0), (2, 1))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1)]

    # Unreachable node
    graph.add_node((5, 5))
    path_unreachable = graph.shortest_topological_path((0, 0), (5, 5))
    assert path_unreachable is None

    # Unknown node
    assert graph.shortest_topological_path((0, 0), (99, 99)) is None


def test_topological_memory_reachability_ratio():
    graph = TopologicalMemoryGraph()
    assert graph.reachability_ratio((0, 0)) == 0.0

    # Component 1: (0, 0) -> (1, 0)
    graph.record_transition((0, 0), 1, (1, 0))
    # Disconnected node: (5, 5)
    graph.add_node((5, 5))

    # From (0, 0), can reach (0, 0) and (1, 0) -> 2 out of 3 nodes
    assert abs(graph.reachability_ratio((0, 0)) - (2.0 / 3.0)) < 1e-6
    # From (5, 5), can only reach itself -> 1 out of 3 nodes
    assert abs(graph.reachability_ratio((5, 5)) - (1.0 / 3.0)) < 1e-6


def test_tri_tier_memory_topological_integration():
    mem = TriTierMemory(capacity=32, lesion_mode="intact")
    mem.record_step(step=0, agent_pos=(1, 1), action=0, reward=0.0, state_features=[0.0] * 8)
    mem.record_step(step=1, agent_pos=(1, 2), action=2, reward=0.1, state_features=[0.1] * 8,
                    transition_start=(1, 1))
    mem.record_step(step=2, agent_pos=(2, 2), action=1, reward=1.0, state_features=[0.2] * 8,
                    transition_start=(1, 2))

    assert len(mem.topological_graph.nodes) == 3
    assert mem.topological_graph.total_transitions == 2
    path = mem.topological_graph.shortest_topological_path((1, 1), (2, 2))
    assert path == [(1, 1), (1, 2), (2, 2)]

    topo_dict = mem.topological_graph.to_dict()
    assert topo_dict["num_nodes"] == 3
    assert topo_dict["num_edges"] == 2
    assert topo_dict["total_transitions"] == 2

    # Reset clears topological graph
    mem.reset()
    assert len(mem.topological_graph.nodes) == 0
    assert mem.topological_graph.total_transitions == 0


def test_explicit_start_records_first_action_and_legacy_steps_do_not_invent_edges():
    mem = TriTierMemory()
    mem.record_step(1, (2, 1), 1, 0.125, [0.0] * 8, transition_start=(1, 1))
    assert mem.topological_graph.transition_probability((1, 1), 1, (2, 1)) == 1.0
    assert mem.topological_graph.total_transitions == 1
    # A caller without a pre-action pose has observations, not an experienced edge.
    mem.record_step(2, (8, 8), 3, 0.0, [0.0] * 8)
    mem.record_step(3, (7, 8), 3, 0.0, [0.0] * 8)
    assert mem.topological_graph.total_transitions == 1


@pytest.mark.parametrize("start,action,end,reward", [
    ((-1, 0), 1, (0, 0), 0.0),
    ((30, 0), 1, (31, 0), 0.0),
    ((0.5, 0), 1, (1, 0), 0.0),
    ((True, 0), 1, (1, 0), 0.0),
    ((0, 0), True, (1, 0), 0.0),
    ((0, 0), 4, (1, 0), 0.0),
    ((0, 0), 1, (2, 0), 0.0),
    ((0, 0), 1, (0, 1), 0.0),
    ((0, 0), 1, (1, 0), float("nan")),
    ((0, 0), 1, (1, 0), float("inf")),
    ((0, 0), 1, (1, 0), True),
])
def test_invalid_transition_is_rejected_atomically(start, action, end, reward):
    graph = TopologicalMemoryGraph()
    graph.record_transition((1, 1), 1, (2, 1), 0.25)
    before = graph.to_state()
    with pytest.raises(ValueError):
        graph.record_transition(start, action, end, reward)
    assert graph.to_state() == before


def test_canonical_graph_roundtrip_preserves_counts_rewards_and_directed_paths():
    graph = TopologicalMemoryGraph()
    graph.add_node((30, 30))
    graph.record_transition((1, 1), 1, (2, 1), 0.125)
    graph.record_transition((1, 1), 1, (1, 1), -0.1)
    graph.record_transition((1, 1), 1, (2, 1), 0.5)
    graph.record_transition((2, 1), 2, (2, 2), 1.0)
    state = graph.to_state()
    restored = TopologicalMemoryGraph.from_state(state)
    assert restored.to_state() == state
    assert restored.transition_probability((1, 1), 1, (2, 1)) == 2 / 3
    assert restored.reward_totals[(2, 1)] == 0.625
    assert restored.shortest_topological_path((1, 1), (2, 2)) == [(1, 1), (2, 1), (2, 2)]
    assert restored.shortest_topological_path((2, 2), (1, 1)) is None
    state["nodes"][0][0] = 20
    assert restored.to_state() == graph.to_state()


def test_bfs_tie_break_is_coordinate_order_independent_of_recording_order():
    graph = TopologicalMemoryGraph()
    for start, action, end in [((1, 1), 1, (2, 1)), ((2, 1), 2, (2, 2)),
                               ((1, 1), 2, (1, 2)), ((1, 2), 1, (2, 2))]:
        graph.record_transition(start, action, end)
    assert graph.shortest_topological_path((1, 1), (2, 2)) == [(1, 1), (1, 2), (2, 2)]


def test_memory_free_never_grows_topological_or_spatial_observers():
    mem = TriTierMemory(lesion_mode="memory_free")
    mem.record_step(1, (2, 1), 1, 0.1, [0.0] * 8, transition_start=(1, 1))
    assert not mem.topological_graph.nodes
    assert not mem.spatial_map
    assert not mem.episodes


def test_spatial_reset_preserves_episodic_memory_without_linking_maps():
    mem = TriTierMemory()
    mem.record_step(1, (2, 1), 1, 0.1, [0.0] * 8, transition_start=(1, 1))
    mem.reset_spatial()
    assert not mem.topological_graph.nodes and not mem.spatial_map
    assert len(mem.episodes) == 1
    assert mem.total_recorded_steps == 1
