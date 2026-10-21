"""Experienced-edge continuation, migration and validation regressions."""
import copy

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest
from neurograft.memory import TopologicalMemoryGraph


def session():
    return LiveSession(make_fixture_bundle(seed=5, nodes_per_source=3), LiveConfig(
        seed=19, hidden_dim=4, threads=1, batch_size=2, train_every=2,
        unroll=3, replay_capacity=8))


def test_live_first_executed_action_records_pre_and_post_pose():
    live = session()
    before = live.world._agent
    live.tick(manual_action=1)
    after = live.world._agent
    graph = live.memory.topological_graph
    assert graph.total_transitions == 1
    assert graph.transition_probability(before, 1, after) == 1.0


def test_topological_graph_continues_exactly_after_checkpoint(tmp_path):
    live = session()
    live.tick(5)
    path = tmp_path / "topological.pt"
    live.save(path)
    restored = LiveSession.load(path, live.bundle)
    assert restored.memory.topological_graph.total_transitions == 5
    assert restored.memory.topological_graph.to_state() == live.memory.topological_graph.to_state()
    live.tick(3)
    restored.tick(3)
    assert restored.memory.topological_graph.to_state() == live.memory.topological_graph.to_state()
    assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())


@pytest.mark.parametrize("version", [2, 3])
def test_old_auxiliary_without_topological_state_loads_empty_graph(tmp_path, version):
    live = session()
    live.tick(4)
    payload = copy.deepcopy(live._payload())
    payload["version"] = version
    payload["auxiliary"].pop("topological_memory", None)
    if version == 2:
        payload.pop("navigation", None)
    path = tmp_path / "old.pt"
    torch.save({"payload": payload, "sha256": _checkpoint_digest(payload)}, path)
    restored = LiveSession.load(path, live.bundle)
    assert restored.memory.topological_graph.total_transitions == 0
    assert not restored.memory.topological_graph.nodes
    assert restored.memory.episodes
    restored.tick()
    assert restored.memory.topological_graph.total_transitions == 1


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(total_transitions=3),
    lambda s: s["transitions"][0].update(destination=[30, 30]),
    lambda s: s["transitions"][0].update(count=True),
    lambda s: s["transitions"].append(copy.deepcopy(s["transitions"][0])),
    lambda s: s["reward_totals"][0].update(total_reward=float("nan")),
    lambda s: s["nodes"].append([31, 0]),
])
def test_invalid_graph_state_is_rejected_before_session_changes(mutate):
    live = session()
    live.tick()
    payload = copy.deepcopy(live._payload())
    mutate(payload["auxiliary"]["topological_memory"])
    target = session()
    before = _checkpoint_digest(target._payload())
    with pytest.raises(ValueError):
        target._restore_validated(payload)
    assert _checkpoint_digest(target._payload()) == before


def test_graph_transition_limit_rejects_increment_without_partial_changes():
    graph = TopologicalMemoryGraph()
    graph.record_transition((1, 1), 1, (2, 1), 0.25)
    state = graph.to_state()
    state["total_transitions"] = 2**53 - 1
    state["transitions"][0]["count"] = 2**53 - 1
    graph = TopologicalMemoryGraph.from_state(state)
    with pytest.raises(ValueError):
        graph.record_transition((2, 1), 1, (3, 1), 0.25)
    assert graph.to_state() == state


def test_map_replacement_clears_spatial_graph_but_preserves_episodes():
    live = session()
    live.tick(3)
    episodes = list(live.memory.episodes)
    live.resize_world(17, "chambers")
    assert not live.memory.spatial_map
    assert not live.memory.topological_graph.nodes
    assert list(live.memory.episodes) == episodes
    live.tick(manual_action=0)
    assert live.memory.topological_graph.total_transitions == 1


def test_same_map_reset_preserves_edges_and_next_tick_has_no_teleport_edge():
    live = session()
    live.tick(3)
    old_count = live.memory.topological_graph.total_transitions
    live.chat("reset world")
    before = live.world._agent
    live.tick(manual_action=2)
    assert live.memory.topological_graph.total_transitions == old_count + 1
    assert live.memory.topological_graph.transition_probability(before, 2, live.world._agent) > 0


def test_timeout_reset_does_not_record_teleport_as_experienced_edge():
    live = session()
    live.world.episode_steps = live.world.MAX_EPISODE_STEPS - 1
    live.tick(manual_action=live.world.expert_action() ^ 2)
    assert live.memory.topological_graph.total_transitions == 0


def test_graph_telemetry_is_observer_only_and_side_effect_free():
    live = session()
    live.tick(2)
    before = _checkpoint_digest(live._payload())
    summary = live.telemetry()["memory"]["topological"]
    assert summary["total_transitions"] == 2
    assert "observer" in summary["role"]
    assert _checkpoint_digest(live._payload()) == before
