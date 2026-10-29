"""Behavioral integration of empirical replay, controllers and continuation."""
import copy

import pytest

from neurograft import navigation_controller as controllers
from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession
from neurograft.memory import TopologicalMemoryGraph
from neurograft.service import LiveApp


def test_replay_updates_change_the_actual_legal_decision():
    from neurograft.replay import ReplayPlanner
    graph = TopologicalMemoryGraph()
    graph.record_transition((2, 2), 1, (3, 2))
    graph.record_transition((3, 2), 1, (4, 2))
    graph.record_transition((4, 2), 1, (5, 2))
    graph.record_transition((2, 2), 3, (1, 2))
    observation = [[0.0] * 4, [0.0] * 4, [1, 0, 1, 0], [0.0] * 4]
    logits = [0.0, 0.0, 0.0, 10.0]
    planner = ReplayPlanner()
    choose = controllers.replay_steering_action
    assert choose(observation, logits, planner, graph, (2, 2), (5, 2)) == 3
    planner.replay(graph, (5, 2), budget=8)
    assert choose(observation, logits, planner, graph, (2, 2), (5, 2)) == 1
    observation[2][1] = 1
    assert choose(observation, logits, planner, graph, (2, 2), (5, 2)) == 3


def _session():
    return LiveSession(make_fixture_bundle(nodes_per_source=3), LiveConfig(hidden_dim=4, unroll=2))


def test_live_replay_reads_are_pure_and_checkpoint_continuation_is_exact(tmp_path):
    session = _session()
    session.learning = False
    origin = tuple(session.world._agent)
    for _ in range(16):
        session.tick(manual_action=session.world.expert_action())
    # The first source may have no incoming experience on a directed path.
    # Select an actually reached destination so the goal has observed support.
    goal = next(destination for counts in session.memory.topological_graph.transitions.values()
                for destination in counts if destination != origin)
    session.set_goal(x=goal[0], z=goal[1])
    session.set_navigation_mode('replay_steer')
    result = session.replay_memory(budget=64)
    assert result['goal_known'] is True
    assert result['backups'] > 0
    before = copy.deepcopy(session.replay_planner.to_state())
    first = session.snapshot()
    session.telemetry()
    session.snapshot()
    assert session.replay_planner.to_state() == before
    assert first['navigation']['replay']['total_backups'] > 0
    assert 'empirical' in first['navigation']['reason'].lower()
    checkpoint = tmp_path / 'continued.pt'
    session.save(checkpoint)
    restored = LiveSession.load(checkpoint, session.bundle)
    assert restored.replay_planner.to_state() == before
    session.tick(steps=4)
    restored.tick(steps=4)
    assert restored.replay_planner.to_state() == session.replay_planner.to_state()
    assert restored.world.to_dict() == session.world.to_dict()
    assert restored.last_action_source == 'replay_steer'


def test_memory_free_bypasses_replay_and_world_reset_clears_values():
    session = _session()
    session.set_navigation_mode('replay_steer')
    session.memory.set_lesion_mode('memory_free')
    assert session.replay_memory(budget=64)['backups'] == 0
    before = copy.deepcopy(session.replay_planner.to_state())
    session.tick(steps=3)
    assert session.replay_planner.to_state() == before
    session.resize_world(size=11, preset='chambers')
    assert session.replay_planner.goal is None
    assert session.replay_planner.values == {}


def test_legacy_navigation_without_planner_still_loads_and_invalid_planner_is_atomic():
    session = _session()
    legacy = copy.deepcopy(session._payload())
    legacy['navigation'].pop('replay_planner', None)
    session._restore_validated(legacy)
    assert session.replay_planner.values == {}
    malformed = copy.deepcopy(session._payload())
    malformed['navigation']['replay_planner']['values'] = [{'position': [1, 1], 'value': float('nan')}]
    before = session.snapshot()
    with pytest.raises(ValueError):
        session._restore_validated(malformed)
    assert session.snapshot() == before


def test_service_manual_replay_uses_controller_planner_without_advancing(tmp_path):
    app = LiveApp(make_fixture_bundle(nodes_per_source=3), tmp_path / 'isolated.pt', background=False)
    try:
        app.control({'action': 'pause'})
        app.control({'action': 'navigation', 'mode': 'replay_steer'})
        before = app.state()
        result = app.control({'action': 'replay', 'sweeps': 2})
        assert result['replay']['budget'] == 16
        assert result['replay']['backups'] <= 16
        assert result['steps'] == before['steps']
        assert result['model_hash'] == before['model_hash']
        assert 'goal-conditioned' in app.session.chat('replay')['reply']
    finally:
        app.close(save=False)


def test_replay_study_commands_produce_and_verify_independent_artifacts(tmp_path, capsys):
    from neurograft.cli import main
    destination = tmp_path / 'study'
    assert main(['study-replay', '--out', str(destination), '--seeds', '11,12', '--budgets', '0,4']) == 0
    capsys.readouterr()
    assert main(['verify-replay-study', str(destination), '--replay']) == 0
    assert 'true' in capsys.readouterr().out.lower()


def test_goal_and_new_experience_mark_cached_replay_diagnostics_stale():
    session = _session()
    session.learning = False
    for _ in range(8):
        session.tick(manual_action=session.world.expert_action())
    destinations = sorted({v for counts in session.memory.topological_graph.transitions.values() for v in counts})
    first, second = destinations[0], destinations[-1]
    assert first != second
    session.set_navigation_mode('replay_steer')
    session.set_goal(x=first[0], z=first[1])
    session.replay_memory(budget=128)
    assert session.snapshot()['navigation']['replay']['stale'] is False
    before = copy.deepcopy(session.replay_planner.to_state())
    session.set_goal(x=second[0], z=second[1])
    diagnostic = session.snapshot()['navigation']['replay']
    assert diagnostic['goal_changed'] is True
    assert diagnostic['stale'] is True
    assert session.replay_planner.to_state() == before
    session.replay_memory(budget=128)
    assert session.snapshot()['navigation']['replay']['stale'] is False
    session.tick()
    assert session.snapshot()['navigation']['replay']['stale'] is True
