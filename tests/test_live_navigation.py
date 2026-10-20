"""Opt-in observation-only control and strict session continuation."""
import copy

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest


def session():
    return LiveSession(make_fixture_bundle(seed=5, nodes_per_source=3), LiveConfig(
        seed=19, hidden_dim=4, threads=1, batch_size=2, train_every=2,
        unroll=3, replay_capacity=8, teacher_probability=1.0))


@pytest.mark.parametrize('mode', ['masked_policy', 'observed_map'])
def test_opt_in_control_uses_observation_and_bypasses_teacher_mixing(mode, monkeypatch):
    live = session()
    live.learning = False
    live.world._agent = next(cell for cell in live.world._beacons.values() if cell != live.world._goal)
    live.world._trail = [live.world._agent]
    live.set_navigation_mode(mode)
    # The teacher deliberately recommends a visible wall. Its label may still
    # enter replay, but must never supply an action in either opt-in mode.
    wall = next(i for i, blocked in enumerate(live.world.observe()[2]) if blocked)
    monkeypatch.setattr(live.world, 'expert_action', lambda: wall)
    before_rng = live.rng.getstate()
    before_reward = live.world.total_reward
    state = live.tick()
    assert state['last_action_source'] == mode
    assert state['last_action'] != wall
    assert live.world.total_reward > before_reward - 0.1
    assert live.rng.getstate() == before_rng
    assert state['navigation']['teacher_probability'] == 0.0
    assert state['navigation']['mode'] == mode


def test_default_policy_retains_teacher_mixing_and_does_not_build_map():
    live = session()
    expected_action = live.world.expert_action()
    state = live.tick()
    assert state['last_action_source'] == 'teacher'
    assert state['last_action'] == expected_action
    assert state['navigation']['mode'] == 'policy'
    assert state['navigation']['teacher_probability'] == 1.0
    assert not live.navigator.known_free


def test_manual_actions_observe_map_and_goal_change_discards_episodic_knowledge():
    live = session()
    live.learning = False
    live.set_navigation_mode('observed_map')
    start = live.world._agent
    action = live.world.expert_action()
    live.tick(manual_action=action)
    assert live.last_action_source == 'manual'
    assert start in live.navigator.known_free
    assert live.navigator.visits[start] == 1
    live.set_goal(x=start[0], z=start[1])
    assert not live.navigator.known_free and not live.navigator.visits
    assert live.navigation_mode == 'observed_map'


def test_mode_validation_is_atomic_and_resize_rebuilds_memory():
    live = session()
    live.set_navigation_mode('observed_map')
    live.tick()
    before = _checkpoint_digest(live._payload())
    for invalid in ('oracle', None, True, 1):
        with pytest.raises(ValueError):
            live.set_navigation_mode(invalid)
        assert _checkpoint_digest(live._payload()) == before
    live.resize_world(7, 'chambers')
    assert live.navigator.size == 7
    assert not live.navigator.known_free
    live.tick()
    live.set_navigation_mode('policy')
    assert not live.navigator.known_free


def test_chat_names_the_controller_and_effective_teacher_assistance():
    live = session()
    live.set_navigation_mode('observed_map')
    live.tick()
    summary = live.chat('navigation')['reply']
    assert 'observed_map' in summary and '0%' in summary
    assert 'reason' in summary.lower()
    explanation = live.chat('why')['reply']
    assert 'observed_map' in explanation and '0%' in explanation
    assert 'teacher actions bypassed' in explanation.lower()


@pytest.mark.parametrize('mode', ['policy', 'masked_policy', 'observed_map'])
def test_every_control_mode_resumes_exact_next_actions_and_state(tmp_path, mode):
    live = session()
    live.set_navigation_mode(mode)
    live.tick(4)
    path = tmp_path / (mode + '.pt')
    live.save(path)
    restored = LiveSession.load(path, live.bundle)
    assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())
    for count in (1, 3):
        live.tick(count)
        restored.tick(count)
        assert live.snapshot()['navigation'] == restored.snapshot()['navigation']
        assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())


def test_legacy_v2_restores_default_policy_with_exact_existing_continuation(tmp_path):
    live = session()
    live.tick(4)
    payload = copy.deepcopy(live._payload())
    payload.pop('navigation')
    payload['version'] = 2
    path = tmp_path / 'v2.pt'
    torch.save({'payload': payload, 'sha256': _checkpoint_digest(payload)}, path)
    restored = LiveSession.load(path, live.bundle)
    assert restored.navigation_mode == 'policy'
    assert restored.checkpoint_compatibility == {'loaded_format': 2, 'auxiliary_state_restored': True}
    assert not restored.navigator.known_free
    live.tick(3)
    restored.tick(3)
    assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())


@pytest.mark.parametrize('mutation', [
    lambda nav: nav.update(mode='unknown'),
    lambda nav: nav['navigator'].update(size=9),
    lambda nav: nav['navigator']['known_free'].append([0, 0]),
    lambda nav: nav.update(extra=True),
])
def test_bad_navigation_checkpoint_fails_before_installation(mutation):
    live = session()
    live.set_navigation_mode('observed_map')
    live.tick(2)
    payload = copy.deepcopy(live._payload())
    mutation(payload['navigation'])
    target = session()
    before = _checkpoint_digest(target._payload())
    with pytest.raises(ValueError):
        target._restore_validated(payload)
    assert _checkpoint_digest(target._payload()) == before


def test_navigation_visit_history_cannot_exceed_executed_session_steps():
    live = session()
    live.set_navigation_mode('observed_map')
    live.tick(2)
    payload = copy.deepcopy(live._payload())
    payload['navigation']['navigator']['visits'][0]['count'] = 3
    target = session()
    before = _checkpoint_digest(target._payload())
    with pytest.raises(ValueError, match='visit'):
        target._restore_validated(payload)
    assert _checkpoint_digest(target._payload()) == before


@pytest.mark.parametrize('displacement', ['far', 'missed_move'])
def test_navigation_saved_pose_must_match_the_executed_transition(displacement):
    live = session()
    live.learning = False
    live.set_navigation_mode('observed_map')
    live.tick()
    payload = copy.deepcopy(live._payload())
    sensed = live.navigator.current
    assert live.world._agent != sensed
    displaced = sensed if displacement == 'missed_move' else next(
        (x, z) for x in range(1, live.world.size - 1) for z in range(1, live.world.size - 1)
        if live.world._free((x, z)) and abs(x - sensed[0]) + abs(z - sensed[1]) > 1)
    payload['world']['agent'].update(x=displaced[0], z=displaced[1])
    payload['world']['trail'] = [{'x': displaced[0], 'z': displaced[1]}]
    target = session()
    before = _checkpoint_digest(target._payload())
    with pytest.raises(ValueError, match='navigation.*position|navigation.*transition'):
        target._restore_validated(payload)
    assert _checkpoint_digest(target._payload()) == before


@pytest.mark.parametrize('collision', [False, True])
def test_navigation_manual_transition_and_collision_continue_exactly(tmp_path, collision):
    live = session()
    live.learning = False
    live.world._agent = next(cell for cell in live.world._beacons.values() if cell != live.world._goal)
    live.world._trail = [live.world._agent]
    live.set_navigation_mode('observed_map')
    manual = next(i for i, wall in enumerate(live.world.observe()[2]) if bool(wall) == collision)
    previous = live.world._agent
    live.tick(manual_action=manual)
    assert (live.world._agent == previous) == collision
    assert live.last_action_source == 'manual'
    path = tmp_path / 'manual.pt'
    live.save(path)
    restored = LiveSession.load(path, live.bundle)
    assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())
    live.tick()
    restored.tick()
    assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())
