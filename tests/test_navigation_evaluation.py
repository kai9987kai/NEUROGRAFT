"""Frozen navigation trials must measure behavior without training the live model."""
import json
import random

import pytest
import torch


def _session():
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    return LiveSession(make_fixture_bundle(nodes_per_source=4), LiveConfig(hidden_dim=4, batch_size=2))


def test_frozen_trials_are_matched_deterministic_and_leave_live_state_and_rng_untouched():
    from neurograft.navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation
    from neurograft.live import _checkpoint_digest
    session = _session()
    session.tick(3)
    before = _checkpoint_digest(session._payload())
    config = NavigationEvalConfig(seeds=(10001, 10002), size=5, max_steps=16)
    rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    first = frozen_navigation_evaluation(session, config)
    second = frozen_navigation_evaluation(session, config)
    assert first == second
    assert _checkpoint_digest(session._payload()) == before
    assert random.getstate() == rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert set(first['variants']) == {'policy', 'no_graft', 'no_recurrence', 'random', 'shortest_path'}
    initial_worlds = [e['initial_world_hash'] for e in first['variants']['policy']['episodes']]
    for variant in first['variants'].values():
        assert [e['initial_world_hash'] for e in variant['episodes']] == initial_worlds
    oracle = first['variants']['shortest_path']
    assert oracle['summary']['success_rate'] == 1 and oracle['summary']['spl'] == 1
    assert oracle['summary']['collision_rate'] == 0
    assert first['teacher_assistance'] == 0 and first['learning'] is False
    assert first['checkpoint_steps'] == 3 and len(first['model_hash']) == 64
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize('values', [
    {'seeds': ()}, {'seeds': (1, 1)}, {'seeds': (True,)}, {'seeds': tuple(range(17))},
    {'seeds': (1.5,)}, {'size': True}, {'size': 32}, {'size': 4},
    {'max_steps': 0}, {'max_steps': 129}, {'max_steps': True}, {'preset': 'missing'},
])
def test_invalid_protocol_is_rejected(values):
    from neurograft.navigation_evaluation import NavigationEvalConfig
    with pytest.raises(ValueError):
        NavigationEvalConfig(**values)


def test_evaluation_identity_changes_with_protocol_and_weights():
    from neurograft.navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation
    session = _session()
    config = NavigationEvalConfig(seeds=(7,), size=5, max_steps=1)
    report = frozen_navigation_evaluation(session, config)
    other = frozen_navigation_evaluation(session, NavigationEvalConfig(seeds=(8,), size=5, max_steps=1))
    assert other['protocol_id'] != report['protocol_id']
    with torch.no_grad():
        session.model.policy.bias[0] += 1
    changed = frozen_navigation_evaluation(session, config)
    assert changed['model_hash'] != report['model_hash']
    assert changed['protocol_id'] == report['protocol_id']


def test_short_horizon_reports_failures_and_attempts_without_nan():
    from neurograft.navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation
    report = frozen_navigation_evaluation(_session(), NavigationEvalConfig(seeds=(11,), size=11, max_steps=1))
    for value in report['variants'].values():
        episode = value['episodes'][0]
        assert episode['steps'] == 1 and len(episode['actions']) == 1
        assert episode['shortest_distance'] > 1
        assert episode['success'] is False and episode['spl'] == 0
        assert 0 <= value['summary']['collision_rate'] <= 1


def test_cli_evaluates_checkpoint_and_preserves_existing_output(tmp_path, capsys):
    from neurograft.cli import main
    session = _session()
    bundle = tmp_path / 'bundle.json'
    session.bundle.save(bundle)
    checkpoint = tmp_path / 'session.pt'
    session.save(checkpoint)
    before = checkpoint.read_bytes()
    out = tmp_path / 'result.json'
    args = ['evaluate-live', '--bundle', str(bundle), '--checkpoint', str(checkpoint),
            '--seeds', '10001', '--size', '5', '--max-steps', '2', '--out', str(out)]
    assert main(args) == 0
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report['checkpoint_steps'] == 0 and report['variants']['policy']['summary']['episodes'] == 1
    assert checkpoint.read_bytes() == before
    result_before = out.read_bytes()
    assert main(args) == 2
    assert out.read_bytes() == result_before
    assert 'already exists' in capsys.readouterr().err


def test_world_save_restores_preset_and_accepts_legacy_missing_preset():
    from neurograft.environment import NavigationWorld
    world = NavigationWorld(seed=7, size=11, preset='hierarchical')
    saved = world.to_dict()
    assert NavigationWorld.from_dict(saved).preset == 'hierarchical'
    saved.pop('preset')
    assert NavigationWorld.from_dict(saved).preset == 'labyrinth'
