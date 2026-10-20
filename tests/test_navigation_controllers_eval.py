import copy
import json

import pytest
import torch


def session():
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    return LiveSession(make_fixture_bundle(nodes_per_source=3), LiveConfig(hidden_dim=4, unroll=2))


def test_extended_trials_isolate_masking_memory_and_policy_prior():
    from neurograft.navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation
    from neurograft.live import _checkpoint_digest
    live = session()
    before = _checkpoint_digest(live._payload())
    report = frozen_navigation_evaluation(live, NavigationEvalConfig(
        seeds=(10000, 10001), size=5, max_steps=32, include_controllers=True))
    assert set(report['variants']) == {'policy', 'no_graft', 'no_recurrence', 'random',
        'shortest_path', 'masked_policy', 'observed_map', 'map_without_policy'}
    assert report['schema_version'] == 2
    assert report['variants']['masked_policy']['summary']['collision_rate'] == 0
    for key in ('observed_map', 'map_without_policy'):
        item = report['variants'][key]
        assert item['summary']['success_rate'] == 1 and item['summary']['collision_rate'] == 0
        for episode in item['episodes']:
            assert len(episode['controller_trace']) == episode['steps']
            assert episode['controller_trace'][-1]['known_free'] <= 9
            assert episode['policy_prior_used'] is (key == 'observed_map')
    hashes = [episode['initial_world_hash'] for episode in report['variants']['policy']['episodes']]
    assert all([episode['initial_world_hash'] for episode in value['episodes']] == hashes
               for value in report['variants'].values())
    assert _checkpoint_digest(live._payload()) == before
    assert frozen_navigation_evaluation(live, NavigationEvalConfig(
        seeds=(10000, 10001), size=5, max_steps=32, include_controllers=True)) == report
    json.dumps(report, allow_nan=False)


def test_legacy_trial_protocol_and_config_stay_unchanged():
    from neurograft.navigation_evaluation import NavigationEvalConfig, navigation_protocol, frozen_navigation_evaluation
    config = NavigationEvalConfig(seeds=(17,), size=5, max_steps=2)
    protocol = navigation_protocol(config, 2)
    assert protocol['schema_version'] == 1
    assert 'include_controllers' not in protocol['config']
    report = frozen_navigation_evaluation(session(), config)
    assert report['schema_version'] == 1 and len(report['variants']) == 5
    assert 'controller_trace' not in report['variants']['policy']['episodes'][0]


@pytest.mark.parametrize('value', [None, 1, 'true', []])
def test_controller_suite_flag_is_strict_boolean(value):
    from neurograft.navigation_evaluation import NavigationEvalConfig
    with pytest.raises(ValueError):
        NavigationEvalConfig(include_controllers=value)
