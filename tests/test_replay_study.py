"""Matched replay schedules must retain exposure, controls and verifiable evidence."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def study_module():
    assert importlib.util.find_spec('neurograft.replay_study'), 'matched replay study is missing'
    return importlib.import_module('neurograft.replay_study')


@pytest.fixture(scope='module')
def archived(tmp_path_factory):
    module = study_module()
    directory = tmp_path_factory.mktemp('replay-study') / 'run'
    report = module.run_replay_study(directory, seeds=(17, 29), budgets=(0, 4, 64))
    return module, directory, report


def test_study_matches_history_exposure_and_goal_relocation(archived):
    _, _, report = archived
    assert len(report['instances']) == 2
    for instance in report['instances']:
        assert instance['stochastic_outcomes'] > 0
        assert [stage['id'] for stage in instance['exposures']] == ['partial', 'full']
        for stage in instance['exposures']:
            assert 0 < stage['experience_fraction'] <= 1
            assert stage['history_sha256']
            for phase in stage['phases']:
                assert {arm['strategy'] for arm in phase['arms']} == {'prioritized', 'uniform', 'none'}
                assert phase['reference_residual'] < 1e-10
                for arm in phase['arms']:
                    assert [point['budget'] for point in arm['curve']] == [0, 4, 64]
                    for point in arm['curve']:
                        assert 0 <= point['backups'] <= point['budget']
                        assert point['value_mae'] >= 0
                        assert point['action_agreement'] is None or 0 <= point['action_agreement'] <= 1
                        if arm['strategy'] == 'none' or point['budget'] == 0:
                            assert point['backups'] == 0
                if phase['id'] == 'relocated_goal':
                    assert phase['warm_goal'] != phase['goal']
                    assert phase['warm_backups'] > 0
            assert stage['unknown_goal']['backups'] == 0
            assert stage['unknown_goal']['max_value'] == 0


def test_reports_are_exactly_repeatable_and_full_replay_verifies(archived, tmp_path):
    module, directory, report = archived
    second = tmp_path / 'second'
    assert module.run_replay_study(second, seeds=(17, 29), budgets=(0, 4, 64)) == report
    assert (second / 'report.json').read_bytes() == (directory / 'report.json').read_bytes()
    assert module.verify_replay_study(directory, replay=True)['ok'] is True
    assert 'Empirical' in (directory / 'report.html').read_text(encoding='utf-8')


def test_existing_output_is_never_overwritten(archived):
    module, directory, _ = archived
    before = (directory / 'report.json').read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        module.run_replay_study(directory, seeds=(17,), budgets=(0,))
    assert (directory / 'report.json').read_bytes() == before


def test_independent_oracle_handles_stochastic_collision_self_loop():
    module = study_module()
    from neurograft.memory import TopologicalMemoryGraph
    graph = TopologicalMemoryGraph()
    graph.record_transition((1, 1), 1, (2, 1))
    graph.record_transition((1, 1), 1, (2, 1))
    graph.record_transition((1, 1), 1, (1, 1))
    graph.record_transition((2, 1), 1, (3, 1))
    values, _, residual = module._reference(graph, (3, 1), .9)
    assert values[(1, 1)] == pytest.approx(.6 / .7)
    assert values[(2, 1)] == pytest.approx(1.)
    assert values[(3, 1)] == 0.
    assert residual < 1e-10


def test_resealed_derived_metric_tampering_is_rejected(archived, tmp_path):
    import shutil
    module, directory, _ = archived
    copy = tmp_path / 'tampered'
    shutil.copytree(directory, copy)
    path = copy / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['instances'][0]['exposures'][0]['phases'][0]['arms'][0]['curve'][0]['value_mae'] += .125
    path.write_text(json.dumps(report), encoding='utf-8')
    manifest = json.loads((copy / 'manifest.json').read_text())
    manifest['files']['report.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (copy / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='derived|content|render|metric'):
        module.verify_replay_study(copy)


def test_exact_replay_rejects_resealed_continuation_metadata(archived, tmp_path):
    import shutil
    module, directory, _ = archived
    copy = tmp_path / 'resealed'
    shutil.copytree(directory, copy)
    path = copy / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['instances'][0]['exposures'][0]['phases'][1]['warm_backups'] += 1
    path.write_text(json.dumps(report), encoding='utf-8')
    manifest = json.loads((copy / 'manifest.json').read_text())
    manifest['files']['report.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (copy / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='replay content'):
        module.verify_replay_study(copy, replay=True)


@pytest.mark.parametrize('kwargs', [
    {'seeds': ()}, {'seeds': (1, 1)}, {'seeds': (True,)},
    {'budgets': (-1,)}, {'budgets': (4097,)}, {'budgets': (4, 0)},
    {'discount': 1.}, {'discount': float('nan')},
])
def test_invalid_protocol_rejected_before_creating_output(tmp_path, kwargs):
    module = study_module()
    out = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        module.run_replay_study(out, **kwargs)
    assert not out.exists()
