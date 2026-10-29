"""Controlled revision must separate stale-model error from planning error."""
import hashlib
import importlib
import importlib.util
import json
import shutil

import pytest


def module():
    assert importlib.util.find_spec('neurograft.revision_study'), 'controlled revision study is missing'
    return importlib.import_module('neurograft.revision_study')


def test_independent_current_dynamics_oracle_handles_collision_probability():
    study = module()
    model = {(1, 1): {1: {(2, 1): .75, (1, 1): .25}},
             (2, 1): {1: {(3, 1): 1.}}, (3, 1): {}}
    values, residual = study._reference(model, (3, 1))
    assert values[(1, 1)] == pytest.approx(.675 / .775)
    assert values[(2, 1)] == pytest.approx(1.)
    assert values[(3, 1)] == 0.
    assert residual < 1e-12


def test_unobserved_change_stages_perform_no_new_backups():
    study = module()
    report = study._compute(study._config((17,), 8, 64))
    rows = [r for r in report['aggregates'] if r['stage'].endswith('_unobserved')]
    assert rows
    assert all(r['mean_backups'] == 0 for r in rows)


@pytest.fixture(scope='module')
def archived(tmp_path_factory):
    study = module()
    out = tmp_path_factory.mktemp('revision') / 'run'
    report = study.run_revision_study(out, seeds=(14000, 14001), window=8, budget=64)
    return study, out, report


def test_crossed_arms_share_explicit_stream_and_distinguish_true_dynamics(archived):
    _, _, report = archived
    assert len(report['instances']) == 2
    for instance in report['instances']:
        assert instance['geometry']['width'] >= 3
        assert instance['geometry']['height'] >= 2
        revision, stationary = instance['cases']
        assert [s['id'] for s in revision['stages']] == [
            'baseline', 'closed_unobserved', 'closed_observed', 'reopened_unobserved',
            'reopened_observed', 'goal_relocated']
        assert [s['id'] for s in stationary['stages']] == [
            'baseline', 'stationary_sample', 'stationary_adverse_run']
        for case in instance['cases']:
            for stage in case['stages']:
                assert len(stage['arms']) == 6
                assert {(a['model'], a['strategy']) for a in stage['arms']} == {
                    (m, s) for m in ('cumulative', 'recent')
                    for s in ('prioritized', 'uniform', 'none')}
                assert stage['reference_residual'] < 1e-12
                for arm in stage['arms']:
                    assert arm['empirical']['value_mae'] >= 0
                    current = arm['current_dynamics']
                    assert current['transition_brier'] >= 0
                    assert current['start_regret'] >= -1e-11
                    assert arm['summary']['backups'] <= arm['summary']['budget']
        before, hidden = revision['stages'][:2]
        assert before['models'] == hidden['models']
        assert [a['values'] for a in before['arms']] == [a['values'] for a in hidden['arms']]
        assert hidden['commands'] == []
        assert hidden['truth'] != before['truth']


def test_stationary_adverse_control_reports_the_recency_tradeoff(archived):
    _, _, report = archived
    for instance in report['instances']:
        stage = instance['cases'][1]['stages'][-1]
        arms = {(a['model'], a['strategy']): a for a in stage['arms']}
        assert arms['recent', 'prioritized']['current_dynamics']['target_action_brier'] > (
            arms['cumulative', 'prioritized']['current_dynamics']['target_action_brier'])
    assert 'conditional' in report['protocol']['stationary_adverse_control'].lower()


def test_archive_is_repeatable_and_replays_without_live_checkpoint(archived, tmp_path, monkeypatch):
    study, out, report = archived
    monkeypatch.setattr('neurograft.live.LiveSession.load', lambda *_: pytest.fail('study opened live state'))
    second = tmp_path / 'second'
    assert study.run_revision_study(second, seeds=(14000, 14001), window=8, budget=64) == report
    assert (out / 'report.json').read_bytes() == (second / 'report.json').read_bytes()
    assert study.verify_revision_study(out, replay=True)['deterministic_replay'] is True
    assert 'Current dynamics' in (out / 'report.html').read_text(encoding='utf-8')


def test_existing_directory_is_not_overwritten(archived):
    study, out, _ = archived
    original = (out / 'report.json').read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        study.run_revision_study(out)
    assert (out / 'report.json').read_bytes() == original


def test_plain_verification_rejects_resealed_derived_metric_tampering(archived, tmp_path):
    study, out, _ = archived
    copy = tmp_path / 'tampered'
    shutil.copytree(out, copy)
    path = copy / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['instances'][0]['cases'][0]['stages'][0]['arms'][0]['current_dynamics']['start_regret'] += .1
    path.write_text(json.dumps(report), encoding='utf-8')
    manifest = json.loads((copy / 'manifest.json').read_text(encoding='utf-8'))
    manifest['files']['report.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (copy / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='metric|derived|content|render'):
        study.verify_revision_study(copy)


def test_strict_replay_rejects_resealed_outcome_history(archived, tmp_path):
    study, out, _ = archived
    copy = tmp_path / 'tampered'
    shutil.copytree(out, copy)
    path = copy / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['instances'][0]['cases'][0]['stages'][2]['commands'][0]['count'] -= 1
    path.write_text(json.dumps(report), encoding='utf-8')
    manifest = json.loads((copy / 'manifest.json').read_text(encoding='utf-8'))
    manifest['files']['report.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (copy / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        study.verify_revision_study(copy, replay=True)


@pytest.mark.parametrize('mutation', ['duplicate_value', 'duplicate_node', 'nonfinite_value'])
def test_plain_verification_rejects_malformed_resealed_state(archived, tmp_path, mutation):
    study, out, _ = archived
    copy = tmp_path / mutation
    shutil.copytree(out, copy)
    path = copy / 'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    stage = report['instances'][0]['cases'][0]['stages'][0]
    if mutation == 'duplicate_value':
        stage['arms'][0]['values'].append(dict(stage['arms'][0]['values'][0]))
    elif mutation == 'duplicate_node':
        stage['models']['cumulative']['nodes'].append(stage['models']['cumulative']['nodes'][0])
    else:
        stage['arms'][0]['values'][0]['value'] = float('inf')
    path.write_text(json.dumps(report), encoding='utf-8')
    manifest = json.loads((copy / 'manifest.json').read_text(encoding='utf-8'))
    manifest['files']['report.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (copy / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='invalid|nonfinite'):
        study.verify_revision_study(copy)


@pytest.mark.parametrize('kwargs', [
    {'seeds': ()}, {'seeds': (True,)}, {'seeds': (1, 1)}, {'seeds': (-1,)},
    {'window': 3}, {'window': True}, {'budget': -1}, {'budget': 257}, {'budget': True},
])
def test_invalid_protocol_leaves_no_output(tmp_path, kwargs):
    out = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        module().run_revision_study(out, **kwargs)
    assert not out.exists()
