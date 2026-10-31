"""Causal forecast, held-out evaluation and sealed adaptation evidence."""
import hashlib
import importlib
import importlib.util
import json
import math
import shutil

import pytest


def study():
    assert importlib.util.find_spec('neurograft.adaptation_study'), 'adaptive streaming study is missing'
    return importlib.import_module('neurograft.adaptation_study')


@pytest.fixture(scope='module')
def archived(tmp_path_factory):
    module = study()
    out = tmp_path_factory.mktemp('adaptation') / 'run'
    return module, out, module.run_adaptation_study(out, seeds=(15000,), budget=16)


def test_declared_streams_score_before_update_and_keep_all_comparison_models(archived):
    _, _, report = archived
    instance = report['instances'][0]
    assert {case['id'] for case in instance['cases']} == {
        'stationary', 'reversals', 'gradual', 'conditional_adverse', 'sparse'}
    for case in instance['cases']:
        assert case['event_count'] == len(case['events'])
        assert case['forecast_totals']['adaptive']['count'] == len(case['events'])
        for event in case['events']:
            moved = float(event['source'] != event['destination'])
            assert set(event['predictions']) == {'cumulative', 'recent', 'guarded_recent', 'adaptive'}
            for model, prediction in event['predictions'].items():
                assert event['brier'][model] == pytest.approx((prediction - moved) ** 2)
                assert event['log_loss'][model] == pytest.approx(-math.log(prediction if moved else 1 - prediction))
                assert event['absolute_error'][model] == pytest.approx(abs(prediction - event['truth_p_move']))
        for stage in case['stages']:
            assert len(stage['arms']) == 9
            assert stage['reference_residual'] < 1e-12
            for arm in stage['arms']:
                assert arm['stage_backups'] <= stage['requested_backups']
                assert arm['current_dynamics']['start_regret'] >= -1e-11
    first = instance['cases'][0]['events'][0]
    # Untouched experts must forecast the same baseline before seeing the first outcome.
    assert len(set(first['predictions'].values())) == 1


def test_matched_prefix_truth_cannot_change_memory_and_sparse_actions_stay_untouched(archived):
    _, _, report = archived
    instance = report['instances'][0]
    ambiguity = instance['ambiguity']
    assert ambiguity['truths_differ'] is True
    assert ambiguity['observations_equal'] is True
    assert ambiguity['memory_states_equal'] is True
    assert ambiguity['left_predictions'] == ambiguity['right_predictions']
    sparse = next(case for case in instance['cases'] if case['id'] == 'sparse')
    assert sparse['untouched_pair_preserved'] is True
    assert any(stage['new_events'] == 0 for stage in sparse['stages'][1:])
    for stage in sparse['stages']:
        if stage['new_events'] == 0 and stage['id'] != 'baseline':
            assert stage['requested_backups'] == 0
            assert all(arm['stage_backups'] == 0 for arm in stage['arms'])


def test_held_out_solver_evaluates_collision_policy_without_planner_feedback():
    module = study()
    truth = {(0, 0): {1: {(0, 0): .25, (1, 0): .75}}, (1, 0): {}}
    values, residual = module._reference(truth, (1, 0))
    assert values[(0, 0)] == pytest.approx(.75 / .775)
    assert values[(1, 0)] == 0
    assert residual < 1e-12


def test_selection_delay_is_relative_to_stage_entry_and_separates_existing_selection():
    module = study()
    rows = [
        {'source': [0, 0], 'action': 1, 'index': 11, 'selected_before': 'cumulative',
         'selected_after': 'cumulative', 'switch': False},
        {'source': [0, 0], 'action': 1, 'index': 12, 'selected_before': 'cumulative',
         'selected_after': 'recent', 'switch': True},
    ]
    summary = module._selection_summary(rows, event_offset=10)[0]
    assert summary['recent_at_entry'] is False
    assert summary['first_recent_visit'] == 2
    assert summary['first_recent_event'] == 12
    assert summary['first_recent_event_relative'] == 2
    assert summary['first_recent_switch_visit'] == 2
    rows[0]['selected_before'] = rows[0]['selected_after'] = 'recent'
    rows[1]['selected_before'] = 'recent'
    rows[1]['switch'] = False
    summary = module._selection_summary(rows, event_offset=10)[0]
    assert summary['recent_at_entry'] is True
    assert summary['first_recent_visit'] == 1
    assert summary['first_recent_switch_visit'] is None


def test_archive_verifies_strict_replay_and_refuses_overwrite(archived):
    module, out, report = archived
    result = module.verify_adaptation_study(out, replay=True)
    assert result['ok'] is True
    assert result['deterministic_replay'] is True
    assert result['promotion_performed'] is False
    with pytest.raises(FileExistsError):
        module.run_adaptation_study(out, seeds=(15000,), budget=16)
    assert json.loads((out / 'report.json').read_text()) == report


def test_protocol_is_saved_before_execution_and_failure_retains_declaration(tmp_path, monkeypatch):
    module = study()
    out = tmp_path / 'interrupted'

    def interrupted(config):
        assert (out / 'plan.json').is_file(), 'protocol was not declared before execution'
        plan = json.loads((out / 'plan.json').read_text())
        assert plan['config']['seeds'] == [15000]
        assert plan['config']['budget'] == 0
        raise RuntimeError('controlled execution failure')

    monkeypatch.setattr(module, '_compute', interrupted)
    with pytest.raises(RuntimeError, match='controlled execution failure'):
        module.run_adaptation_study(out, seeds=(15000,), budget=0)
    assert (out / 'plan.json').is_file()
    assert not (out / 'manifest.json').exists()
    with pytest.raises(ValueError, match='invalid adaptation study archive'):
        module.verify_adaptation_study(out)


def reseal(module, out, report):
    (out / 'report.json').write_text(module._json(report), encoding='utf-8')
    (out / 'report.html').write_text(module._render(report), encoding='utf-8')
    manifest = json.loads((out / 'manifest.json').read_text())
    for name in ('report.json', 'report.html'):
        manifest['files'][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(module._json(manifest), encoding='utf-8')


def test_plain_verification_rejects_resealed_derived_loss_tampering(archived, tmp_path):
    module, origin, _ = archived
    out = tmp_path / 'tampered'
    shutil.copytree(origin, out)
    report = json.loads((out / 'report.json').read_text())
    report['instances'][0]['cases'][0]['events'][0]['brier']['adaptive'] += .1
    reseal(module, out, report)
    with pytest.raises(ValueError, match='loss|derived|forecast'):
        module.verify_adaptation_study(out)


def test_strict_replay_rejects_coordinated_resealing_of_seeded_outcomes(archived, tmp_path, monkeypatch):
    module, origin, _ = archived
    out = tmp_path / 'tampered'
    shutil.copytree(origin, out)
    declared = json.loads((out / 'report.json').read_text())['protocol']['config']
    original = module._blocks

    def changed_stream(seed, case_id, truth, targets, window):
        blocks = original(seed, case_id, truth, targets, window)
        if case_id == 'stationary':
            first = blocks[0]['events'][0]
            source = first['source']
            first['destination'] = (list(module._move_destination(tuple(source), first['action']))
                                    if first['destination'] == source else list(source))
        return blocks

    # Recompute every dependent score, model, metric, aggregate and HTML seal for
    # a changed outcome: plain consistency can hold, seeded provenance cannot.
    with monkeypatch.context() as patch:
        patch.setattr(module, '_blocks', changed_stream)
        report = module._compute(declared)
    reseal(module, out, report)
    assert module.verify_adaptation_study(out)['ok'] is True
    with pytest.raises(ValueError, match='replay|seed|content'):
        module.verify_adaptation_study(out, replay=True)


@pytest.mark.parametrize('kwargs', [
    {'seeds': ()}, {'seeds': (True,)}, {'seeds': (1, 1)}, {'seeds': (-1,)},
    {'seeds': tuple(range(33))}, {'window': 3}, {'window': True},
    {'budget': -1}, {'budget': True}, {'budget': 257},
])
def test_invalid_configuration_creates_no_archive(tmp_path, kwargs):
    module = study()
    out = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        module.run_adaptation_study(out, **kwargs)
    assert not out.exists()
