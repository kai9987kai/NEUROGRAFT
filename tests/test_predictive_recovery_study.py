"""Predictive studies must bind their plan, paired training and probe records."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys

import pytest
import torch


def study_module():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'study_predictive_recovery.py'
    assert path.is_file(), 'the predictive recovery study driver is missing'
    spec = importlib.util.spec_from_file_location('predictive_recovery_study_tool', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def archived_study(tmp_path_factory):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    from neurograft.navigation_training import CurriculumConfig
    tool = study_module()
    directory = tmp_path_factory.mktemp('predictive-recovery')
    bundle = make_fixture_bundle(nodes_per_source=3)
    checkpoint = directory / 'source.pt'
    LiveSession(bundle, LiveConfig(hidden_dim=4, batch_size=2, unroll=2, threads=1)).save(checkpoint)
    before = checkpoint.read_bytes()
    config = CurriculumConfig(rounds=1, worlds_per_round=2, updates_per_round=2,
                              batch_size=2, replay_capacity=8, max_steps=3,
                              sizes=(5,), presets=('labyrinth',), eval_size=5,
                              validation_seeds=(55000,), test_seeds=(65000,))
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    result = tool.run_study(bundle, checkpoint, directory / 'study', config,
                            seeds=(17,), probe_transitions=3)
    assert checkpoint.read_bytes() == before
    assert python_rng == random.getstate()
    assert torch.equal(torch_rng, torch.get_rng_state())
    return directory / 'study', result


def reseal(directory, relative, value):
    path = directory / relative
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')


def test_four_arms_are_predeclared_and_share_data_within_each_auxiliary_pair(archived_study):
    directory, result = archived_study
    plan = json.loads((directory / 'plan.json').read_text())
    assert [arm['id'] for arm in result['arms']] == [
        'expert_aux0', 'expert_aux05', 'recovery_aux0', 'recovery_aux05']
    assert len(plan['runs']) == 4
    assert {run['config']['predictive_weight'] for run in plan['runs']} == {0.0, 0.5}
    assert all(run['config']['predictive_head'] and run['config']['readout'] == 'context_residual'
               and run['config']['rounds'] == 1 and run['config']['init'] == 'fresh'
               for run in plan['runs'])
    assert len(result['matched_pairs']) == 2
    assert all(pair['data_sha256'] and pair['batch_sha256'] for pair in result['matched_pairs'])
    assert plan['test_selection_performed'] is result['promotion_performed'] is False
    assert (directory / 'plan.json').stat().st_mtime_ns <= min(
        (directory / run['directory'] / 'report.json').stat().st_mtime_ns for run in plan['runs'])
    for arm in result['arms']:
        assert set(arm['variants']) == {'policy', 'no_graft', 'no_recurrence', 'no_context', 'random', 'shortest_path'}
        assert arm['variants']['policy']['summary']['episodes'] == 1
    probe = json.loads((directory / 'prediction-probes.json').read_text())
    assert probe['valid_transitions'] == 12
    assert set(probe['baselines']) == {'copy_current', 'kinematic'}
    assert len(probe['runs']) == 4
    assert all(set(run['metrics']['groups']) == {'position', 'heading', 'goal', 'walls', 'reward_clock'}
               for run in probe['runs'])
    assert study_module().verify_study(directory, replay=True)['replay_verified'] is True


@pytest.mark.parametrize('artifact', ['aggregate.json', 'prediction-probes.json'])
def test_resealed_false_aggregate_or_probe_metrics_are_rejected(tmp_path, archived_study, artifact):
    original, _ = archived_study
    directory = tmp_path / artifact.replace('.json', '')
    shutil.copytree(original, directory)
    data = json.loads((directory / artifact).read_text())
    if artifact == 'aggregate.json':
        data['arms'][0]['variants']['policy']['summary']['success_rate'] = .123
    else:
        data['runs'][0]['metrics']['overall_mse'] += 1
    reseal(directory, artifact, data)
    with pytest.raises(ValueError, match='aggregate|probe'):
        study_module().verify_study(directory)


def test_resealed_mismatched_auxiliary_pair_is_rejected(tmp_path, archived_study):
    original, _ = archived_study
    directory = tmp_path / 'mismatched-pair'
    shutil.copytree(original, directory)
    plan = json.loads((directory / 'plan.json').read_text())
    child = directory / plan['runs'][1]['directory']
    report = json.loads((child / 'report.json').read_text())
    report['progress'][1]['training']['batch_sha256'] = '0' * 64
    reseal(child, 'progress.json', report['progress'])
    reseal(child, 'report.json', report)
    manifest = json.loads((directory / 'manifest.json').read_text())
    relative = plan['runs'][1]['directory'] + '/manifest.json'
    manifest['files'][relative] = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='pair|matched|batch'):
        study_module().verify_study(directory)


def test_probe_kinematics_respects_walls_and_sets_action_heading():
    tool = study_module()
    observation = [[.5, .5, 0., -1.], [.8, .2, .3, -.3], [0., 1., 0., 0.], [.2, .25, .5, 1.]]
    predicted = tool.kinematic_prediction(observation, 1, 11)
    assert predicted[0] == [.5, .5, 1., 0.]
    assert predicted[1][:2] == [.8, .2]
    assert predicted[2] == observation[2]
    assert predicted[3] == [-.1, .25 + 1 / 256, .5, 1.]


def test_study_bounds_reject_multiple_rounds_and_more_than_two_seeds(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.navigation_training import CurriculumConfig
    tool = study_module()
    bundle = make_fixture_bundle(nodes_per_source=3)
    with pytest.raises(ValueError, match='one|round'):
        tool.run_study(bundle, tmp_path / 'missing.pt', tmp_path / 'rounds', CurriculumConfig(rounds=2))
    with pytest.raises(ValueError, match='1-2|two'):
        tool.run_study(bundle, tmp_path / 'missing.pt', tmp_path / 'seeds', seeds=(0, 1000, 2000))
    assert not (tmp_path / 'rounds').exists()
    assert not (tmp_path / 'seeds').exists()
