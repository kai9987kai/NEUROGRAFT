"""Matched study receipts must bind their plan, child archives and summaries."""
import hashlib
from html import escape
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def study_module():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'study_navigation_readouts.py'
    assert path.is_file(), 'the reusable matched readout study tool is missing'
    spec = importlib.util.spec_from_file_location('readout_study_tool', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def archived_study(tmp_path_factory):
    tool = study_module()
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    from neurograft.navigation_training import CurriculumConfig
    directory = tmp_path_factory.mktemp('readout-study')
    bundle = make_fixture_bundle(nodes_per_source=3)
    checkpoint = directory / 'source.pt'
    LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2, batch_size=2, threads=1)).save(checkpoint)
    before = checkpoint.read_bytes()
    config = CurriculumConfig(rounds=1, worlds_per_round=2, updates_per_round=1,
                              batch_size=2, max_steps=2, replay_capacity=8,
                              sizes=(5,), presets=('labyrinth',), eval_size=5,
                              validation_seeds=(50000,), test_seeds=(60000, 60001))
    result = tool.run_study(bundle, checkpoint, directory / 'study', config, seeds=(17, 1017))
    assert checkpoint.read_bytes() == before
    return directory / 'study', result


def test_four_arms_use_one_frozen_input_and_derive_summaries(archived_study):
    directory, result = archived_study
    plan = json.loads((directory / 'plan.json').read_text())
    assert [arm['id'] for arm in result['arms']] == [
        'human_plain', 'human_rotated', 'context_residual_plain', 'context_residual_rotated']
    assert len(plan['runs']) == 8
    assert all(run['config']['collection_policy'] == 'expert' and run['config']['init'] == 'fresh'
               and run['config']['seed'] == run['config']['train_seed_start'] == run['seed']
               for run in plan['runs'])
    assert {run['config']['readout'] for run in plan['runs']} == {'human', 'context_residual'}
    assert {run['config']['rotation_augmentation'] for run in plan['runs']} == {True, False}
    assert plan['test_selection_performed'] is False
    assert result['promotion_performed'] is False
    assert result['matched_receipts'][0]['seed'] == 17
    assert [receipt['seed'] for receipt in result['matched_receipts']] == [17, 1017]
    assert plan['parameter_counts']['context_residual']['total'] > plan['parameter_counts']['human']['total']
    assert plan['parameter_counts']['context_residual']['common'] == plan['parameter_counts']['human']['common']
    assert (directory / 'plan.json').stat().st_mtime_ns <= min(
        (directory / run['directory'] / 'report.json').stat().st_mtime_ns for run in plan['runs'])
    for arm in result['arms']:
        policy = arm['variants']['policy']
        assert policy['summary']['episodes'] == len(policy['episodes']) == 4
        assert [episode['seed'] for episode in policy['episodes']] == [60000, 60001, 60000, 60001]
        assert [episode['initialization_seed'] for episode in policy['episodes']] == [17, 17, 1017, 1017]
        assert policy['summary']['success_rate'] == sum(e['success'] for e in policy['episodes']) / 4
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert {'plan.json', 'aggregate.json', 'report.html'} <= set(manifest['files'])
    assert all(f"{run['directory']}/manifest.json" in manifest['files'] for run in plan['runs'])
    assert study_module().verify_study(directory, replay=True)['replay_verified'] is True


def reseal(directory, relative, value):
    path = directory / relative
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')


def test_resealed_false_aggregate_is_rejected(tmp_path, archived_study):
    original, _ = archived_study
    directory = tmp_path / 'wrong-aggregate'
    shutil.copytree(original, directory)
    report = json.loads((directory / 'aggregate.json').read_text())
    report['arms'][0]['variants']['policy']['summary']['success_rate'] = .123
    reseal(directory, 'aggregate.json', report)
    with pytest.raises(ValueError, match='aggregate'):
        study_module().verify_study(directory)


@pytest.mark.parametrize('change', ['config', 'data_hash'])
def test_resealed_child_mismatches_are_rejected(tmp_path, archived_study, change):
    original, _ = archived_study
    directory = tmp_path / change
    shutil.copytree(original, directory)
    plan = json.loads((directory / 'plan.json').read_text())
    run = plan['runs'][0]['directory']
    child = directory / run
    report = json.loads((child / 'report.json').read_text())
    if change == 'config':
        report['config']['learning_rate'] = .006
        reseal(child, 'config.json', report['config'])
    else:
        report['progress'][1]['collection']['data_sha256'] = '0' * 64
        reseal(child, 'progress.json', report['progress'])
    reseal(child, 'report.json', report)
    outer_manifest = json.loads((directory / 'manifest.json').read_text())
    relative = f'{run}/manifest.json'
    outer_manifest['files'][relative] = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
    (directory / 'manifest.json').write_text(json.dumps(outer_manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='config|matched|receipt|data'):
        study_module().verify_study(directory)


@pytest.mark.parametrize('change', ['schedule', 'teacher', 'worlds', 'transitions',
                                   'updates', 'batch_size', 'rotation_counts', 'rotation_plain'])
def test_matched_receipts_must_also_obey_the_declared_schedule_and_budget(archived_study, change):
    directory, _ = archived_study
    plan = json.loads((directory / 'plan.json').read_text())
    reports = {run['directory']: json.loads((directory / run['directory'] / 'report.json').read_text())
               for run in plan['runs']}
    for report in reports.values():
        entry = report['progress'][1]
        collection, training = entry['collection'], entry['training']
        if change == 'schedule':
            collection['world_seeds'] = [50000, 50001]  # Leaks validation even when every arm agrees.
        elif change == 'teacher':
            collection['learner_steps'] = 1
            collection['teacher_steps'] -= 1
        elif change == 'worlds':
            collection['worlds'] = 99
        elif change == 'transitions':
            collection['transitions'] += 1
        elif change == 'updates':
            training['updates'] = 7
            entry['updates'] = 7
        elif change == 'batch_size':
            training['batch_size'] = 1
        elif change == 'rotation_counts':
            training['rotation_counts'] = [-1, 3, 0, 0]
        elif not report['config']['rotation_augmentation']:
            training['rotation_counts'] = [0, 2, 0, 0]
    with pytest.raises(ValueError, match='receipt|schedule|training|collection|rotation|budget'):
        study_module()._aggregate(plan, reports, hashlib.sha256((directory / 'plan.json').read_bytes()).hexdigest())


def test_existing_output_and_more_than_three_initializations_are_rejected(tmp_path):
    tool = study_module()
    from neurograft.fixtures import make_fixture_bundle
    bundle = make_fixture_bundle(nodes_per_source=3)
    existing = tmp_path / 'occupied'
    existing.mkdir()
    sentinel = existing / 'sentinel'
    sentinel.write_text('keep')
    with pytest.raises(ValueError, match='already exists'):
        tool.run_study(bundle, tmp_path / 'missing.pt', existing)
    assert sentinel.read_text() == 'keep'
    with pytest.raises(ValueError, match='1.?3|three'):
        tool.run_study(bundle, tmp_path / 'missing.pt', tmp_path / 'invalid', seeds=(0, 1000, 2000, 3000))
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('relative', ['../outside.json', 'runs/../../outside.json'])
def test_archive_file_paths_cannot_traverse_outside_the_directory(tmp_path, relative):
    with pytest.raises(ValueError, match='path|archive'):
        study_module()._safe_file(tmp_path, relative)


def test_archive_file_paths_reject_directory_junctions(tmp_path):
    if os.name != 'nt':
        pytest.skip('Windows junction regression')
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'manifest.json').write_text('{}')
    link = tmp_path / 'junction'
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                            capture_output=True, text=True)
    if result.returncode:
        pytest.skip('junction creation is unavailable in this environment')
    with pytest.raises(ValueError, match='junction|path'):
        study_module()._safe_file(tmp_path, 'junction/manifest.json')


def test_one_retained_world_cannot_claim_two_training_rounds():
    from dataclasses import asdict
    from neurograft.navigation_training import CurriculumConfig
    config = CurriculumConfig(rounds=2, worlds_per_round=1, updates_per_round=1,
                              batch_size=1, max_steps=1, replay_capacity=4,
                              sizes=(5,), presets=('labyrinth',), collection_policy='expert')
    progress = [{'round': 0, 'updates': 0}]
    for round_idx in (1, 2):
        progress.append({'round': round_idx, 'updates': round_idx,
            'collection': {'worlds': 1, 'world_seeds': [round_idx - 1],
                'condition_counts': [{'size': 5, 'preset': 'labyrinth', 'worlds': 1}],
                'transitions': 1, 'action_counts': {'north': 1, 'east': 0, 'south': 0, 'west': 0},
                'teacher_probability': 1.0, 'teacher_steps': 1, 'learner_steps': 0, 'data_sha256': '0' * 64},
            'replay': {'strategy': 'per-action-algorithm-r', 'capacity': 4, 'seen_total': round_idx,
                'retained_total': 1, 'action_quotas': {'north': 1, 'east': 1, 'south': 1, 'west': 1},
                'action_counts': {'north': 1, 'east': 0, 'south': 0, 'west': 0},
                'seen_action_counts': {'north': round_idx, 'east': 0, 'south': 0, 'west': 0},
                'retained_worlds': 1, 'retained_rounds': list(range(1, round_idx + 1))},
            'training': {'updates': 1, 'batch_size': 1, 'losses': [1.0], 'batch_sha256': '0' * 64,
                         'rotation_counts': [1, 0, 0, 0]}})
    with pytest.raises(ValueError, match='replay'):
        study_module()._round_receipts({'config': json.loads(json.dumps(asdict(config))), 'progress': progress})


def test_captured_earliest_v1_scope_and_html_remain_verifiable(tmp_path, archived_study):
    original, _ = archived_study
    directory = tmp_path / 'earliest-v1'
    shutil.copytree(original, directory)
    plan = json.loads((directory / 'plan.json').read_text())
    current_scope = plan['evidence_scope']
    legacy_scope = current_scope.replace(
        "Each seed sets both model initialization and the training-world seed offset; arms share that seed's corpus. ", '')
    assert legacy_scope != current_scope
    plan['evidence_scope'] = legacy_scope
    reseal(directory, 'plan.json', plan)
    aggregate = json.loads((directory / 'aggregate.json').read_text())
    aggregate['evidence_scope'] = legacy_scope
    aggregate['plan_sha256'] = hashlib.sha256((directory / 'plan.json').read_bytes()).hexdigest()
    reseal(directory, 'aggregate.json', aggregate)
    html = (directory / 'report.html').read_bytes().decode('utf-8')
    html = html.replace(escape(current_scope), escape(legacy_scope)).replace(
        'table{display:block;overflow-x:auto;', 'table{')
    reseal(directory, 'report.html', html.encode('utf-8'))
    assert study_module().verify_study(directory)['ok'] is True
    plan['evidence_scope'] = 'Unrecognized study interpretation.'
    reseal(directory, 'plan.json', plan)
    with pytest.raises(ValueError, match='scope|plan'):
        study_module().verify_study(directory)
