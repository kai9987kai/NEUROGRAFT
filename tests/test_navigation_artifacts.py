import hashlib
import json

import pytest


def run(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    from neurograft.navigation_training import CurriculumConfig, run_navigation_training
    bundle = make_fixture_bundle(nodes_per_source=3)
    source = tmp_path / 'source.pt'
    LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2, batch_size=2)).save(source)
    before = source.read_bytes()
    directory = tmp_path / 'run'
    result = run_navigation_training(bundle, source, directory, CurriculumConfig(
        rounds=1, worlds_per_round=2, updates_per_round=1, batch_size=2,
        max_steps=2, sizes=(5,), presets=('labyrinth',), eval_size=5,
        validation_seeds=(20000,), test_seeds=(30000,), replay_capacity=8))
    assert source.read_bytes() == before
    return directory, result, before


def test_run_archives_exact_inputs_raw_results_and_can_replay_without_source(tmp_path):
    directory, result, before = run(tmp_path)
    assert (directory / 'frozen-input.pt').read_bytes() == before
    assert json.loads((directory / 'report.json').read_text())['selected_model_hash'] == result['selected_model_hash']
    assert json.loads((directory / 'progress.json').read_text()) == result['progress']
    (tmp_path / 'source.pt').unlink()
    from neurograft.navigation_artifacts import verify_navigation_run
    receipt = verify_navigation_run(directory, replay=True)
    assert receipt['ok'] is True and receipt['replay_verified'] is True
    assert receipt['selected_model_hash'] == result['selected_model_hash']
    assert receipt['promotion_performed'] is False


def test_verifier_rejects_changed_missing_and_escaping_artifacts(tmp_path):
    directory, _, _ = run(tmp_path)
    from neurograft.navigation_artifacts import verify_navigation_run
    original = (directory / 'report.json').read_bytes()
    (directory / 'report.json').write_bytes(original + b' ')
    with pytest.raises(ValueError, match='hash'):
        verify_navigation_run(directory)
    (directory / 'report.json').write_bytes(original)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['../source.pt'] = hashlib.sha256((tmp_path / 'source.pt').read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='artifact|manifest'):
        verify_navigation_run(directory)
    del manifest['files']['../source.pt']
    del manifest['files']['candidate.pt']
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='artifact|manifest'):
        verify_navigation_run(directory)


def test_verifier_rejects_inconsistent_identity_even_with_updated_file_hash(tmp_path):
    directory, _, _ = run(tmp_path)
    from neurograft.navigation_artifacts import verify_navigation_run
    result = json.loads((directory / 'report.json').read_text())
    result['selected_model_hash'] = '0' * 64
    (directory / 'report.json').write_text(json.dumps(result))
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files']['report.json'] = hashlib.sha256((directory / 'report.json').read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='model|identity'):
        verify_navigation_run(directory)


def test_verify_navigation_cli_exposes_integrity_and_replay_modes():
    from neurograft.cli import parser
    args = parser().parse_args(['verify-navigation', 'my-run', '--replay'])
    assert args.command == 'verify-navigation' and args.run == 'my-run' and args.replay


def test_verifier_derives_protocol_identity_instead_of_trusting_matching_strings(tmp_path):
    directory, _, _ = run(tmp_path)
    from neurograft.navigation_artifacts import verify_navigation_run
    path = directory / 'report.json'
    result = json.loads(path.read_text())
    result['protocol_id'] = result['baseline']['protocol_id'] = result['candidate']['protocol_id'] = '0' * 64
    path.write_text(json.dumps(result))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['report.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='protocol'):
        verify_navigation_run(directory)


def test_verifier_rejects_missing_declared_control_even_with_resealed_report(tmp_path):
    directory, _, _ = run(tmp_path)
    from neurograft.navigation_artifacts import verify_navigation_run
    path = directory / 'report.json'
    result = json.loads(path.read_text())
    for name in ('baseline', 'candidate'):
        del result[name]['variants']['no_graft']
    path.write_text(json.dumps(result))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['report.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='variant|control'):
        verify_navigation_run(directory)
