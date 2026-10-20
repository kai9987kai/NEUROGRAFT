"""Fixed-weight controller studies archive enough evidence to replay alone."""
import copy
import hashlib
import json

import pytest


def make_study(tmp_path, directory='study', sizes=(5, 7)):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    from neurograft.navigation_study import StudyConfig, run_navigation_study
    bundle = make_fixture_bundle(nodes_per_source=3)
    checkpoint = tmp_path / 'original.pt'
    if not checkpoint.exists():
        LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2, batch_size=2)).save(checkpoint)
    before = checkpoint.read_bytes()
    destination = tmp_path / directory
    result = run_navigation_study(bundle, checkpoint, destination,
                                  StudyConfig(sizes=sizes, presets=('labyrinth',), seeds=(41000,), max_steps=2))
    assert checkpoint.read_bytes() == before
    return destination, result, before


def reseal(directory, name, value):
    path = directory / name
    path.write_text(json.dumps(value), encoding='utf-8')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')


def test_study_repeats_fixed_weights_and_archives_exact_source_bytes(tmp_path):
    directory, result, before = make_study(tmp_path)
    other, repeated, _ = make_study(tmp_path, 'repeat')
    assert result == repeated
    assert result['schema'] == 'navigation-controller-study-v1'
    assert result['learning'] is False and result['teacher_assistance'] == 0
    assert result['promotion_performed'] is False
    assert len(result['conditions']) == 2
    assert (directory / 'frozen-input.pt').read_bytes() == before
    assert json.loads((directory / 'report.json').read_text()) == result
    assert (other / 'report.html').is_file()
    assert set(result['variants']) == {'policy', 'no_graft', 'no_recurrence', 'random', 'shortest_path',
                                       'masked_policy', 'observed_map', 'map_without_policy'}
    for variant in result['variants'].values():
        assert variant['summary']['episodes'] == 2
        assert [episode['size'] for episode in variant['episodes']] == [5, 7]
        assert all(episode['preset'] == 'labyrinth' for episode in variant['episodes'])
    declaration = json.loads((directory / 'protocol.json').read_text())
    assert declaration['study_id'] == result['study_id']
    assert declaration['model_hash'] == result['model_hash']


def test_study_replays_without_original_checkpoint_or_bundle_paths(tmp_path):
    from neurograft.navigation_study import verify_navigation_study
    directory, result, _ = make_study(tmp_path)
    (tmp_path / 'original.pt').unlink()
    receipt = verify_navigation_study(directory, replay=True)
    assert receipt['ok'] is True and receipt['replay_verified'] is True
    assert receipt['study_id'] == result['study_id']
    assert receipt['model_hash'] == result['model_hash']
    assert receipt['promotion_performed'] is False
    assert receipt['conditions_verified'] == 2


def test_existing_output_directory_is_preserved(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.navigation_study import run_navigation_study
    directory = tmp_path / 'existing'
    directory.mkdir()
    sentinel = directory / 'keep.txt'
    sentinel.write_text('keep', encoding='utf-8')
    with pytest.raises(ValueError, match='already exists'):
        run_navigation_study(make_fixture_bundle(nodes_per_source=3), tmp_path / 'missing.pt', directory)
    assert sentinel.read_text() == 'keep' and list(directory.iterdir()) == [sentinel]


@pytest.mark.parametrize('values', [
    {'sizes': ()}, {'sizes': [7]}, {'sizes': (7, 7)}, {'sizes': (True,)}, {'sizes': (4,)},
    {'sizes': (32,)}, {'presets': ()}, {'presets': ('labyrinth', 'labyrinth')}, {'presets': ['labyrinth']},
    {'presets': ('unknown',)}, {'seeds': ()}, {'seeds': (1, 1)}, {'seeds': [1]},
    {'seeds': (True,)}, {'seeds': (-1,)}, {'seeds': (2**63,)}, {'seeds': tuple(range(17))},
    {'max_steps': 0}, {'max_steps': 129}, {'max_steps': True},
    {'sizes': (5, 7), 'seeds': tuple(range(6))},  # 2*6*6=72 worlds exceeds the 64-world cap.
])
def test_invalid_study_configuration_is_rejected(values):
    from neurograft.navigation_study import StudyConfig
    with pytest.raises(ValueError):
        StudyConfig(**values)


def test_integrity_rejects_changed_artifact_and_manifest_path_escape(tmp_path):
    from neurograft.navigation_study import verify_navigation_study
    directory, _, _ = make_study(tmp_path, sizes=(5,))
    path = directory / 'trial_000.json'
    original = path.read_bytes()
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='hash'):
        verify_navigation_study(directory)
    path.write_bytes(original)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['../original.pt'] = hashlib.sha256((tmp_path / 'original.pt').read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest|artifact'):
        verify_navigation_study(directory)


@pytest.mark.parametrize('identity', ['model_hash', 'graph_hash', 'study_id'])
def test_resealed_report_identity_tampering_is_rejected(tmp_path, identity):
    from neurograft.navigation_study import verify_navigation_study
    directory, result, _ = make_study(tmp_path, sizes=(5,))
    changed = copy.deepcopy(result)
    changed[identity] = '0' * 64
    reseal(directory, 'report.json', changed)
    with pytest.raises(ValueError, match='identity|model|graph|study|aggregate'):
        verify_navigation_study(directory)


def test_verifier_derives_protocol_and_recomputes_aggregate(tmp_path):
    from neurograft.navigation_study import verify_navigation_study
    directory, result, _ = make_study(tmp_path, sizes=(5,))
    trial_path = directory / 'trial_000.json'
    trial = json.loads(trial_path.read_text())
    trial['protocol_id'] = '0' * 64
    reseal(directory, trial_path.name, trial)
    with pytest.raises(ValueError, match='protocol|identity'):
        verify_navigation_study(directory)
    # Restore the trial, then tamper only with the report's aggregate.
    from neurograft.navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation
    from neurograft.graphs import GraphBundle
    from neurograft.live import LiveSession
    bundle = GraphBundle.load(directory / 'source.bundle.json')
    session = LiveSession.load(directory / 'frozen-input.pt', bundle)
    actual = frozen_navigation_evaluation(session, NavigationEvalConfig(
        seeds=(41000,), size=5, max_steps=2, include_controllers=True))
    reseal(directory, trial_path.name, actual)
    changed = copy.deepcopy(result)
    changed['variants']['policy']['summary']['success_rate'] = 0.123
    reseal(directory, 'report.json', changed)
    with pytest.raises(ValueError, match='aggregate|identity'):
        verify_navigation_study(directory)


def test_study_cli_runs_and_verifies_without_overwriting(tmp_path, capsys):
    from neurograft.cli import main, parser
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    bundle = make_fixture_bundle(nodes_per_source=3)
    source = tmp_path / 'bundle.json'
    checkpoint = tmp_path / 'source.pt'
    bundle.save(source)
    LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2, batch_size=2)).save(checkpoint)
    args = ['study-navigation', '--bundle', str(source), '--checkpoint', str(checkpoint),
            '--out', str(tmp_path / 'cli-study'), '--sizes', '5', '--presets', 'labyrinth',
            '--seeds', '41000', '--max-steps', '2']
    assert parser().parse_args(args).sizes == '5'
    assert main(args) == 0
    capsys.readouterr()
    assert main(['verify-navigation-study', str(tmp_path / 'cli-study'), '--replay']) == 0
    assert json.loads(capsys.readouterr().out)['replay_verified'] is True
    assert main(args) == 2
    assert 'already exists' in capsys.readouterr().err


@pytest.fixture(scope='module')
def small_archived_study(tmp_path_factory):
    return make_study(tmp_path_factory.mktemp('controller-study-validation'), sizes=(5,))[0]


def forge_episode_and_reseal(directory, variant, alter):
    """Keep all redundant summaries coherent so validation must inspect trials."""
    from neurograft.navigation_evaluation import _summary
    trial = json.loads((directory / 'trial_000.json').read_text())
    alter(trial['variants'][variant]['episodes'][0])
    values = trial['variants'][variant]
    values['summary'] = _summary(values['episodes'])
    report = json.loads((directory / 'report.json').read_text())
    episode = values['episodes'][0]
    report['variants'][variant]['episodes'][0] = {
        **episode, 'size': 5, 'preset': 'labyrinth', 'condition_index': 0}
    report['variants'][variant]['summary'] = _summary(report['variants'][variant]['episodes'])
    report['conditions'][0]['summaries'][variant] = values['summary']
    reseal(directory, 'trial_000.json', trial)
    reseal(directory, 'report.json', report)


@pytest.mark.parametrize(('variant', 'alter'), [
    ('policy', lambda episode: episode.update(success=False, spl=2.0)),
    ('policy', lambda episode: episode.update(spl=-0.1)),
    ('policy', lambda episode: episode.update(shortest_distance=-1)),
    ('policy', lambda episode: episode.update(shortest_distance=True)),
    ('policy', lambda episode: episode.update(terminal_distance=-1)),
    ('policy', lambda episode: episode.update(terminal_distance=True)),
    ('policy', lambda episode: episode.update(teacher_agreement=1.1)),
    ('policy', lambda episode: episode.update(teacher_agreement=True)),
    ('policy', lambda episode: episode.update(total_reward=True)),
    ('policy', lambda episode: episode.update(total_reward=2.5)),
    ('policy', lambda episode: episode.update(policy_proposals=0)),
    ('policy', lambda episode: episode.update(proposed_wall_actions=episode['steps'] + 1)),
    ('policy', lambda episode: episode.update(policy_overrides=1)),
    ('policy', lambda episode: episode.update(policy_prior_used=1)),
    ('map_without_policy', lambda episode: episode.update(policy_prior_used=True)),
    ('random', lambda episode: episode.update(policy_prior_used=True)),
    ('policy', lambda episode: episode['controller_trace'].pop()),
    ('policy', lambda episode: episode['controller_trace'][0].update(action=(episode['actions'][0] + 1) % 4)),
    ('policy', lambda episode: episode['controller_trace'][0].update(policy_action=True)),
    ('policy', lambda episode: episode['controller_trace'][0]['position'].update(x=99)),
    ('policy', lambda episode: episode['controller_trace'][0]['position'].update(direction=True)),
    ('observed_map', lambda episode: episode['controller_trace'][0].update(known_free=99)),
    ('observed_map', lambda episode: episode['controller_trace'][0].update(visited=True)),
    ('observed_map', lambda episode: episode['controller_trace'][0].update(frontiers=-1)),
])
def test_resealed_impossible_metrics_and_controller_metadata_are_rejected(
        tmp_path, small_archived_study, variant, alter):
    import shutil
    from neurograft.navigation_study import verify_navigation_study
    directory = tmp_path / 'forged-study'
    shutil.copytree(small_archived_study, directory)
    forge_episode_and_reseal(directory, variant, alter)
    with pytest.raises(ValueError):
        verify_navigation_study(directory)


def test_resealed_spatially_impossible_but_bounded_metrics_are_rejected(tmp_path, small_archived_study):
    import shutil
    from neurograft.navigation_study import verify_navigation_study
    directory = tmp_path / 'forged-geometry'
    shutil.copytree(small_archived_study, directory)
    def alter(episode):
        # This remains in the legal scalar domain and leaves failure SPL zero.
        episode['terminal_distance'] = (episode['terminal_distance'] + 1) % 5
    forge_episode_and_reseal(directory, 'policy', alter)
    with pytest.raises(ValueError, match='geometry|distance|metric'):
        verify_navigation_study(directory)


@pytest.mark.parametrize(('variant', 'alter'), [
    ('policy', lambda episode: episode.update(success=False, spl=0.5)),
    ('shortest_path', lambda episode: episode.update(spl=0.5)),
    ('policy', lambda episode: episode.update(policy_proposals=True)),
    ('policy', lambda episode: episode.update(
        proposed_wall_actions=(episode['proposed_wall_actions'] + 1) % (episode['steps'] + 1))),
    ('policy', lambda episode: episode['controller_trace'][0]['position'].update(
        x=3 if episode['controller_trace'][0]['position']['x'] != 3 else 1)),
    ('policy', lambda episode: episode['controller_trace'][0].update(
        policy_action=(episode['controller_trace'][0]['policy_action'] + 1) % 4)),
    ('observed_map', lambda episode: episode['controller_trace'][0].update(
        frontiers=episode['controller_trace'][0]['frontiers'] + 1)),
])
def test_resealed_in_domain_metric_formula_and_trace_consistency_is_enforced(
        tmp_path, small_archived_study, variant, alter):
    import shutil
    from neurograft.navigation_study import verify_navigation_study
    directory = tmp_path / 'forged-in-domain'
    shutil.copytree(small_archived_study, directory)
    forge_episode_and_reseal(directory, variant, alter)
    with pytest.raises(ValueError):
        verify_navigation_study(directory)
