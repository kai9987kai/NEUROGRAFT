import json
import random

import pytest
import torch


@pytest.mark.parametrize('values', [
    {'rounds': 0}, {'rounds': True}, {'rounds': 21}, {'batch_size': 0},
    {'worlds_per_round': 257}, {'updates_per_round': 0}, {'sizes': (4,)},
    {'presets': ('unknown',)}, {'init': 'magic'}, {'learning_rate': float('nan')},
    {'validation_seeds': (100,)}, {'test_seeds': (20000,)},
    {'validation_seeds': (8, 8)}, {'dense_warmup_rounds': 5},
    {'train_seed_start': 2**63-1}, {'replay_capacity': 3},
    {'batch_size': 17}, {'updates_per_round': 257}, {'replay_capacity': 4097},
    {'rounds': 20, 'updates_per_round': 256},
    {'rounds': 20, 'worlds_per_round': 256, 'max_steps': 128},
    {'learning_rate': 0.5}, {'learning_rate': 1e-9},
    {'validation_seeds': tuple(range(20000, 20017))},
    {'test_seeds': tuple(range(30000, 30017))},
    {'sizes': (5, 5)}, {'presets': ('labyrinth', 'labyrinth')},
])
def test_curriculum_rejects_unsafe_or_overlapping_protocol(values):
    from neurograft.navigation_training import CurriculumConfig
    with pytest.raises(ValueError):
        CurriculumConfig(**values)


def test_candidate_run_is_reproducible_preserves_source_and_exports_valid_session(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest
    from neurograft.navigation_training import CurriculumConfig, run_navigation_training
    bundle = make_fixture_bundle(nodes_per_source=3)
    source = LiveSession(bundle, LiveConfig(hidden_dim=4, batch_size=2, unroll=2))
    source.tick(3)
    checkpoint = tmp_path / 'source.pt'
    source.save(checkpoint)
    before = checkpoint.read_bytes()
    config = CurriculumConfig(rounds=2, worlds_per_round=4, updates_per_round=2,
                              max_steps=4, sizes=(5,), presets=('labyrinth',),
                              validation_seeds=(20000,), test_seeds=(30000,),
                              eval_size=5, batch_size=2, replay_capacity=32)
    py_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    first = run_navigation_training(bundle, checkpoint, tmp_path / 'first', config)
    second = run_navigation_training(bundle, checkpoint, tmp_path / 'second', config)
    assert checkpoint.read_bytes() == before
    assert random.getstate() == py_rng and torch.equal(torch.get_rng_state(), torch_rng)
    for key in ('progress', 'selected_model_hash', 'candidate', 'baseline', 'protocol_id'):
        assert first[key] == second[key]
    assert first['promotion_performed'] is False
    assert first['candidate']['teacher_assistance'] == 0
    assert first['candidate']['protocol_id'] == first['baseline']['protocol_id']
    assert first['progress'][1]['collection']['worlds'] == 4
    assert first['progress'][2]['replay']['seen_total'] > first['progress'][1]['replay']['seen_total']
    assert first['progress'][2]['replay']['retained_total'] <= config.replay_capacity
    assert len(first['progress'][2]['training']['losses']) == config.updates_per_round
    assert first['candidate']['variants']['policy']['episodes'][0]['initial_world_hash'] == first['baseline']['variants']['policy']['episodes'][0]['initial_world_hash']
    loaded = LiveSession.load(tmp_path / 'first' / 'candidate.pt', bundle)
    assert _checkpoint_digest(loaded.model.state_dict()) == first['selected_model_hash']
    assert loaded.steps == loaded.updates == 0
    assert not loaded.running and not loaded.learning
    assert loaded.model.config.top_k == 2
    assert _checkpoint_digest(loaded.anchor_model.state_dict()) == first['selected_model_hash']
    assert json.loads((tmp_path / 'first' / 'status.json').read_text())['status'] == 'COMPLETE'
    assert (tmp_path / 'first' / 'report.html').is_file()
    assert 'candidate.pt' in json.loads((tmp_path / 'first' / 'manifest.json').read_text())['files']
    with pytest.raises(ValueError, match='already exists'):
        run_navigation_training(bundle, checkpoint, tmp_path / 'first', config)
    assert checkpoint.read_bytes() == before


def test_round_selection_uses_validation_only_and_retains_earlier_tie():
    from neurograft.navigation_training import _selection_key
    good = dict(success_rate=.5, spl=.4, collision_rate=.1)
    bad = dict(success_rate=.25, spl=.9, collision_rate=0)
    assert _selection_key(good) > _selection_key(bad)
    assert _selection_key(good) == _selection_key(dict(good, test_success_rate=1))


def test_failed_run_is_marked_without_touching_source(tmp_path, monkeypatch):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    import neurograft.navigation_training as training
    bundle = make_fixture_bundle(nodes_per_source=3)
    path = tmp_path / 'input.pt'
    LiveSession(bundle, LiveConfig(hidden_dim=4)).save(path)
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError('injected training error')
    monkeypatch.setattr(training, '_collect_round', fail)
    with pytest.raises(RuntimeError, match='injected'):
        training.run_navigation_training(bundle, path, tmp_path / 'failed', training.CurriculumConfig(
            rounds=1, worlds_per_round=1, updates_per_round=1, max_steps=1,
            validation_seeds=(20000,), test_seeds=(30000,), eval_size=5))
    status = json.loads((tmp_path / 'failed' / 'status.json').read_text())
    assert status['status'] == 'FAILED' and 'injected' in status['error']
    assert path.read_bytes() == before
    assert not (tmp_path / 'failed' / 'candidate.pt').exists()


def test_world_schedule_covers_cartesian_conditions_independently_of_seed_offset():
    from neurograft.navigation_training import CurriculumConfig, _world_schedule
    config = CurriculumConfig(rounds=2, worlds_per_round=6, sizes=(5, 11),
                              presets=('labyrinth', 'chambers', 'pillars'),
                              train_seed_start=123)
    first = _world_schedule(config, 1)
    second = _world_schedule(config, 2)
    assert [item['seed'] for item in first + second] == list(range(123, 135))
    expected = {(5, 'labyrinth'), (11, 'labyrinth'), (5, 'chambers'),
                (11, 'chambers'), (5, 'pillars'), (11, 'pillars')}
    assert {(item['size'], item['preset']) for item in first} == expected
    assert {(item['size'], item['preset']) for item in second} == expected


def test_reservoir_is_bounded_preserves_action_classes_and_retains_earlier_worlds():
    from neurograft.navigation_training import _ActionReservoir
    first = _ActionReservoir(capacity=20, rng=random.Random(7))
    second = _ActionReservoir(capacity=20, rng=random.Random(7))
    # Four early worlds supply one class each. Later worlds overwhelm only north.
    for action in range(4):
        for index in range(40):
            sample = {'action': action, 'world_seed': action, 'round': 1, 'id': (action, index)}
            first.add(sample)
            second.add(sample)
            assert len(first.samples) <= 20
    for index in range(1000):
        sample = {'action': 0, 'world_seed': 99, 'round': 2, 'id': (99, index)}
        first.add(sample)
        second.add(sample)
        assert len(first.samples) <= 20
    assert first.samples == second.samples
    receipt = first.summary()
    assert receipt['seen_total'] == 1160
    assert receipt['retained_total'] == 20
    assert receipt['action_counts'] == {'north': 5, 'east': 5, 'south': 5, 'west': 5}
    assert {1, 2, 3} <= {sample['world_seed'] for sample in first.samples}
    batch = first.balanced_batch(16, random.Random(11))
    assert [sum(sample['action'] == action for sample in batch) for action in range(4)] == [4, 4, 4, 4]


def test_round_receipt_records_actual_coverage_replay_and_finite_training_losses():
    import math
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, LiveSession
    from neurograft.navigation_training import CurriculumConfig, _ActionReservoir, _collect_round
    bundle = make_fixture_bundle(nodes_per_source=3)
    model = LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2, threads=1)).model
    config = CurriculumConfig(rounds=2, worlds_per_round=2, updates_per_round=2,
                              sizes=(5, 7), presets=('labyrinth',), max_steps=3,
                              batch_size=4, replay_capacity=16)
    replay = _ActionReservoir(config.replay_capacity, random.Random(5))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    receipt = _collect_round(bundle, model, config, 1, replay, optimizer, random.Random(9), 2)
    assert receipt['collection']['world_seeds'] == [0, 1]
    assert receipt['collection']['condition_counts'] == [
        {'size': 5, 'preset': 'labyrinth', 'worlds': 1},
        {'size': 7, 'preset': 'labyrinth', 'worlds': 1}]
    assert receipt['collection']['transitions'] == receipt['replay']['seen_total']
    assert sum(receipt['collection']['action_counts'].values()) == receipt['collection']['transitions']
    assert receipt['collection']['teacher_steps'] == receipt['collection']['transitions']
    assert receipt['collection']['learner_steps'] == 0
    assert receipt['training']['updates'] == 2
    assert len(receipt['training']['losses']) == 2
    assert all(math.isfinite(value) and value > 0 for value in receipt['training']['losses'])
    assert receipt['replay']['retained_total'] <= 16


def test_reservoir_reserves_remainder_slots_and_learns_late_action_classes():
    from neurograft.navigation_training import _ActionReservoir
    replay = _ActionReservoir(capacity=7, rng=random.Random(31))
    for index in range(100):
        replay.add({'action': 0, 'world_seed': 0, 'round': 1, 'id': index})
        assert len(replay.samples) <= 7
    early_ids = {sample['id'] for sample in replay.samples}
    assert replay.summary()['action_counts'] == {'north': 2, 'east': 0, 'south': 0, 'west': 0}
    for action in (1, 2, 3):
        for index in range(100):
            replay.add({'action': action, 'world_seed': action, 'round': 2, 'id': index})
            assert len(replay.samples) <= 7
    assert {sample['id'] for sample in replay.samples if sample['action'] == 0} == early_ids
    assert replay.summary()['action_counts'] == {'north': 2, 'east': 2, 'south': 2, 'west': 1}
    # A one-item update must not always choose the first populated action class.
    rng = random.Random(11)
    choices = [replay.balanced_batch(1, rng)[0]['action'] for _ in range(40)]
    assert set(choices) == {0, 1, 2, 3}
