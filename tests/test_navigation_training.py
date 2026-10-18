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
