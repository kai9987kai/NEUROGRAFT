"""Geometry and matched-exposure checks for optional curriculum factors."""
from dataclasses import replace
import random

import pytest
import torch

from neurograft.environment import NavigationWorld
from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession
from neurograft.navigation_training import CurriculumConfig


def test_quarter_turn_rotates_geometry_actions_and_preserves_padding():
    import neurograft.navigation_training as training
    # Heading north, goal east, wall north; one preceding padding frame.
    observation = [[.25, .5, 0., -1.], [.75, .5, .5, 0.], [1., 0., 0., 0.], [.2, .3, .4, 1.]]
    sequence = torch.tensor([[[[0.] * 4 for _ in range(4)], observation]])
    original = sequence.clone()
    rotated, actions = training._rotate_navigation_batch(sequence, torch.tensor([1]), torch.tensor([1]))
    assert torch.equal(sequence, original)
    assert torch.equal(rotated[0, 0], sequence[0, 0])
    assert torch.allclose(rotated[0, 1], torch.tensor([
        [.5, .25, 1., 0.], [.5, .75, 0., .5], [0., 1., 0., 0.], [.2, .3, .4, 1.]]))
    assert actions.tolist() == [2]
    for _ in range(3):
        rotated, actions = training._rotate_navigation_batch(rotated, actions, torch.tensor([1]))
    assert torch.allclose(rotated, original, atol=1e-7)
    assert actions.tolist() == [1]


def test_all_rotations_preserve_local_action_feasibility_and_goal_displacement():
    import neurograft.navigation_training as training
    for seed in range(12):
        world = NavigationWorld(seed, 11, 'corridors')
        sequence = torch.tensor([[world.observe()]]).repeat(4, 1, 1, 1)
        expert = world.expert_action()
        rotations = torch.arange(4)
        rotated, labels = training._rotate_navigation_batch(sequence, torch.full((4,), expert), rotations)
        assert labels.tolist() == [(expert + turn) % 4 for turn in range(4)]
        for index in range(4):
            assert rotated[index, 0, 2, labels[index]] == 0
            assert torch.allclose(rotated[index, 0, 1, :2] - rotated[index, 0, 0, :2],
                                  rotated[index, 0, 1, 2:], atol=1e-7)


@pytest.mark.parametrize('values', [
    {'readout': 'unknown'}, {'readout': True}, {'rotation_augmentation': 1},
    {'collection_policy': 'unknown'}, {'collection_policy': None},
    {'include_readout_controls': 1},
])
def test_curriculum_factors_reject_invalid_values(values):
    with pytest.raises(ValueError):
        CurriculumConfig(**values)


def test_matched_expert_collection_and_base_batches_do_not_depend_on_rotations_or_weights():
    import neurograft.navigation_training as training
    bundle = make_fixture_bundle(nodes_per_source=3)
    common = CurriculumConfig(rounds=2, worlds_per_round=3, updates_per_round=2,
        sizes=(5, 7), presets=('labyrinth',), max_steps=5, batch_size=4,
        replay_capacity=24, collection_policy='expert')
    receipts = []
    for augmented in (False, True):
        model = LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2, threads=1)).model
        if augmented:
            with torch.no_grad():
                model.policy.bias[0].add_(100)
        config = replace(common, rotation_augmentation=augmented)
        replay = training._ActionReservoir(24, random.Random(5))
        rng = random.Random(9)
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        receipts.append([training._collect_round(bundle, model, config, index, replay, optimizer, rng, 2)
                         for index in (1, 2)])
    for first, second in zip(*receipts):
        assert first['collection'] == second['collection']
        assert first['collection']['learner_steps'] == 0
        assert first['training']['batch_sha256'] == second['training']['batch_sha256']
        assert first['training']['rotation_counts'] == [8, 0, 0, 0]
        assert sum(second['training']['rotation_counts']) == 8
        assert sum(second['training']['rotation_counts'][1:]) > 0


def test_curriculum_exports_selected_context_readout_and_replays(tmp_path):
    from neurograft.navigation_artifacts import verify_navigation_run
    from neurograft.navigation_training import run_navigation_training
    bundle = make_fixture_bundle(nodes_per_source=3)
    path = tmp_path / 'input.pt'
    LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2)).save(path)
    before = path.read_bytes()
    result = run_navigation_training(bundle, path, tmp_path / 'run', CurriculumConfig(
        rounds=1, worlds_per_round=2, updates_per_round=2, max_steps=2,
        validation_seeds=(20000,), test_seeds=(30000,), eval_size=5,
        readout='context_residual', rotation_augmentation=True, collection_policy='expert'))
    candidate = LiveSession.load(tmp_path / 'run' / 'candidate.pt', bundle)
    assert candidate.config.readout == 'context_residual'
    assert len(result['initial_common_model_hash']) == 64
    assert verify_navigation_run(tmp_path / 'run', replay=True)['ok']
    assert path.read_bytes() == before
