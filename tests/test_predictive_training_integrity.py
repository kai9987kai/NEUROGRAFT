"""Predict executed transitions, retaining old sampling and rotation semantics."""
import copy
import random

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession
from neurograft.navigation_training import CurriculumConfig


@pytest.mark.parametrize('values', [
    {'predictive_weight': True}, {'predictive_weight': float('nan')},
    {'predictive_weight': float('inf')}, {'predictive_weight': .5},
])
def test_invalid_or_inactive_predictive_objective_is_rejected(values):
    with pytest.raises(ValueError, match='predictive'):
        CurriculumConfig(**values)


def test_predictive_batch_rotates_executed_action_and_next_observation_together():
    import neurograft.navigation_training as training
    obs = [[.25, .5, 0., -1.], [.75, .5, .5, 0.], [1., 0., 0., 0.], [.2, .3, .4, 1.]]
    sample = {'observations': [[[0.] * 4 for _ in range(4)], obs], 'action': 1,
              'executed_action': 2, 'next_observation': obs, 'transition_valid': False}
    original = copy.deepcopy(sample)
    seq, labels, executed, targets, valid = training._prepare_navigation_batch(
        [sample], torch.tensor([1]), torch.float32, True)
    assert labels.tolist() == [2] and executed.tolist() == [3]
    assert valid.tolist() == [False]
    assert torch.equal(seq[0, 0], torch.zeros(4, 4))
    assert torch.allclose(targets[0], torch.tensor([
        [.5, .25, 1., 0.], [.5, .75, 0., .5], [0., 1., 0., 0.], [.2, .3, .4, 1.]]))
    assert sample == original


def test_terminal_targets_cannot_contaminate_mse_or_encoder_gradients():
    from neurograft.navigation_training import _masked_prediction_loss
    prediction = torch.zeros(2, 4, 4, requires_grad=True)
    targets = torch.stack((torch.ones(4, 4), torch.full((4, 4), 1e20)))
    loss = _masked_prediction_loss(prediction, targets, torch.tensor([True, False]))
    assert loss.item() == 1.0
    loss.backward()
    assert prediction.grad[0].abs().sum() > 0
    assert torch.count_nonzero(prediction.grad[1]) == 0
    assert _masked_prediction_loss(prediction, targets, torch.tensor([False, False])) is None


def test_legacy_expert_collection_preserves_historical_minibatch_rng_stream():
    import neurograft.navigation_training as training
    bundle = make_fixture_bundle(nodes_per_source=3)
    model = LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2, threads=1)).model
    cfg = CurriculumConfig(rounds=2, worlds_per_round=2, updates_per_round=2,
        sizes=(5,), presets=('labyrinth',), max_steps=4, batch_size=4,
        replay_capacity=16, collection_policy='expert')
    rng = random.Random(9)
    receipt = training._collect_round(bundle, model, cfg, 1,
        training._ActionReservoir(16, random.Random(5)),
        torch.optim.Adam(model.parameters(), lr=cfg.learning_rate), rng, 2)
    assert receipt['collection']['data_sha256'] == '751f945a6a542e7081af49ca534f29098d0a9a108c71b46a1a1544aefdf1ee36'
    assert receipt['training']['batch_sha256'] == '02e428f84068bbfbd7224ce89674fe150bdc46c2109fff62569c4ca233cde020'
    assert rng.random() == 0.05133402430227041


def test_predictive_training_reports_separate_losses_and_excludes_goal_boundaries(tmp_path):
    from neurograft.navigation_training import run_navigation_training
    bundle = make_fixture_bundle(nodes_per_source=3)
    source = tmp_path / 'source.pt'
    LiveSession(bundle, LiveConfig(hidden_dim=4, unroll=2)).save(source)
    result = run_navigation_training(bundle, source, tmp_path / 'run', CurriculumConfig(
        rounds=1, worlds_per_round=6, updates_per_round=2, max_steps=32,
        sizes=(5,), presets=('labyrinth',), eval_size=5, batch_size=4,
        validation_seeds=(20000,), test_seeds=(30000,), predictive_head=True,
        predictive_weight=.5, collection_policy='expert'))
    receipt = result['progress'][1]
    assert receipt['collection']['prediction_excluded_transitions'] == 6
    assert receipt['collection']['prediction_valid_transitions'] > 0
    objective = receipt['training']['prediction']
    assert len(objective['losses']) == len(objective['total_losses']) == 2
    assert objective['valid_examples'] > 0
    for ce, pred, total in zip(receipt['training']['losses'], objective['losses'], objective['total_losses']):
        assert total == pytest.approx(ce + .5 * pred)


def test_checkpoint_initialization_rejects_predictor_architecture_mismatch(tmp_path):
    from neurograft.navigation_training import run_navigation_training
    bundle = make_fixture_bundle(nodes_per_source=3)
    source = tmp_path / 'source.pt'
    LiveSession(bundle, LiveConfig(hidden_dim=4)).save(source)
    with pytest.raises(ValueError, match='predictive_head'):
        run_navigation_training(bundle, source, tmp_path / 'run', CurriculumConfig(
            init='checkpoint', predictive_head=True, rounds=1, worlds_per_round=1,
            updates_per_round=1, max_steps=1, eval_size=5,
            validation_seeds=(20000,), test_seeds=(30000,)))
