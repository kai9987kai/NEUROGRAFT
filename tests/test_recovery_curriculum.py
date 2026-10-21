"""Tests for Recovery-augmented DAgger and Predictive Multi-Task Training Curriculum."""
import pytest
import torch
from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession
from neurograft.navigation_training import CurriculumConfig, run_navigation_training


def test_curriculum_config_validation():
    cfg = CurriculumConfig(collection_policy="recovery", predictive_head=True, predictive_weight=0.5)
    assert cfg.collection_policy == "recovery"
    assert cfg.predictive_head is True
    assert cfg.predictive_weight == 0.5

    with pytest.raises(ValueError, match="collection_policy must be 'mixed', 'expert' or 'recovery'"):
        CurriculumConfig(collection_policy="invalid")

    with pytest.raises(ValueError, match="predictive_head must be boolean"):
        CurriculumConfig(predictive_head="yes")  # type: ignore

    with pytest.raises(ValueError, match="predictive_weight must be a finite float in"):
        CurriculumConfig(predictive_weight=-1.0)


def test_recovery_curriculum_smoke(tmp_path):
    bundle = make_fixture_bundle(nodes_per_source=3)
    session = LiveSession(bundle, LiveConfig(hidden_dim=4, batch_size=2, unroll=2))
    checkpoint = tmp_path / "source.pt"
    session.save(checkpoint)

    cfg = CurriculumConfig(
        rounds=1,
        worlds_per_round=2,
        updates_per_round=2,
        batch_size=2,
        sizes=(5,),
        presets=("labyrinth",),
        max_steps=4,
        validation_seeds=(20000,),
        test_seeds=(30000,),
        eval_size=5,
        collection_policy="recovery",
        seed=123,
    )
    result = run_navigation_training(bundle, checkpoint, tmp_path / "recovery_run", cfg)
    assert "progress" in result
    coll = result["progress"][1]["collection"]
    assert coll["teacher_steps"] + coll["learner_steps"] > 0
    assert len(result["progress"][1]["training"]["losses"]) == 2


def test_predictive_multitask_training_smoke(tmp_path):
    bundle = make_fixture_bundle(nodes_per_source=3)
    session = LiveSession(bundle, LiveConfig(hidden_dim=4, batch_size=2, unroll=2))
    checkpoint = tmp_path / "source.pt"
    session.save(checkpoint)

    cfg = CurriculumConfig(
        rounds=1,
        worlds_per_round=2,
        updates_per_round=2,
        batch_size=2,
        sizes=(5,),
        presets=("labyrinth",),
        max_steps=4,
        validation_seeds=(20000,),
        test_seeds=(30000,),
        eval_size=5,
        predictive_head=True,
        predictive_weight=1.0,
        seed=456,
    )
    result = run_navigation_training(bundle, checkpoint, tmp_path / "pred_run", cfg)
    assert "progress" in result
    losses = result["progress"][1]["training"]["losses"]
    assert len(losses) == 2
    for l in losses:
        assert torch.isfinite(torch.tensor(l))
