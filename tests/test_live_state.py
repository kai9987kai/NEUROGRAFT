"""Regressions for complete continuation and observational diagnostics."""
import copy
import random

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest


def session():
    return LiveSession(make_fixture_bundle(seed=5, nodes_per_source=3), LiveConfig(
        seed=19, hidden_dim=4, threads=1, batch_size=2, train_every=2,
        unroll=3, replay_capacity=8))


def test_synced_anchor_and_auxiliary_state_continue_exactly(tmp_path):
    live = session()
    live.tick(6)
    live.chat("anchor sync")
    live.memory.set_lesion_mode("shifted")
    live.memory.working_context[0] = 0.123456789123456
    with torch.no_grad():
        live.byte_bridge.encoder[0].weight.add_(0.125)
        live.consolidation.block.router.bias.add_(torch.arange(4) * 0.25)
    live.memory.working_context[0] = 0.123456789123456
    with torch.no_grad():
        live.byte_bridge.encoder[0].weight.add_(0.125)
        live.consolidation.block.router.bias.add_(torch.arange(4) * 0.25)
    path = tmp_path / "full.pt"
    live.save(path)
    restored = LiveSession.load(path, live.bundle)
    assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())
    assert live.telemetry() == restored.telemetry()
    for count in (1, 3):
        live.tick(count)
        restored.tick(count)
        assert _checkpoint_digest(live._payload()) == _checkpoint_digest(restored._payload())
        assert live.telemetry() == restored.telemetry()


def test_telemetry_has_no_side_effects_or_global_random_draws():
    live = session()
    live.tick(6)
    live.memory.set_lesion_mode("random_retrieval")
    before = _checkpoint_digest(live._payload())
    phase_before = live.phase_plane.to_dict()
    random_before = random.getstate()
    first = live.telemetry()
    second = live.telemetry()
    assert first == second
    assert random.getstate() == random_before
    assert live.phase_plane.to_dict() == phase_before
    assert _checkpoint_digest(live._payload()) == before
    assert live.archaeology._last_report is None


def test_intervention_uses_one_current_weight_baseline(monkeypatch):
    live = session()
    live.tick(6)
    original = live.model.forward

    def disabled(sequence, **kwargs):
        # Simulate a circuit whose intervention flag is an exact identity.
        return original(sequence)

    monkeypatch.setattr(live.model, "forward", disabled)
    result = live.telemetry()
    assert result["graft_impact"] == 0.0
    assert all(value == 0.0 for value in result["source_ablation"].values())


def test_policy_probe_matches_current_world_and_assay_cache(monkeypatch):
    live = session()
    live.tick(6)
    calls = []
    original = type(live.archaeology).run_resection_assay

    def assay(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(live.archaeology), "run_resection_assay", assay)
    window = (list(live.observation_history) + [live.world.observe()])[-live.config.unroll:]
    with torch.no_grad():
        expected = live.model(torch.tensor([window], dtype=torch.float32)).logits.softmax(-1)[0]
    result = live.telemetry()
    assert result["action_probs"] == pytest.approx(expected.tolist())
    assert result["expert_action"] == live.world.expert_action()
    assert result["environment"]["policy_probe"]["updates"] == live.updates
    live.telemetry()
    assert len(calls) == 1
    live.tick()
    live.telemetry()
    assert len(calls) == 2


def test_policy_probe_matches_current_world_and_assay_cache(monkeypatch):
    live = session()
    live.tick(6)
    calls = []
    original = type(live.archaeology).run_resection_assay

    def assay(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(live.archaeology), "run_resection_assay", assay)
    window = (list(live.observation_history) + [live.world.observe()])[-live.config.unroll:]
    with torch.no_grad():
        expected = live.model(torch.tensor([window], dtype=torch.float32)).logits.softmax(-1)[0]
    result = live.telemetry()
    assert result["action_probs"] == pytest.approx(expected.tolist())
    assert result["expert_action"] == live.world.expert_action()
    assert result["environment"]["policy_probe"]["updates"] == live.updates
    live.telemetry()
    assert len(calls) == 1
    live.tick()
    live.telemetry()
    assert len(calls) == 2


def test_learning_off_freezes_self_model_parameters_and_optimizer():
    live = session()
    live.tick(4)
    live.learning = False
    weights = _checkpoint_digest(dict(live.self_model.named_parameters()))
    optimizer = _checkpoint_digest(live.self_model.optimizer.state_dict())
    live.tick(4)
    assert _checkpoint_digest(dict(live.self_model.named_parameters())) == weights
    # Audits may record observations while frozen; trainable weights and Adam remain fixed.
    assert _checkpoint_digest(live.self_model.optimizer.state_dict()) == optimizer


def test_compass_and_forward_observer_use_the_executed_transition(monkeypatch):
    live = session()
    angular_velocities = []
    original_step = live.compass.step

    def compass_step(*args, **kwargs):
        angular_velocities.append(kwargs["angular_velocity"])
        return original_step(*args, **kwargs)

    audited_actions = []
    original_audit = live.self_model.step_audit

    def audit(latent, action, next_latent, **kwargs):
        audited_actions.append(action)
        return original_audit(latent, action, next_latent, **kwargs)

    monkeypatch.setattr(live.compass, "step", compass_step)
    monkeypatch.setattr(live.self_model, "step_audit", audit)
    live.world.direction = 3
    live.tick(manual_action=0)
    live.tick(manual_action=1)
    assert angular_velocities == pytest.approx([torch.pi / 2 / 0.2] * 2)
    assert audited_actions == [0]


def test_legacy_checkpoint_reports_missing_auxiliary_state(tmp_path):
    live = session()
    live.tick(4)
    payload = copy.deepcopy(live._payload())
    payload.pop("auxiliary", None)
    payload.pop("navigation", None)
    payload["version"] = 1
    path = tmp_path / "legacy.pt"
    torch.save({"payload": payload, "sha256": _checkpoint_digest(payload)}, path)
    with pytest.warns(RuntimeWarning, match="auxiliary"):
        restored = LiveSession.load(path, live.bundle)
    assert restored.checkpoint_compatibility["auxiliary_state_restored"] is False
    assert restored.snapshot()["model_hash"] == live.snapshot()["model_hash"]


@pytest.mark.parametrize("mutation", [
    lambda state: state["stp"]["u"][0].fill_(float("nan")),
    lambda state: state["compass"]["state"].append(0.0),
    lambda state: state["memory"].update(capacity=10_000_000),
    lambda state: state["self_model_optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
    lambda state: state["byte_bridge"]["encoder.0.weight"].fill_(3e38),
])
def test_invalid_auxiliary_state_is_rejected_before_installation(mutation):
    live = session()
    live.tick(4)
    payload = copy.deepcopy(live._payload())
    mutation(payload["auxiliary"])
    restored = session()
    before = _checkpoint_digest(restored._payload())
    with pytest.raises(ValueError):
        restored._restore_validated(payload)
    assert _checkpoint_digest(restored._payload()) == before


@pytest.mark.parametrize("preset", ["concentric", "hierarchical"])
def test_chat_supports_every_environment_preset(preset):
    live = session()
    assert preset in live.chat("preset " + preset)["reply"]
    assert live.world.preset == preset


def test_goal_discontinuity_resets_phase_history_without_advancing_world():
    live = session()
    live.tick()
    agent = live.world._agent
    steps_before = live.steps
    live.set_goal(x=agent[0], z=agent[1])
    assert live._phase_summary is None
    assert live.phase_plane.last_state is None
    first = live.telemetry()["phase_plane"]
    assert first["state"]["proximity"] == 1.0
    assert first["lyapunov_drift"] == 0.0
    assert live.telemetry()["phase_plane"] == first
    assert live.steps == steps_before
