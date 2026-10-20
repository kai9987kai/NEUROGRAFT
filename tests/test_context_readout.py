"""Optional direct source readout preserves legacy weights and exact continuation."""

import pytest
import torch
import torch.nn.functional as F

from neurograft.dynamics import DynamicsConfig, GraphCircuit
from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest


def _bundle():
    return make_fixture_bundle(seed=5, nodes_per_source=3)


def _model(readout="context_residual"):
    torch.manual_seed(31)
    return GraphCircuit(_bundle(), DynamicsConfig(hidden_dim=4, output_dim=4, readout=readout))


def _sequence():
    return torch.randn(3, 3, 4, 4, generator=torch.Generator().manual_seed(17))


def _equal(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor) and torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _equal(left, right)
    else:
        assert first == second


def test_optional_readout_preserves_initial_logits_shared_weights_and_rng():
    human = _model("human")
    human_rng = torch.get_rng_state().clone()
    residual = _model()
    residual_rng = torch.get_rng_state().clone()
    assert torch.equal(human_rng, residual_rng)
    assert DynamicsConfig().readout == LiveConfig().readout == "human"
    assert not any(key.startswith("context_policy.") for key in human.state_dict())
    assert set(residual.state_dict()) - set(human.state_dict()) == {
        "context_policy.weight", "context_policy.bias"}
    for name, value in human.state_dict().items():
        assert torch.equal(value, residual.state_dict()[name]), name
    assert torch.count_nonzero(residual.context_policy.weight) == 0
    assert torch.count_nonzero(residual.context_policy.bias) == 0
    for graft_enabled in (True, False):
        first = human(_sequence(), graft_enabled=graft_enabled)
        second = residual(_sequence(), graft_enabled=graft_enabled)
        assert torch.equal(first.logits, second.logits)
        assert torch.equal(first.logits, human.policy(first.context[:, -4:]))


def test_residual_readout_learns_from_every_source_without_graft():
    model = _model()
    sequence = _sequence()
    labels = torch.tensor([0, 1, 2])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = F.cross_entropy(model(sequence, graft_enabled=False).logits, labels)
    loss.backward()
    for source in range(4):
        gradient = model.context_policy.weight.grad[:, source * 4:(source + 1) * 4]
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sequence = sequence.clone().requires_grad_()
    F.cross_entropy(model(sequence, graft_enabled=False).logits, labels).backward()
    for source in range(4):
        gradient = sequence.grad[:, :, source]
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    for source in range(3):
        changed = sequence.detach().clone()
        changed[:, :, source] += 2
        assert not torch.equal(model(sequence, graft_enabled=False).logits,
                               model(changed, graft_enabled=False).logits)


def test_disabling_residual_restores_legacy_core_logits_without_changing_weights():
    human = _model("human")
    residual = _model()
    with torch.no_grad():
        residual.context_policy.weight.fill_(0.2)
        residual.context_policy.bias.fill_(0.3)
    before = _checkpoint_digest(residual.state_dict())
    sequence = _sequence()
    assert torch.equal(human(sequence).logits, human(sequence, context_enabled=False).logits)
    assert torch.equal(human(sequence).logits, residual(sequence, context_enabled=False).logits)
    assert not torch.equal(human(sequence).logits, residual(sequence).logits)
    assert _checkpoint_digest(residual.state_dict()) == before


@pytest.mark.parametrize("source", range(4))
@pytest.mark.parametrize("graft_enabled", [True, False])
def test_residual_source_lesion_blocks_direct_and_routed_input(source, graft_enabled):
    model = _model()
    with torch.no_grad():
        model.context_policy.weight.copy_(torch.randn(
            model.context_policy.weight.shape, generator=torch.Generator().manual_seed(9)))
    sequence = _sequence()
    changed = sequence.clone()
    changed[:, :, source] += 3
    first = model(sequence, graft_enabled=graft_enabled, ablate_source=source)
    second = model(changed, graft_enabled=graft_enabled, ablate_source=source)
    assert torch.equal(first.logits, second.logits)
    assert torch.count_nonzero(first.context[:, source * 4:(source + 1) * 4]) == 0
    assert not torch.equal(model(sequence, graft_enabled=graft_enabled).logits,
                           model(changed, graft_enabled=graft_enabled).logits)


@pytest.mark.parametrize("readout", ["", "all", None, True, 1, []])
@pytest.mark.parametrize("config_type", [DynamicsConfig, LiveConfig])
def test_invalid_readout_is_rejected(readout, config_type):
    with pytest.raises(ValueError, match="readout"):
        config_type(readout=readout)


def _session(readout):
    return LiveSession(_bundle(), LiveConfig(seed=19, hidden_dim=4, threads=1,
        batch_size=2, train_every=2, unroll=3, replay_capacity=8, readout=readout))


def test_residual_live_checkpoint_restores_exact_optimizer_and_future_update(tmp_path):
    session = _session("context_residual")
    assert session.anchor_model.config.readout == session.model.config.readout == "context_residual"
    assert session.snapshot()["navigation"]["readout"] == "context_residual"
    session.tick(4)
    path = tmp_path / "residual.pt"
    session.save(path)
    restored = LiveSession.load(path, session.bundle)
    assert restored.config.readout == "context_residual"
    _equal(session._payload(), restored._payload())
    session.tick(2)
    restored.tick(2)
    assert session.updates == restored.updates == 3
    _equal(session._payload(), restored._payload())


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_checkpoints_without_readout_restore_human_mode(tmp_path, version):
    session = _session("human")
    session.tick(4)
    payload = session._payload()
    payload["version"] = version
    payload["config"].pop("readout")
    if version < 3:
        payload.pop("navigation")
    if version == 1:
        payload.pop("auxiliary")
    path = tmp_path / f"legacy-v{version}.pt"
    torch.save({"payload": payload, "sha256": _checkpoint_digest(payload)}, path)
    if version == 1:
        with pytest.warns(RuntimeWarning, match="Legacy"):
            restored = LiveSession.load(path, session.bundle)
    else:
        restored = LiveSession.load(path, session.bundle)
    assert restored.config.readout == restored.model.config.readout == "human"
    _equal(session.model.state_dict(), restored.model.state_dict())
    sequence = _sequence()
    assert torch.equal(session.model(sequence).logits, restored.model(sequence).logits)
    if version >= 2:
        session.tick(2)
        restored.tick(2)
        _equal(session.model.state_dict(), restored.model.state_dict())
        _equal(session.optimizer.state_dict(), restored.optimizer.state_dict())
