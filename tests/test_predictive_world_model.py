"""Tests for Connectome Predictive World Model in GraphCircuit."""
import pytest
import torch
from neurograft.dynamics import DynamicsConfig, GraphCircuit, CircuitOutput
from neurograft.fixtures import make_fixture_bundle


@pytest.fixture
def minimal_bundle():
    return make_fixture_bundle(seed=7, nodes_per_source=5)


def test_predictive_head_config_validation():
    cfg = DynamicsConfig(predictive_head=False)
    assert cfg.predictive_head is False
    cfg_true = DynamicsConfig(predictive_head=True)
    assert cfg_true.predictive_head is True

    with pytest.raises(ValueError, match="predictive_head must be a boolean"):
        DynamicsConfig(predictive_head="invalid")  # type: ignore


def test_predictive_head_zero_initialization_invariance(minimal_bundle):
    # Test that model with predictive_head=False has no predictor
    torch.manual_seed(42)
    circuit_base = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=False, output_dim=4))
    assert circuit_base.predictor is None

    # Test that model with predictive_head=True creates zero-initialized output predictor
    torch.manual_seed(42)
    circuit_pred = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, output_dim=4))
    assert circuit_pred.predictor is not None
    assert torch.all(circuit_pred.predictor[2].weight == 0.0)
    assert torch.all(circuit_pred.predictor[2].bias == 0.0)

    # Policy logits must be bit-identical between both on same inputs
    obs = torch.randn(1, 4, 4)
    out_base = circuit_base.step(obs)
    out_pred = circuit_pred.step(obs)
    torch.testing.assert_close(out_base.logits, out_pred.logits)
    assert out_base.predicted_obs is None
    assert out_pred.predicted_obs is None  # Action not provided, so predicted_obs is None


def test_predict_transition_forward_and_backward(minimal_bundle):
    circuit = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, hidden_dim=12, input_dim=4, output_dim=4))
    obs = torch.randn(2, 4, 4)
    action = torch.tensor([1, 2])

    out = circuit.step(obs, action=action)
    assert out.predicted_obs is not None
    # Target observation shape is [batch, 4, input_dim] = [2, 4, 4]
    assert out.predicted_obs.shape == (2, 4, 4)

    # Initial prediction is exactly zero because final layer weights and bias are initialized to zero
    torch.testing.assert_close(out.predicted_obs, torch.zeros(2, 4, 4))

    # Test gradient flow through predictor
    target_next_obs = torch.randn(2, 4, 4)
    loss = torch.nn.functional.mse_loss(out.predicted_obs, target_next_obs)
    loss.backward()

    assert circuit.predictor[2].weight.grad is not None
    assert circuit.predictor[2].bias.grad is not None
    assert torch.any(circuit.predictor[2].weight.grad != 0.0)


def test_predictive_head_state_dict_roundtrip(minimal_bundle):
    circuit = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, output_dim=4))
    sd = circuit.state_dict()
    assert "predictor.0.weight" in sd
    assert "predictor.2.weight" in sd
    assert "predictor.2.bias" in sd

    circuit2 = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, output_dim=4))
    circuit2.load_state_dict(sd)
    assert torch.all(circuit2.predictor[2].weight == circuit.predictor[2].weight)


@pytest.mark.parametrize("context", [
    [[0.0] * 48],
    torch.zeros(48),
    torch.zeros(0, 48),
    torch.zeros(2, 47),
    torch.zeros(2, 48, dtype=torch.int64),
    torch.zeros(2, 48, dtype=torch.float64),
    torch.full((2, 48), float("nan")),
    torch.full((2, 48), float("inf")),
    torch.zeros(2, 48).to_sparse(),
    torch.empty(2, 48, device="meta"),
])
def test_prediction_rejects_invalid_context(minimal_bundle, context):
    circuit = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, hidden_dim=12, output_dim=4))
    with pytest.raises(ValueError, match="context"):
        circuit.predict_transition(context, 1)


@pytest.mark.parametrize("action", [
    True, False, -1, 4, 1.5, [1, 2], [[0, 1, 0, 0], [0, 0, 1, 0]],
    torch.tensor(1),
    torch.tensor([1]),
    torch.tensor([0, 1, 2]),
    torch.tensor([[1], [2]]),
    torch.tensor([True, False]),
    torch.tensor([1.0, 2.0]),
    torch.tensor([float("nan"), 1.0]),
    torch.tensor([1, 2], dtype=torch.int16),
    torch.tensor([-1, 2]),
    torch.tensor([1, 4]),
    torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
    torch.tensor([1, 2]).to_sparse(),
    torch.empty(2, dtype=torch.int64, device="meta"),
])
def test_prediction_rejects_ambiguous_or_invalid_actions(minimal_bundle, action):
    circuit = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, hidden_dim=12, output_dim=4))
    with pytest.raises(ValueError, match="action"):
        circuit.predict_transition(torch.zeros(2, 48), action)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_prediction_accepts_scalar_broadcast_and_batched_indices(minimal_bundle, dtype):
    circuit = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, hidden_dim=12, output_dim=4)).double()
    # Nonzero final weights make equality depend on correct action conditioning.
    torch.nn.init.normal_(circuit.predictor[-1].weight)
    context = torch.randn(2, 48, dtype=torch.float64)
    scalar = circuit.predict_transition(context, 2)
    batched = circuit.predict_transition(context, torch.tensor([2, 2], dtype=dtype))
    assert scalar.shape == (2, 4, 4)
    assert scalar.dtype == torch.float64
    assert torch.equal(scalar, batched)


@pytest.mark.parametrize("readout", ["human", "context_residual"])
def test_predictor_preserves_exact_common_weights_rng_and_policy(minimal_bundle, readout):
    torch.manual_seed(1729)
    baseline = GraphCircuit(minimal_bundle, DynamicsConfig(output_dim=4, readout=readout))
    after_baseline = torch.get_rng_state().clone()
    torch.manual_seed(1729)
    predictive = GraphCircuit(minimal_bundle, DynamicsConfig(output_dim=4, readout=readout, predictive_head=True))
    assert torch.equal(after_baseline, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, predictive.state_dict()[name]), name
    observations = torch.randn(2, 3, 4, 4)
    assert torch.equal(baseline(observations).logits, predictive(observations, action=1).logits)


def test_predictive_loss_reaches_encoders_after_head_warmup(minimal_bundle):
    torch.manual_seed(17)
    circuit = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, hidden_dim=6, output_dim=4))
    optimizer = torch.optim.SGD(circuit.parameters(), lr=0.1)
    observations = torch.randn(2, 3, 4, 4)
    target = torch.randn(2, 4, 4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = circuit(observations, action=torch.tensor([0, 3])).predicted_obs
        torch.nn.functional.mse_loss(prediction, target).backward()
        optimizer.step()
    encoder_gradients = [parameter.grad for parameter in circuit.encoders.parameters()]
    assert any(gradient is not None and bool((gradient != 0).any()) for gradient in encoder_gradients)


def test_sequence_prediction_uses_final_context(minimal_bundle):
    circuit = GraphCircuit(minimal_bundle, DynamicsConfig(predictive_head=True, output_dim=4))
    torch.nn.init.normal_(circuit.predictor[-1].weight)
    observations = torch.randn(2, 3, 4, 4)
    action = torch.tensor([0, 3])
    output = circuit(observations, action=action)
    state = None
    for frame in observations.unbind(dim=1):
        step_output = circuit.step(frame, state)
        state = step_output.state
    expected = circuit.predict_transition(step_output.context, action)
    assert torch.equal(output.predicted_obs, expected)
