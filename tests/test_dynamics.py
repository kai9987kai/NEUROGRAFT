"""Behavioral checks for sparse trainable graph dynamics (synthetic inputs only)."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from neurograft.dynamics import DynamicsConfig, GraphCircuit, SparseRelation
from neurograft.fixtures import make_fixture_bundle
from neurograft.graphs import Edge, GraphBundle


torch.set_num_threads(2)


def _circuit(top_k=3):
    torch.manual_seed(41)
    bundle = make_fixture_bundle(seed=7, nodes_per_source=5)
    return GraphCircuit(bundle, DynamicsConfig(hidden_dim=8, top_k=top_k)), bundle


def _sequence(batch=3, steps=4):
    generator = torch.Generator().manual_seed(91)
    return torch.randn(batch, steps, 4, 4, generator=generator)


def test_forward_has_external_node_state_and_resets_each_episode():
    model, bundle = _circuit()
    sequence = _sequence()
    first = model(sequence)
    model(sequence.flip(1))
    second = model(sequence)
    assert first.logits.shape == (3, 2)
    assert first.context.shape == (3, 32)
    assert first.routes.shape == (3, 4, 5)
    assert isinstance(first.state, tuple)
    assert [tuple(state.shape) for state in first.state] == [(3, 5, 8)] * 4
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.context, second.context)
    assert model.context_dim == 32
    assert bundle.digest() == model.bundle_digest


def test_step_does_not_mutate_supplied_state_or_source_graph():
    model, bundle = _circuit()
    sequence = _sequence()
    original_digest = bundle.digest()
    state = model.step(sequence[:, 0]).state
    state_copies = tuple(tensor.detach().clone() for tensor in state)
    next_output = model.step(sequence[:, 1], state)
    assert any(not torch.equal(new, old) for new, old in zip(next_output.state, state))
    assert all(torch.equal(old, copy) for old, copy in zip(state, state_copies))
    assert bundle.digest() == original_digest


def test_cross_source_routes_mask_self_before_normalization_and_keep_null():
    model, _ = _circuit(top_k=1)
    with torch.no_grad():
        model.route_bias.fill_diagonal_(1000)
    output = model(_sequence())
    routes = output.routes
    assert torch.equal(routes[:, range(4), range(4)], torch.zeros(3, 4))
    assert torch.allclose(routes.sum(-1), torch.ones(3, 4), atol=1e-7)
    assert torch.equal((routes[..., :4] > 0).sum(-1), torch.ones(3, 4, dtype=torch.long))
    assert torch.all(routes[..., 4] > 0)
    assert torch.isfinite(routes).all()


def test_sender_selection_is_per_receiver_and_per_example():
    model, _ = _circuit(top_k=1)
    with torch.no_grad():
        for projection in tuple(model.queries) + tuple(model.keys):
            projection.weight.copy_(torch.eye(8))
    pooled = torch.zeros(2, 4, 8)
    pooled[:, 3, 0] = 1
    pooled[0, 0, 0], pooled[0, 1, 0] = 2, -2
    pooled[1, 0, 0], pooled[1, 1, 0] = -2, 2
    _, routes = model._route(pooled)
    assert routes[0, 3, 0] > 0 and routes[0, 3, 1] == 0
    assert routes[1, 3, 1] > 0 and routes[1, 3, 0] == 0


def test_null_only_routes_equal_graft_bypass_without_changing_parameters():
    model, _ = _circuit(top_k=0)
    sequence = _sequence()
    parameters_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    enabled = model(sequence)
    disabled = model(sequence, graft_enabled=False)
    assert torch.equal(enabled.logits, disabled.logits)
    assert torch.equal(enabled.context, disabled.context)
    assert torch.equal(enabled.routes[..., 4], torch.ones(3, 4))
    assert torch.equal(enabled.routes[..., :4], torch.zeros(3, 4, 4))
    assert all(torch.equal(parameter, parameters_before[name]) for name, parameter in model.named_parameters())


def test_no_graft_human_policy_is_independent_of_other_source_inputs():
    model, _ = _circuit()
    sequence = _sequence()
    altered = sequence.clone()
    altered[:, :, :3] += 10
    first = model(sequence, graft_enabled=False)
    second = model(altered, graft_enabled=False)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.context[:, -8:], second.context[:, -8:])
    assert not torch.equal(first.context[:, :8], second.context[:, :8])
    assert torch.equal(first.routes[..., 4], torch.ones(3, 4))


def test_no_recurrence_forgets_earlier_observations():
    model, _ = _circuit()
    sequence = _sequence()
    altered = sequence.clone()
    altered[:, :-1] += 5
    first = model(sequence, recurrence=False)
    second = model(altered, recurrence=False)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.context, second.context)
    assert not torch.allclose(model(sequence).logits, model(altered).logits)


def test_gradients_reach_graph_states_nonhuman_inputs_routing_and_recurrent_weights():
    model, _ = _circuit()
    sequence = _sequence().requires_grad_()
    initial = tuple(tensor.requires_grad_() for tensor in model.initial_state(3))
    output = model.step(sequence[:, 0], initial)
    for time in range(1, sequence.shape[1]):
        output = model.step(sequence[:, time], output.state)
    loss = F.cross_entropy(output.logits, torch.tensor([0, 1, 0]))
    loss.backward()
    assert sequence.grad is not None and torch.isfinite(sequence.grad).all()
    assert sequence.grad[:, 0, :3].abs().sum().item() > 0
    assert all(state.grad is not None and state.grad.abs().sum().item() > 0 for state in initial)
    parameters = dict(model.named_parameters())
    for prefix in (
        "encoders.0.gru", "encoders.3.gru", "queries.3", "keys.0", "values.0",
        "encoders.0.relations.chemical.transform", "encoders.3.relations.macro_functional.transform",
    ):
        gradients = [parameter.grad for name, parameter in parameters.items() if name.startswith(prefix)]
        assert gradients and any(gradient is not None and gradient.abs().sum().item() > 0 for gradient in gradients), prefix
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_optimizer_changes_weights_and_reduces_loss_on_bounded_example():
    model, _ = _circuit()
    sequence = _sequence(batch=6)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    before = model.policy.weight.detach().clone()
    initial_loss = F.cross_entropy(model(sequence).logits, labels).item()
    for _ in range(12):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(sequence).logits, labels)
        loss.backward()
        optimizer.step()
    final_loss = F.cross_entropy(model(sequence).logits, labels).item()
    assert not torch.equal(before, model.policy.weight)
    assert final_loss < initial_loss


def test_signed_sparse_relation_uses_absolute_incoming_weight_normalization():
    relation = SparseRelation(3, [0, 1], [2, 2], [2.0, -1.0], hidden_dim=1)
    with torch.no_grad():
        relation.transform.weight.fill_(1.0)
    states = torch.tensor([[[3.0], [6.0], [0.0]], [[9.0], [3.0], [0.0]]], requires_grad=True)
    output = relation(states)
    assert torch.allclose(output, torch.tensor([[[0.0], [0.0], [0.0]], [[0.0], [0.0], [5.0]]]))
    output.sum().backward()
    assert torch.allclose(states.grad[:, :, 0], torch.tensor([[2 / 3, -1 / 3, 0.0]] * 2))
    assert dict(relation.named_buffers())["src"].shape == (2,)


def test_relation_weight_scaling_does_not_change_normalized_messages():
    first = SparseRelation(3, [0, 1], [2, 2], [2.0, -1.0], hidden_dim=2)
    second = SparseRelation(3, [0, 1], [2, 2], [20.0, -10.0], hidden_dim=2)
    with torch.no_grad():
        second.transform.weight.copy_(first.transform.weight)
    states = torch.tensor([[[3.0, 1.0], [6.0, 4.0], [0.0, 0.0]]])
    assert torch.equal(first(states), second(states))


def test_undirected_edges_expand_for_messages_but_self_loop_is_not_doubled():
    _, bundle = _circuit()
    human = bundle.graphs[3]
    edges = (
        Edge("human:0", "human:1", "macro_functional", 0.5, "human", False),
        Edge("human:0", "human:0", "macro_functional", 0.5, "human", False),
    )
    changed = replace(human, edges=edges)
    changed_bundle = GraphBundle(tuple(bundle.graphs[:3]) + (changed,))
    circuit = GraphCircuit(changed_bundle, DynamicsConfig(hidden_dim=8))
    relation = circuit.encoders[3].relations["macro_functional"]
    with torch.no_grad():
        relation.transform.weight.copy_(torch.eye(8))
    states = torch.zeros(1, 5, 8)
    states[0, :2, 0] = torch.tensor([1.0, 3.0])
    assert torch.equal(relation(states)[0, :, 0], torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0]))
    assert len(changed.edges) == 2


def test_node_states_and_observed_edges_affect_policy_output():
    model, bundle = _circuit()
    sequence = _sequence()
    initial = model.initial_state(3)
    altered = list(initial)
    altered[3] = torch.randn_like(altered[3])
    assert not torch.allclose(model.step(sequence[:, 0], initial).logits, model.step(sequence[:, 0], tuple(altered)).logits)
    human = bundle.graphs[3]
    changed = replace(human, edges=tuple(
        replace(edge, weight=-edge.weight) if edge.relation == "macro_functional" else edge
        for edge in human.edges
    ))
    other_bundle = GraphBundle(tuple(bundle.graphs[:3]) + (changed,))
    other = GraphCircuit(other_bundle, model.config)
    other_parameters = dict(other.named_parameters())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            other_parameters[name].copy_(parameter)
    assert not torch.allclose(model(sequence).logits, other(sequence).logits)


@pytest.mark.parametrize("kwargs", [{"hidden_dim": 0}, {"input_dim": 0}, {"output_dim": 0}, {"top_k": -1}, {"top_k": 4}])
def test_invalid_dynamics_config_rejected(kwargs):
    with pytest.raises(ValueError):
        DynamicsConfig(**kwargs)


def test_input_and_state_contracts_fail_before_silent_broadcast():
    model, _ = _circuit()
    with pytest.raises(ValueError):
        model(torch.zeros(3, 0, 4, 4))
    with pytest.raises(ValueError):
        model.step(torch.zeros(3, 3, 4))
    with pytest.raises(ValueError):
        model.step(torch.full((3, 4, 4), float("nan")))
    bad_state = model.initial_state(2)
    with pytest.raises(ValueError):
        model.step(torch.zeros(3, 4, 4), bad_state)
