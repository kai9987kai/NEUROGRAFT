import pytest
import torch


def tiny_backbone(kind='qwen3'):
    if kind == 'qwen3':
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8)
        return Qwen3ForCausalLM(config)
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=8,
                              layer_types=['linear_attention', 'full_attention'],
                              linear_conv_kernel_dim=2, linear_key_head_dim=4,
                              linear_value_head_dim=4, linear_num_key_heads=1,
                              linear_num_value_heads=2,
                              max_position_embeddings=128,
                              rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                               'mrope_section': [1, 1, 2]})
    return Qwen3_5ForCausalLM(config)


@pytest.mark.parametrize('kind', ['qwen3', 'qwen3_5'])
def test_real_qwen_interface_identity_and_graft_gradient(kind):
    from neurograft.llm import FrozenLanguageGraft
    torch.manual_seed(7)
    torch.set_num_threads(2)
    base = tiny_backbone(kind).eval()
    graft = FrozenLanguageGraft(base, context_dim=12)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(tokens)
    context = torch.randn(2, 12, requires_grad=True)
    expected = base(input_ids=tokens, attention_mask=mask, use_cache=False).logits
    bypass = graft(tokens, context, attention_mask=mask, enabled=False, use_cache=False).logits
    zero = graft(tokens, context, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(bypass, expected, rtol=0, atol=0)
    torch.testing.assert_close(zero, expected, rtol=0, atol=0)
    with torch.no_grad():
        graft.gate.fill_(0.2)
    out = graft(tokens, context, attention_mask=mask, use_cache=False)
    loss = out.logits.square().mean()
    loss.backward()
    assert context.grad is not None and context.grad.abs().sum() > 0
    assert graft.projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in base.parameters())
    graft.train()
    assert base.training is False


def test_generation_matches_cache_free_full_prefix_reference():
    from neurograft.llm import FrozenLanguageGraft
    torch.manual_seed(5)
    graft = FrozenLanguageGraft(tiny_backbone(), context_dim=8)
    with torch.no_grad():
        graft.gate.fill_(0.3)
    ids = torch.tensor([[1, 2]])
    context = torch.randn(1, 8)
    expected = ids.clone()
    for _ in range(3):
        out = graft(expected, context, use_cache=False)
        expected = torch.cat([expected, out.logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    actual = graft.generate_greedy(ids, context, max_new_tokens=3)
    assert torch.equal(actual, expected)


def test_model_loader_requires_immutable_revision_before_loading():
    from neurograft.llm import load_pretrained
    with pytest.raises(ValueError, match='revision'):
        load_pretrained('Qwen/Qwen3.5-0.8B', 'main', context_dim=48)


def test_explicit_injection_mask_cannot_override_padding():
    from neurograft.llm import FrozenLanguageGraft
    torch.manual_seed(4)
    graft = FrozenLanguageGraft(tiny_backbone(), 8)
    with torch.no_grad():
        graft.gate.fill_(.3)
    ids = torch.tensor([[0, 1, 2]])
    expected = graft.backbone.get_input_embeddings()(ids).detach()
    captured = {}
    def capture(module, args, kwargs):
        captured['embeddings'] = kwargs['inputs_embeds'].detach()
    handle = graft.backbone.register_forward_pre_hook(capture, with_kwargs=True)
    graft(ids, torch.randn(1, 8), attention_mask=torch.tensor([[0, 1, 1]]),
          injection_mask=torch.ones_like(ids), use_cache=False)
    handle.remove()
    torch.testing.assert_close(captured['embeddings'][:, 0], expected[:, 0], rtol=0, atol=0)
    assert not torch.equal(captured['embeddings'][:, 1:], expected[:, 1:])


def test_actual_graph_context_receives_language_gradient_and_base_stays_frozen():
    from neurograft.llm import FrozenLanguageGraft, NeurograftLanguageModel
    from neurograft.dynamics import DynamicsConfig, GraphCircuit
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.evaluation import generate_task
    torch.manual_seed(4)
    circuit = GraphCircuit(make_fixture_bundle(nodes_per_source=4), DynamicsConfig(hidden_dim=4))
    language = FrozenLanguageGraft(tiny_backbone(), circuit.context_dim)
    model = NeurograftLanguageModel(circuit, language)
    with torch.no_grad():
        language.gate.fill_(.2)
    base_before = {n: p.detach().clone() for n, p in language.backbone.named_parameters()}
    sequence, _ = generate_task(4, 2, 7)
    tokens = torch.tensor([[1,2,3]] * 4)
    loss = model(tokens, sequence, labels=tokens).loss
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in circuit.named_parameters() if 'relations' in n)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
    optimizer.step()
    assert all(torch.equal(p, base_before[n]) for n, p in language.backbone.named_parameters())
