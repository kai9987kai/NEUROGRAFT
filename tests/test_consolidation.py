import pytest
import torch

from neurograft.consolidation import (
    ConsolidationConfig,
    SwiGLULatentExpert,
    LatentAssociativeMemory,
    NativeConsolidationBlock,
    ConnectomeConsolidationStack,
    TeacherConsolidationFusion,
)


def test_consolidation_config_validation():
    cfg = ConsolidationConfig(hidden_size=8, shared_dim=32, latent_experts=4)
    assert cfg.hidden_size == 8
    assert cfg.latent_top_k == 2

    with pytest.raises(ValueError):
        ConsolidationConfig(hidden_size=-1)
    with pytest.raises(ValueError):
        ConsolidationConfig(latent_top_k=10, latent_experts=4)


def test_swiglu_expert():
    expert = SwiGLULatentExpert(dim=32, rank=16)
    x = torch.randn(2, 32)
    out = expert(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_latent_associative_memory():
    mem = LatentAssociativeMemory(slots=8, dim=32)
    z = torch.randn(2, 32)
    gated, attn = mem(z)
    assert gated.shape == (2, 32)
    assert attn.shape == (2, 8)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_native_consolidation_block_identity_at_birth():
    cfg = ConsolidationConfig(hidden_size=8, shared_dim=32, latent_experts=4, memory_slots=4)
    block = NativeConsolidationBlock(cfg)
    block.eval()
    x = torch.randn(3, 8)
    with torch.no_grad():
        out, aux = block(x)
    # Since out_gate is initialized to zeros and requires_grad is false, output is exact identity
    assert torch.equal(out, x)
    assert "route_probs" in aux
    assert "router_balance" in aux
    assert "memory_attn" in aux
    assert aux["gate_abs_mean"] == 0.0


def test_native_consolidation_stack():
    cfg = ConsolidationConfig(hidden_size=8, shared_dim=32)
    stack = ConnectomeConsolidationStack(cfg)
    pooled = torch.randn(2, 4, 8)
    out, aux = stack(pooled)
    assert out.shape == (2, 8)
    assert aux["router_balance"] >= 1.0


def test_teacher_consolidation_fusion():
    fusion = TeacherConsolidationFusion(dim=8, shared_dim=32)
    sources = [torch.randn(2, 8) for _ in range(4)]
    student_latent = torch.randn(2, 32)
    res = fusion(sources, student_latent)
    assert res["fused_consensus"].shape == (2, 32)
    assert len(res["source_weights"]) == 2
    assert torch.isfinite(res["alignment_loss"])
