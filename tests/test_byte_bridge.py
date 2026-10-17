"""Tests for Connectome Byte-Level Semantic Guidance Bridge."""

import pytest
import torch

from neurograft.byte_bridge import ByteBridgeConfig, ConnectomeByteBridge


def test_byte_bridge_config_validation():
    with pytest.raises(ValueError, match="context_dim"):
        ByteBridgeConfig(context_dim=0)
    with pytest.raises(ValueError, match="byte_vocab_size"):
        ByteBridgeConfig(byte_vocab_size=0)
    with pytest.raises(ValueError, match="temperature"):
        ByteBridgeConfig(temperature=-0.5)


def test_byte_bridge_forward_and_decode():
    bridge = ConnectomeByteBridge(ByteBridgeConfig(context_dim=16, byte_vocab_size=256))
    batch_size = 2
    context = torch.randn(batch_size, 16)

    logits, probs = bridge.forward(context)
    assert logits.shape == (batch_size, 256)
    assert probs.shape == (batch_size, 256)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)

    summary = bridge.decode_semantic_intent(context)
    assert "semantic_intent" in summary
    assert isinstance(summary["semantic_intent"], str)
    assert summary["confidence"] >= 0.0
    assert summary["byte_entropy"] > 0.0
    assert len(summary["top_byte_chars"]) == 6
    assert 0.0 <= summary["vocab_spread"] <= 1.0


def test_byte_bridge_dynamic_dim_adaptation():
    bridge = ConnectomeByteBridge(ByteBridgeConfig(context_dim=32, byte_vocab_size=256))

    # Passing smaller dimension (16 instead of 32)
    smaller = torch.randn(2, 16)
    logits, probs = bridge.forward(smaller)
    assert logits.shape == (2, 256)

    # Passing larger dimension (48 instead of 32)
    larger = torch.randn(2, 48)
    logits2, probs2 = bridge.forward(larger)
    assert logits2.shape == (2, 256)
