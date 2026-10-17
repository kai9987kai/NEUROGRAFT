"""Unit tests for Tri-Tier Cognitive Memory and Matched Lesion Assays."""
import pytest
from neurograft.memory import TriTierMemory, LESION_MODES


def test_tri_tier_memory_recording_and_spatial_map():
    mem = TriTierMemory(capacity=64, lesion_mode="intact")
    assert mem.lesion_mode == "intact"
    # Record several steps
    for step in range(10):
        pos = (step % 4, step // 4)
        mem.record_step(
            step=step,
            agent_pos=pos,
            action=step % 4,
            reward=0.1 * step,
            state_features=[float(x) for x in range(8)],
            surprise=0.2,
        )
    assert len(mem.episodes) == 10
    assert len(mem.spatial_map) > 0
    ctx = mem.retrieve_context([0.0] * 8)
    assert ctx["mode"] == "intact"
    assert ctx["mapped_cells"] == len(mem.spatial_map)
    assert ctx["retrieved_item"] is not None


@pytest.mark.parametrize("mode", LESION_MODES)
def test_tri_tier_memory_lesion_modes(mode):
    mem = TriTierMemory(capacity=32, lesion_mode=mode)
    mem.record_step(step=1, agent_pos=(2, 2), action=1, reward=1.0, state_features=[0.5] * 8, surprise=0.1)
    ctx = mem.retrieve_context([0.5] * 8)
    assert ctx["mode"] == mode
    if mode in ("memory_free", "erased"):
        assert ctx["retrieved_item"] is None
        assert ctx["working_context"] == [0.0] * 8
    else:
        assert ctx["retrieved_item"] is not None


def test_tri_tier_memory_serialization_roundtrip():
    mem = TriTierMemory(capacity=32, lesion_mode="shifted")
    mem.record_step(step=1, agent_pos=(3, 4), action=2, reward=0.5, state_features=[0.1] * 8, surprise=0.3)
    data = mem.to_dict()
    assert data["lesion_mode"] == "shifted"
    restored = TriTierMemory.from_dict(data)
    assert restored.lesion_mode == "shifted"
    assert restored.capacity == 32
    assert len(restored.episodes) == len(mem.episodes)
    assert restored.total_recorded_steps == mem.total_recorded_steps
