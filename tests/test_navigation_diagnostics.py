"""Small deterministic checks for navigation root-cause measurements."""
import copy
import importlib
import random

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest


def _session():
    return LiveSession(make_fixture_bundle(seed=5, nodes_per_source=3), LiveConfig(
        seed=19, hidden_dim=4, threads=1, batch_size=2, train_every=2,
        unroll=3, replay_capacity=8))


def _api():
    return importlib.import_module("neurograft.navigation_diagnostics").navigation_diagnostics


def test_diagnostics_report_known_replay_and_routing_values_without_mutation():
    live = _session()
    live.tick(6)
    before = _checkpoint_digest(live._payload())
    replay = copy.deepcopy(list(live.replay))
    function = _api()
    report = function(live)
    assert report["replay"]["action_counts"] == {
        name: sum(item["action"] == index for item in replay)
        for index, name in enumerate(("north", "east", "south", "west"))}
    progress = [item["observations"][-1][3][2] for item in replay]
    assert report["replay"]["goal_progress_feature_range"] == {"min": min(progress), "max": max(progress)}
    with torch.no_grad():
        reference = live.model(torch.tensor([item["observations"] for item in replay], dtype=torch.float32))
    expected = reference.routes[:, 3, :].mean(0).tolist()
    assert list(report["human_routing"]["mean_mass"].values()) == pytest.approx(expected)
    assert report["human_routing"]["goal_source_mean_mass"] == expected[1]
    assert report["saved_world"]["size"] == live.world.size
    assert report["saved_world"]["preset"] == live.world.preset
    assert report["representative_windows"]["count"] == len(replay)
    assert _checkpoint_digest(live._payload()) == before
    assert live.archaeology._last_report is None


def test_empty_replay_is_explicit_and_finite_json():
    import json
    live = _session()
    report = _api()(live)
    assert report["replay"]["size"] == 0
    assert report["replay"]["goal_progress_feature_range"] is None
    assert report["human_routing"] is None
    assert report["policy"] is None
    assert report["representative_windows"]["indices"] == []
    json.dumps(report, allow_nan=False)


def test_representative_sampling_is_bounded_repeatable_and_uses_no_global_rng():
    live = _session()
    live.tick(4)
    sample = copy.deepcopy(live.replay[-1])
    live.replay = __import__("collections").deque([copy.deepcopy(sample) for _ in range(512)], maxlen=512)
    python_before = random.getstate()
    torch_before = torch.random.get_rng_state().clone()
    function = _api()
    first, second = function(live), function(live)
    assert first == second
    assert first["representative_windows"]["count"] == 32
    indices = first["representative_windows"]["indices"]
    assert indices[0] == 0 and indices[-1] == 511 and len(set(indices)) == 32
    assert random.getstate() == python_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_corrupt_replay_is_rejected_without_mutation():
    live = _session()
    live.tick(2)
    live.replay[-1]["observations"][-1][0][0] = float("nan")
    with pytest.raises(ValueError, match="observation"):
        _api()(live)


@pytest.mark.parametrize('mutation', ['model', 'replay'])
def test_diagnostics_use_one_frozen_capture_when_live_state_changes_after_unlock(mutation):
    live = _session()
    live.tick(4)
    live.model.train()
    live.model.encoders[0].eval()
    modes_before = [module.training for module in live.model.modules()]
    expected = _api()(live)
    python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
    original_lock = live.lock

    class ChangeAfterCapture:
        held = False

        def __enter__(self):
            original_lock.acquire()
            self.held = True

        def __exit__(self, *exc):
            self.held = False
            original_lock.release()
            if mutation == 'model':
                with torch.no_grad():
                    live.model.policy.bias.add_(100)
            else:
                live.replay[-1]['observations'][-1][3][2] = 0.987

    lock = ChangeAfterCapture()
    live.lock = lock

    def require_probe_outside_lock(module, args):
        assert not lock.held, 'diagnostic inference must release the live-session lock'

    hook = live.model.register_forward_pre_hook(require_probe_outside_lock)
    try:
        actual = _api()(live)
    finally:
        hook.remove()
        live.lock = original_lock
    assert actual == expected
    assert [module.training for module in live.model.modules()] == modes_before
    assert random.getstate() == python_before
    assert torch.equal(torch.get_rng_state(), torch_before)
