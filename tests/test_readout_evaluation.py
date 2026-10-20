"""Optional readout controls preserve archived protocols and isolate inference."""

import json
import random

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest
from neurograft.navigation_evaluation import (
    NavigationEvalConfig,
    frozen_navigation_evaluation,
    navigation_protocol,
)


def _session(readout="context_residual"):
    session = LiveSession(make_fixture_bundle(nodes_per_source=3),
        LiveConfig(hidden_dim=4, batch_size=2, unroll=2, readout=readout))
    with torch.no_grad():
        session.model.policy.weight.zero_()
        session.model.policy.bias.copy_(torch.tensor([9., -9., -9., -9.]))
        if session.model.context_policy is not None:
            session.model.context_policy.weight.zero_()
            session.model.context_policy.bias.copy_(torch.tensor([0., 30., 0., 0.]))
    return session


@pytest.mark.parametrize("controllers", [False, True])
def test_omitting_readout_controls_preserves_legacy_protocol_and_report(controllers):
    config = NavigationEvalConfig(seeds=(17,), size=5, max_steps=2,
        include_controllers=controllers, include_readout_controls=False)
    expected = {
        "schema_version": 2 if controllers else 1,
        "config": {"seeds": [17], "size": 5, "preset": "labyrinth", "max_steps": 2},
        "unroll": 2,
        "variants": ["policy", "no_graft", "no_recurrence", "random", "shortest_path"],
        "success_rule": "first arrival at initial goal",
        "path_length": "successful cell moves",
    }
    if controllers:
        expected["config"]["include_controllers"] = True
        expected["variants"] += ["masked_policy", "observed_map", "map_without_policy"]
        expected["controller"] = "observed-frontier-v1; local walls; policy prior only breaks ties"
    assert navigation_protocol(config, 2) == expected
    session = _session("human")
    report = frozen_navigation_evaluation(session, config)
    implicit = frozen_navigation_evaluation(session, NavigationEvalConfig(
        seeds=(17,), size=5, max_steps=2, include_controllers=controllers))
    assert report == implicit
    assert "include_readout_controls" not in report["config"]
    assert "no_context" not in report["variants"]
    assert "no_context" not in report["paired_effects"]
    assert "context residual" not in report["evidence_scope"]


@pytest.mark.parametrize("controllers", [False, True])
def test_readout_control_bypasses_only_residual_and_leaves_live_state_untouched(controllers):
    session = _session()
    before = _checkpoint_digest(session._payload())
    rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    config = NavigationEvalConfig(seeds=(17, 23), size=5, max_steps=2,
        include_controllers=controllers, include_readout_controls=True)
    report = frozen_navigation_evaluation(session, config)
    assert report == frozen_navigation_evaluation(session, config)
    assert _checkpoint_digest(session._payload()) == before
    assert random.getstate() == rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert report["schema_version"] == 3
    assert report["config"]["include_readout_controls"] is True
    assert report["teacher_assistance"] == 0 and report["learning"] is False
    variants = report["variants"]
    for name in ("policy", "no_graft", "no_recurrence"):
        assert all(action == 1 for episode in variants[name]["episodes"]
                   for action in episode["actions"])
    assert all(action == 0 for episode in variants["no_context"]["episodes"]
               for action in episode["actions"])
    initial = [episode["initial_world_hash"] for episode in variants["policy"]["episodes"]]
    assert all([episode["initial_world_hash"] for episode in variant["episodes"]] == initial
               for variant in variants.values())
    assert set(report["paired_effects"]["no_context"]) == {"success_rate_delta", "spl_delta"}
    assert "no_graft retains" in report["evidence_scope"]
    assert "all-source" in report["evidence_scope"]
    assert "no_context bypasses" in report["evidence_scope"]
    assert "inference-time" in report["evidence_scope"]
    json.dumps(report, allow_nan=False)


def test_readout_control_is_noop_for_legacy_human_readout():
    report = frozen_navigation_evaluation(_session("human"), NavigationEvalConfig(
        seeds=(17,), size=5, max_steps=2, include_readout_controls=True))
    assert report["variants"]["policy"] == report["variants"]["no_context"]


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_readout_control_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="include_readout_controls"):
        NavigationEvalConfig(include_readout_controls=value)


def test_evaluate_live_cli_enables_readout_controls_without_changing_checkpoint(tmp_path):
    from neurograft.cli import main
    session = _session()
    bundle = tmp_path / "bundle.json"
    checkpoint = tmp_path / "session.pt"
    out = tmp_path / "readout-evaluation.json"
    session.bundle.save(bundle)
    session.save(checkpoint)
    before = checkpoint.read_bytes()
    assert main(["evaluate-live", "--bundle", str(bundle), "--checkpoint", str(checkpoint),
        "--out", str(out), "--seeds", "17", "--size", "5", "--max-steps", "2",
        "--include-readout-controls"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == 3
    assert "no_context" in report["variants"]
    assert checkpoint.read_bytes() == before
