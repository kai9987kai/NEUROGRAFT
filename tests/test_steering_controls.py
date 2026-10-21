"""Steering assistance is optional, legal, reproducible and explicitly separated."""

import json
import math

import pytest
import torch

from neurograft.compass import CXPFLSteering
from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest
from neurograft.navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation, navigation_protocol


@pytest.mark.parametrize("name", ["steering_weight", "gain_scale"])
@pytest.mark.parametrize("value", [-1., math.nan, math.inf, True, "1"])
def test_steering_rejects_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        CXPFLSteering(**{name: value})


@pytest.mark.parametrize("values", [(math.nan, 1., 0.), (0., math.inf, 0.), (0., 1., True)])
def test_steering_rejects_nonfinite_or_nonnumeric_geometry(values):
    with pytest.raises(ValueError):
        CXPFLSteering().compute_steering_prior(*values)


@pytest.mark.parametrize("walls", [[0, 0, 0], [0, 0, 0, math.nan], [0, .5, 0, 0], [1, 1, 1, 1]])
def test_steering_rejects_malformed_or_all_blocked_walls(walls):
    with pytest.raises(ValueError):
        CXPFLSteering().steer(0., 1., 0., walls, [0.] * 4)


@pytest.mark.parametrize("logits", [[0.] * 3, [0., 0., math.inf, 0.], [False, 0., 0., 0.]])
def test_steering_requires_four_finite_policy_logits(logits):
    with pytest.raises(ValueError):
        CXPFLSteering().steer(0., 1., 0., [0] * 4, logits)


def test_zero_goal_still_masks_walls_and_zero_weight_is_legal_policy():
    steer = CXPFLSteering(steering_weight=0.)
    prior = steer.compute_steering_prior(0., 0., 0., [1, 0, 0, 0])
    assert prior == [-1e6, 0., 0., 0.]
    assert steer.steer(0., 0., 0., [1, 0, 0, 0], [100., 1., 2., 0.]) == 2


@pytest.mark.parametrize("value", [None, 0, 1, "true", []])
def test_steering_control_flag_is_strict_boolean(value):
    with pytest.raises(ValueError, match="include_steering_controls"):
        NavigationEvalConfig(include_steering_controls=value)


@pytest.mark.parametrize("controllers,readout,schema", [(False, False, 1), (True, False, 2), (False, True, 3), (True, True, 3)])
def test_disabled_steering_preserves_protocol_and_report(controllers, readout, schema):
    implicit = NavigationEvalConfig(seeds=(17,), size=5, max_steps=2,
        include_controllers=controllers, include_readout_controls=readout)
    explicit = NavigationEvalConfig(seeds=(17,), size=5, max_steps=2,
        include_controllers=controllers, include_readout_controls=readout, include_steering_controls=False)
    protocol = navigation_protocol(explicit, 2)
    assert protocol == navigation_protocol(implicit, 2)
    assert protocol["schema_version"] == schema
    assert "include_steering_controls" not in protocol["config"]
    assert "cx_steer" not in protocol["variants"]
    session = LiveSession(make_fixture_bundle(nodes_per_source=3), LiveConfig(hidden_dim=4, unroll=2))
    assert frozen_navigation_evaluation(session, explicit) == frozen_navigation_evaluation(session, implicit)


@pytest.mark.parametrize("controllers,readout", [(False, False), (True, True)])
def test_steering_controls_share_worlds_and_do_not_mutate_live_state(controllers, readout):
    session = LiveSession(make_fixture_bundle(nodes_per_source=3), LiveConfig(hidden_dim=4, unroll=2))
    before = _checkpoint_digest(session._payload())
    config = NavigationEvalConfig(seeds=(17, 23), size=5, max_steps=8,
        include_controllers=controllers, include_readout_controls=readout, include_steering_controls=True)
    report = frozen_navigation_evaluation(session, config)
    assert report["schema_version"] == 4
    assert report["config"]["include_steering_controls"] is True
    assert _checkpoint_digest(session._payload()) == before
    assert report == frozen_navigation_evaluation(session, config)
    worlds = [episode["initial_world_hash"] for episode in report["variants"]["policy"]["episodes"]]
    assert all([episode["initial_world_hash"] for episode in variant["episodes"]] == worlds
               for variant in report["variants"].values())
    for name in ("cx_steer", "cx_without_policy"):
        variant = report["variants"][name]
        assert variant["summary"]["collision_rate"] == 0
        for episode in variant["episodes"]:
            assert len(episode["controller_trace"]) == episode["steps"]
            assert episode["policy_prior_used"] is (name == "cx_steer")
            assert all(step["reason"] == ("cx_goal_heading_prior" if name == "cx_steer" else "cx_goal_heading_without_policy")
                       for step in episode["controller_trace"])
    assert "engineered assistance" in report["evidence_scope"]
    assert "cell-specific" in report["evidence_scope"]
    json.dumps(report, allow_nan=False)


def test_steering_without_policy_ignores_learned_logits():
    session = LiveSession(make_fixture_bundle(nodes_per_source=3), LiveConfig(hidden_dim=4, unroll=2))
    config = NavigationEvalConfig(seeds=(17,), size=7, max_steps=8, include_steering_controls=True)
    with torch.no_grad():
        session.model.policy.weight.zero_()
        session.model.policy.bias.copy_(torch.tensor([100., -100., -100., -100.]))
    north = frozen_navigation_evaluation(session, config)
    with torch.no_grad():
        session.model.policy.bias.copy_(torch.tensor([-100., 100., -100., -100.]))
    east = frozen_navigation_evaluation(session, config)
    assert north["variants"]["policy"]["episodes"][0]["actions"] != east["variants"]["policy"]["episodes"][0]["actions"]
    assert north["variants"]["cx_without_policy"]["episodes"][0]["actions"] == east["variants"]["cx_without_policy"]["episodes"][0]["actions"]


def test_steering_control_is_available_through_http_and_cli(tmp_path):
    from test_server import service, request
    from neurograft.cli import parser
    args = parser().parse_args(["evaluate-live", "--bundle", "b.json", "--checkpoint", "c.pt",
                               "--out", "o.json", "--include-steering-controls"])
    assert args.include_steering_controls is True
    with service(tmp_path) as (app, url):
        request(url + "/api/control", {"action": "pause"})
        status, state = request(url + "/api/control", {"action": "navigation", "mode": "cx_steer"})
        assert status == 200 and state["navigation"]["mode"] == "cx_steer"
        before = _checkpoint_digest(app.session._payload())
        status, report = request(url + "/api/evaluate", {"seeds": [17], "size": 5,
            "max_steps": 2, "include_steering_controls": True})
        assert status == 200 and report["schema_version"] == 4
        assert "cx_without_policy" in report["variants"]
        assert request(url + "/api/evaluate", {"include_steering_controls": 1})[0] == 400
        assert _checkpoint_digest(app.session._payload()) == before


def test_cli_writes_steering_comparison_without_modifying_checkpoint(tmp_path, capsys):
    from neurograft.cli import main
    session = LiveSession(make_fixture_bundle(nodes_per_source=3), LiveConfig(hidden_dim=4, unroll=2))
    bundle, checkpoint, out = tmp_path / "bundle.json", tmp_path / "source.pt", tmp_path / "steering.json"
    session.bundle.save(bundle)
    session.save(checkpoint)
    before = checkpoint.read_bytes()
    assert main(["evaluate-live", "--bundle", str(bundle), "--checkpoint", str(checkpoint),
        "--out", str(out), "--seeds", "17", "--size", "5", "--max-steps", "2",
        "--include-steering-controls"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == 4
    assert {"cx_steer", "cx_without_policy"} <= report["variants"].keys()
    assert report == json.loads(capsys.readouterr().out)
    assert checkpoint.read_bytes() == before
