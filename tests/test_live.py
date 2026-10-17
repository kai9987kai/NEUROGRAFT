"""Small CPU checks for real online updates and complete live continuation."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json

import pytest
import torch

from neurograft.fixtures import make_fixture_bundle


def _api():
    assert importlib.util.find_spec("neurograft.live") is not None, "live session implementation is missing"
    from neurograft.live import LiveConfig, LiveSession
    return LiveConfig, LiveSession


def _session(**kwargs):
    config_type, session_type = _api()
    defaults = dict(seed=19, hidden_dim=4, threads=1, batch_size=2, train_every=2, unroll=3, replay_capacity=8)
    defaults.update(kwargs)
    return session_type(make_fixture_bundle(seed=5, nodes_per_source=3), config_type(**defaults))


def _equal(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor) and torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _equal(a, b)
    else:
        assert first == second


def test_live_tick_performs_real_finite_updates_with_detached_recurrence():
    session = _session()
    digest = session.bundle.digest()
    before = session.model.policy.weight.detach().clone()
    session.tick(8)
    assert session.steps == 8 and session.updates == 4
    assert not torch.equal(before, session.model.policy.weight)
    assert session.last_loss > 0
    assert all(torch.isfinite(p).all() for p in session.model.parameters())
    assert all(not state.requires_grad and state.grad_fn is None for state in session.recurrent_state)
    assert session.bundle.digest() == digest
    assert len(session.replay) <= 8


def test_paused_explicit_tick_runs_but_learning_off_freezes_optimizer_and_weights():
    session = _session()
    session.tick(4)
    session.running = False
    session.learning = False
    weights = copy.deepcopy(session.model.state_dict())
    optimizer = copy.deepcopy(session.optimizer.state_dict())
    session.tick(5)
    assert session.steps == 9 and session.updates == 2
    _equal(weights, session.model.state_dict())
    _equal(optimizer, session.optimizer.state_dict())


def test_snapshot_is_finite_json_and_reports_world_counters_and_honest_scope():
    session = _session()
    session.tick(3)
    snap = session.snapshot()
    json.dumps(snap, allow_nan=False)
    expected = {"version", "running", "learning", "steps", "updates", "episodes", "goals_reached",
                "last_loss", "loss_history", "reward_history", "policy_accuracy", "last_action",
                "last_action_source", "teacher_probability", "world", "sources", "routes", "activity",
                "facts", "chat", "backend", "graph_hash", "model_hash", "last_saved", "evidence_scope"}
    assert set(snap) == expected
    assert snap["backend"] == "local-memory-and-control"
    assert "supervised" in snap["evidence_scope"].lower()
    assert snap["episodes"] == snap["world"]["episode"]
    assert snap["goals_reached"] == snap["world"]["goals_reached"]
    assert len(snap["routes"]) == 4 and all(len(row) == 5 for row in snap["routes"])
    assert len(snap["activity"]) == 4 and 0 <= snap["policy_accuracy"] <= 1
    snap["facts"]["outside"] = "mutation"
    assert "outside" not in session.snapshot()["facts"]


def test_private_randomness_does_not_depend_on_global_torch_or_python_rng():
    import random
    first = _session()
    second = _session()
    first.tick(6)
    random.seed(876)
    torch.manual_seed(876)
    torch.rand(31)
    second.tick(6)
    _equal(first.model.state_dict(), second.model.state_dict())
    _equal(first.world.to_dict(), second.world.to_dict())
    _equal(first.last_logits, second.last_logits)


@pytest.mark.parametrize("kwargs", [{"hidden_dim": True}, {"hidden_dim": 0}, {"hidden_dim": 1024},
    {"learning_rate": float("nan")}, {"learning_rate": 0}, {"batch_size": 0}, {"train_every": 0},
    {"unroll": 0}, {"replay_capacity": 1, "batch_size": 2}, {"teacher_probability": 1.1}, {"threads": 0}])
def test_unsafe_config_values_fail_before_allocating_a_circuit(kwargs):
    config_type, _ = _api()
    with pytest.raises(ValueError):
        config_type(**kwargs)


@pytest.mark.parametrize("manual", [True, -1, 4, "1", 1.5])
def test_bad_manual_actions_do_not_advance_or_train(manual):
    session = _session()
    before = session.snapshot()
    with pytest.raises(ValueError):
        session.tick(2, manual_action=manual)
    _equal(before, session.snapshot())


def test_manual_moves_are_not_counted_as_policy_accuracy_and_goal_resets_memory():
    session = _session()
    session.tick(3)
    accuracy = session.snapshot()["policy_accuracy"]
    session.running = False
    session.tick(2, manual_action=0)
    assert session.snapshot()["policy_accuracy"] == accuracy
    assert session.snapshot()["last_action_source"] == "manual"
    session.set_goal(color="blue")
    assert all(torch.count_nonzero(tensor) == 0 for tensor in session.recurrent_state)
    assert not session.observation_history
    before = session.snapshot()
    with pytest.raises(ValueError):
        session.set_goal(x=-1, z=0)
    _equal(before, session.snapshot())


def test_chat_controls_facts_goals_and_unknown_backend_are_honest():
    session = _session()
    assert session.chat("pause")["backend"] == "local-memory-and-control" and session.running is False
    session.chat("remember favourite = blue")
    assert "blue" in session.chat("what is favourite")["reply"]
    session.chat("remember owner is Kai")
    assert "Kai" in session.chat("recall owner")["reply"]
    session.chat("go to amber")
    assert session.world.snapshot()["goal"]["color"] == "amber"
    session.chat("resume")
    assert session.running is True
    assert session.chat("save")["action"] == "save"
    assert "local" in session.chat("write me a novel")["reply"].lower()
    assert session.snapshot()["facts"] == {"favourite": "blue", "owner": "Kai"}
    with pytest.raises(ValueError):
        session.chat("x" * 5000)


def test_save_restores_exact_future_updates_world_replay_and_chat(tmp_path):
    session = _session()
    session.chat("remember experiment = exact continuation")
    session.tick(7)
    session.running = False
    path = tmp_path / "session.pt"
    receipt = session.save(path)
    _, session_type = _api()
    restored = session_type.load(path, session.bundle)
    assert receipt["bytes"] == path.stat().st_size and len(receipt["sha256"]) == 64
    _equal(session.snapshot(), restored.snapshot())
    _equal(session.optimizer.state_dict(), restored.optimizer.state_dict())
    _equal(session.recurrent_state, restored.recurrent_state)
    for count in (1, 3, 6):
        session.tick(count)
        restored.tick(count)
        _equal(session.snapshot(), restored.snapshot())
        _equal(session.model.state_dict(), restored.model.state_dict())
        _equal(session.optimizer.state_dict(), restored.optimizer.state_dict())
        _equal(session.world.to_dict(), restored.world.to_dict())
        _equal(session.recurrent_state, restored.recurrent_state)
        _equal(session.replay, restored.replay)
        _equal(session.last_logits, restored.last_logits)
    assert not list(tmp_path.glob("*.tmp"))


def test_save_load_rejects_corruption_mismatched_graph_and_oversize(tmp_path):
    session = _session()
    path = tmp_path / "session.pt"
    session.save(path)
    _, session_type = _api()
    with pytest.raises(ValueError, match="graph"):
        session_type.load(path, make_fixture_bundle(seed=6, nodes_per_source=3))
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x33
    path.write_bytes(data)
    with pytest.raises(ValueError):
        session_type.load(path, session.bundle)
    from neurograft.live import MAX_CHECKPOINT_BYTES
    huge = tmp_path / "oversize.pt"
    with huge.open("wb") as handle:
        handle.truncate(MAX_CHECKPOINT_BYTES + 1)
    with pytest.raises(ValueError, match="size|large"):
        session_type.load(huge, session.bundle)


@pytest.mark.parametrize("mutate", ["topology", "optimizer", "replay", "rng", "facts", "counter"])
def test_valid_checksum_cannot_bypass_deep_candidate_validation(tmp_path, mutate):
    session = _session()
    session.tick(4)
    path = tmp_path / "session.pt"
    session.save(path)
    from neurograft.live import _checkpoint_digest
    _, session_type = _api()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    payload = checkpoint["payload"]
    if mutate == "topology":
        payload["model"]["encoders.0.node_features"][0, 0] += 1
    elif mutate == "optimizer":
        next(iter(payload["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    elif mutate == "replay":
        payload["replay"][0]["action"] = 4
    elif mutate == "rng":
        payload["rng"] = [3, [1], None]
    elif mutate == "facts":
        payload["facts"] = {"bad": "x" * 5000}
    else:
        payload["steps"] = -1
    checkpoint["sha256"] = _checkpoint_digest(payload)
    torch.save(checkpoint, path)
    before = session.snapshot()
    with pytest.raises(ValueError):
        session_type.load(path, session.bundle)
    _equal(before, session.snapshot())


def test_save_and_ticks_share_lock_and_produce_loadable_consistent_checkpoints(tmp_path):
    session = _session()
    _, session_type = _api()
    def save_one(index):
        path = tmp_path / f"slot-{index}.pt"
        session.save(path)
        restored = session_type.load(path, session.bundle)
        assert restored.steps == restored.world.steps
        restored.tick()
    with ThreadPoolExecutor(max_workers=2) as pool:
        moving = pool.submit(session.tick, 8)
        saving = pool.submit(lambda: [save_one(index) for index in range(3)])
        moving.result(timeout=30)
        saving.result(timeout=30)
    assert session.steps == 8


def test_telemetry_shortest_path_and_introspective_chat():
    session = _session()
    path = session.world.shortest_path()
    assert isinstance(path, list)
    obstacles = {(w["x"], w["z"]) for w in session.world.snapshot()["obstacles"]}
    for cell in path:
        assert 0 <= cell["x"] < session.world.size and 0 <= cell["z"] < session.world.size
        assert (cell["x"], cell["z"]) not in obstacles
    session.tick(4)
    telemetry = session.telemetry()
    assert set(telemetry) == {
        "action_probs", "policy_confidence", "expert_action", "expert_path",
        "graft_gates", "route_entropy", "weight_delta_norm", "graft_impact", "top_nodes",
    }
    assert len(telemetry["action_probs"]) == 4
    assert abs(sum(telemetry["action_probs"]) - 1.0) < 1e-4
    assert telemetry["weight_delta_norm"] > 0.0
    assert telemetry["graft_impact"] >= 0.0
    assert len(telemetry["top_nodes"]) == len(session.bundle.graphs)
    assert all(len(group) >= 1 for group in telemetry["top_nodes"].values())
    before_steps = session.steps
    session.chat("evolve 6")
    assert session.steps == before_steps + 6
    assert "Top firing" in session.chat("top neurons")["reply"]
    assert "Synthetic routing telemetry" in session.chat("routes")["reply"]
    assert "Decision probe" in session.chat("why")["reply"]

