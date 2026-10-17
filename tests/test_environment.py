"""Navigation behavior, reachability, and exact JSON save continuation."""

import copy
import importlib
import importlib.util
import json
import math
import random
from collections import deque

import pytest


def world_class():
    # The first RED run reports the missing behavior as an assertion failure.
    assert importlib.util.find_spec("neurograft.environment") is not None, "navigation world is missing"
    return importlib.import_module("neurograft.environment").NavigationWorld


def distance(snapshot, start, end):
    blocked = {(item["x"], item["z"]) for item in snapshot["obstacles"]}
    pending = deque([(start, 0)])
    visited = {start}
    while pending:
        (x, z), length = pending.popleft()
        if (x, z) == end:
            return length
        for dx, dz in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            cell = (x + dx, z + dz)
            if 0 <= cell[0] < snapshot["size"] and 0 <= cell[1] < snapshot["size"] and cell not in blocked and cell not in visited:
                visited.add(cell)
                pending.append((cell, length + 1))
    return None


def fixture_world():
    cls = world_class()
    state = cls(seed=19, size=7).to_dict()
    state["obstacles"] = [
        {"x": x, "z": z}
        for z in range(7)
        for x in range(7)
        if x in (0, 6) or z in (0, 6) or (x == 3 and z < 5)
    ]
    state["beacons"] = [{"x": 5, "z": 1, "color": "blue"}, {"x": 1, "z": 5, "color": "amber"}, {"x": 5, "z": 5, "color": "rose"}]
    state["agent"] = {"x": 1, "z": 1, "direction": 0}
    state["goal"] = {"x": 5, "z": 1, "color": "blue"}
    state["trail"] = [{"x": 1, "z": 1}]
    return cls.from_dict(state)


def test_seeded_world_and_observation_are_deterministic_without_global_random_mutation():
    cls = world_class()
    before = random.getstate()
    first, second = cls(seed=41), cls(seed=41)
    assert first.to_dict() == second.to_dict()
    assert random.getstate() == before
    assert cls(seed=42).snapshot()["obstacles"] != first.snapshot()["obstacles"]
    observed = first.observe()
    assert len(observed) == 4 and all(len(row) == 4 for row in observed)
    assert all(type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1 for row in observed for value in row)


@pytest.mark.parametrize("size", [5, 7, 11, 21, 31])
@pytest.mark.parametrize("seed", [0, 3, 99])
def test_every_beacon_and_free_target_is_accessible(size, seed):
    world = world_class()(seed=seed, size=size)
    snapshot = world.snapshot()
    start = (snapshot["agent"]["x"], snapshot["agent"]["z"])
    obstacles = {(item["x"], item["z"]) for item in snapshot["obstacles"]}
    assert {beacon["color"] for beacon in snapshot["beacons"]} == {"blue", "amber", "rose"}
    for z in range(1, size - 1):
        for x in range(1, size - 1):
            if (x, z) not in obstacles:
                assert distance(snapshot, start, (x, z)) is not None
    for beacon in snapshot["beacons"]:
        world.set_goal(color=beacon["color"])
        assert world.snapshot()["goal"] == beacon


def test_expert_follows_a_shortest_detour_and_finishes_episode():
    world = fixture_world()
    start = world.snapshot()
    assert distance(start, (1, 1), (5, 1)) == 12
    assert world.expert_action() == 1
    for remaining in range(12, 0, -1):
        snapshot = world.snapshot()
        position = (snapshot["agent"]["x"], snapshot["agent"]["z"])
        assert distance(snapshot, position, (5, 1)) == remaining
        result = world.step(world.expert_action())
        assert not result["collision"]
        assert result["goal_reached"] == (remaining == 1)
        assert result["episode_done"] == (remaining == 1)
    assert world.snapshot()["agent"]["x"] == 5
    assert world.snapshot()["agent"]["z"] == 1
    assert world.snapshot()["goals_reached"] == 1
    assert world.snapshot()["episode"] == 1
    assert world.snapshot()["goal"] != start["goal"]


def test_collision_has_penalty_and_direction_and_does_not_cross_walls():
    world = fixture_world()
    result = world.step(0)
    assert result == {"reward": -0.1, "goal_reached": False, "episode_done": False, "collision": True}
    assert world.snapshot()["agent"] == {"x": 1, "z": 1, "direction": 0}
    assert world.snapshot()["steps"] == 1
    assert world.snapshot()["last_reward"] == -0.1


@pytest.mark.parametrize("action", [-1, 4, 1.0, True, "1", None])
def test_invalid_actions_do_not_change_world_or_random_state(action):
    world = fixture_world()
    before = world.to_dict()
    with pytest.raises(ValueError):
        world.step(action)
    assert world.to_dict() == before


@pytest.mark.parametrize("kwargs", [{"color": "green"}, {"x": 0, "z": 2}, {"x": 3, "z": 2}, {"x": 2}, {"x": 2.0, "z": 2}, {"x": True, "z": 2}, {"color": "blue", "x": 2, "z": 2}])
def test_invalid_goals_do_not_change_world_or_random_state(kwargs):
    world = fixture_world()
    before = world.to_dict()
    with pytest.raises(ValueError):
        world.set_goal(**kwargs)
    assert world.to_dict() == before


def test_custom_goal_and_current_cell_goal_complete_safely():
    world = fixture_world()
    world.set_goal(x=2, z=1)
    assert world.snapshot()["goal"] == {"x": 2, "z": 1, "color": "custom"}
    assert world.step(world.expert_action())["goal_reached"]
    world.set_goal(x=2, z=1)
    assert world.step(world.expert_action())["goal_reached"]
    assert world.snapshot()["goals_reached"] == 2


def test_timeout_is_bounded_and_reset_retains_lifetime_counters():
    world = fixture_world()
    for count in range(world.MAX_EPISODE_STEPS):
        result = world.step(0)
        assert result["episode_done"] == (count == world.MAX_EPISODE_STEPS - 1)
    assert world.snapshot()["episode"] == 1
    assert world.snapshot()["steps"] == world.MAX_EPISODE_STEPS
    assert world.snapshot()["goals_reached"] == 0
    assert len(world.snapshot()["trail"]) <= 64
    previous = world.snapshot()
    world.reset()
    reset = world.snapshot()
    assert reset["episode"] == previous["episode"] + 1
    assert (reset["steps"], reset["goals_reached"], reset["total_reward"]) == (previous["steps"], previous["goals_reached"], previous["total_reward"])
    assert len(reset["trail"]) == 1


def test_saved_world_roundtrips_through_json_with_identical_future_rng_and_counters():
    cls = world_class()
    world = cls(seed=735)
    for _ in range(47):
        world.step(world.expert_action())
    restored = cls.from_dict(json.loads(json.dumps(world.to_dict(), allow_nan=False)))
    assert restored.to_dict() == world.to_dict()
    for index in range(300):
        if index % 41 == 0:
            world.reset()
            restored.reset()
        if index % 17 == 0:
            world.set_goal()
            restored.set_goal()
        action = world.expert_action()
        assert restored.expert_action() == action
        assert restored.step(action) == world.step(action)
        assert restored.to_dict() == world.to_dict()


def test_snapshot_and_save_are_detached_from_internal_state():
    world = fixture_world()
    before = world.to_dict()
    snapshot, saved = world.snapshot(), world.to_dict()
    snapshot["agent"]["x"] = 99
    saved["trail"][0]["x"] = 99
    saved["rng_state"][1][0] = 0
    assert world.to_dict() == before


@pytest.mark.parametrize("field,value", [("version", 999), ("size", 5000), ("seed", True), ("steps", -1), ("episode", -1), ("goals_reached", 1), ("last_reward", float("nan")), ("total_reward", float("inf")), ("episode_steps", 256), ("rng_state", [3, [1, 2, 3], None]), ("agent", {"x": 0, "z": 0, "direction": 0}), ("goal", {"x": 2, "z": 1, "color": "blue"}), ("trail", [{"x": 2, "z": 1}]), ("beacons", [])])
def test_malformed_saved_state_is_rejected_without_touching_source(field, value):
    world = fixture_world()
    state = world.to_dict()
    state[field] = value
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        world_class().from_dict(state)
    if field not in ("last_reward", "total_reward"):
        assert state == before
    assert world.snapshot()["steps"] == 0


def test_disconnected_saved_layout_is_rejected():
    world = fixture_world()
    saved = world.to_dict()
    saved["obstacles"].append({"x": 3, "z": 5})
    with pytest.raises(ValueError, match="reach|connect"):
        world_class().from_dict(saved)


@pytest.mark.parametrize("changes", [
    {"last_reward": 1.0},
    {"last_reward": -0.1},
    {"steps": 2, "episode_steps": 2, "last_reward": -0.01, "total_reward": 1.9},
    {"steps": 2, "episode": 1, "goals_reached": 1, "episode_steps": 1, "last_reward": 1.0, "total_reward": 0.99},
    {"steps": 2, "episode_steps": 1, "last_reward": 0.0, "total_reward": -0.02},
])
def test_saved_reward_and_episode_counters_must_be_consistent(changes):
    saved = fixture_world().to_dict()
    saved.update(changes)
    with pytest.raises(ValueError):
        world_class().from_dict(saved)


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": "0"}, {"size": True}, {"size": 4}, {"size": 32}, {"size": 7.5}, {"preset": "unknown"}])
def test_constructor_rejects_unsafe_or_nonintegral_configuration(kwargs):
    with pytest.raises(ValueError):
        world_class()(**kwargs)


@pytest.mark.parametrize("preset", ["labyrinth", "chambers", "corridors", "pillars", "concentric", "hierarchical"])
def test_architectural_presets_and_environment_telemetry(preset):
    cls = world_class()
    world = cls(seed=17, size=17, preset=preset)
    snap = world.snapshot()
    env = world.environment_telemetry()
    assert snap["size"] == 17
    assert env["preset"] == preset
    assert len(env["lidar_8"]) == 8 and all(0.0 <= v <= 1.0 for v in env["lidar_8"])
    assert len(env["lidar_rays"]) == 8
    assert env["detour_index"] >= 1.0
    assert 0.0 <= env["chemoattractant"] <= 1.0
    assert isinstance(env["choke_points"], list)
    assert len(env["plume_gradient"]) == 2
    assert env["plume_width"] > 0.0
    assert 0.0 <= env["plume_intensity"] <= 1.0
    assert isinstance(env["gate_open"], bool)
    assert env["topological_entropy"] >= 0.0
    start = (snap["agent"]["x"], snap["agent"]["z"])
    for beacon in snap["beacons"]:
        assert distance(snap, start, (beacon["x"], beacon["z"])) is not None
