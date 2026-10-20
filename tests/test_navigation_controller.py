"""Observation-only navigation separates policy proposals from controller aid."""
import copy
import json
import random

import pytest


ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def observation(size=7, agent=(3, 3), goal=(5, 1), walls=(0, 0, 0, 0)):
    scale = size - 1
    x, z = agent
    gx, gz = goal
    return [[x / scale, z / scale, 0.0, -1.0],
            [gx / scale, gz / scale, (gx - x) / scale, (gz - z) / scale],
            list(walls), [0.0, 0.0, 0.0, 1.0]]


class TinyGrid:
    def __init__(self, size, free, start, goal):
        self.size, self.free, self.agent, self.goal = size, set(free), start, goal
        self.trail = [start]

    def observe(self):
        x, z = self.agent
        walls = [(x + dx, z + dz) not in self.free for dx, dz in ACTIONS]
        return observation(self.size, self.agent, self.goal, tuple(int(v) for v in walls))

    def move(self, action):
        dx, dz = ACTIONS[action]
        target = (self.agent[0] + dx, self.agent[1] + dz)
        assert target in self.free, 'controller attempted a collision'
        self.agent = target
        self.trail.append(target)


def test_legal_mask_selects_best_available_action_and_preserves_inputs():
    from neurograft.navigation_controller import MODES, legal_action
    obs = observation(walls=(1, 0, 0, 1))
    logits = [100.0, 2.0, 3.0, 20.0]
    before = copy.deepcopy((obs, logits))
    assert MODES == ('policy', 'masked_policy', 'observed_map')
    assert legal_action(obs, logits) == 2
    assert (obs, logits) == before
    assert legal_action(obs, [0, 0, 0, 0]) == 1


@pytest.mark.parametrize('change', [
    lambda obs: obs.pop(),
    lambda obs: obs[0].pop(),
    lambda obs: obs[0].__setitem__(0, float('nan')),
    lambda obs: obs[1].__setitem__(2, float('inf')),
    lambda obs: obs[2].__setitem__(1, 0.5),
    lambda obs: obs[2].__setitem__(1, True),
    lambda obs: obs.__setitem__(2, [1, 1, 1, 1]),
])
def test_mask_rejects_invalid_sensor_data(change):
    from neurograft.navigation_controller import legal_action
    obs = observation()
    change(obs)
    with pytest.raises(ValueError):
        legal_action(obs, [0, 1, 2, 3])


@pytest.mark.parametrize('logits', [[1, 2, 3], [0, 1, 2, float('nan')],
                                    [0, 1, 2, float('inf')], [0, 1, 2, True], '0123'])
def test_invalid_prior_is_rejected(logits):
    from neurograft.navigation_controller import legal_action
    with pytest.raises(ValueError):
        legal_action(observation(), logits)


def test_mapping_records_only_current_and_four_neighbors():
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    obs = observation(walls=(1, 0, 1, 0))
    before = copy.deepcopy(obs)
    rng = random.getstate()
    action = navigator.choose(obs, [9, 8, 7, 6])
    assert action == 1
    assert navigator.known_free == {(3, 3), (4, 3), (2, 3)}
    assert navigator.known_blocked == {(3, 2), (3, 4)}
    assert navigator.visits == {(3, 3): 1}
    assert obs == before and random.getstate() == rng
    snapshot = navigator.snapshot()
    assert snapshot['mode'] == 'observed_map'
    assert snapshot['policy_action'] == 0 and snapshot['last_action'] == 1
    assert snapshot['overridden'] is True
    assert snapshot['known_free'] == 3 and snapshot['known_blocked'] == 2
    assert snapshot['visited'] == 1 and snapshot['frontiers'] == 2
    assert snapshot['path'][0] == {'x': 3, 'z': 3}
    assert snapshot['size'] == 7
    assert snapshot['position'] == [3, 3] and snapshot['goal'] == [5, 1]
    assert snapshot['free_cells'] == [[2, 3], [3, 3], [4, 3]]
    assert snapshot['blocked_cells'] == [[3, 2], [3, 4]]
    assert snapshot['visited_cells'] == [[3, 3]]
    snapshot['free_cells'].clear()
    assert len(navigator.known_free) == 3
    json.dumps(snapshot, allow_nan=False)


def test_equal_frontiers_use_prior_and_map_only_control_ignores_it():
    from neurograft.navigation_controller import ObservedMapNavigator
    obs = observation(goal=(5, 1))
    north = ObservedMapNavigator(7)
    east = ObservedMapNavigator(7)
    assert north.choose(obs, [9, 8, 0, 0]) == 0
    assert east.choose(obs, [8, 9, 0, 0]) == 1
    left = ObservedMapNavigator(7)
    right = ObservedMapNavigator(7)
    assert left.choose(obs, [9, 8, 0, 0], use_policy=False) == 0
    assert right.choose(obs, [8, 9, 0, 0], use_policy=False) == 0
    # The raw proposal remains separately visible even when it is not consulted.
    assert left.snapshot()['policy_action'] == 0
    assert right.snapshot()['policy_action'] == 1


def test_known_goal_route_overrides_a_strong_wrong_prior():
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    assert navigator.choose(observation(goal=(4, 3)), [100, 0, 0, 0]) == 1
    assert navigator.snapshot()['reason'] == 'known_goal'
    assert navigator.snapshot()['path'] == [{'x': 3, 'z': 3}, {'x': 4, 'z': 3}]


def test_equal_known_goal_routes_use_prior_but_keep_shortest_path():
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    for position in ((3, 3), (3, 2), (4, 2)):
        navigator.choose(observation(agent=position), [0, 1, 2, 3])
    north = ObservedMapNavigator.from_dict(navigator.to_dict())
    east = ObservedMapNavigator.from_dict(navigator.to_dict())
    obs = observation(agent=(3, 3), goal=(4, 2))
    assert north.choose(obs, [9, 8, 0, 0]) == 0
    assert east.choose(obs, [8, 9, 0, 0]) == 1
    assert len(north.snapshot()['path']) == len(east.snapshot()['path']) == 3


def test_discontinuous_unseen_position_requires_episodic_reset():
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    navigator.choose(observation(agent=(2, 2)), [0, 1, 2, 3])
    before = navigator.to_dict()
    with pytest.raises(ValueError, match='position'):
        navigator.choose(observation(agent=(4, 4)), [0, 1, 2, 3])
    assert navigator.to_dict() == before


def test_goal_cannot_conflict_with_sensed_wall_and_failure_is_atomic():
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    before = navigator.to_dict()
    with pytest.raises(ValueError, match='goal'):
        navigator.choose(observation(goal=(4, 3), walls=(0, 1, 0, 0)), [0, 1, 2, 3])
    assert navigator.to_dict() == before


def test_map_exploration_reaches_goal_around_a_wall_without_collisions():
    from neurograft.navigation_controller import ObservedMapNavigator
    free = {(x, z) for x in range(1, 6) for z in range(1, 6)} - {(3, 2), (3, 3), (3, 4)}
    world = TinyGrid(7, free, (2, 3), (4, 3))
    navigator = ObservedMapNavigator(7)
    for _ in range(80):
        world.move(navigator.choose(world.observe(), [0, 10, 0, 0]))
        if world.agent == world.goal:
            break
    assert world.agent == world.goal
    assert len(world.trail) - 1 > 2
    assert navigator.known_free <= free
    assert len(navigator.known_free | navigator.known_blocked) <= 49


def test_dead_end_backtracking_keeps_frontier_and_restores_exact_continuation():
    from neurograft.navigation_controller import ObservedMapNavigator
    free = {(1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (4, 3), (5, 3), (5, 2), (5, 1)}
    world = TinyGrid(7, free, (2, 3), (5, 1))
    navigator = ObservedMapNavigator(7)
    committed = None
    restored = None
    for _ in range(30):
        obs = world.observe()
        action = navigator.choose(obs, [10, 9, 0, 0])
        if world.agent == (3, 1):
            committed = navigator.target
            assert committed is not None and committed != world.agent
            state = json.loads(json.dumps(navigator.to_dict()))
            restored = ObservedMapNavigator.from_dict(state)
            assert restored.to_dict() == state
            assert restored.snapshot() == navigator.snapshot()
        elif restored is not None:
            assert restored.choose(obs, [10, 9, 0, 0]) == action
            assert restored.to_dict() == navigator.to_dict()
            if world.agent == (3, 2):
                assert navigator.target == committed
        world.move(action)
        if world.agent == world.goal:
            break
    assert world.agent == world.goal
    assert (3, 1) in world.trail and world.trail.count((3, 2)) >= 2


def test_hidden_cells_do_not_affect_action_or_memory():
    from neurograft.navigation_controller import ObservedMapNavigator
    free = {(x, z) for x in range(1, 6) for z in range(1, 6)}
    first = TinyGrid(7, free, (3, 3), (5, 1))
    second = TinyGrid(7, free - {(1, 1), (5, 5)}, (3, 3), (5, 1))
    left, right = ObservedMapNavigator(7), ObservedMapNavigator(7)
    assert first.observe() == second.observe()
    assert left.choose(first.observe(), [0, 9, 0, 0]) == right.choose(second.observe(), [0, 9, 0, 0])
    assert left.to_dict() == right.to_dict()


@pytest.mark.parametrize('size', [True, 4, 32, 7.0, None])
def test_grid_size_is_strict_and_bounded(size):
    from neurograft.navigation_controller import ObservedMapNavigator
    with pytest.raises(ValueError):
        ObservedMapNavigator(size)


@pytest.mark.parametrize('change', [
    lambda obs: obs[0].__setitem__(0, 0.49),
    lambda obs: obs[1].__setitem__(0, 1.0),
    lambda obs: obs[2].__setitem__(0, 0),  # Previously sensed wall cannot become free.
    lambda obs: obs[3].__setitem__(3, float('nan')),
])
def test_invalid_observation_leaves_all_controller_state_untouched(change):
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    navigator.choose(observation(walls=(1, 0, 0, 0)), [0, 1, 2, 3])
    before = navigator.to_dict()
    obs = observation(walls=(1, 0, 0, 0))
    change(obs)
    with pytest.raises(ValueError):
        navigator.choose(obs, [0, 1, 2, 3])
    assert navigator.to_dict() == before


def test_boundary_free_sensor_and_invalid_prior_are_atomic():
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    before = navigator.to_dict()
    for obs, logits in [(observation(agent=(1, 1)), [0, 1, 2, 3]),
                        (observation(), [0, 1, 2, float('nan')])]:
        with pytest.raises(ValueError):
            navigator.choose(obs, logits)
        assert navigator.to_dict() == before


@pytest.mark.parametrize('mutate', [
    lambda state: state.__setitem__('version', 2),
    lambda state: state.__setitem__('extra', 1),
    lambda state: state['known_free'].append(state['known_free'][0]),
    lambda state: state['known_free'].append([-1, 2]),
    lambda state: state['known_blocked'].append(state['known_free'][0]),
    lambda state: state['visits'][0].__setitem__('count', True),
    lambda state: state['visits'][0].__setitem__('count', 0),
    lambda state: state.__setitem__('target', [1, 1]),
    lambda state: state.__setitem__('last_action', 4),
    lambda state: state.__setitem__('policy_action', True),
    lambda state: state.__setitem__('reason', 'invented'),
    lambda state: state['path'].append([1, 1]),
    lambda state: state.__setitem__('use_policy', 1),
])
def test_restoration_rejects_malformed_and_inconsistent_state_without_mutating_input(mutate):
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(7)
    navigator.choose(observation(), [0, 1, 2, 3])
    state = navigator.to_dict()
    mutate(state)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        ObservedMapNavigator.from_dict(state)
    assert state == before


def test_empty_controller_has_exact_json_roundtrip_and_serialization_is_detached():
    from neurograft.navigation_controller import ObservedMapNavigator
    navigator = ObservedMapNavigator(31)
    state = navigator.to_dict()
    restored = ObservedMapNavigator.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict() == state
    assert restored.snapshot() == navigator.snapshot()
    navigator.choose(observation(31, (15, 15), (29, 1)), [0, 1, 2, 3])
    detached = navigator.to_dict()
    detached['known_free'].clear()
    assert navigator.known_free
