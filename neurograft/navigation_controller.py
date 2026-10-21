"""Deterministic navigation assistance from the policy's local observations.

This controller receives the existing four-by-four observation plus the public
grid size. It never receives a world, teacher action, hidden obstacle map or
environmental telemetry. Its collision avoidance is an engineered legal-action
filter; successful assisted routes do not establish neural navigation ability.
"""
from __future__ import annotations

from collections import deque
import math


MODES = ('policy', 'masked_policy', 'observed_map')
ALL_MODES = ('policy', 'masked_policy', 'observed_map', 'cx_steer')
_ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_REASONS = frozenset({'uninitialized', 'known_goal', 'frontier', 'at_goal', 'explored_fallback'})
_MAX_VISITS = 2**53 - 1


def _finite(value):
    if type(value) not in (int, float):
        raise ValueError('observation and logits must contain finite numbers')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError('observation and logits must contain finite numbers')


def _inputs(observation, logits):
    if (not isinstance(observation, (tuple, list)) or len(observation) != 4
            or any(not isinstance(row, (tuple, list)) or len(row) != 4 for row in observation)):
        raise ValueError('observation must be a four-by-four sequence')
    for row in observation:
        for value in row:
            _finite(value)
    if not isinstance(logits, (tuple, list)) or len(logits) != 4:
        raise ValueError('logits must contain four finite numbers')
    for value in logits:
        _finite(value)
    if any(value not in (0, 1) for value in observation[2]):
        raise ValueError('local walls must be exact zero or one')
    legal = [action for action, wall in enumerate(observation[2]) if wall == 0]
    if not legal:
        raise ValueError('observation has no legal actions')
    return legal


def legal_action(observation, logits) -> int:
    """Mask observed walls and choose the largest logit; ties use N/E/S/W.

    Inputs are Python lists or tuples of built-in finite numbers. Tensor callers
    should pass ``logits[0].tolist()``. Neither input is modified.
    """
    legal = _inputs(observation, logits)
    return max(legal, key=lambda action: (logits[action], -action))


def cx_steering_action(observation, logits, steering_weight: float = 1.5) -> int:
    """Add a synthetic CX-inspired goal/heading prior to learned policy logits.

    Heading and goal displacement come directly from the observation, not the
    diagnostic ring attractor. Local walls are masked exactly as in legal_action.
    This engineered assistance is not a cell-specific circuit reconstruction.
    """
    _inputs(observation, logits)
    head_dx, head_dz = observation[0][2], observation[0][3]
    heading = math.atan2(head_dz, head_dx)
    goal_dx, goal_dz = observation[1][2], observation[1][3]
    walls = observation[2]

    from .compass import CXPFLSteering
    steer_engine = CXPFLSteering(steering_weight=steering_weight)
    return steer_engine.steer(heading, goal_dx, goal_dz, walls, logits)


def _neighbor(cell, action):
    dx, dz = _ACTIONS[action]
    return cell[0] + dx, cell[1] + dz


def _routes(start, free, order):
    """BFS parents. Action ordering breaks equal-length route ties only."""
    distances, parents = {start: 0}, {start: None}
    pending = deque([start])
    while pending:
        cell = pending.popleft()
        for action in order:
            target = _neighbor(cell, action)
            if target in free and target not in distances:
                distances[target] = distances[cell] + 1
                parents[target] = cell
                pending.append(target)
    return distances, parents


def _path(parents, target):
    result = [target]
    while parents[result[-1]] is not None:
        result.append(parents[result[-1]])
    result.reverse()
    return result


def _action(start, end):
    return _ACTIONS.index((end[0] - start[0], end[1] - start[1]))


class ObservedMapNavigator:
    """A bounded episodic free/blocked map with committed-frontier exploration.

    Current and adjacent cells are the only newly mapped cells. A frontier is an
    observed-free cell that has not been visited. A frontier stays committed
    until reached, unless a complete route through known free cells to the goal
    becomes available. This avoids repeatedly changing targets while backtracking.
    ``use_policy=False`` uses fixed N/E/S/W ties, creating a map-only control.

    Callers create a new instance when the arena or episode resets. The public
    cell sets are available for checkpoint validation and should not be mutated.
    """

    def __init__(self, size: int):
        if type(size) is not int or not 5 <= size <= 31:
            raise ValueError('size must be an integer from 5 through 31')
        self.size = size
        self.known_free = set()
        self.known_blocked = set()
        self.visits = {}
        self.current = self.goal = self.target = None
        self.last_action = self.policy_action = None
        self.reason = 'uninitialized'
        self.path = []
        self.use_policy = True

    def _coordinate(self, values):
        result = []
        for value in values:
            if not 0 < value < 1:
                raise ValueError('agent and goal coordinates must be interior normalized cells')
            scaled = value * (self.size - 1)
            rounded = round(scaled)
            if not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-5):
                raise ValueError('normalized coordinates do not match the public grid size')
            if not 1 <= rounded < self.size - 1:
                raise ValueError('agent and goal must be interior cells')
            result.append(rounded)
        return tuple(result)

    def choose(self, observation, logits, use_policy=True) -> int:
        """Ingest one local observation and return an observed-legal action.

        All input and static-map consistency validation precedes mutation. Model
        logits resolve equal route/frontier choices; they cannot override the
        known map or the committed exploration target.
        """
        legal = _inputs(observation, logits)
        if type(use_policy) is not bool:
            raise ValueError('use_policy must be boolean')
        current = self._coordinate(observation[0][:2])
        goal = self._coordinate(observation[1][:2])
        if self.known_free and current not in self.known_free:
            raise ValueError('position is outside the observed map; reset the episodic controller')
        free, blocked = {current}, set()
        for action, wall in enumerate(observation[2]):
            cell = _neighbor(current, action)
            if not wall and not (0 < cell[0] < self.size - 1 and 0 < cell[1] < self.size - 1):
                raise ValueError('a boundary cell cannot be observed free')
            (blocked if wall else free).add(cell)
        if free & self.known_blocked or blocked & self.known_free:
            raise ValueError('observation contradicts the static episodic map; reset the controller')
        if goal in blocked or goal in self.known_blocked:
            raise ValueError('goal conflicts with an observed blocked cell')
        if sum(self.visits.values()) >= _MAX_VISITS:
            raise ValueError('controller visit counter exceeds its limit')

        # Work on detached candidates until every decision is determined.
        free |= self.known_free
        blocked |= self.known_blocked
        visits = dict(self.visits)
        visits[current] = visits.get(current, 0) + 1
        order = sorted(range(4), key=lambda action: (-logits[action], action)) if use_policy else list(range(4))
        ranks = {action: index for index, action in enumerate(order)}
        policy_action = max(range(4), key=lambda action: (logits[action], -action))
        distances, parents = _routes(current, free, order)
        target = None
        if current == goal:
            path, reason = [current], 'at_goal'
            selected = max(legal, key=lambda action: (logits[action], -action)) if use_policy else min(legal)
        elif goal in distances:
            path, reason = _path(parents, goal), 'known_goal'
            selected = _action(path[0], path[1])
        else:
            frontiers = (free - visits.keys()) & distances.keys()
            if self.goal == goal and self.target in frontiers:
                target = self.target
            elif frontiers:
                def priority(cell):
                    distance = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
                    route = _path(parents, cell)
                    first = _action(route[0], route[1])
                    return (distances[cell] + distance, distance, ranks[first], cell[0], cell[1])
                target = min(frontiers, key=priority)
            if target is not None:
                path, reason = _path(parents, target), 'frontier'
                selected = _action(path[0], path[1])
            else:
                # A disconnected observed component may be exhausted. This is
                # explicitly labelled; no hidden-map route is manufactured.
                selected = max(legal, key=lambda action: (logits[action], -action)) if use_policy else min(legal)
                path, reason = [current, _neighbor(current, selected)], 'explored_fallback'
        if selected not in legal:
            raise ValueError('known-map route disagrees with current local walls')
        self.known_free, self.known_blocked, self.visits = free, blocked, visits
        self.current, self.goal, self.target = current, goal, target
        self.last_action, self.policy_action = selected, policy_action
        self.reason, self.path, self.use_policy = reason, path, use_policy
        return selected

    def to_dict(self):
        """Return detached, canonical, JSON-only continuation state."""
        return {
            'version': 1, 'size': self.size,
            'known_free': [list(cell) for cell in sorted(self.known_free)],
            'known_blocked': [list(cell) for cell in sorted(self.known_blocked)],
            'visits': [{'cell': list(cell), 'count': self.visits[cell]} for cell in sorted(self.visits)],
            'current': list(self.current) if self.current is not None else None,
            'goal': list(self.goal) if self.goal is not None else None,
            'target': list(self.target) if self.target is not None else None,
            'last_action': self.last_action, 'policy_action': self.policy_action,
            'reason': self.reason, 'path': [list(cell) for cell in self.path],
            'use_policy': self.use_policy,
        }

    @classmethod
    def from_dict(cls, data):
        """Validate an entire bounded candidate before constructing its state."""
        fields = {'version', 'size', 'known_free', 'known_blocked', 'visits', 'current',
                  'goal', 'target', 'last_action', 'policy_action', 'reason', 'path', 'use_policy'}
        if not isinstance(data, dict) or set(data) != fields or type(data['version']) is not int or data['version'] != 1:
            raise ValueError('invalid observed-map controller fields or version')
        candidate = cls(data['size'])
        size = candidate.size

        def cell(value, interior=False, optional=False):
            if value is None and optional:
                return None
            low, high = (1, size - 2) if interior else (0, size - 1)
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(number) is not int or not low <= number <= high for number in value)):
                raise ValueError('invalid observed-map cell')
            return tuple(value)

        def cells(name, interior=False):
            values = data[name]
            if not isinstance(values, list) or len(values) > size * size:
                raise ValueError('observed-map cell list exceeds bounds')
            parsed = [cell(value, interior) for value in values]
            if len(set(parsed)) != len(parsed):
                raise ValueError('duplicate observed-map cell')
            return parsed

        free, blocked = set(cells('known_free', True)), set(cells('known_blocked'))
        if free & blocked or len(free | blocked) > size * size:
            raise ValueError('inconsistent observed-map occupancy')
        raw_visits = data['visits']
        if not isinstance(raw_visits, list) or len(raw_visits) > size * size:
            raise ValueError('invalid observed-map visit history')
        visits = {}
        for visit in raw_visits:
            if not isinstance(visit, dict) or set(visit) != {'cell', 'count'}:
                raise ValueError('invalid observed-map visit entry')
            location = cell(visit['cell'], True)
            count = visit['count']
            if (location not in free or location in visits or type(count) is not int
                    or not 1 <= count <= _MAX_VISITS):
                raise ValueError('invalid observed-map visit count or cell')
            visits[location] = count
        if sum(visits.values()) > _MAX_VISITS:
            raise ValueError('observed-map visit counters exceed their limit')
        current = cell(data['current'], True, True)
        goal = cell(data['goal'], True, True)
        target = cell(data['target'], True, True)
        if goal in blocked:
            raise ValueError('goal conflicts with the observed blocked map')
        path = cells('path', True)
        for name in ('last_action', 'policy_action'):
            if data[name] is not None and (type(data[name]) is not int or not 0 <= data[name] <= 3):
                raise ValueError('invalid observed-map action')
        if not isinstance(data['reason'], str) or data['reason'] not in _REASONS or type(data['use_policy']) is not bool:
            raise ValueError('invalid observed-map decision metadata')
        reason = data['reason']
        if not free:
            if (blocked or visits or current is not None or goal is not None or target is not None
                    or path or data['last_action'] is not None or data['policy_action'] is not None
                    or reason != 'uninitialized'):
                raise ValueError('uninitialized observed map contains navigation history')
        else:
            if current not in visits or goal is None or not path or path[0] != current or reason == 'uninitialized':
                raise ValueError('observed-map current position/history is inconsistent')
            if data['last_action'] is None or data['policy_action'] is None:
                raise ValueError('observed-map decision actions are missing')
            neighborhood = set(visits)
            for location in visits:
                adjacent = {_neighbor(location, action) for action in range(4)}
                if not adjacent <= free | blocked:
                    raise ValueError('visited cell lacks its complete local wall observation')
                neighborhood |= adjacent
            if not (free | blocked) <= neighborhood:
                raise ValueError('observed map includes cells outside recorded sensor neighborhoods')
            distances, _ = _routes(current, free, range(4))
            if set(distances) != free:
                raise ValueError('observed free map must be connected')
            if (any(location not in free for location in path)
                    or any(abs(left[0] - right[0]) + abs(left[1] - right[1]) != 1
                           for left, right in zip(path, path[1:]))):
                raise ValueError('observed-map explanation path is invalid')
            if reason == 'at_goal':
                if current != goal or len(path) != 1 or target is not None:
                    raise ValueError('invalid at-goal decision')
            elif len(path) < 2 or _action(path[0], path[1]) != data['last_action']:
                raise ValueError('observed-map action disagrees with its path')
            if reason == 'frontier':
                if target not in free or target in visits or path[-1] != target:
                    raise ValueError('invalid committed observed-map frontier')
            elif target is not None:
                raise ValueError('committed frontier is inconsistent with decision reason')
            if reason == 'known_goal' and path[-1] != goal:
                raise ValueError('known-goal route does not reach the goal')
        candidate.known_free, candidate.known_blocked, candidate.visits = free, blocked, visits
        candidate.current, candidate.goal, candidate.target = current, goal, target
        candidate.last_action, candidate.policy_action = data['last_action'], data['policy_action']
        candidate.reason, candidate.path, candidate.use_policy = reason, path, data['use_policy']
        return candidate

    def snapshot(self):
        """Small JSON explanation, keeping the neural proposal separate."""
        return {
            'mode': 'observed_map', 'reason': self.reason,
            'size': self.size,
            'position': list(self.current) if self.current is not None else None,
            'goal': list(self.goal) if self.goal is not None else None,
            'free_cells': [list(cell) for cell in sorted(self.known_free)],
            'blocked_cells': [list(cell) for cell in sorted(self.known_blocked)],
            'visited_cells': [list(cell) for cell in sorted(self.visits)],
            'known_free': len(self.known_free), 'known_blocked': len(self.known_blocked),
            'visited': len(self.visits), 'frontiers': len(self.known_free - self.visits.keys()),
            'last_action': self.last_action, 'policy_action': self.policy_action,
            'overridden': self.last_action is not None and self.last_action != self.policy_action,
            'path': [{'x': x, 'z': z} for x, z in self.path],
            'use_policy': self.use_policy,
        }
