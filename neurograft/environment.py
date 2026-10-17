"""Deterministic, dependency-free navigation world for supervised live sessions.

Coordinates are integer (x, z) cells. Actions are north, east, south, west.
This module supplies route labels; it does not perform learning itself.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any


class NavigationWorld:
    """A connected seeded grid with bounded episodes and JSON-safe saves."""

    VERSION = 1
    MAX_EPISODE_STEPS = 256
    MAX_TRAIL = 64
    COLORS = ("blue", "amber", "rose")
    PRESETS = ("labyrinth", "chambers", "corridors", "pillars")
    ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
    RAYS_8 = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
    _SAVE_KEYS = {
        "version", "seed", "size", "agent", "goal", "beacons", "obstacles",
        "trail", "steps", "episode", "goals_reached", "total_reward",
        "last_reward", "episode_steps", "max_episode_steps", "rng_state",
    }

    def __init__(self, seed: int = 0, size: int = 15, preset: str = "labyrinth"):
        self.seed = self._integer(seed, "seed", -(2**63), 2**63 - 1)
        self.size = self._integer(size, "size", 5, 31)
        if type(preset) is not str or preset not in self.PRESETS:
            raise ValueError(f"preset must be one of {self.PRESETS}")
        self.preset = preset
        self._rng = random.Random(self.seed)
        self.steps = self.episode = self.goals_reached = self.episode_steps = 0
        self.total_reward = self.last_reward = 0.0
        edge = self.size - 1
        self._obstacles = {
            (x, z) for z in range(self.size) for x in range(self.size)
            if x in (0, edge) or z in (0, edge)
        }
        corners = [(1, 1), (edge - 1, 1), (1, edge - 1), (edge - 1, edge - 1)]
        self._beacons = dict(zip(self.COLORS, self._rng.sample(corners, 3)))
        self._agent = (self.size // 2, self.size // 2)
        protected = set(self._beacons.values()) | {self._agent}
        candidates = [
            (x, z) for z in range(1, edge) for x in range(1, edge)
            if (x, z) not in protected
        ]
        self._rng.shuffle(candidates)
        if self.size >= 11:
            candidates.sort(key=lambda cell: not self._is_motif_cell(cell, self.preset))
        density = 0.22 if self.size >= 11 else 0.15
        target = max(1, int((self.size - 2) ** 2 * density))
        added = 0
        for cell in candidates:
            self._obstacles.add(cell)
            if len(self._reachable(self._agent)) != (self.size - 2) ** 2 - added - 1:
                self._obstacles.remove(cell)
                continue
            added += 1
            if added == target:
                break
        self.direction = 0
        self._trail = [self._agent]
        self._goal = self._agent
        self._goal_color = "custom"
        self.set_goal()

    def _is_motif_cell(self, cell: tuple[int, int], preset: str) -> bool:
        x, z = cell
        mid = self.size // 2
        q1 = max(2, self.size // 4)
        q3 = min(self.size - 3, self.size - 1 - q1)
        if preset == "chambers":
            on_cross = (x == mid and z not in (q1, mid, q3)) or (z == mid and x not in (q1, mid, q3))
            on_ring = (x in (q1, q3) and z in (q1, q3))
            return on_cross or on_ring
        if preset == "corridors":
            return (x % 3 == 0 and 2 <= z <= self.size - 3 and (z + x) % 4 != 0)
        if preset == "pillars":
            return (x % 2 == 0 and z % 2 == 0) or (abs(x - mid) == abs(z - mid) and x not in (1, mid, self.size - 2))
        # Default "labyrinth": multi-chamber partitions + inner canyon baffles
        on_partition = (x == mid and z not in (q1, mid, q3)) or (z == mid and x not in (q1, mid, q3))
        on_baffle = ((x in (q1, q3) and abs(z - mid) <= max(1, self.size // 6) and z != mid)
                     or (z in (q1, q3) and abs(x - mid) <= max(1, self.size // 6) and x != mid))
        return on_partition or on_baffle

    @staticmethod
    def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        return value

    @staticmethod
    def _number(value: Any, name: str) -> float:
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a finite number")
        try:
            result = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    def _free(self, cell: tuple[int, int]) -> bool:
        return 0 < cell[0] < self.size - 1 and 0 < cell[1] < self.size - 1 and cell not in self._obstacles

    def _reachable(self, start: tuple[int, int]) -> set[tuple[int, int]]:
        visited = {start}
        pending = deque([start])
        while pending:
            x, z = pending.popleft()
            for dx, dz in self.ACTIONS:
                cell = (x + dx, z + dz)
                if self._free(cell) and cell not in visited:
                    visited.add(cell)
                    pending.append(cell)
        return visited

    def observe(self) -> list[list[float]]:
        """Return four source-feature rows, each with four normalized values.

        Rows contain agent position/heading, goal position/displacement,
        neighboring walls (N/E/S/W), and reward/episode-clock/lifetime progress.
        """
        x, z = self._agent
        gx, gz = self._goal
        scale = float(self.size - 1)
        dx, dz = self.ACTIONS[self.direction]
        return [
            [x / scale, z / scale, float(dx), float(dz)],
            [gx / scale, gz / scale, (gx - x) / scale, (gz - z) / scale],
            [float(not self._free((x + ax, z + az))) for ax, az in self.ACTIONS],
            [self.last_reward, self.episode_steps / self.MAX_EPISODE_STEPS,
             self.goals_reached / (self.goals_reached + 1), 1.0],
        ]

    def expert_action(self) -> int:
        """Return the first action on a shortest path, with fixed N/E/S/W ties.

        A goal at the current cell completes on the next tick without movement.
        """
        if self._agent == self._goal:
            return self.direction
        pending = deque([(self._agent, None)])
        visited = {self._agent}
        while pending:
            (x, z), first = pending.popleft()
            for action, (dx, dz) in enumerate(self.ACTIONS):
                cell = (x + dx, z + dz)
                if cell in visited or not self._free(cell):
                    continue
                selected = action if first is None else first
                if cell == self._goal:
                    return selected
                visited.add(cell)
                pending.append((cell, selected))
        raise RuntimeError("world invariant broken: goal is not reachable")

    def shortest_path(self) -> list[dict[str, int]]:
        """Return ordered interior cells along the shortest path from agent to goal."""
        if self._agent == self._goal:
            return [self._position(self._goal)]
        pending = deque([(self._agent, [self._agent])])
        visited = {self._agent}
        while pending:
            (x, z), path = pending.popleft()
            for dx, dz in self.ACTIONS:
                cell = (x + dx, z + dz)
                if cell in visited or not self._free(cell):
                    continue
                next_path = [*path, cell]
                if cell == self._goal:
                    return [self._position(step) for step in next_path]
                visited.add(cell)
                pending.append((cell, next_path))
        return [self._position(self._agent)]

    def environment_telemetry(self) -> dict[str, Any]:
        """Compute spatial complexity, 8-ray optic/LiDAR clearances, choke points, and plume gradient."""
        path = self.shortest_path()
        geodesic = max(0, len(path) - 1)
        ax, az = self._agent
        gx, gz = self._goal
        manhattan = abs(ax - gx) + abs(az - gz)
        detour = float(geodesic) / max(1, manhattan) if geodesic > 0 else 1.0
        interior = (self.size - 2) ** 2
        boundary_count = 4 * self.size - 4
        obstacle_cells = max(0, len(self._obstacles) - boundary_count)
        free_cells = interior - obstacle_cells
        corridors = junctions = dead_ends = 0
        for z in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                if (x, z) in self._obstacles:
                    continue
                deg = sum(1 for dx, dz in self.ACTIONS if self._free((x + dx, z + dz)))
                if deg <= 1:
                    dead_ends += 1
                elif deg == 2:
                    corridors += 1
                else:
                    junctions += 1
        choke_points = []
        for step in path[1:-1]:
            sx, sz = step["x"], step["z"]
            deg = sum(1 for dx, dz in self.ACTIONS if self._free((sx + dx, sz + dz)))
            if deg <= 2:
                choke_points.append({"x": sx, "z": sz})
        horizon = max(1, self.size - 2)
        lidar_8 = []
        lidar_rays = []
        for dx, dz in self.RAYS_8:
            steps = 0
            cx, cz = ax, az
            while self._free((cx + dx, cz + dz)) and steps < horizon:
                cx += dx
                cz += dz
                steps += 1
            norm_d = float(steps) / float(horizon)
            lidar_8.append(norm_d)
            lidar_rays.append({"x": cx + dx, "z": cz + dz, "free_x": cx, "free_z": cz, "clearance": norm_d})
        chemoattractant = float(math.exp(-geodesic / max(2.0, 0.55 * self.size)))
        return {
            "preset": getattr(self, "preset", "labyrinth"),
            "size": self.size,
            "interior_cells": interior,
            "free_cells": free_cells,
            "obstacle_cells": obstacle_cells,
            "obstacle_density": float(obstacle_cells) / max(1, interior),
            "corridor_cells": corridors,
            "branch_junctions": junctions,
            "dead_ends": dead_ends,
            "geodesic_distance": geodesic,
            "manhattan_distance": manhattan,
            "detour_index": detour,
            "choke_points": choke_points,
            "chemoattractant": chemoattractant,
            "lidar_8": lidar_8,
            "lidar_rays": lidar_rays,
        }

    def set_goal(self, color: str | None = None, x: int | None = None, z: int | None = None) -> dict:
        """Choose a beacon or a free interior cell; validate before mutation."""
        if color is not None:
            if type(color) is not str or color not in self.COLORS or x is not None or z is not None:
                raise ValueError("goal requires one beacon color or both integer coordinates")
            cell, selected_color = self._beacons[color], color
        elif x is not None or z is not None:
            gx = self._integer(x, "goal x", 1, self.size - 2)
            gz = self._integer(z, "goal z", 1, self.size - 2)
            cell, selected_color = (gx, gz), "custom"
            if not self._free(cell):
                raise ValueError("goal must be a free, reachable interior cell")
        else:
            choices = [
                item for item in self._beacons.items()
                if item[1] != self._agent and item[1] != self._goal
            ]
            if not choices:
                choices = [item for item in self._beacons.items() if item[1] != self._agent]
            selected_color, cell = self._rng.choice(choices)
        self._goal, self._goal_color = cell, selected_color
        return {"x": cell[0], "z": cell[1], "color": selected_color}

    def reset(self) -> dict:
        """Start a new episode on the same map and retain lifetime counters."""
        choices = [
            (x, z) for z in range(1, self.size - 1) for x in range(1, self.size - 1)
            if self._free((x, z)) and (x, z) not in self._beacons.values()
        ]
        self._agent = self._rng.choice(choices)
        self.direction = 0
        self._trail = [self._agent]
        self.episode += 1
        self.episode_steps = 0
        self.last_reward = 0.0
        self.set_goal()
        return self.snapshot()

    def step(self, action: int) -> dict:
        action = self._integer(action, "action", 0, 3)
        self.direction = action
        already_at_goal = self._agent == self._goal
        x, z = self._agent
        dx, dz = self.ACTIONS[action]
        cell = (x + dx, z + dz)
        collision = not already_at_goal and not self._free(cell)
        if not collision and not already_at_goal:
            self._agent = cell
        self.steps += 1
        self.episode_steps += 1
        self._trail.append(self._agent)
        self._trail = self._trail[-self.MAX_TRAIL:]
        reached = self._agent == self._goal
        reward = 1.0 if reached else (-0.1 if collision else -0.01)
        self.total_reward += reward
        done = reached or self.episode_steps >= self.MAX_EPISODE_STEPS
        if reached:
            self.goals_reached += 1
            self.episode += 1
            self.episode_steps = 0
            self.set_goal()
        elif done:
            self.reset()
        self.last_reward = reward
        return {"reward": reward, "goal_reached": reached, "episode_done": done, "collision": collision}

    @staticmethod
    def _position(cell: tuple[int, int]) -> dict[str, int]:
        return {"x": cell[0], "z": cell[1]}

    def snapshot(self) -> dict:
        return {
            "version": self.VERSION, "seed": self.seed, "size": self.size,
            "agent": {**self._position(self._agent), "direction": self.direction},
            "goal": {**self._position(self._goal), "color": self._goal_color},
            "beacons": [{**self._position(self._beacons[color]), "color": color} for color in self.COLORS],
            "obstacles": [self._position(cell) for cell in sorted(self._obstacles, key=lambda cell: (cell[1], cell[0]))],
            "trail": [self._position(cell) for cell in self._trail],
            "steps": self.steps, "episode": self.episode, "goals_reached": self.goals_reached,
            "total_reward": self.total_reward, "last_reward": self.last_reward,
            "episode_steps": self.episode_steps, "max_episode_steps": self.MAX_EPISODE_STEPS,
        }

    def to_dict(self) -> dict:
        state = self._rng.getstate()
        return {**self.snapshot(), "rng_state": [state[0], list(state[1]), state[2]]}

    @classmethod
    def from_dict(cls, data: Any) -> NavigationWorld:
        """Validate a complete candidate before constructing a restored world.

        Only primitive save fields are accepted. No constructor, caller-owned
        object, or global random generator is mutated during validation.
        """
        if type(data) is not dict or set(data) != cls._SAVE_KEYS:
            raise ValueError("world save fields are missing or unknown")
        if type(data["version"]) is not int or data["version"] != cls.VERSION:
            raise ValueError("unsupported world serialization version")
        seed = cls._integer(data["seed"], "seed", -(2**63), 2**63 - 1)
        size = cls._integer(data["size"], "size", 5, 31)
        if type(data["max_episode_steps"]) is not int or data["max_episode_steps"] != cls.MAX_EPISODE_STEPS:
            raise ValueError("invalid episode limit")

        def position(value: Any, name: str, extra: set[str] | None = None) -> tuple[int, int]:
            if type(value) is not dict or set(value) != {"x", "z"} | (extra or set()):
                raise ValueError(f"invalid {name} fields")
            return (cls._integer(value["x"], f"{name} x", 0, size - 1),
                    cls._integer(value["z"], f"{name} z", 0, size - 1))

        obstacles_data = data["obstacles"]
        if type(obstacles_data) is not list or len(obstacles_data) > size * size:
            raise ValueError("invalid obstacles")
        obstacles = {position(item, "obstacle") for item in obstacles_data}
        if len(obstacles) != len(obstacles_data):
            raise ValueError("duplicate obstacle")
        boundary = {(x, z) for z in range(size) for x in range(size) if x in (0, size - 1) or z in (0, size - 1)}
        if not boundary <= obstacles:
            raise ValueError("world boundary must be blocked")

        agent = position(data["agent"], "agent", {"direction"})
        direction = cls._integer(data["agent"]["direction"], "direction", 0, 3)
        goal = position(data["goal"], "goal", {"color"})
        goal_color = data["goal"]["color"]
        if type(goal_color) is not str or goal_color not in (*cls.COLORS, "custom"):
            raise ValueError("invalid goal color")
        beacons_data = data["beacons"]
        if type(beacons_data) is not list or len(beacons_data) != len(cls.COLORS):
            raise ValueError("three colored beacons are required")
        beacons = {}
        for item in beacons_data:
            cell = position(item, "beacon", {"color"})
            color = item["color"]
            if type(color) is not str or color not in cls.COLORS or color in beacons:
                raise ValueError("invalid or repeated beacon color")
            beacons[color] = cell
        if len(set(beacons.values())) != len(beacons):
            raise ValueError("beacons must have distinct cells")
        if goal_color != "custom" and goal != beacons[goal_color]:
            raise ValueError("goal color must match its beacon coordinates")

        trail_data = data["trail"]
        if type(trail_data) is not list or not 1 <= len(trail_data) <= cls.MAX_TRAIL:
            raise ValueError("invalid trail length")
        trail = [position(item, "trail") for item in trail_data]
        if trail[-1] != agent or any(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1 for a, b in zip(trail, trail[1:])):
            raise ValueError("trail must follow adjacent cells and end at the agent")
        for cell in [agent, goal, *beacons.values(), *trail]:
            if cell in obstacles or not 0 < cell[0] < size - 1 or not 0 < cell[1] < size - 1:
                raise ValueError("agent, goals, beacons and trail must occupy free interior cells")
        steps = cls._integer(data["steps"], "steps", 0, 2**63 - 1)
        episode = cls._integer(data["episode"], "episode", 0, 2**63 - 1)
        goals = cls._integer(data["goals_reached"], "goals_reached", 0, min(steps, episode))
        episode_steps = cls._integer(data["episode_steps"], "episode_steps", 0, min(steps, cls.MAX_EPISODE_STEPS - 1))
        if len(trail) > steps + 1:
            raise ValueError("trail is longer than the step history")
        reward = cls._number(data["last_reward"], "last_reward")
        total_reward = cls._number(data["total_reward"], "total_reward")
        non_goal_steps = steps - goals
        minimum_reward = goals - 0.1 * non_goal_steps
        maximum_reward = goals - 0.01 * non_goal_steps
        tolerance = max(1e-6, steps * 1e-10)
        if reward not in (0.0, -0.01, -0.1, 1.0) or not minimum_reward - tolerance <= total_reward <= maximum_reward + tolerance:
            raise ValueError("reward counters exceed possible navigation rewards")
        if ((steps == 0 and reward != 0.0)
                or (reward == 1.0 and (goals == 0 or episode_steps != 0))
                or (reward == 0.0 and episode_steps != 0)):
            raise ValueError("last reward is inconsistent with episode counters")
        rng_data = data["rng_state"]
        if type(rng_data) is not list or len(rng_data) != 3 or type(rng_data[0]) is not int or rng_data[0] != 3 or type(rng_data[1]) is not list or len(rng_data[1]) != 625 or rng_data[2] is not None:
            raise ValueError("invalid private random state")
        random_words = [cls._integer(word, "random state word", 0, 2**32 - 1) for word in rng_data[1][:-1]]
        random_index = cls._integer(rng_data[1][-1], "random state index", 0, 624)
        rng = random.Random()
        rng.setstate((3, (*random_words, random_index), None))

        candidate = cls.__new__(cls)
        candidate.seed, candidate.size = seed, size
        candidate.preset = "labyrinth"
        candidate._obstacles = obstacles
        candidate._agent, candidate.direction = agent, direction
        candidate._goal, candidate._goal_color = goal, goal_color
        candidate._beacons, candidate._trail = beacons, trail
        candidate.steps, candidate.episode, candidate.goals_reached = steps, episode, goals
        candidate.episode_steps = episode_steps
        candidate.total_reward, candidate.last_reward = total_reward, reward
        candidate._rng = rng
        free_count = (size - 2) ** 2 - len(obstacles - boundary)
        if free_count < len(cls.COLORS) + 1 or len(candidate._reachable(agent)) != free_count:
            raise ValueError("all free cells and beacons must be connected and reachable")
        return candidate
