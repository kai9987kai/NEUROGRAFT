"""Tri-Tier Cognitive Memory Architecture with Matched Lesion Assays.

Inspired by Drosophila mushroom body sparse associative memory and the Mnemosyne
cognitive architecture in Supermix-Poseidon:
1. Tier 1 (Working Memory): Fast recurrent context vector updated with prediction surprise.
2. Tier 2 (Episodic Store): Value-gated experiential memory weighted by reward, surprise, and recency.
3. Tier 3 (Cognitive Spatial Map): Online topological graph of visited coordinates,
   cell clearance statistics, and beacon landmark confidence.

Includes 5 matched lesion modes:
- intact: Full 3-tier memory active.
- erased: Memory wiped at every tick (zero context retention).
- shifted: Circular permutation of episodic memory keys (breaks semantic alignment).
- random_retrieval: Queries select randomly from history, ignoring value/similarity.
- memory_free: Memory completely bypassed (pure reactive baseline).
"""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

LESION_MODES = ("intact", "erased", "shifted", "random_retrieval", "memory_free")


@dataclass
class EpisodicItem:
    step: int
    state: list[float]
    action: int
    reward: float
    surprise: float



class TopologicalMemoryGraph:
    """Bounded empirical graph of experienced grid transitions.

    Counts describe observed P(destination | source, executed action); BFS follows
    only directed experienced edges. This observer is neither a learned successor
    representation nor evidence for biological place cells. The legacy ``discount``
    constructor argument is accepted for compatibility, validated, and unused.
    """

    MAX_COORDINATE = 30
    MAX_TRANSITIONS = 2**53 - 1
    ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def __init__(self, discount: float = 0.9) -> None:
        self._number(discount, "discount")
        if not 0 <= discount <= 1:
            raise ValueError("invalid topological discount")
        self.nodes: set[tuple[int, int]] = set()
        self.transitions: dict[tuple[tuple[int, int], int], dict[tuple[int, int], int]] = {}
        self.adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self.total_transitions: int = 0
        self.reward_totals: dict[tuple[int, int], float] = {}

    @classmethod
    def _position(cls, value, *, bounded=True) -> tuple[int, int]:
        if (type(value) not in (tuple, list) or len(value) != 2
                or any(type(part) is not int for part in value)
                or (bounded and any(not 0 <= part <= cls.MAX_COORDINATE for part in value))):
            raise ValueError("invalid topological position")
        return tuple(value)

    @staticmethod
    def _number(value, name):
        if type(value) not in (int, float):
            raise ValueError(f"invalid topological {name}")
        try:
            result = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f"invalid topological {name}") from None
        if not math.isfinite(result):
            raise ValueError(f"invalid topological {name}")
        return result

    @classmethod
    def _transition(cls, u, action, v):
        u, v = cls._position(u), cls._position(v)
        if type(action) is not int or not 0 <= action < len(cls.ACTIONS):
            raise ValueError("invalid topological action")
        dx, dz = cls.ACTIONS[action]
        if v != u and v != (u[0] + dx, u[1] + dz):
            raise ValueError("topological transition must be an executed neighbor move or collision")
        return u, v

    def add_node(self, pos: tuple[int, int]) -> None:
        pos = self._position(pos)
        if pos not in self.nodes:
            self.nodes.add(pos)
            self.adjacency[pos] = set()
            self.reward_totals[pos] = 0.0

    def record_transition(
        self,
        u: tuple[int, int],
        action: int,
        v: tuple[int, int],
        reward: float = 0.0,
    ) -> None:
        u, v = self._transition(u, action, v)
        reward = self._number(reward, "reward")
        reward_total = self._number(self.reward_totals.get(v, 0.0) + reward, "reward total")
        if self.total_transitions >= self.MAX_TRANSITIONS:
            raise ValueError("topological transition count exceeds limit")
        self.add_node(u)
        self.add_node(v)
        self.total_transitions += 1
        self.reward_totals[v] = reward_total
        self.adjacency[u].add(v)

        key = (u, action)
        if key not in self.transitions:
            self.transitions[key] = {}
        self.transitions[key][v] = self.transitions[key].get(v, 0) + 1

    def transition_probability(self, u: tuple[int, int], action: int, v: tuple[int, int]) -> float:
        u, v = self._position(u, bounded=False), self._position(v, bounded=False)
        if type(action) is not int or not 0 <= action < len(self.ACTIONS):
            raise ValueError("invalid topological action")
        key = (u, action)
        counts = self.transitions.get(key, {})
        total = sum(counts.values())
        return counts.get(v, 0) / total if total > 0 else 0.0

    def shortest_topological_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        start = self._position(start, bounded=False)
        goal = self._position(goal, bounded=False)
        if start not in self.nodes or goal not in self.nodes:
            return None
        if start == goal:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = collections.deque([start])
        while queue:
            curr = queue.popleft()
            if curr == goal:
                path = []
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
            for nxt in sorted(self.adjacency.get(curr, set())):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = curr
                    queue.append(nxt)
        return None

    def reachability_ratio(self, start: tuple[int, int]) -> float:
        start = self._position(start, bounded=False)
        if not self.nodes or start not in self.nodes:
            return 0.0
        visited = set()
        queue = collections.deque([start])
        while queue:
            curr = queue.popleft()
            if curr not in visited:
                visited.add(curr)
                queue.extend(self.adjacency.get(curr, set()) - visited)
        return len(visited) / len(self.nodes)

    def to_state(self) -> dict[str, Any]:
        """Return a detached, canonical, versioned continuation state."""
        return {
            "version": 1,
            "nodes": [list(pos) for pos in sorted(self.nodes)],
            "transitions": [
                {"source": list(source), "action": action, "destination": list(destination), "count": count}
                for (source, action), counts in sorted(self.transitions.items())
                for destination, count in sorted(counts.items())
            ],
            "reward_totals": [
                {"position": list(pos), "total_reward": self.reward_totals[pos]}
                for pos in sorted(self.nodes)
            ],
            "total_transitions": self.total_transitions,
        }

    @classmethod
    def from_state(cls, data: dict[str, Any]) -> TopologicalMemoryGraph:
        """Validate a complete state before returning an isolated graph."""
        def keys(value, expected):
            if type(value) is not dict or set(value) != set(expected):
                raise ValueError("invalid topological state fields")

        keys(data, ("version", "nodes", "transitions", "reward_totals", "total_transitions"))
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported topological state version")
        total = data["total_transitions"]
        if type(total) is not int or not 0 <= total <= cls.MAX_TRANSITIONS:
            raise ValueError("invalid topological transition total")
        limit = (cls.MAX_COORDINATE + 1) ** 2
        if type(data["nodes"]) is not list or len(data["nodes"]) > limit:
            raise ValueError("invalid topological nodes")
        graph = cls()
        for position in data["nodes"]:
            position = cls._position(position)
            if position in graph.nodes:
                raise ValueError("duplicate topological node")
            graph.add_node(position)
        transitions = data["transitions"]
        if type(transitions) is not list or len(transitions) > limit * 8:
            raise ValueError("invalid topological transitions")
        for item in transitions:
            keys(item, ("source", "action", "destination", "count"))
            source, destination = cls._transition(item["source"], item["action"], item["destination"])
            count = item["count"]
            if (source not in graph.nodes or destination not in graph.nodes
                    or type(count) is not int or not 1 <= count <= cls.MAX_TRANSITIONS):
                raise ValueError("invalid topological transition entry")
            counts = graph.transitions.setdefault((source, item["action"]), {})
            if destination in counts:
                raise ValueError("duplicate topological transition")
            counts[destination] = count
            graph.adjacency[source].add(destination)
            graph.total_transitions += count
            if graph.total_transitions > cls.MAX_TRANSITIONS:
                raise ValueError("topological transition count exceeds limit")
        if graph.total_transitions != total:
            raise ValueError("topological transition total disagrees with counts")
        rewards = data["reward_totals"]
        if type(rewards) is not list or len(rewards) != len(graph.nodes):
            raise ValueError("invalid topological rewards")
        seen = set()
        for item in rewards:
            keys(item, ("position", "total_reward"))
            position = cls._position(item["position"])
            if position not in graph.nodes or position in seen:
                raise ValueError("invalid topological reward position")
            seen.add(position)
            graph.reward_totals[position] = cls._number(item["total_reward"], "reward total")
        return graph

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": sum(len(neighbors) for neighbors in self.adjacency.values()),
            "total_transitions": self.total_transitions,
            "nodes": [list(pos) for pos in sorted(self.nodes)],
        }


class TriTierMemory:
    """Biologically grounded 3-tier memory system with matched lesion assays."""

    def __init__(
        self,
        capacity: int = 128,
        lesion_mode: str = "intact",
    ) -> None:
        if lesion_mode not in LESION_MODES:
            raise ValueError(f"lesion_mode must be one of {LESION_MODES}, got {lesion_mode}")
        self.capacity = int(capacity)
        self.lesion_mode = lesion_mode

        # Tier 1: Working Memory context vector
        self.working_context = [0.0] * 8
        self.last_predicted_state: list[float] | None = None

        # Tier 2: Episodic memory store
        self.episodes: collections.deque[EpisodicItem] = collections.deque(maxlen=self.capacity)

        # Tier 3: Cognitive spatial map (x, z) -> stats & Topological graph
        self.spatial_map: dict[str, dict[str, Any]] = {}
        self.topological_graph = TopologicalMemoryGraph()
        self.total_recorded_steps: int = 0

    def reset(self) -> None:
        self.working_context = [0.0] * 8
        self.last_predicted_state = None
        self.episodes.clear()
        self.reset_spatial()
        self.total_recorded_steps = 0

    def reset_spatial(self) -> None:
        """Discard coordinates when the arena changes, retaining episodic policy."""
        self.spatial_map.clear()
        self.topological_graph = TopologicalMemoryGraph()

    def set_lesion_mode(self, mode: str) -> None:
        if mode not in LESION_MODES:
            raise ValueError(f"lesion_mode must be one of {LESION_MODES}, got {mode}")
        self.lesion_mode = mode

    def record_step(
        self,
        step: int,
        agent_pos: tuple[int, int],
        action: int,
        reward: float,
        state_features: Sequence[float],
        surprise: float = 0.0,
        *,
        transition_start: tuple[int, int] | None = None,
    ) -> None:
        """Record an observation and, when supplied, its actual pre-action pose.

        Legacy callers without ``transition_start`` still record observations;
        consecutive post-action poses alone do not establish an executed edge.
        """
        agent_pos = TopologicalMemoryGraph._position(agent_pos)
        reward = TopologicalMemoryGraph._number(reward, "reward")
        if type(action) is not int or not 0 <= action < 4:
            raise ValueError("invalid memory action")
        if self.lesion_mode == "memory_free":
            self.total_recorded_steps += 1
            return
        if transition_start is not None:
            self.topological_graph.record_transition(transition_start, action, agent_pos, reward)
        else:
            self.topological_graph.add_node(agent_pos)
        self.total_recorded_steps += 1
        pos_key = f"{agent_pos[0]},{agent_pos[1]}"

        # Tier 3: Spatial Map update
        if pos_key not in self.spatial_map:
            self.spatial_map[pos_key] = {
                "visits": 0,
                "first_step": step,
                "last_step": step,
                "total_reward": 0.0,
            }
        self.spatial_map[pos_key]["visits"] += 1
        self.spatial_map[pos_key]["last_step"] = step
        self.spatial_map[pos_key]["total_reward"] += float(reward)

        # Tier 1: Working Memory update
        if self.lesion_mode == "erased":
            self.working_context = [0.0] * 8
        else:
            # Recurrent leaky context update driven by surprise
            alpha = max(0.1, min(0.9, 0.4 + 0.5 * surprise))
            for i in range(min(8, len(state_features))):
                self.working_context[i] = (1.0 - alpha) * self.working_context[i] + alpha * float(state_features[i])

        # Tier 2: Episodic Store update
        item = EpisodicItem(
            step=step,
            state=[round(float(v), 4) for v in state_features[:8]],
            action=int(action),
            reward=float(reward),
            surprise=float(surprise),
        )
        self.episodes.append(item)

    def retrieve_context(self, current_features: Sequence[float]) -> dict[str, Any]:
        """Query episodic store and spatial map for relevant context."""
        if self.lesion_mode == "memory_free":
            return {
                "working_context": [0.0] * 8,
                "retrieved_item": None,
                "spatial_familiarity": 0.0,
                "mode": self.lesion_mode,
            }
        if self.lesion_mode == "erased":
            return {
                "working_context": [0.0] * 8,
                "retrieved_item": None,
                "spatial_familiarity": 0.0,
                "mode": self.lesion_mode,
            }

        if not self.episodes:
            return {
                "working_context": list(self.working_context),
                "retrieved_item": None,
                "spatial_familiarity": 0.0,
                "mode": self.lesion_mode,
            }

        if self.lesion_mode == "random_retrieval":
            chosen = random.choice(list(self.episodes))
        elif self.lesion_mode == "shifted":
            # Circular shift by half capacity
            items = list(self.episodes)
            idx = (len(items) // 2) % len(items)
            chosen = items[idx]
        else:  # "intact"
            # Value-gated similarity matching
            best_score = -float("inf")
            chosen = None
            cur = list(current_features[:8])
            for item in reversed(list(self.episodes)):
                # Cosine similarity approximation
                diff_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(cur, item.state)))
                sim = 1.0 / (1.0 + diff_norm)
                # Value weight: reward + surprise + recency
                age = max(1, self.total_recorded_steps - item.step)
                recency = math.exp(-age / 32.0)
                score = sim * (1.0 + abs(item.reward) * 2.0 + item.surprise * 1.5 + recency * 0.5)
                if score > best_score:
                    best_score = score
                    chosen = item

        retrieved_dict = {
            "step": chosen.step,
            "action": chosen.action,
            "reward": chosen.reward,
            "surprise": chosen.surprise,
        } if chosen else None

        # Familiarity: fraction of mapped cells with >= 2 visits
        mapped_count = len(self.spatial_map)
        familiar_count = sum(1 for v in self.spatial_map.values() if v["visits"] >= 2)
        familiarity = familiar_count / max(1, mapped_count)

        return {
            "working_context": list(self.working_context),
            "retrieved_item": retrieved_dict,
            "spatial_familiarity": round(familiarity, 3),
            "mapped_cells": mapped_count,
            "mode": self.lesion_mode,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "lesion_mode": self.lesion_mode,
            "working_context": [round(v, 4) for v in self.working_context],
            "episodes": [
                {
                    "step": ep.step,
                    "state": ep.state,
                    "action": ep.action,
                    "reward": ep.reward,
                    "surprise": ep.surprise,
                }
                for ep in self.episodes
            ],
            "spatial_map": dict(self.spatial_map),
            "total_recorded_steps": self.total_recorded_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriTierMemory:
        if not isinstance(data, dict):
            raise ValueError("TriTierMemory data must be a dictionary")
        capacity = int(data.get("capacity", 128))
        lesion_mode = str(data.get("lesion_mode", "intact"))
        mem = cls(capacity=capacity, lesion_mode=lesion_mode)
        ctx = data.get("working_context")
        if isinstance(ctx, list):
            mem.working_context = [float(v) for v in ctx]
        ep_list = data.get("episodes")
        if isinstance(ep_list, list):
            for ep in ep_list:
                mem.episodes.append(
                    EpisodicItem(
                        step=int(ep["step"]),
                        state=[float(v) for v in ep["state"]],
                        action=int(ep["action"]),
                        reward=float(ep["reward"]),
                        surprise=float(ep.get("surprise", 0.0)),
                    )
                )
        sp = data.get("spatial_map")
        if isinstance(sp, dict):
            mem.spatial_map = {str(k): dict(v) for k, v in sp.items()}
        mem.total_recorded_steps = int(data.get("total_recorded_steps", 0))
        return mem
