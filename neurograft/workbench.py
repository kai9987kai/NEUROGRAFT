"""Isolated empirical evidence revision crossed with bounded value replay.

Captured aggregate counts have no chronology. A recent model retains those
counts for each source/action until its first explicitly supplied new outcome;
that action then uses only its last W new outcomes. No live session, neural
weights, or policy is mutated. Receipt hashes check consistency, not authorship.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import random
from typing import Any

from .memory import TopologicalMemoryGraph
from .replay import ReplayPlanner


Position = tuple[int, int]


class ReplayWorkbench:
    """Six independent planners sharing a bounded, ordered observation stream.

    ``stale`` says that an arm's replay summary predates supplied observations.
    The current Bellman residual separately measures its value inconsistency.
    MAE includes every empirical node, including the zero-valued terminal goal.
    Action agreement uses the lowest-index greedy action at states with a unique
    empirical oracle optimum (gap > 1e-10); no decisive states gives null.
    Unknown goals give null metrics and no invented values.

    In addition to the specified snapshot fields, ``fixture`` gives optional
    synthetic-fixture control hints, or null for supplied graphs. Reads may fill
    a pure oracle cache but never change experiment state or exported hashes.
    """

    VERSION = 1
    MAX_NODES = 256
    MAX_OPERATIONS = 128
    MAX_EVENTS = 2048
    MAX_BUDGET = 256
    MAX_REQUESTED_BACKUPS = 4096
    MAX_RECEIPT_BYTES = 2 * 1024 * 1024
    MODELS = ("cumulative", "recent")
    SCHEDULES = ("prioritized", "uniform", "none")
    EVIDENCE_SCOPE = (
        "Synthetic or frozen captured empirical transitions with explicitly supplied experimental observations. "
        "Recent windows contain only newly supplied ordered outcomes, never invented pre-capture chronology. "
        "Empirical oracle agreement and Bellman residual measure model planning consistency; they do not establish "
        "current-world accuracy, biological replay, policy learning, or neural training. "
        "Receipt SHA256 hashes check deterministic consistency, not authorship."
    )

    def __init__(self, graph: TopologicalMemoryGraph | None = None,
                 goal: tuple | None = None, provenance: dict | None = None,
                 window: int = 8, seed: int = 0) -> None:
        if type(window) is not int or window not in (4, 8, 16):
            raise ValueError("workbench window must be 4, 8, or 16")
        self._integer(seed, "seed", 2**32 - 1)
        if graph is None:
            graph = self._fixture_graph(seed)
            if goal is None:
                goal = (5, 1)
            if provenance is None:
                provenance = {"kind": "synthetic-fixture", "seed": seed}
        if not isinstance(graph, TopologicalMemoryGraph):
            raise ValueError("invalid workbench graph")
        try:
            isolated = TopologicalMemoryGraph.from_state(graph.to_state())
        except (KeyError, TypeError, OverflowError):
            raise ValueError("invalid workbench graph state") from None
        if len(isolated.nodes) > self.MAX_NODES:
            raise ValueError("workbench node limit exceeded")
        if goal is None:
            goal = max(isolated.nodes, default=(0, 0))
        self.goal = TopologicalMemoryGraph._position(goal)
        self.window = window
        if provenance is None:
            provenance = {"kind": "supplied-graph"}
        if type(provenance) is not dict:
            raise ValueError("invalid workbench provenance")
        self.provenance = self._json_copy(provenance)
        self._initial = {"graph": isolated.to_state(), "goal": list(self.goal),
                         "window": self.window, "provenance": copy.deepcopy(self.provenance)}
        if len(self._canonical(self._initial)) > self.MAX_RECEIPT_BYTES:
            raise ValueError("workbench initial state exceeds receipt size limit")
        self._models = {model: TopologicalMemoryGraph.from_state(isolated.to_state())
                        for model in self.MODELS}
        self._planners = {model: {schedule: ReplayPlanner() for schedule in self.SCHEDULES}
                          for model in self.MODELS}
        self._stale = {model: dict.fromkeys(self.SCHEDULES, False) for model in self.MODELS}
        self._windows: dict[tuple[Position, int], list[Position]] = {}
        self._history: list[dict[str, Any]] = []
        self._events = 0
        self._requested_backups = 0
        self._oracle_cache: dict[str, tuple[dict[Position, float], dict[Position, tuple[float, ...]]]] = {}
        self._refresh(0)

    @staticmethod
    def _integer(value: Any, name: str, upper: int, lower: int = 0) -> int:
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"invalid workbench {name}")
        return value

    @staticmethod
    def _fields(value: Any, expected: tuple[str, ...]) -> None:
        if type(value) is not dict or set(value) != set(expected):
            raise ValueError("invalid workbench fields")

    @staticmethod
    def _canonical(value: Any) -> bytes:
        """Canonical UTF-8 JSON; reject types JSON would silently coerce."""
        def check(item: Any, depth: int = 0) -> None:
            if depth > 64:
                raise ValueError("workbench JSON nesting exceeds limit")
            if item is None or type(item) in (str, bool, int):
                return
            if type(item) is float and math.isfinite(item):
                return
            if type(item) is list:
                for part in item:
                    check(part, depth + 1)
                return
            if type(item) is dict and all(type(key) is str for key in item):
                for part in item.values():
                    check(part, depth + 1)
                return
            raise ValueError("workbench data must contain finite JSON values")

        try:
            check(value)
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("utf-8")
        except (TypeError, OverflowError, RecursionError, UnicodeError):
            raise ValueError("invalid workbench JSON data") from None

    @classmethod
    def _json_copy(cls, value: Any) -> Any:
        return json.loads(cls._canonical(value))

    @classmethod
    def _digest(cls, value: Any) -> str:
        return hashlib.sha256(cls._canonical(value)).hexdigest()

    @classmethod
    def fixture(cls, seed: int = 0, window: int = 8) -> ReplayWorkbench:
        return cls(seed=seed, window=window)

    @staticmethod
    def _fixture_graph(seed: int) -> TopologicalMemoryGraph:
        rng = random.Random(seed)
        graph = TopologicalMemoryGraph()
        routes = (
            ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
            ((1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 2), (5, 1)),
        )
        for route in routes:
            for source, destination in zip(route, route[1:]):
                delta = (destination[0] - source[0], destination[1] - source[1])
                action = TopologicalMemoryGraph.ACTIONS.index(delta)
                for _ in range(rng.randint(6, 12)):
                    graph.record_transition(source, action, destination)
        # A genuine collision outcome alongside a successful executed move.
        for _ in range(rng.randint(1, 3)):
            graph.record_transition((2, 1), 1, (2, 1))
        return graph

    def _fixture_details(self) -> dict[str, Any] | None:
        if self.provenance.get("kind") != "synthetic-fixture":
            return None
        seed = self.provenance.get("seed")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            return None
        return {"seed": seed, "start": [1, 1], "gate": [3, 1],
                "suggested_observation": {"action": "observe", "source": [3, 1],
                                          "action_index": 1, "destination": [3, 1],
                                          "count": self.window}}

    def _refresh(self, budget: int) -> None:
        for model in self.MODELS:
            for schedule in self.SCHEDULES:
                self._planners[model][schedule].replay(
                    self._models[model], self.goal, budget=budget, strategy=schedule)
                self._stale[model][schedule] = False

    @classmethod
    def _command(cls, command: Any) -> dict[str, Any]:
        if type(command) is not dict or type(command.get("action")) is not str:
            raise ValueError("invalid workbench command")
        action = command["action"]
        if action == "reset":
            cls._fields(command, ("action",))
        elif action == "goal":
            cls._fields(command, ("action", "x", "z"))
            TopologicalMemoryGraph._position([command["x"], command["z"]])
        elif action == "replay":
            cls._fields(command, ("action", "budget"))
            cls._integer(command["budget"], "budget", cls.MAX_BUDGET)
        elif action == "observe":
            cls._fields(command, ("action", "source", "action_index", "destination", "count"))
            source, destination = TopologicalMemoryGraph._transition(
                command["source"], command["action_index"], command["destination"])
            cls._integer(command["count"], "observation count", 16, 1)
            command = {**command, "source": list(source), "destination": list(destination)}
        else:
            raise ValueError("unknown workbench action")
        return cls._json_copy(command)

    def apply(self, command: dict) -> dict[str, Any]:
        """Validate and execute on a complete candidate before installing it."""
        command = self._command(command)
        if command["action"] == "reset":
            candidate = type(self)(TopologicalMemoryGraph.from_state(self._initial["graph"]),
                                   goal=self._initial["goal"], window=self._initial["window"],
                                   provenance=self._initial["provenance"])
        else:
            if len(self._history) >= self.MAX_OPERATIONS:
                raise ValueError("workbench operation limit exceeded; reset or export the experiment")
            candidate = copy.deepcopy(self)
            candidate._execute(command)
            candidate._history.append(command)
        result = candidate.snapshot()
        # A successful experiment must remain exportable as a bounded receipt.
        candidate.export_receipt()
        self.__dict__ = candidate.__dict__
        return result

    def _execute(self, command: dict[str, Any]) -> None:
        action = command["action"]
        if action == "goal":
            self.goal = (command["x"], command["z"])
            self._refresh(0)
        elif action == "replay":
            budget = command["budget"]
            if self._requested_backups + budget > self.MAX_REQUESTED_BACKUPS:
                raise ValueError("workbench requested backup limit exceeded")
            self._requested_backups += budget
            self._refresh(budget)
        else:
            count = command["count"]
            if self._events + count > self.MAX_EVENTS:
                raise ValueError("workbench event limit exceeded")
            source, destination = tuple(command["source"]), tuple(command["destination"])
            action_index = command["action_index"]
            nodes = self._models["cumulative"].nodes | {source, destination}
            if len(nodes) > self.MAX_NODES:
                raise ValueError("workbench node limit exceeded")
            for _ in range(count):
                self._models["cumulative"].record_transition(source, action_index, destination)
            recent = self._models["recent"]
            recent.add_node(source)
            recent.add_node(destination)
            key = (source, action_index)
            outcomes = (self._windows.get(key, []) + [destination] * count)[-self.window:]
            self._windows[key] = outcomes
            recent.transitions[key] = dict(Counter(outcomes))
            recent.total_transitions = sum(sum(counts.values()) for counts in recent.transitions.values())
            recent.adjacency = {position: set() for position in recent.nodes}
            for (origin, _), counts in recent.transitions.items():
                recent.adjacency[origin].update(counts)
            self._events += count
            for model in self.MODELS:
                for schedule in self.SCHEDULES:
                    self._stale[model][schedule] = True

    @classmethod
    def _model_snapshot(cls, graph: TopologicalMemoryGraph) -> dict[str, Any]:
        model = {"nodes": [list(position) for position in sorted(graph.nodes)],
                 "edges": [{"source": list(source), "action": action,
                            "destination": list(destination), "count": count}
                           for (source, action), counts in sorted(graph.transitions.items())
                           for destination, count in sorted(counts.items())],
                 "total_transitions": graph.total_transitions}
        return {**model, "sha256": cls._digest(model)}

    @staticmethod
    def _action_values(graph: TopologicalMemoryGraph, goal: Position,
                       values: dict[Position, float], position: Position) -> tuple[float, ...]:
        if position == goal or goal not in graph.nodes:
            return (0.0, 0.0, 0.0, 0.0)
        scores = []
        for action in range(4):
            counts = graph.transitions.get((position, action), {})
            total = sum(counts.values())
            scores.append(min(1.0, math.fsum(
                (count / total) * (1.0 if destination == goal else 0.9 * values.get(destination, 0.0))
                for destination, count in sorted(counts.items()))) if total else 0.0)
        return tuple(scores)

    def _oracle(self, graph: TopologicalMemoryGraph, digest: str):
        key = f"{digest}:{self.goal[0]},{self.goal[1]}"
        if key not in self._oracle_cache:
            # Independent synchronous contraction iteration, never planner replay.
            values = dict.fromkeys(sorted(graph.nodes), 0.0)
            for _ in range(512):
                updated = {position: max(self._action_values(graph, self.goal, values, position))
                           for position in values}
                delta = max((abs(updated[position] - values[position]) for position in values), default=0.0)
                values = updated
                if delta <= 1e-14:
                    break
            q_values = {position: self._action_values(graph, self.goal, values, position)
                        for position in values}
            # Bound a cache whose contents have no role in exported state.
            if len(self._oracle_cache) >= 8:
                self._oracle_cache.clear()
            self._oracle_cache[key] = (values, q_values)
        return self._oracle_cache[key]

    def _metrics(self, graph: TopologicalMemoryGraph, digest: str,
                 planner: ReplayPlanner) -> dict[str, Any]:
        if self.goal not in graph.nodes:
            return {"value_mae": None, "bellman_residual": None,
                    "action_agreement": None, "decisive_states": 0}
        oracle, q_oracle = self._oracle(graph, digest)
        mae = math.fsum(abs(planner.values.get(position, 0.0) - value)
                        for position, value in oracle.items()) / len(oracle)
        residual, decisive, agreed = 0.0, 0, 0
        for position in sorted(graph.nodes):
            if position == self.goal:
                continue
            scores = self._action_values(graph, self.goal, planner.values, position)
            residual = max(residual, abs(max(scores) - planner.values.get(position, 0.0)))
            ranked = sorted(range(4), key=lambda action: (-q_oracle[position][action], action))
            if q_oracle[position][ranked[0]] - q_oracle[position][ranked[1]] > 1e-10:
                decisive += 1
                selected = max(range(4), key=lambda action: (scores[action], -action))
                agreed += selected == ranked[0]
        return {"value_mae": mae, "bellman_residual": residual,
                "action_agreement": agreed / decisive if decisive else None,
                "decisive_states": decisive}

    def snapshot(self) -> dict[str, Any]:
        """Return detached diagnostics without replay or experiment mutation."""
        models = {name: self._model_snapshot(graph) for name, graph in self._models.items()}
        arms = {}
        for model in self.MODELS:
            graph = self._models[model]
            arms[model] = {}
            for schedule in self.SCHEDULES:
                planner = self._planners[model][schedule]
                arms[model][schedule] = {
                    "values": [{"position": list(position), "value": planner.values.get(position, 0.0)}
                               for position in sorted(graph.nodes)] if self.goal in graph.nodes else [],
                    "total_backups": planner.total_backups,
                    "summary": copy.deepcopy(planner.last_summary),
                    "stale": self._stale[model][schedule],
                    "metrics": self._metrics(graph, models[model]["sha256"], planner),
                }
        return {"version": self.VERSION, "goal": list(self.goal), "window": self.window,
                "provenance": copy.deepcopy(self.provenance), "events": self._events,
                "operations": len(self._history), "models": models, "arms": arms,
                "history": copy.deepcopy(self._history), "evidence_scope": self.EVIDENCE_SCOPE,
                "fixture": self._fixture_details()}

    def export_receipt(self) -> dict[str, Any]:
        receipt = {"version": self.VERSION, "initial": copy.deepcopy(self._initial),
                   "operations": copy.deepcopy(self._history),
                   "final_sha256": self._digest(self.snapshot())}
        if len(self._canonical(receipt)) > self.MAX_RECEIPT_BYTES:
            raise ValueError("workbench receipt exceeds size limit")
        return receipt

    @classmethod
    def from_receipt(cls, receipt: dict) -> ReplayWorkbench:
        """Re-execute a strict bounded ledger and verify its complete snapshot."""
        cls._fields(receipt, ("version", "initial", "operations", "final_sha256"))
        if len(cls._canonical(receipt)) > cls.MAX_RECEIPT_BYTES:
            raise ValueError("workbench receipt exceeds size limit")
        if type(receipt["version"]) is not int or receipt["version"] != cls.VERSION:
            raise ValueError("unsupported workbench receipt version")
        initial = receipt["initial"]
        cls._fields(initial, ("graph", "goal", "window", "provenance"))
        TopologicalMemoryGraph._position(initial["goal"])
        if type(initial["provenance"]) is not dict:
            raise ValueError("invalid workbench initial provenance")
        digest = receipt["final_sha256"]
        if (type(digest) is not str or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("invalid workbench final digest")
        operations = receipt["operations"]
        if type(operations) is not list or len(operations) > cls.MAX_OPERATIONS:
            raise ValueError("invalid workbench operation ledger")
        graph = TopologicalMemoryGraph.from_state(initial["graph"])
        candidate = cls(graph, goal=initial["goal"], window=initial["window"], provenance=initial["provenance"])
        for operation in operations:
            if type(operation) is dict and operation.get("action") == "reset":
                raise ValueError("reset cannot occur inside a retained workbench ledger")
            candidate.apply(operation)
        if candidate._digest(candidate.snapshot()) != digest:
            raise ValueError("workbench receipt final digest mismatch")
        return candidate
