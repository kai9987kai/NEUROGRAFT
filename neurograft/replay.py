"""Bounded empirical goal-conditioned value replay.

This is model-based tabular planning over experienced transitions. It does not
change neural weights or model biological replay. In particular, even disabled
replay can assign Q=1 to an observed transition directly into the public goal:
that immediate outcome comes from the empirical model, without a value backup.
"""
from __future__ import annotations

import copy
import heapq
import math
from typing import Any

from .memory import TopologicalMemoryGraph


Position = tuple[int, int]
Model = dict[Position, dict[int, tuple[tuple[Position, float], ...]]]


class ReplayPlanner:
    """One bounded goal-conditioned value vector and deterministic replay cursor.

    For an observed action, Q(u,a) = sum_v P(v|u,a) *
    [1(v=goal) + discount * 1(v!=goal) * V(v)]. Unobserved actions
    have Q=0, and the terminal goal has V=0. Prioritized replay schedules the
    largest absolute Bellman residual, breaking ties by coordinate. Uniform
    replay cycles through sorted nonterminal states, including zero residuals.
    A budget bounds state backups; it does not promise convergence.
    """

    MAX_BUDGET = 4096
    MAX_NODES = (TopologicalMemoryGraph.MAX_COORDINATE + 1) ** 2
    MAX_COUNTER = 2**53 - 1
    TRACE_LIMIT = 32
    TOLERANCE = 1e-12
    STRATEGIES = ("prioritized", "uniform", "none")
    SUMMARY_FIELDS = frozenset((
        "strategy", "budget", "backups", "residual_before", "residual_after",
        "value_delta", "converged", "goal_known", "model_nodes", "model_transitions",
    ))

    def __init__(self, discount: float = 0.9) -> None:
        self.discount = self._number(discount, "discount", upper=1.0, exclusive=True)
        self.values: dict[Position, float] = {}
        self.goal: Position | None = None
        self.uniform_cursor = 0
        self.total_backups = 0
        self.last_summary: dict[str, Any] | None = None

    @staticmethod
    def _number(value: Any, name: str, *, upper: float = 1.0,
                exclusive: bool = False) -> float:
        if type(value) not in (int, float):
            raise ValueError(f"invalid replay {name}")
        try:
            result = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f"invalid replay {name}") from None
        if not math.isfinite(result) or result < 0 or (result >= upper if exclusive else result > upper):
            raise ValueError(f"invalid replay {name}")
        return result

    @staticmethod
    def _integer(value: Any, name: str, upper: int) -> int:
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError(f"invalid replay {name}")
        return value

    @staticmethod
    def _position(value: Any) -> Position:
        return TopologicalMemoryGraph._position(value)

    @staticmethod
    def _fields(value: Any, fields: set[str] | frozenset[str] | tuple[str, ...]) -> None:
        if type(value) is not dict or set(value) != set(fields):
            raise ValueError("invalid replay state fields")

    @classmethod
    def _model(cls, graph: TopologicalMemoryGraph) -> tuple[Model, int]:
        """Copy only observed transitions, validating before planner mutation.

        Adjacency, rewards and the graph's legacy estimated values are not used.
        The observation total is recomputed from the current count model so edge
        removal also works when a caller replaces or edits its empirical graph.
        """
        if not isinstance(graph, TopologicalMemoryGraph):
            raise ValueError("invalid replay graph")
        if type(graph.nodes) is not set or len(graph.nodes) > cls.MAX_NODES:
            raise ValueError("invalid replay graph nodes")
        nodes = {cls._position(position) for position in graph.nodes}
        if type(graph.transitions) is not dict or len(graph.transitions) > cls.MAX_NODES * 4:
            raise ValueError("invalid replay graph transitions")
        model: Model = {position: {} for position in sorted(nodes)}
        observed_total = 0
        # Validate keys before sorting to turn malformed keys into ValueErrors.
        entries = []
        for key, counts in graph.transitions.items():
            if type(key) is not tuple or len(key) != 2:
                raise ValueError("invalid replay transition key")
            source, action = cls._position(key[0]), key[1]
            cls._integer(action, "action", 3)
            if source not in nodes or type(counts) is not dict or not 1 <= len(counts) <= 2:
                raise ValueError("invalid replay transition outcomes")
            checked_counts = {}
            for destination, count in counts.items():
                destination = cls._position(destination)
                TopologicalMemoryGraph._transition(source, action, destination)
                if destination not in nodes or type(count) is not int or not 1 <= count <= cls.MAX_COUNTER:
                    raise ValueError("invalid replay transition outcome")
                checked_counts[destination] = count
            total = sum(checked_counts.values())
            observed_total += total
            if observed_total > cls.MAX_COUNTER:
                raise ValueError("replay observation count exceeds limit")
            entries.append((source, action, checked_counts, total))
        for source, action, counts, total in sorted(entries):
            model[source][action] = tuple((destination, count / total)
                                          for destination, count in sorted(counts.items()))
        return model, observed_total

    def _q(self, model: Model, position: Position, goal: Position,
           values: dict[Position, float]) -> dict[int, float]:
        scores = dict.fromkeys(range(4), 0.0)
        if position == goal or position not in model or goal not in model:
            return scores
        for action, outcomes in model[position].items():
            scores[action] = min(1.0, math.fsum(
                probability * (1.0 if destination == goal
                               else self.discount * values.get(destination, 0.0))
                for destination, probability in outcomes
            ))
        return scores

    def action_values(self, graph: TopologicalMemoryGraph, current: Position,
                      goal: Position) -> dict[int, float]:
        """Read observed action expectations without backups or state changes.

        A different goal reads an empty value vector. Observed immediate goal
        outcomes still contribute reward, including when replay is disabled.
        """
        current, goal = self._position(current), self._position(goal)
        model, _ = self._model(graph)
        values = self.values if goal == self.goal else {}
        return self._q(model, current, goal, values)

    def replay(self, graph: TopologicalMemoryGraph, goal: Position, budget: int = 32,
               strategy: str = "prioritized") -> dict[str, Any]:
        """Apply at most ``budget`` state backups and return detached diagnostics."""
        goal = self._position(goal)
        budget = self._integer(budget, "budget", self.MAX_BUDGET)
        if type(strategy) is not str or strategy not in self.STRATEGIES:
            raise ValueError("invalid replay strategy")
        model, observation_count = self._model(graph)
        known = goal in model
        if known and strategy != "none" and self.total_backups + budget > self.MAX_COUNTER:
            raise ValueError("replay backup counter exceeds limit")

        # Reconcile against the supplied model; never retain removed nodes or
        # carry an old goal's values into a relocated-goal experiment.
        if goal != self.goal:
            self.values = {}
            self.uniform_cursor = 0
        self.goal = goal
        self.values = {position: (0.0 if position == goal else self.values.get(position, 0.0))
                       for position in model} if known else {}
        states = [position for position in model if position != goal] if known else []
        self.uniform_cursor = self.uniform_cursor % len(states) if states else 0
        initial = self.values.copy()

        def target(position: Position) -> float:
            return max(self._q(model, position, goal, self.values).values())

        def residual(position: Position) -> float:
            return abs(target(position) - self.values[position])

        def max_residual() -> float:
            return max((residual(position) for position in states), default=0.0)

        before = max_residual()
        trace: list[dict[str, Any]] = []
        backups = 0

        def backup(position: Position) -> None:
            nonlocal backups
            old_value = self.values[position]
            self.values[position] = target(position)
            backups += 1
            if len(trace) < self.TRACE_LIMIT:
                trace.append({"position": list(position), "before": old_value,
                              "after": self.values[position]})

        if strategy == "uniform" and states:
            for _ in range(budget):
                backup(states[self.uniform_cursor])
                self.uniform_cursor = (self.uniform_cursor + 1) % len(states)
        elif strategy == "prioritized" and states and budget:
            predecessors: dict[Position, set[Position]] = {position: set() for position in model}
            for source, actions in model.items():
                for outcomes in actions.values():
                    for destination, _ in outcomes:
                        predecessors[destination].add(source)
            # Versions invalidate earlier heap entries after a predecessor's
            # residual changes, including decreases and collision self-loops.
            versions = dict.fromkeys(states, 0)
            heap: list[tuple[float, Position, int]] = []

            def enqueue(position: Position) -> None:
                if position == goal:
                    return
                versions[position] += 1
                priority = residual(position)
                if priority > self.TOLERANCE:
                    heapq.heappush(heap, (-priority, position, versions[position]))

            for position in states:
                enqueue(position)
            while heap and backups < budget:
                _, position, version = heapq.heappop(heap)
                if version != versions[position]:
                    continue
                backup(position)
                # A backup affects its own residual and all predecessors. Sort
                # them so graph insertion order cannot influence continuation.
                for predecessor in sorted(predecessors[position] | {position}):
                    enqueue(predecessor)

        after = max_residual()
        self.total_backups += backups
        self.last_summary = {
            "strategy": strategy, "budget": budget, "backups": backups,
            "residual_before": before, "residual_after": after,
            "value_delta": max((abs(value - initial[position])
                                for position, value in self.values.items()), default=0.0),
            "converged": known and after <= self.TOLERANCE,
            "goal_known": known, "model_nodes": len(model),
            "model_transitions": observation_count, "trace": trace,
        }
        return copy.deepcopy(self.last_summary)

    def to_state(self) -> dict[str, Any]:
        """Return detached canonical continuation data, including the replay cursor."""
        return {
            "version": 1, "discount": self.discount,
            "goal": list(self.goal) if self.goal is not None else None,
            "values": [{"position": list(position), "value": value}
                       for position, value in sorted(self.values.items())],
            "uniform_cursor": self.uniform_cursor, "total_backups": self.total_backups,
            "last_summary": copy.deepcopy(self.last_summary),
        }

    @classmethod
    def _summary(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if (type(value) is not dict or not cls.SUMMARY_FIELDS <= set(value)
                or set(value) - cls.SUMMARY_FIELDS - {"trace"}):
            raise ValueError("invalid replay summary fields")
        if type(value["strategy"]) is not str or value["strategy"] not in cls.STRATEGIES:
            raise ValueError("invalid replay summary strategy")
        budget = cls._integer(value["budget"], "summary budget", cls.MAX_BUDGET)
        backups = cls._integer(value["backups"], "summary backups", budget)
        if value["strategy"] == "none" and backups:
            raise ValueError("disabled replay summary contains backups")
        for name in ("residual_before", "residual_after", "value_delta"):
            cls._number(value[name], name)
        for name in ("goal_known", "converged"):
            if type(value[name]) is not bool:
                raise ValueError(f"invalid replay summary {name}")
        cls._integer(value["model_nodes"], "summary node count", cls.MAX_NODES)
        cls._integer(value["model_transitions"], "summary observation count", cls.MAX_COUNTER)
        if value["converged"] != (value["goal_known"] and value["residual_after"] <= cls.TOLERANCE):
            raise ValueError("inconsistent replay convergence summary")
        if not value["goal_known"] and backups:
            raise ValueError("unknown replay goal summary contains backups")
        trace = value.get("trace", [])
        if type(trace) is not list or len(trace) > min(backups, cls.TRACE_LIMIT):
            raise ValueError("invalid replay trace")
        for entry in trace:
            cls._fields(entry, ("position", "before", "after"))
            cls._position(entry["position"])
            cls._number(entry["before"], "trace value")
            cls._number(entry["after"], "trace value")
        return copy.deepcopy(value)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ReplayPlanner:
        """Validate the complete bounded state before returning an isolated planner."""
        cls._fields(state, ("version", "discount", "goal", "values", "uniform_cursor",
                            "total_backups", "last_summary"))
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("unsupported replay state version")
        discount = cls._number(state["discount"], "discount", exclusive=True)
        goal = None if state["goal"] is None else cls._position(state["goal"])
        if type(state["values"]) is not list or len(state["values"]) > cls.MAX_NODES:
            raise ValueError("invalid replay values")
        values = {}
        for entry in state["values"]:
            cls._fields(entry, ("position", "value"))
            position = cls._position(entry["position"])
            value = cls._number(entry["value"], "value")
            if position in values:
                raise ValueError("duplicate replay value position")
            values[position] = value
        if values and (goal is None or goal not in values or values[goal] != 0.0):
            raise ValueError("invalid replay terminal goal value")
        cursor = cls._integer(state["uniform_cursor"], "uniform cursor", cls.MAX_NODES - 1)
        if (not values and cursor) or (values and cursor >= max(1, len(values) - 1)):
            raise ValueError("replay cursor exceeds state count")
        total = cls._integer(state["total_backups"], "backup counter", cls.MAX_COUNTER)
        summary = cls._summary(state["last_summary"])
        if summary is not None:
            if (goal is None or summary["backups"] > total
                    or summary["goal_known"] != bool(values)
                    or (values and summary["model_nodes"] != len(values))):
                raise ValueError("inconsistent replay continuation summary")
        # No input objects are retained, and no live planner is partially
        # installed if any validation above rejects hostile checkpoint data.
        result = cls(discount=discount)
        result.goal, result.values = goal, values
        result.uniform_cursor, result.total_backups = cursor, total
        result.last_summary = summary
        return result

    def snapshot(self) -> dict[str, Any]:
        """Small JSON diagnostics; reading this method never performs replay."""
        return {
            "discount": self.discount,
            "goal": list(self.goal) if self.goal is not None else None,
            "value_count": len(self.values), "max_value": max(self.values.values(), default=0.0),
            "uniform_cursor": self.uniform_cursor, "total_backups": self.total_backups,
            "last_summary": copy.deepcopy(self.last_summary),
        }
