"""Behavioral checks for isolated, ordered evidence revision experiments."""
import copy
import json

import pytest

from neurograft.memory import TopologicalMemoryGraph


def workbench(*args, **kwargs):
    from neurograft.workbench import ReplayWorkbench

    return ReplayWorkbench(*args, **kwargs)


def restore(receipt):
    from neurograft.workbench import ReplayWorkbench

    return ReplayWorkbench.from_receipt(receipt)


def chain():
    graph = TopologicalMemoryGraph()
    for x in range(3):
        graph.record_transition((x, 0), 1, (x + 1, 0))
    return graph


def values(snapshot, model="recent", schedule="prioritized"):
    return {tuple(row["position"]): row["value"]
            for row in snapshot["arms"][model][schedule]["values"]}


def counts(snapshot, model, source, action):
    return {tuple(row["destination"]): row["count"]
            for row in snapshot["models"][model]["edges"]
            if row["source"] == list(source) and row["action"] == action}


def observe(bench, source=(0, 0), destination=(0, 0), count=1, action=1):
    return bench.apply({"action": "observe", "source": list(source),
                        "action_index": action, "destination": list(destination), "count": count})


def test_six_arms_receive_same_budget_and_chain_values_match_hand_solution():
    bench = workbench(chain(), goal=(3, 0))
    initial = bench.snapshot()
    metrics = initial["arms"]["recent"]["prioritized"]["metrics"]
    assert metrics["value_mae"] == pytest.approx(0.6775)
    assert metrics["bellman_residual"] == 1.0
    assert metrics["decisive_states"] == 3
    assert metrics["action_agreement"] == pytest.approx(1 / 3)
    snapshot = bench.apply({"action": "replay", "budget": 12})
    for model in ("cumulative", "recent"):
        assert values(snapshot, model) == {(0, 0): 0.81, (1, 0): 0.9, (2, 0): 1.0, (3, 0): 0.0}
        assert values(snapshot, model, "uniform") == values(snapshot, model)
        assert all(value == 0.0 for value in values(snapshot, model, "none").values())
        for schedule, want_backups in (("prioritized", 3), ("uniform", 12), ("none", 0)):
            arm = snapshot["arms"][model][schedule]
            assert arm["summary"]["budget"] == 12
            assert arm["total_backups"] == want_backups
        assert snapshot["arms"][model]["prioritized"]["metrics"]["value_mae"] == 0.0


def test_recent_evidence_starts_at_first_new_event_for_each_action():
    graph = chain()
    for _ in range(11):
        graph.record_transition((0, 0), 1, (1, 0))
    bench = workbench(graph, goal=(3, 0), window=4)
    before = bench.snapshot()
    assert before["models"]["recent"] == before["models"]["cumulative"]
    after = observe(bench, count=2)
    assert counts(after, "cumulative", (0, 0), 1) == {(0, 0): 2, (1, 0): 12}
    assert counts(after, "recent", (0, 0), 1) == {(0, 0): 2}
    assert counts(after, "recent", (1, 0), 1) == {(2, 0): 1}
    # New chronology is [collision, collision, success, success, success].
    after = observe(bench, destination=(1, 0), count=3)
    assert counts(after, "recent", (0, 0), 1) == {(0, 0): 1, (1, 0): 3}
    assert after["events"] == 5
    assert after["models"]["recent"]["total_transitions"] == 6


def test_barrier_closure_retains_values_until_replay_then_reopening_restores_them():
    graph = TopologicalMemoryGraph()
    for _ in range(12):
        graph.record_transition((0, 0), 1, (1, 0))
    bench = workbench(graph, goal=(1, 0), window=4)
    bench.apply({"action": "replay", "budget": 8})
    revised = observe(bench, count=4)
    assert values(revised)[(0, 0)] == 1.0
    assert revised["arms"]["recent"]["prioritized"]["stale"]
    assert revised["arms"]["recent"]["prioritized"]["metrics"]["value_mae"] == 0.5
    for _ in range(2):
        revised = bench.apply({"action": "replay", "budget": 256})
    assert values(revised)[(0, 0)] == pytest.approx(0.0, abs=1e-11)
    assert values(revised, "cumulative")[(0, 0)] == pytest.approx(30 / 31, abs=1e-11)
    assert not revised["arms"]["recent"]["prioritized"]["stale"]
    observe(bench, destination=(1, 0), count=4)
    reopened = bench.apply({"action": "replay", "budget": 8})
    assert values(reopened)[(0, 0)] == 1.0
    assert reopened["arms"]["recent"]["none"]["total_backups"] == 0


def test_goal_changes_clear_every_arm_without_backups_and_unknown_goals_are_explicit():
    bench = workbench(chain(), goal=(3, 0))
    before = bench.apply({"action": "replay", "budget": 16})
    switched = bench.apply({"action": "goal", "x": 0, "z": 0})
    for model in switched["arms"]:
        for schedule, arm in switched["arms"][model].items():
            assert all(row["value"] == 0 for row in arm["values"])
            assert arm["total_backups"] == before["arms"][model][schedule]["total_backups"]
            assert arm["summary"]["budget"] == arm["summary"]["backups"] == 0
    unknown = bench.apply({"action": "goal", "x": 30, "z": 30})
    for model in unknown["arms"].values():
        for arm in model.values():
            assert not arm["summary"]["goal_known"]
            assert arm["values"] == []
            assert arm["metrics"] == {"value_mae": None, "bellman_residual": None,
                                       "action_agreement": None, "decisive_states": 0}


def test_source_graph_reads_and_returned_payloads_are_fully_detached():
    graph, provenance = chain(), {"kind": "capture", "details": {"tick": 19}}
    before_graph = graph.to_state()
    bench = workbench(graph, goal=(3, 0), provenance=provenance)
    before = bench.export_receipt()
    for _ in range(3):
        bench.snapshot()
        assert bench.export_receipt() == before
    graph.record_transition((3, 0), 3, (2, 0))
    provenance["details"]["tick"] = 99
    snapshot = bench.snapshot()
    snapshot["models"]["recent"]["edges"][0]["count"] = 100
    snapshot["provenance"]["details"]["tick"] = 100
    receipt = bench.export_receipt()
    receipt["initial"]["graph"]["nodes"].clear()
    assert bench.export_receipt() == before
    observe(bench)
    assert graph.to_state()["total_transitions"] == before_graph["total_transitions"] + 1


def test_receipts_replay_exactly_and_reset_restores_original_experiment():
    bench = workbench(chain(), goal=(3, 0), provenance={"kind": "test"}, window=4)
    original = bench.snapshot()
    bench.apply({"action": "replay", "budget": 2})
    observe(bench, count=3)
    bench.apply({"action": "replay", "budget": 19})
    bench.apply({"action": "goal", "x": 2, "z": 0})
    receipt = json.loads(json.dumps(bench.export_receipt(), allow_nan=False))
    imported = restore(receipt)
    assert imported.snapshot() == bench.snapshot()
    assert imported.export_receipt() == bench.export_receipt()
    imported.apply({"action": "replay", "budget": 1})
    assert imported.snapshot() != bench.snapshot()
    assert bench.apply({"action": "reset"}) == original


@pytest.mark.parametrize("command", [
    None, [], {}, {"action": "unknown"}, {"action": "reset", "extra": 1},
    {"action": "replay", "budget": True}, {"action": "replay", "budget": -1},
    {"action": "replay", "budget": 257}, {"action": "replay", "budget": 1, "extra": 1},
    {"action": "goal", "x": 31, "z": 0}, {"action": "goal", "x": True, "z": 0},
    {"action": "goal", "position": [0, 0]},
    {"action": "observe", "source": [0, 0], "action_index": 1, "destination": [2, 0], "count": 1},
    {"action": "observe", "source": [0, 0], "action_index": True, "destination": [1, 0], "count": 1},
    {"action": "observe", "source": [0, 0], "action_index": 1, "destination": [1, 0], "count": 17},
    {"action": "observe", "source": [0, 0], "action_index": 1, "destination": [1, 0], "count": 0},
])
def test_invalid_commands_cannot_partially_change_experiment(command):
    bench = workbench(chain(), goal=(3, 0))
    before = bench.export_receipt()
    with pytest.raises(ValueError):
        bench.apply(command)
    assert bench.export_receipt() == before


def test_requested_budget_and_event_limits_hold_even_when_planner_does_no_backups():
    bench = workbench(chain(), goal=(30, 30))
    for _ in range(16):
        bench.apply({"action": "replay", "budget": 256})
    before = bench.export_receipt()
    with pytest.raises(ValueError):
        bench.apply({"action": "replay", "budget": 1})
    assert bench.export_receipt() == before
    bench.apply({"action": "reset"})
    for _ in range(128):
        observe(bench, count=16)
    assert bench.snapshot()["events"] == 2048
    before = bench.export_receipt()
    with pytest.raises(ValueError):
        observe(bench)
    assert bench.export_receipt() == before
    assert bench.apply({"action": "reset"})["operations"] == 0


def test_node_bound_and_count_sum_consistency_are_validated_before_use():
    graph = TopologicalMemoryGraph()
    for x in range(16):
        for z in range(16):
            graph.add_node((x, z))
    bench = workbench(graph, goal=(0, 0))
    before = bench.export_receipt()
    with pytest.raises(ValueError):
        observe(bench, source=(15, 15), destination=(16, 15))
    assert bench.export_receipt() == before
    graph.add_node((16, 15))
    with pytest.raises(ValueError):
        workbench(graph, goal=(0, 0))
    inconsistent = chain()
    inconsistent.total_transitions += 1
    with pytest.raises(ValueError):
        workbench(inconsistent, goal=(3, 0))


@pytest.mark.parametrize("mutation", ["count", "digest", "unknown", "initial", "oversized", "reset"])
def test_tampered_or_malformed_receipts_are_rejected_without_touching_original(mutation):
    bench = workbench(chain(), goal=(3, 0))
    observe(bench, count=2)
    before = bench.export_receipt()
    receipt = copy.deepcopy(before)
    if mutation == "count":
        receipt["operations"][0]["count"] = 3
    elif mutation == "digest":
        receipt["final_sha256"] = "0" * 64
    elif mutation == "unknown":
        receipt["extra"] = 1
    elif mutation == "initial":
        receipt["initial"]["graph"]["total_transitions"] += 1
    elif mutation == "oversized":
        receipt["initial"]["provenance"] = {"text": "x" * (2 * 1024 * 1024)}
    else:
        receipt["operations"].append({"action": "reset"})
    with pytest.raises(ValueError):
        restore(receipt)
    assert bench.export_receipt() == before


@pytest.mark.parametrize("kwargs", [{"window": 3}, {"window": True}, {"seed": True},
                                    {"provenance": []}, {"provenance": {"x": float("nan")}},
                                    {"goal": (31, 0)}])
def test_constructor_rejects_malformed_experiment_inputs(kwargs):
    with pytest.raises(ValueError):
        workbench(**kwargs)


def test_seeded_fixture_is_reproducible_and_has_two_routes_and_a_collision_loop():
    first, again = workbench(seed=7), workbench(seed=7)
    assert first.export_receipt() == again.export_receipt()
    snapshot = first.snapshot()
    graph = snapshot["models"]["recent"]
    assert any(row["source"] == row["destination"] for row in graph["edges"])
    start = snapshot["fixture"]["start"]
    assert len({edge["action"] for edge in graph["edges"] if edge["source"] == start}) >= 2
    observation = snapshot["fixture"]["suggested_observation"]
    changed = first.apply(observation)
    assert changed["events"] == observation["count"]


@pytest.mark.parametrize("field", ["goal", "provenance"])
def test_receipt_initial_fields_do_not_accept_constructor_default_shortcuts(field):
    # Both nulls currently resolve to the same defaults, so hash verification
    # alone would accept them despite the receipt's strict initial-state schema.
    receipt = workbench(chain(), goal=(3, 0)).export_receipt()
    receipt["initial"][field] = None
    with pytest.raises(ValueError):
        restore(receipt)


def test_overflowing_captured_counts_reject_new_observation_atomically():
    graph = TopologicalMemoryGraph()
    graph.record_transition((0, 0), 1, (1, 0))
    graph.transitions[((0, 0), 1)][(1, 0)] = 2**53 - 1
    graph.total_transitions = 2**53 - 1
    bench = workbench(graph, goal=(1, 0))
    before = bench.export_receipt()
    with pytest.raises(ValueError):
        observe(bench)
    assert bench.export_receipt() == before
