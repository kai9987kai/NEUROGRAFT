"""Measure complete SR and replay32 controller calls on synthetic directed grids.

Run from a source checkout, for example:
    python tools/benchmark_replay.py --out reports/v13-replay-latency.json

This is a small descriptive controller benchmark, not a neural inference or
full live-tick benchmark. It neither loads nor changes a live checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directed_grid(side: int):
    """Observe each east/south edge once; no reverse edges are fabricated."""
    from neurograft.memory import TopologicalMemoryGraph

    graph = TopologicalMemoryGraph()
    for x in range(side):
        for y in range(side):
            graph.add_node((x, y))
            if x + 1 < side:
                graph.record_transition((x, y), 1, (x + 1, y))
            if y + 1 < side:
                graph.record_transition((x, y), 2, (x, y + 1))
    return graph


def describe(samples: list[dict]) -> dict:
    values = [sample["elapsed_ms"] for sample in samples]
    return {
        "samples": samples,
        "count": len(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def benchmark() -> dict:
    import torch
    from neurograft.navigation_controller import replay_steering_action, sr_steering_action
    from neurograft.replay import ReplayPlanner

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # All actions are deliberately unmasked, as requested. This observation is
    # a synthetic controller input, not a physically complete corner-cell view.
    observation = [[0.0] * 4 for _ in range(4)]
    logits = [0.0] * 4
    cases = []
    method_calls = {"legacy_sr": 0, "replay32": 0}
    for side in (8, 16, 31):
        graph = directed_grid(side)
        initial_model = graph.to_state()
        current, goal = (0, 0), (side - 1, side - 1)
        legacy_samples = []
        legacy_repetitions = 1 if side == 31 else 3
        for repetition in range(legacy_repetitions):
            started = time.perf_counter_ns()
            action = sr_steering_action(observation, logits, graph, current, goal)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            legacy_samples.append({
                "repetition": repetition + 1,
                "elapsed_ms": elapsed,
                "action": action,
                "cold_warm": "first_call_at_this_shape" if repetition == 0 else "repeated_same_shape",
                "process_method_call_index": method_calls["legacy_sr"] + 1,
                "state": "stateless_full_dense_SR_rebuild",
            })
            method_calls["legacy_sr"] += 1

        planner = ReplayPlanner(discount=0.9)
        replay_samples = []
        for repetition in range(3):
            prior_backups = planner.total_backups
            started = time.perf_counter_ns()
            summary = planner.replay(graph, goal, budget=32, strategy="prioritized")
            action = replay_steering_action(observation, logits, planner, graph, current, goal)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            replay_samples.append({
                "repetition": repetition + 1,
                "elapsed_ms": elapsed,
                "action": action,
                "cold_warm": "cold_empty_planner" if repetition == 0 else "warm_continued_planner",
                "process_method_call_index": method_calls["replay32"] + 1,
                "backups_before_call": prior_backups,
                "backups_this_call": summary["backups"],
                "backups_after_call": planner.total_backups,
                "residual_before": summary["residual_before"],
                "residual_after": summary["residual_after"],
                "converged": summary["converged"],
                "goal_known": summary["goal_known"],
                "start_state_value_after_call": planner.values[current],
            })
            method_calls["replay32"] += 1
        if graph.to_state() != initial_model:
            raise RuntimeError("a benchmarked controller modified its empirical model")
        cases.append({
            "nodes": len(graph.nodes), "side": side,
            "observed_transition_outcomes": graph.total_transitions,
            "current": list(current), "goal": list(goal),
            "legacy_sr": describe(legacy_samples),
            "replay32_plus_action": describe(replay_samples),
            "empirical_graph_unchanged": True,
        })

    source_files = (
        "tools/benchmark_replay.py", "neurograft/replay.py",
        "neurograft/navigation_controller.py", "neurograft/memory.py",
    )
    return {
        "schema": "neurograft.replay-controller-latency.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": "python tools/benchmark_replay.py --out reports/v13-replay-latency.json",
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "timer": "time.perf_counter_ns wall clock, converted to milliseconds",
        },
        "source_sha256": {name: digest(ROOT / name) for name in source_files},
        "setup": {
            "graph": "east/south directed square grid, one observation per edge",
            "sizes": [64, 256, 961],
            "goal": "known node at opposite corner from current=(0,0)",
            "discount": 0.9,
            "local_observation": observation,
            "logits": logits,
            "legacy_parameters": {"sr_weight": 1.2, "epistemic_weight": 0.25},
            "replay_parameters": {"strategy": "prioritized", "budget": 32},
            "legacy_repetitions_by_nodes": {"64": 3, "256": 3, "961": 1},
            "replay_repetitions_per_size": 3,
            "ordering": "sizes ascending; legacy calls first, then replay calls at each size",
            "warmup_calls": 0,
            "legacy_warmth": "first call at each shape then repeated dense rebuilds; earlier sizes already ran in the same process",
            "replay_warmth": "new empty planner per size; later calls continue values with 32 more allowed backups each",
        },
        "timed_work": [
            "actual legacy sr_steering_action including dense SR construction/inversion and action selection",
            "actual ReplayPlanner.replay budget32 plus replay_steering_action",
            "replay graph validation/copy, queue construction, predecessor updates, before/after residual scans, summary copy",
            "replay action read including its separate graph validation/copy and local legal-action selection",
        ],
        "excluded_work": [
            "synthetic graph construction and planner construction",
            "all GraphCircuit neural inference, feature preparation, memory recording, world stepping and browser rendering",
            "checkpoint loading/saving, serialization of this report, source hashing and output logging",
        ],
        "caveats": [
            "Small sequential local timing sample; scheduler, CPU clocks and other workloads can change results.",
            "Cold/warm labels describe method/planner state, not a fresh OS process or cleared hardware caches.",
            "Legacy SR and replay optimize different quantities and do different amounts of work; this is not an equal-compute comparison.",
            "A 32-backup replay call can leave most state values unresolved, including the current state's distant-goal value.",
            "Three replay samples mix an empty vector with continuing vectors; inspect every raw sample rather than treating their median as a stationary estimate.",
            "Only one legacy sample is collected at 961 nodes; it does not estimate latency variance.",
            "The synthetic all-legal observation does not represent the boundary walls at the grid corner.",
            "These controller timings do not establish live-tick latency, navigation quality, general speedup or neural-learning improvement.",
        ],
        "live_state_accessed": False,
        "latency_threshold_asserted": False,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/v13-replay-latency.json",
                        help="new JSON receipt path; existing files are preserved")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    report = benchmark()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "receipt": str(output),
        "cases": [{"nodes": case["nodes"],
                   "legacy_raw_ms": [entry["elapsed_ms"] for entry in case["legacy_sr"]["samples"]],
                   "replay32_raw_ms": [entry["elapsed_ms"] for entry in case["replay32_plus_action"]["samples"]]}
                  for case in report["cases"]],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
