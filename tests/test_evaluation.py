"""Behavioral checks for paired synthetic evaluation and replay receipts."""

import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest


def test_evaluation_module_is_available():
    # A missing evaluator prevents any reproducible benchmark from being run.
    assert importlib.util.find_spec("neurograft.evaluation") is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": ()}, {"seeds": (1, 1)}, {"seeds": (True,)},
        {"seeds": (-1,)}, {"epochs": 0}, {"epochs": True},
        {"train_size": 0}, {"val_size": -1}, {"test_size": 0},
        {"batch_size": 0}, {"delay": -1}, {"hidden_dim": 0},
        {"learning_rate": float("nan")}, {"learning_rate": 0},
        {"data_seed": -1}, {"threads": 0}, {"threads": 3},
    ],
)
def test_invalid_configs_are_rejected_before_training(kwargs):
    from neurograft.evaluation import BenchmarkConfig

    with pytest.raises((TypeError, ValueError)):
        BenchmarkConfig(**kwargs)


def test_task_is_seeded_delayed_xor_and_cannot_leak_to_human():
    import torch
    from neurograft.evaluation import generate_task

    observations, labels = generate_task(16, delay=3, seed=102)
    again, again_labels = generate_task(16, delay=3, seed=102)
    other, _ = generate_task(16, delay=3, seed=103)
    assert observations.shape == (16, 5, 4, 4)
    assert labels.dtype == torch.long
    assert torch.equal(observations, again)
    assert torch.equal(labels, again_labels)
    assert not torch.equal(observations, other)
    assert torch.equal(labels, ((observations[:, 0, 0, 0] > 0) ^
                                (observations[:, 0, 1, 0] > 0)).long())
    assert labels.bincount(minlength=2).tolist() == [8, 8]
    assert torch.count_nonzero(observations[:, 1:, :2, :2]) == 0
    assert torch.count_nonzero(observations[:, :, 2]) == 0
    expected_human = torch.zeros(5, 4)
    expected_human[-1, 2] = 1
    assert torch.equal(observations[:, :, 3], expected_human.expand(16, -1, -1))


@pytest.mark.parametrize("args", [(0, 1, 0), (4, -1, 0), (4, 1, -1), (True, 1, 0)])
def test_invalid_task_dimensions_are_rejected(args):
    from neurograft.evaluation import generate_task

    with pytest.raises((TypeError, ValueError)):
        generate_task(*args)


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def completed_benchmark(tmp_path_factory):
    from neurograft.evaluation import BenchmarkConfig, run_benchmark
    from neurograft.fixtures import make_fixture_bundle

    bundle = make_fixture_bundle(seed=2, nodes_per_source=6)
    config = BenchmarkConfig(seeds=(7, 11), epochs=2, train_size=16,
                             val_size=8, test_size=12, batch_size=8,
                             delay=1, hidden_dim=4, threads=2)
    output = tmp_path_factory.mktemp("paired-benchmark") / "run"
    before = bundle.digest()
    report = run_benchmark(bundle, config, output)
    assert bundle.digest() == before
    return bundle, config, output, report


def test_benchmark_pairs_weights_data_steps_and_parameters(completed_benchmark):
    bundle, config, output, report = completed_benchmark
    assert len(report["runs"]) == 8
    assert report["parameter_counts_equal"] is True
    assert report["split_seeds"] == {"train": 5187, "validation": 5188, "test": 5189}
    assert report["input_graph_hash"] == bundle.digest()
    assert json.loads((output / "report.json").read_text()) == report
    for seed in config.seeds:
        rows = [row for row in report["runs"] if row["seed"] == seed]
        assert len({row["initial_state_hash"] for row in rows}) == 1
        assert len({row["data_hashes"]["train"] for row in rows}) == 1
        assert len({row["batch_order_hash"] for row in rows}) == 1
        assert {row["optimizer_steps"] for row in rows} == {4}
        assert len({row["parameter_count"] for row in rows}) == 1
        assert len({row["test_evaluations"] for row in rows}) == 1
        for row in rows:
            assert row["test_evaluations"] == 1
            assert row["selected_epoch"] in (1, 2)
            assert 0 <= row["test_accuracy"] <= 1
            assert row["elapsed_seconds"] > 0
            assert row["memory"]["max_sampled_rss_bytes"] > 0
            metrics = [json.loads(line) for line in
                       (output / row["metrics_path"]).read_text().splitlines()]
            assert [m["epoch"] for m in metrics] == [1, 2]
            assert all("validation_accuracy" in m for m in metrics)
            assert all("test_accuracy" not in m for m in metrics)
        rewired = next(row for row in rows if row["variant"] == "rewired")
        assert rewired["rewire_receipt"]
        assert rewired["graph_hash"] != bundle.digest()
    assert len(report["paired_comparisons"]) == 3
    for comparison in report["paired_comparisons"]:
        assert comparison["n_paired_seeds"] == 2
        assert "small" in comparison["caveat"].lower()
        assert comparison["bootstrap_95_ci"][0] <= comparison["mean_accuracy_difference"]
        assert comparison["mean_accuracy_difference"] <= comparison["bootstrap_95_ci"][1]


def test_checkpoint_replay_matches_recorded_logits_exactly(completed_benchmark):
    import torch
    from neurograft.evaluation import generate_task, load_checkpoint

    bundle, config, output, report = completed_benchmark
    task, _ = generate_task(config.test_size, config.delay, report["split_seeds"]["test"])
    for row in report["runs"]:
        model, metadata = load_checkpoint(output / row["checkpoint_path"], bundle)
        with torch.no_grad():
            replay = model(task, graft_enabled=row["variant"] != "no_graft",
                           recurrence=row["variant"] != "no_recurrence").logits
        assert _tensor_hash(replay) == row["test_logits_hash"]
        assert metadata["selected_epoch"] == row["selected_epoch"]
        assert metadata["graph_hash"] == row["graph_hash"]


def test_report_readiness_and_summary_reflect_fixture_runs(completed_benchmark):
    bundle, _, _, report = completed_benchmark
    assert report["source_readiness"] == bundle.stats()
    assert report["source_readiness"]["authentic_four_source_ready"] is False
    assert {source["status"] for source in report["source_readiness"]["sources"].values()} == {"fixture"}
    assert set(report["summary"]) == {"grafted", "no_graft", "no_recurrence", "rewired"}
    for variant, summary in report["summary"].items():
        scores = [run["test_accuracy"] for run in report["runs"] if run["variant"] == variant]
        assert summary["n_seeds"] == 2
        assert summary["mean_test_accuracy"] == pytest.approx(statistics.mean(scores))
        assert summary["std_test_accuracy"] == pytest.approx(statistics.stdev(scores))
        assert summary["std_ddof"] == 1


def test_repeating_benchmark_reproduces_scores_and_weights(completed_benchmark, tmp_path):
    from neurograft.evaluation import run_benchmark

    bundle, config, _, report = completed_benchmark
    repeated = run_benchmark(bundle, config, tmp_path / "repeat")
    fields = ("seed", "variant", "initial_state_hash", "selected_state_hash",
              "test_logits_hash", "selected_epoch", "test_accuracy", "graph_hash")
    assert [[row[field] for field in fields] for row in repeated["runs"]] == [
        [row[field] for field in fields] for row in report["runs"]]


def test_nonempty_output_and_unknown_variant_are_rejected(completed_benchmark, tmp_path):
    from neurograft.evaluation import run_benchmark

    bundle, config, output, _ = completed_benchmark
    with pytest.raises(FileExistsError):
        run_benchmark(bundle, config, output)
    with pytest.raises(ValueError):
        run_benchmark(bundle, config, tmp_path / "bad", variants=("typo",))
    assert not (tmp_path / "bad").exists()


def test_checkpoint_rejects_wrong_graph_and_corrupt_metadata(completed_benchmark, tmp_path):
    import torch
    from neurograft.evaluation import load_checkpoint
    from neurograft.fixtures import make_fixture_bundle

    bundle, _, output, report = completed_benchmark
    source = output / report["runs"][0]["checkpoint_path"]
    with pytest.raises(ValueError):
        load_checkpoint(source, make_fixture_bundle(seed=99, nodes_per_source=6))
    for field, value in [("graph_hash", "0" * 64), ("selected_epoch", 0),
                         ("variant", "unknown"), ("seed", 123456)]:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        payload[field] = value
        corrupt = tmp_path / f"corrupt-{field}.pt"
        torch.save(payload, corrupt)
        with pytest.raises(ValueError):
            load_checkpoint(corrupt, bundle)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    key = next(name for name, tensor in payload["model_state"].items()
               if tensor.is_floating_point())
    payload["model_state"][key].reshape(-1)[0] = float("nan")
    corrupt = tmp_path / "corrupt-state.pt"
    torch.save(payload, corrupt)
    with pytest.raises(ValueError):
        load_checkpoint(corrupt, bundle)
