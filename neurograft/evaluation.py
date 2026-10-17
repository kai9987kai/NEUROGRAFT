"""Bounded, paired synthetic evaluation of source communication and memory.

This task tests engineered routing and recurrence. It does not establish a
biological effect, topology advantage, or pretrained-language-model capability.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import statistics
import tempfile
import threading
import time
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .dynamics import DynamicsConfig, GraphCircuit
from .graphs import GraphBundle, rewire_graph


VARIANTS = ("grafted", "no_graft", "no_recurrence", "rewired")
TASK_VERSION = "delayed_xor_v1"
CHECKPOINT_VERSION = 1
_MAX_TENSOR_BYTES = 256 * 1024 * 1024
_SEED_MAX = 2**31 - 1


def _integer(name: str, value: Any, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum or 'the supported limit'}")
    return value


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 20
    train_size: int = 128
    val_size: int = 64
    test_size: int = 128
    batch_size: int = 32
    delay: int = 4
    hidden_dim: int = 12
    learning_rate: float = 0.01
    data_seed: int = 1729
    threads: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, (tuple, list)) or not self.seeds:
            raise ValueError("seeds must be a nonempty sequence")
        seeds = tuple(_integer("seed", seed, 0, _SEED_MAX) for seed in self.seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("model seeds must be unique")
        if len(seeds) > 32:
            raise ValueError("this bounded CPU benchmark accepts at most 32 seeds")
        object.__setattr__(self, "seeds", seeds)
        for name in ("epochs", "train_size", "val_size", "test_size", "batch_size", "hidden_dim"):
            _integer(name, getattr(self, name))
        _integer("delay", self.delay, 0, 128)
        _integer("hidden_dim", self.hidden_dim, 1, 128)
        _integer("epochs", self.epochs, 1, 1000)
        _integer("data_seed", self.data_seed, 0, _SEED_MAX)
        _integer("threads", self.threads, 1, 2)
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (float, int)):
            raise TypeError("learning_rate must be a finite positive number")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        total_input_bytes = (self.train_size + self.val_size + self.test_size) * (self.delay + 2) * 16 * 4
        if total_input_bytes > _MAX_TENSOR_BYTES:
            raise ValueError("requested task inputs exceed the 256 MiB benchmark allocation guard")


def generate_task(size: int, delay: int, seed: int) -> tuple[Tensor, Tensor]:
    """Return delayed XOR inputs [B, delay+2, 4, 4] and long labels [B].

    Worm and female fly receive separate signed bits at t=0. The human module
    sees only a common final query indicator. Male fly sees zeros. Small
    label-independent nuisance values occur only in the two cue modules.
    Four cue pairs are balanced when size is divisible by four; split draws
    have disjoint PRNG seeds, although the four underlying cue patterns recur.
    """
    _integer("size", size)
    _integer("delay", delay, 0, 128)
    _integer("seed", seed, 0, 3 * _SEED_MAX + 2)
    if size * (delay + 2) * 16 * 4 > _MAX_TENSOR_BYTES:
        raise ValueError("task allocation exceeds 256 MiB")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pairs = (torch.arange(size) % 4)[torch.randperm(size, generator=generator)]
    bits_a, bits_b = (pairs // 2), (pairs % 2)
    inputs = torch.zeros((size, delay + 2, 4, 4), dtype=torch.float32)
    inputs[:, 0, 0, 0] = bits_a * 2 - 1
    inputs[:, 0, 1, 0] = bits_b * 2 - 1
    inputs[:, 0, :2, 1] = 1
    inputs[:, :, :2, 3] = torch.randn((size, delay + 2, 2), generator=generator) * 0.05
    inputs[:, -1, 3, 2] = 1
    return inputs, torch.bitwise_xor(bits_a, bits_b).long()


def _split_seeds(config: BenchmarkConfig) -> dict[str, int]:
    return dict(zip(("train", "validation", "test"), (3 * config.data_seed + i for i in range(3))))


def _tensor_hash(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_hash(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(bytes.fromhex(_tensor_hash(value)))
    return digest.hexdigest()


def _topology_hash(bundle: GraphBundle) -> str:
    return bundle.topology_digest()


def _clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _atomic_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _rss_bytes() -> int:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
            return counters.WorkingSetSize
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")


class _MemorySamples:
    """Sample process RSS; not a tensor-only allocator or exact peak reading."""

    def __enter__(self):
        self.before = _rss_bytes()
        self.maximum = self.before
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self):
        while not self._stop.wait(0.05):
            self.maximum = max(self.maximum, _rss_bytes())

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.after = _rss_bytes()
        self.maximum = max(self.maximum, self.after)

    def to_dict(self):
        return {"before_rss_bytes": self.before, "after_rss_bytes": self.after,
                "max_sampled_rss_bytes": self.maximum, "sampling_interval_seconds": 0.05,
                "scope": "whole process RSS including PyTorch and preceding runs; sampled, not exact peak"}


@contextmanager
def _deterministic_cpu(threads: int):
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def _flags(variant: str) -> dict[str, bool]:
    return {"graft_enabled": variant != "no_graft", "recurrence": variant != "no_recurrence"}


def _evaluate(model: GraphCircuit, inputs: Tensor, labels: Tensor, variant: str) -> tuple[float, float, Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(inputs, **_flags(variant)).logits
        loss = F.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    if not math.isfinite(loss) or not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite held-out outputs")
    return loss, accuracy, logits


def _rewired_bundle(bundle: GraphBundle, seed: int) -> tuple[GraphBundle, list[dict[str, Any]]]:
    controls, receipts = [], []
    for index, graph in enumerate(bundle.graphs):
        rewire_seed = seed * 4 + index
        control = rewire_graph(graph, seed=rewire_seed)
        controls.append(control)
        receipts.append({"source_id": graph.manifest.source_id, "seed": rewire_seed,
                         "original_graph_hash": graph.digest(), "control_graph_hash": control.digest(),
                         "processing": list(control.manifest.processing)})
    return GraphBundle(tuple(controls)), receipts


def _paired_comparisons(runs: list[dict[str, Any]], variants: tuple[str, ...]) -> list[dict[str, Any]]:
    if "grafted" not in variants:
        return []
    grafted = {run["seed"]: run["test_accuracy"] for run in runs if run["variant"] == "grafted"}
    comparisons = []
    for variant in variants:
        if variant == "grafted":
            continue
        control = {run["seed"]: run["test_accuracy"] for run in runs if run["variant"] == variant}
        seeds = sorted(set(grafted) & set(control))
        differences = [grafted[seed] - control[seed] for seed in seeds]
        mean = sum(differences) / len(differences)
        interval = None
        if len(seeds) >= 2:
            rng = random.Random(702731)
            bootstrap = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                               for _ in range(2000))
            interval = [bootstrap[49], bootstrap[1949]]
        comparisons.append({"comparison": f"grafted minus {variant}", "control": variant,
                            "seeds": seeds, "n_paired_seeds": len(seeds),
                            "seed_accuracy_differences": differences, "mean_accuracy_difference": mean,
                            "bootstrap_95_ci": interval, "bootstrap_resamples": 2000 if interval else 0,
                            "caveat": "Small seed samples give unstable descriptive bootstrap intervals, not a powered significance test. "
                                      "All seeds use one fixed set of task splits; intervals reflect run-seed variation in initialization/training and, for rewired controls, topology rewiring. They do not measure data-split uncertainty."})
    return comparisons


def run_benchmark(bundle: GraphBundle, config: BenchmarkConfig, output_dir: str | Path,
                  variants: tuple[str, ...] = VARIANTS) -> dict[str, Any]:
    """Train paired CPU controls, select by validation, and evaluate test once.

    A nonempty destination is refused. Finished files are atomically replaced;
    an interrupted directory remains inspectable and cannot silently be reused.
    Checkpoints are inference/replay artifacts, not training-resume snapshots.
    """
    if not isinstance(bundle, GraphBundle) or not isinstance(config, BenchmarkConfig):
        raise TypeError("bundle and config must be GraphBundle and BenchmarkConfig")
    if not isinstance(variants, (tuple, list)) or not variants or any(v not in VARIANTS for v in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    variants = tuple(variants)
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique")
    node_count = sum(len(graph.nodes) for graph in bundle.graphs)
    estimated_working_bytes = min(config.batch_size, config.train_size) * node_count * config.hidden_dim * (config.delay + 2) * 4 * 40
    estimated_heldout_bytes = max(config.val_size, config.test_size) * node_count * config.hidden_dim * 4 * 40
    if max(estimated_working_bytes, estimated_heldout_bytes) > _MAX_TENSOR_BYTES:
        raise ValueError("estimated recurrent training tensors exceed the 256 MiB CPU benchmark guard; reduce batch, delay or hidden size")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"benchmark destination is not empty: {output}")
    # Exclusive creation closes the race between two writers that saw an empty directory.
    with (output / ".benchmark.lock").open("xb") as lock:
        lock.write(str(os.getpid()).encode("ascii"))
    _atomic_bytes(output / "RUNNING.json", _json_bytes({"config": asdict(config), "status": "running"}))
    started = time.perf_counter()
    split_seeds = _split_seeds(config)
    splits = {name: generate_task(size, config.delay, split_seeds[name]) for name, size in
              (("train", config.train_size), ("validation", config.val_size), ("test", config.test_size))}
    hashes = {name: _state_hash({"inputs": tensors[0], "labels": tensors[1]}) for name, tensors in splits.items()}
    label_counts = {name: tensors[1].bincount(minlength=2).tolist() for name, tensors in splits.items()}
    model_config = DynamicsConfig(input_dim=4, hidden_dim=config.hidden_dim, output_dim=2, top_k=2)
    runs = []
    with _deterministic_cpu(config.threads):
        for seed in config.seeds:
            torch.manual_seed(seed)
            template = GraphCircuit(bundle, model_config)
            initial_parameters = _clone_state(dict(template.named_parameters()))
            initial_hash = _state_hash(initial_parameters)
            del template
            order_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
            batch_orders = [torch.randperm(config.train_size, generator=order_generator) for _ in range(config.epochs)]
            order_hash = _tensor_hash(torch.stack(batch_orders))
            for variant in variants:
                run_started = time.perf_counter()
                with _MemorySamples() as memory:
                    graph, rewire_receipt = _rewired_bundle(bundle, seed) if variant == "rewired" else (bundle, [])
                    torch.manual_seed(seed)
                    model = GraphCircuit(graph, model_config)
                    # Topology is carried in buffers; copying state_dict here would undo rewiring.
                    with torch.no_grad():
                        for name, parameter in model.named_parameters():
                            if name not in initial_parameters or parameter.shape != initial_parameters[name].shape:
                                raise RuntimeError("control variant changed parameter allocation")
                            parameter.copy_(initial_parameters[name])
                    if _state_hash(dict(model.named_parameters())) != initial_hash:
                        raise RuntimeError("control initial weights do not match")
                    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                    metrics = []
                    best_accuracy, best_loss, selected_epoch, selected_state = -1.0, math.inf, 0, None
                    optimizer_steps = 0
                    relative = Path("runs") / f"seed_{seed}" / variant
                    train_inputs, train_labels = splits["train"]
                    for epoch, order in enumerate(batch_orders, 1):
                        epoch_started = time.perf_counter()
                        model.train()
                        total_loss, total_correct = 0.0, 0
                        for offset in range(0, config.train_size, config.batch_size):
                            indices = order[offset:offset + config.batch_size]
                            optimizer.zero_grad(set_to_none=True)
                            logits = model(train_inputs[indices], **_flags(variant)).logits
                            loss = F.cross_entropy(logits, train_labels[indices])
                            if not torch.isfinite(loss):
                                raise RuntimeError("nonfinite training loss")
                            loss.backward()
                            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                                raise RuntimeError("nonfinite training gradient")
                            optimizer.step()
                            optimizer_steps += 1
                            total_loss += loss.item() * len(indices)
                            total_correct += int((logits.argmax(-1) == train_labels[indices]).sum())
                        val_loss, val_accuracy, _ = _evaluate(model, *splits["validation"], variant)
                        # Accuracy is primary; validation CE breaks ties. Test remains untouched.
                        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
                            best_accuracy, best_loss, selected_epoch = val_accuracy, val_loss, epoch
                            selected_state = _clone_state(model.state_dict())
                        metrics.append({"seed": seed, "variant": variant, "epoch": epoch,
                                        "train_loss": total_loss / config.train_size,
                                        "train_accuracy": total_correct / config.train_size,
                                        "validation_loss": val_loss, "validation_accuracy": val_accuracy,
                                        "optimizer_steps": optimizer_steps,
                                        "elapsed_seconds": time.perf_counter() - epoch_started})
                        _atomic_bytes(output / relative / "metrics.jsonl", b"".join(
                            (json.dumps(metric, sort_keys=True, allow_nan=False) + "\n").encode() for metric in metrics))
                    assert selected_state is not None
                    model.load_state_dict(selected_state, strict=True)
                    test_loss, test_accuracy, test_logits = _evaluate(model, *splits["test"], variant)
                    selected_hash = _state_hash(dict(model.named_parameters()))
                    logits_hash = _tensor_hash(test_logits)
                    checkpoint = {"format_version": CHECKPOINT_VERSION, "task_version": TASK_VERSION,
                                  "model_state": selected_state, "model_state_hash": _state_hash(selected_state),
                                  "model_config": asdict(model_config), "graph_bundle": graph.to_dict(),
                                  "graph_hash": graph.digest(), "input_graph_hash": bundle.digest(),
                                  "topology_hash": _topology_hash(graph),
                                  "benchmark_config": asdict(config), "seed": seed, "variant": variant,
                                  "selected_epoch": selected_epoch, "selected_validation_accuracy": best_accuracy,
                                  "selected_validation_loss": best_loss, "split_seeds": split_seeds,
                                  "initial_state_hash": initial_hash, "selected_state_hash": selected_hash,
                                  "test_logits_hash": logits_hash, "data_hashes": hashes,
                                  "rewire_receipt": rewire_receipt, "resume_training_supported": False}
                    encoded = io.BytesIO()
                    torch.save(checkpoint, encoded)
                    _atomic_bytes(output / relative / "checkpoint.pt", encoded.getvalue())
                    parameter_count = sum(parameter.numel() for parameter in model.parameters())
                runs.append({"seed": seed, "variant": variant, "parameter_count": parameter_count,
                             "initial_state_hash": initial_hash, "selected_state_hash": selected_hash,
                             "graph_hash": graph.digest(), "rewire_receipt": rewire_receipt,
                             "topology_hash": _topology_hash(graph),
                             "topology_changed": _topology_hash(graph) != _topology_hash(bundle),
                             "data_hashes": hashes, "batch_order_hash": order_hash,
                             "optimizer_steps": optimizer_steps, "selected_epoch": selected_epoch,
                             "selected_validation_accuracy": best_accuracy, "selected_validation_loss": best_loss,
                             "test_loss": test_loss, "test_accuracy": test_accuracy, "test_evaluations": 1,
                             "test_logits_hash": logits_hash, "elapsed_seconds": time.perf_counter() - run_started,
                             "memory": memory.to_dict(), "metrics_path": (relative / "metrics.jsonl").as_posix(),
                             "checkpoint_path": (relative / "checkpoint.pt").as_posix()})
                del model, optimizer, selected_state, test_logits, encoded
    parameter_counts_equal = len({run["parameter_count"] for run in runs}) == 1
    if not parameter_counts_equal:
        raise RuntimeError("control trainable parameter counts differ")
    summary = {}
    for variant in variants:
        scores = [run["test_accuracy"] for run in runs if run["variant"] == variant]
        summary[variant] = {"mean_test_accuracy": statistics.mean(scores),
                            "std_test_accuracy": statistics.stdev(scores) if len(scores) > 1 else None,
                            "n_seeds": len(scores), "std_ddof": 1}
    report = {"format_version": 1, "task": TASK_VERSION, "config": asdict(config),
              "variants": list(variants), "input_graph_hash": bundle.digest(), "split_seeds": split_seeds,
              "source_readiness": bundle.stats(), "summary": summary,
              "input_topology_hash": _topology_hash(bundle), "data_hashes": hashes,
              "split_label_counts": label_counts, "runs": runs, "parameter_counts_equal": parameter_counts_equal,
              "paired_comparisons": _paired_comparisons(runs, variants),
              "elapsed_seconds": time.perf_counter() - started,
              "estimated_training_tensor_bytes": estimated_working_bytes,
              "estimated_heldout_tensor_bytes": estimated_heldout_bytes,
              "runtime": {"torch": str(torch.__version__), "device": "cpu", "threads": config.threads,
                          "deterministic_algorithms": True},
              "selection": "highest validation accuracy, then lowest validation CE; test evaluated once after selection",
              "evidence_scope": ["Synthetic delayed XOR communication and memory only; no biological validation.",
                                 "This source-communication task is designed to require a graft and recurrence. A graft advantage does not establish a topology benefit.",
                                 "Fixed disjoint PRNG seeds define splits; the four underlying XOR cue patterns recur across splits.",
                                 "All variants allocate identical trainable parameters, weights, data, epoch shuffles and optimizer steps; effective computation and active gradients differ.",
                                 "Variants run sequentially; timing includes initialization, optimizer setup, validation and checkpoint writes. RSS includes prior allocator retention.",
                                 "Checkpoint replay is CPU/runtime-specific; training resume is not implemented."]}
    # Normalize tuple fields once so the returned receipt exactly equals its JSON artifact.
    report = json.loads(_json_bytes(report))
    _atomic_bytes(output / "report.json", _json_bytes(report))
    _atomic_bytes(output / "COMPLETE.json", _json_bytes({"status": "complete", "report": "report.json"}))
    (output / "RUNNING.json").unlink()
    (output / ".benchmark.lock").unlink()
    return report


def load_checkpoint(path: str | Path, bundle: GraphBundle) -> tuple[GraphCircuit, dict[str, Any]]:
    """Load a restricted CPU inference checkpoint, validating graph and weights.

    ``bundle`` is the original benchmark input. Rewired checkpoints reconstruct
    their separately saved control graph and validate it before model loading.
    """
    if not isinstance(bundle, GraphBundle):
        raise TypeError("bundle must be a GraphBundle")
    try:
        if Path(path).stat().st_size > 512 * 1024 * 1024:
            raise ValueError("checkpoint exceeds the 512 MiB CPU loading guard")
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint root must be a mapping")
        if checkpoint["format_version"] != CHECKPOINT_VERSION or checkpoint["task_version"] != TASK_VERSION:
            raise ValueError("unsupported checkpoint format or task")
        if checkpoint["input_graph_hash"] != bundle.digest():
            raise ValueError("checkpoint input graph does not match supplied bundle")
        config = BenchmarkConfig(**checkpoint["benchmark_config"])
        seed = _integer("seed", checkpoint["seed"], 0, _SEED_MAX)
        if seed not in config.seeds or checkpoint["variant"] not in VARIANTS:
            raise ValueError("invalid checkpoint seed or variant")
        _integer("selected_epoch", checkpoint["selected_epoch"], 1, config.epochs)
        val_accuracy = checkpoint["selected_validation_accuracy"]
        val_loss = checkpoint["selected_validation_loss"]
        if not math.isfinite(val_accuracy) or not 0 <= val_accuracy <= 1 or not math.isfinite(val_loss) or val_loss < 0:
            raise ValueError("invalid checkpoint validation metrics")
        if checkpoint["split_seeds"] != _split_seeds(config):
            raise ValueError("checkpoint task split seeds do not match config")
        graph = GraphBundle.from_dict(checkpoint["graph_bundle"])
        if graph.digest() != checkpoint["graph_hash"]:
            raise ValueError("checkpoint graph hash mismatch")
        if _topology_hash(graph) != checkpoint["topology_hash"]:
            raise ValueError("checkpoint topology hash mismatch")
        if checkpoint["variant"] == "rewired":
            expected_graph, expected_receipt = _rewired_bundle(bundle, seed)
            if graph.digest() != expected_graph.digest() or checkpoint["rewire_receipt"] != expected_receipt:
                raise ValueError("rewired checkpoint does not match deterministic control graph")
        elif graph.digest() != bundle.digest() or checkpoint["rewire_receipt"]:
            raise ValueError("non-rewired checkpoint changed input topology")
        model_config = DynamicsConfig(**checkpoint["model_config"])
        expected_config = DynamicsConfig(input_dim=4, hidden_dim=config.hidden_dim, output_dim=2, top_k=2)
        if asdict(model_config) != asdict(expected_config):
            raise ValueError("checkpoint model and benchmark configurations disagree")
        state = checkpoint["model_state"]
        if not isinstance(state, dict) or not state or any(not isinstance(value, Tensor) for value in state.values()):
            raise ValueError("invalid checkpoint model state")
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("checkpoint state contains nonfinite weights")
        if _state_hash(state) != checkpoint["model_state_hash"]:
            raise ValueError("checkpoint weight hash mismatch")
        with torch.random.fork_rng(devices=[]):
            model = GraphCircuit(graph, model_config)
        for name, buffer in model.named_buffers():
            if name not in state or not torch.equal(buffer, state[name]):
                raise ValueError("checkpoint buffers disagree with declared graph")
        model.load_state_dict(state, strict=True)
        if _state_hash(dict(model.named_parameters())) != checkpoint["selected_state_hash"]:
            raise ValueError("checkpoint selected parameter hash mismatch")
        model.eval()
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state"}
        return model, metadata
    except (KeyError, TypeError, RuntimeError, AttributeError, EOFError, OSError, pickle.UnpicklingError) as error:
        raise ValueError(f"invalid checkpoint: {error}") from error
