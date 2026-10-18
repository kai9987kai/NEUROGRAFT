"""Small local commands with lazy imports for the training and model runtimes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def _print(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def _bundle(path):
    from .graphs import GraphBundle
    return GraphBundle.load(path)


def _save(bundle, path):
    destination = Path(path)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(destination)
    _print({"path": str(destination.resolve()), "sha256": bundle.digest(), **bundle.stats()})


def _manifest(path, input_path):
    from .graphs import SourceManifest
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Importers validate their input-size cap, compute real input hashes and
    # replace this transient schema value before any graph can be returned.
    # Do not read the source here: that would bypass their allocation guard.
    values.setdefault("sha256", "0" * 64)
    values.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
    return SourceManifest(**values)


def parser():
    root = argparse.ArgumentParser(prog="neurograft", description="NEUROGRAFT local research instrument")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="lightweight environment and memory diagnostics")
    fixture = commands.add_parser("fixtures", help="create explicitly synthetic four-source test graphs")
    fixture.add_argument("--out", required=True)
    fixture.add_argument("--nodes", type=int, default=8)
    fixture.add_argument("--seed", type=int, default=0)
    inspect = commands.add_parser("inspect", help="validate checksum and inspect source readiness")
    inspect.add_argument("bundle")
    fetch = commands.add_parser("fetch-worm", help="download bounded revision-pinned OpenWorm input")
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--electrical-conflicts", choices=('error', 'exclude'), default='error',
                       help='reject inconsistent mirrored counts, or explicitly exclude those pairs with receipts')
    assemble = commands.add_parser("assemble", help="replace sources in a bundle without relabelling fixtures")
    assemble.add_argument("--base", required=True)
    assemble.add_argument("--source", action="append", required=True)
    assemble.add_argument("--out", required=True)
    edges = commands.add_parser("import-csv", help="import normalized local neuron/region CSV tables")
    edges.add_argument("--nodes", required=True)
    edges.add_argument("--edges", required=True)
    edges.add_argument("--manifest", required=True)
    edges.add_argument("--out", required=True)
    human = commands.add_parser("import-human", help="import an authorized labelled human parcel matrix")
    human.add_argument("--matrix", required=True)
    human.add_argument("--manifest", required=True)
    human.add_argument("--relation", choices=("macro_functional", "macro_structural"), default="macro_functional")
    human.add_argument("--threshold", type=float, default=0)
    human.add_argument("--out", required=True)
    bench = commands.add_parser("benchmark", help="train and compare matched graph-only controls")
    bench.add_argument("--bundle", required=True)
    bench.add_argument("--out", required=True)
    bench.add_argument("--seeds", default="0,1,2,3,4")
    bench.add_argument("--epochs", type=int, default=20)
    bench.add_argument("--train-size", type=int, default=128)
    bench.add_argument("--val-size", type=int, default=64)
    bench.add_argument("--test-size", type=int, default=128)
    bench.add_argument("--batch-size", type=int, default=32)
    bench.add_argument("--delay", type=int, default=4)
    bench.add_argument("--hidden-dim", type=int, default=12)
    bench.add_argument("--threads", type=int, default=2)
    bench.add_argument("--learning-rate", type=float, default=.01)
    bench.add_argument("--data-seed", type=int, default=1729)
    bench.add_argument("--variants", default="grafted,no_graft,no_recurrence,rewired")
    replay = commands.add_parser("replay", help="verify checkpoint and regenerate held-out task logits")
    replay.add_argument("--checkpoint", required=True)
    replay.add_argument("--bundle", required=True)
    nav = commands.add_parser("evaluate-live", help="evaluate a frozen live checkpoint on matched navigation trials")
    nav.add_argument("--bundle", required=True)
    nav.add_argument("--checkpoint", required=True)
    nav.add_argument("--out", required=True)
    nav.add_argument("--seeds", default="10000,10001,10002")
    nav.add_argument("--size", type=int, default=11)
    nav.add_argument("--preset", default="labyrinth")
    nav.add_argument("--max-steps", type=int, default=48)
    curriculum = commands.add_parser('train-navigation', help='train an isolated navigation candidate and compare frozen trials')
    curriculum.add_argument('--bundle', required=True)
    curriculum.add_argument('--checkpoint', required=True, help='captured baseline checkpoint; never overwritten')
    curriculum.add_argument('--out', required=True, help='new experiment directory')
    curriculum.add_argument('--config', help='JSON object containing CurriculumConfig fields; explicit CLI options override it')
    curriculum.add_argument('--rounds', type=int, default=None)
    curriculum.add_argument('--worlds-per-round', '--worlds', dest='worlds_per_round', type=int, default=None)
    curriculum.add_argument('--updates-per-round', '--updates', dest='updates_per_round', type=int, default=None)
    curriculum.add_argument('--seed', type=int, default=None)
    curriculum.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=None)
    curriculum.add_argument('--init', choices=('fresh', 'checkpoint'), default=None)
    report = commands.add_parser("report", help="create a standalone local HTML experiment report")
    report.add_argument("report")
    report.add_argument("--out", required=True)
    model = commands.add_parser("model-probe", help="explicit pinned pretrained loading and zero-graft probe")
    model.add_argument("--model-id", choices=("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"), default="Qwen/Qwen3.5-0.8B")
    model.add_argument("--revision", required=True)
    model.add_argument("--download", action="store_true", help="allow network model download (requires free RAM)")
    model.add_argument("--prompt", default="Describe a recurrent neural circuit in one sentence.")
    live = commands.add_parser("live", help="launch persistent continuous learning service and 3D browser UI")
    live.add_argument("--bundle", default=None)
    live.add_argument("--state", default="data/live/session.pt")
    live.add_argument("--port", type=int, default=8768)
    live.add_argument("--interval", type=float, default=0.25)
    live.add_argument("--autosave", type=float, default=15)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from .runtime import doctor
            _print(doctor())
        elif args.command == "fixtures":
            from .fixtures import make_fixture_bundle
            _save(make_fixture_bundle(args.seed, args.nodes), args.out)
        elif args.command == "inspect":
            _print(_bundle(args.bundle).stats())
        elif args.command == "fetch-worm":
            from .acquire import fetch_worm
            _print(fetch_worm(args.out, electrical_conflicts=args.electrical_conflicts))
        elif args.command == "assemble":
            from .graphs import GraphBundle
            sources = {g.manifest.source_id: g for g in _bundle(args.base).graphs}
            replaced = set()
            for path in args.source:
                for graph in _bundle(path).graphs:
                    if graph.manifest.source_id in replaced:
                        raise ValueError("multiple replacement inputs declare the same source")
                    sources[graph.manifest.source_id] = graph
                    replaced.add(graph.manifest.source_id)
            _save(GraphBundle(tuple(sources.values())), args.out)
        elif args.command in ("import-csv", "import-human"):
            from .graphs import GraphBundle
            from .ingest import import_edge_csv, import_human_matrix
            if args.command == "import-csv":
                graph = import_edge_csv(args.nodes, args.edges, _manifest(args.manifest, args.edges))
            else:
                graph = import_human_matrix(args.matrix, _manifest(args.manifest, args.matrix),
                                            args.relation, args.threshold)
            _save(GraphBundle((graph,)), args.out)
        elif args.command == "benchmark":
            from .evaluation import BenchmarkConfig, run_benchmark
            bundle = _bundle(args.bundle)
            if sum(len(g.nodes) for g in bundle.graphs) > 5000 or sum(len(g.edges) for g in bundle.graphs) > 100000:
                raise ValueError("local CLI benchmark cap is 5,000 nodes / 100,000 observed edges; import a bounded subgraph")
            config = BenchmarkConfig(seeds=tuple(int(s) for s in args.seeds.split(',')), epochs=args.epochs,
                                     train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
                                     batch_size=args.batch_size, delay=args.delay, hidden_dim=args.hidden_dim,
                                     threads=args.threads, learning_rate=args.learning_rate, data_seed=args.data_seed)
            _print(run_benchmark(bundle, config, args.out, variants=tuple(args.variants.split(','))))
        elif args.command == "evaluate-live":
            from .navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation
            from .live import LiveSession
            destination = Path(args.out)
            if destination.exists():
                raise ValueError(f"output already exists: {destination}; choose a new path")
            config = NavigationEvalConfig(seeds=tuple(int(s) for s in args.seeds.split(',')),
                                          size=args.size, preset=args.preset, max_steps=args.max_steps)
            bundle = _bundle(args.bundle)
            if sum(len(g.nodes) for g in bundle.graphs) > 2000 or sum(len(g.edges) for g in bundle.graphs) > 50000:
                raise ValueError("navigation evaluation cap is 2,000 nodes and 50,000 observations")
            result = frozen_navigation_evaluation(LiveSession.load(args.checkpoint, bundle), config)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('x', encoding='utf-8') as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.write('\n')
            _print(result)
        elif args.command == 'train-navigation':
            from dataclasses import fields
            from .navigation_training import CurriculumConfig, run_navigation_training
            values = {}
            if args.config is not None:
                config_path = Path(args.config)
                if config_path.stat().st_size > 65536:
                    raise ValueError('curriculum config exceeds 64 KiB')
                values = json.loads(config_path.read_text(encoding='utf-8-sig'))
                if not isinstance(values, dict):
                    raise ValueError('curriculum config must be a JSON object')
                unknown = set(values) - {field.name for field in fields(CurriculumConfig)}
                if unknown:
                    raise ValueError('unknown curriculum config fields: ' + ', '.join(sorted(unknown)))
                for name in ('validation_seeds', 'test_seeds', 'sizes', 'presets'):
                    if name in values:
                        if not isinstance(values[name], list):
                            raise ValueError(f'{name} must be a JSON array')
                        values[name] = tuple(values[name])
            for name in ('rounds', 'worlds_per_round', 'updates_per_round', 'seed', 'learning_rate', 'init'):
                value = getattr(args, name)
                if value is not None:
                    values[name] = value
            config = CurriculumConfig(**values)
            _print(run_navigation_training(_bundle(args.bundle), args.checkpoint, args.out, config))
        elif args.command == "report":
            from .report import write_html_report
            result = json.loads(Path(args.report).read_text(encoding="utf-8"))
            _print({"path": str(write_html_report(result, args.out).resolve())})
        elif args.command == "replay":
            import torch
            from .evaluation import load_checkpoint, generate_task
            model, metadata = load_checkpoint(args.checkpoint, _bundle(args.bundle))
            config = metadata["benchmark_config"]
            torch.set_num_threads(config["threads"])
            x, y = generate_task(config["test_size"], config["delay"], config["data_seed"] * 3 + 2)
            variant = metadata["variant"]
            model.eval()
            with torch.no_grad():
                logits = model(x, graft_enabled=variant != "no_graft", recurrence=variant != "no_recurrence").logits
            actual_hash = hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()
            matches = actual_hash == metadata["test_logits_hash"]
            _print({"test_accuracy": float((logits.argmax(-1) == y).float().mean()),
                    "test_logits_hash": actual_hash, "expected_test_logits_hash": metadata["test_logits_hash"],
                    "matches_checkpoint": matches, "variant": variant, "seed": metadata["seed"],
                    "evidence_scope": "synthetic task replay"})
            if not matches:
                print("error: replay differs from saved checkpoint predictions", file=sys.stderr)
                return 2
        elif args.command == "model-probe":
            import torch
            from .llm import load_pretrained
            model, tokenizer = load_pretrained(args.model_id, args.revision, 48, local_files_only=not args.download)
            ids = tokenizer(args.prompt, return_tensors="pt")
            context = torch.zeros((1, 48))
            with torch.no_grad():
                plain = model(ids["input_ids"], context, attention_mask=ids["attention_mask"], enabled=False, use_cache=False).logits
                zero = model(ids["input_ids"], context, attention_mask=ids["attention_mask"], use_cache=False).logits
            _print({"model_id": args.model_id, "revision": args.revision,
                    "zero_graft_logits_equal": torch.equal(plain, zero),
                    "status": "pretrained runtime probe; graph-language adapter remains untrained"})
        elif args.command == "live":
            from .fixtures import make_fixture_bundle
            from .server import serve_live
            if args.bundle is not None:
                bundle = _bundle(args.bundle)
            elif Path("data/imported/mixed.bundle.json").is_file():
                bundle = _bundle("data/imported/mixed.bundle.json")
            elif Path("data/fixtures.bundle.json").is_file():
                bundle = _bundle("data/fixtures.bundle.json")
            else:
                bundle = make_fixture_bundle()
            serve_live(bundle, args.state, port=args.port, interval=args.interval, autosave=args.autosave)
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
