"""Immutable, provenance-bearing sparse observations without ML dependencies.

An undirected observation is stored once. Its reverse is a computational arc,
not another biological observation. Observed autapses are permitted; rewiring
leaves them fixed and never proposes a new self-loop.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping


SOURCE_KEYS = ("worm", "fly_f", "fly_m", "human")
RELATIONS = ("chemical", "electrical", "macro_structural", "macro_functional")
SCHEMA_VERSION = 1


def _text(value: Any, label: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value) or value != value.strip():
        raise ValueError(f"{label} must be a {'possibly empty ' if empty else 'nonempty '}trimmed string")


def _source(value: str) -> None:
    if value not in SOURCE_KEYS:
        raise ValueError(f"source_id must be one of {SOURCE_KEYS}")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _fields(data: Any, required: set[str], optional: set[str] = frozenset()) -> dict:
    if not isinstance(data, Mapping):
        raise ValueError("serialized record must be an object")
    if not required <= data.keys() or data.keys() - required - optional:
        raise ValueError("serialized record contains missing or unknown fields")
    return dict(data)


@dataclass(frozen=True)
class SourceManifest:
    source_id: str
    species: str
    version: str
    url: str
    citation: str
    license: str
    scale: str
    status: str
    sha256: str
    retrieved_at: str
    processing: tuple[str, ...] = ()
    authorized: bool = False

    def __post_init__(self) -> None:
        _source(self.source_id)
        for key in ("species", "version", "url", "citation", "license", "retrieved_at"):
            _text(getattr(self, key), key, empty=self.status == "fixture")
        expected_scale = "brain_region" if self.source_id == "human" else "single_neuron"
        if self.scale != expected_scale:
            raise ValueError(f"{self.source_id} requires scale={expected_scale}")
        if self.status not in ("fixture", "imported"):
            raise ValueError("status must be fixture or imported")
        if not isinstance(self.authorized, bool):
            raise ValueError("authorized must be boolean")
        _text(self.sha256, "sha256", empty=self.status == "fixture")
        if self.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("sha256 must contain exactly 64 hexadecimal digits")
        if isinstance(self.processing, str):
            raise ValueError("processing must be a sequence of receipt strings")
        receipts = tuple(self.processing)
        for receipt in receipts:
            _text(receipt, "processing receipt")
        object.__setattr__(self, "processing", receipts)
        object.__setattr__(self, "sha256", self.sha256.lower())


@dataclass(frozen=True)
class Node:
    id: str
    source_id: str
    kind: str
    original_id: str

    def __post_init__(self) -> None:
        _source(self.source_id)
        _text(self.original_id, "original_id")
        if self.id != f"{self.source_id}:{self.original_id}":
            raise ValueError("node id must be source_id:original_id")
        if self.kind not in ("neuron", "region"):
            raise ValueError("node kind must be neuron or region")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    relation: str
    weight: float
    source_id: str
    directed: bool = True
    confidence: float | None = None

    def __post_init__(self) -> None:
        _source(self.source_id)
        _text(self.src, "src")
        _text(self.dst, "dst")
        if self.relation not in RELATIONS:
            raise ValueError(f"unknown relation: {self.relation}")
        if not isinstance(self.directed, bool):
            raise ValueError("directed must be boolean")
        weight = _number(self.weight, "weight")
        if self.relation == "macro_functional":
            if not -1 <= weight <= 1:
                raise ValueError("functional correlation weight must be in [-1, 1]")
        elif weight < 0:
            raise ValueError("count/structural weights must be nonnegative")
        object.__setattr__(self, "weight", weight)
        if self.confidence is not None:
            confidence = _number(self.confidence, "confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be in [0, 1]")
            object.__setattr__(self, "confidence", confidence)


def _edge_key(edge: Edge) -> tuple:
    endpoints = (edge.src, edge.dst) if edge.directed else tuple(sorted((edge.src, edge.dst)))
    return (edge.relation, edge.directed, *endpoints)


@dataclass(frozen=True)
class SourceGraph:
    manifest: SourceManifest
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, SourceManifest):
            raise ValueError("manifest must be a SourceManifest")
        nodes, edges = tuple(self.nodes), tuple(self.edges)
        if not nodes:
            raise ValueError("source graph must have at least one node")
        source = self.manifest.source_id
        expected_kind = "region" if self.manifest.scale == "brain_region" else "neuron"
        ids: set[str] = set()
        for node in nodes:
            if not isinstance(node, Node) or node.source_id != source or node.kind != expected_kind:
                raise ValueError("node source/kind does not match graph scale")
            if node.id in ids:
                raise ValueError(f"duplicate node id: {node.id}")
            ids.add(node.id)
        allowed = {"macro_structural", "macro_functional"} if source == "human" else {"chemical", "electrical"}
        keys = set()
        for edge in edges:
            if not isinstance(edge, Edge) or edge.source_id != source:
                raise ValueError("edge source does not match graph source")
            if edge.src not in ids or edge.dst not in ids:
                raise ValueError("edge endpoint does not resolve to a graph node")
            if edge.relation not in allowed:
                raise ValueError("edge relation does not match source scale")
            if edge.directed != (edge.relation == "chemical"):
                raise ValueError("chemical observations are directed; electrical/macro observations are undirected")
            key = _edge_key(edge)
            if key in keys:
                raise ValueError("duplicate observation; undirected pairs must be stored once")
            keys.add(key)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    def to_dict(self) -> dict:
        return {"manifest": asdict(self.manifest), "nodes": [asdict(n) for n in self.nodes],
                "edges": [asdict(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, data: Mapping) -> SourceGraph:
        data = _fields(data, {"manifest", "nodes", "edges"})
        manifest = SourceManifest(**_fields(data["manifest"],
            {"source_id", "species", "version", "url", "citation", "license", "scale", "status", "sha256", "retrieved_at"},
            {"processing", "authorized"}))
        try:
            nodes = tuple(Node(**_fields(n, {"id", "source_id", "kind", "original_id"})) for n in data["nodes"])
            edges = tuple(Edge(**_fields(e, {"src", "dst", "relation", "weight", "source_id"},
                                        {"directed", "confidence"})) for e in data["edges"])
        except TypeError as exc:
            raise ValueError("serialized nodes/edges must be arrays") from exc
        return cls(manifest, nodes, edges)

    def digest(self) -> str:
        return _digest(self.to_dict())

    def topology_digest(self) -> str:
        """Hash weighted observations independently of record representation.

        Node/edge order and undirected endpoint orientation are immaterial.
        Directed endpoints, relations, weights, and confidence remain relevant.
        The ordinary digest still preserves serialized order and provenance.
        """
        nodes = sorted((asdict(node) for node in self.nodes), key=lambda node: node["id"])
        edges = []
        for edge in self.edges:
            record = asdict(edge)
            if not edge.directed:
                record["src"], record["dst"] = sorted((edge.src, edge.dst))
            edges.append(record)
        # Validated relation/direction/endpoint keys are unique, so no tie can
        # depend on original edge order, weight, or optional confidence types.
        edges.sort(key=lambda edge: (edge["relation"], edge["directed"], edge["src"], edge["dst"]))
        return _digest({"nodes": nodes, "edges": edges})

    def stats(self) -> dict:
        # Sparse O(N+E) connectivity; isolated nodes are retained in components.
        adjacent: dict[str, set[str]] = {node.id: set() for node in self.nodes}
        computational_edges = 0
        for edge in self.edges:
            adjacent[edge.src].add(edge.dst)
            adjacent[edge.dst].add(edge.src)
            computational_edges += 1 if edge.directed or edge.src == edge.dst else 2
        visited: set[str] = set()
        sizes = []
        for node in adjacent:
            if node in visited:
                continue
            stack, size = [node], 0
            visited.add(node)
            while stack:
                current = stack.pop()
                size += 1
                for neighbor in adjacent[current] - visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
            sizes.append(size)
        n = len(self.nodes)
        # Density is unique off-diagonal computational arcs / possible arcs.
        # Multiple relation observations on the same pair do not inflate it.
        arcs = set()
        for edge in self.edges:
            if edge.src != edge.dst:
                arcs.add((edge.src, edge.dst))
                if not edge.directed:
                    arcs.add((edge.dst, edge.src))
        control = any(receipt.startswith("degree_rewire:") for receipt in self.manifest.processing)
        authentic = self.manifest.status == "imported" and not control
        return {"source_id": self.manifest.source_id, "status": self.manifest.status,
                "scale": self.manifest.scale, "node_count": n, "edge_count": len(self.edges),
                "computational_edges": computational_edges,
                "relation_counts": dict(sorted(Counter(e.relation for e in self.edges).items())),
                "weak_components": len(sizes), "isolated_nodes": sum(not neighbors for neighbors in adjacent.values()),
                "largest_component_nodes": max(sizes, default=0),
                "density": len(arcs) / (n * (n - 1)) if n > 1 else 0.0,
                "self_loops": sum(e.src == e.dst for e in self.edges),
                "authentic_ready": authentic, "authorized": self.manifest.authorized,
                "rights_ready": authentic and self.manifest.authorized,
                "degree_rewired_control": control, "digest": self.digest(),
                "topology_digest": self.topology_digest()}


@dataclass(frozen=True)
class GraphBundle:
    graphs: tuple[SourceGraph, ...]

    def __post_init__(self) -> None:
        graphs = tuple(self.graphs)
        if not graphs or len(graphs) > len(SOURCE_KEYS) or any(not isinstance(g, SourceGraph) for g in graphs):
            raise ValueError("bundle requires one to four SourceGraph objects")
        keys = [g.manifest.source_id for g in graphs]
        if len(set(keys)) != len(keys):
            raise ValueError("bundle source_id values must be unique")
        object.__setattr__(self, "graphs", tuple(sorted(graphs, key=lambda g: SOURCE_KEYS.index(g.manifest.source_id))))

    def _payload(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "graphs": [g.to_dict() for g in self.graphs]}

    def digest(self) -> str:
        return _digest(self._payload())

    def topology_digest(self) -> str:
        return _digest({g.manifest.source_id: g.topology_digest() for g in self.graphs})

    def to_dict(self) -> dict:
        return {**self._payload(), "sha256": self.digest()}

    @classmethod
    def from_dict(cls, data: Mapping) -> GraphBundle:
        data = _fields(data, {"schema_version", "graphs", "sha256"})
        if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported graph bundle schema version")
        payload = {"schema_version": data["schema_version"], "graphs": data["graphs"]}
        if data["sha256"] != _digest(payload):
            raise ValueError("graph bundle checksum integrity failure")
        try:
            bundle = cls(tuple(SourceGraph.from_dict(g) for g in data["graphs"]))
        except TypeError as exc:
            raise ValueError("serialized graphs must be an array") from exc
        if bundle.digest() != data["sha256"]:
            raise ValueError("graph bundle checksum changed after normalization")
        return bundle

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GraphBundle:
        def unique_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON key: {key}")
                value[key] = item
            return value
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON number: {value}")))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError("invalid graph bundle JSON") from exc
        return cls.from_dict(data)

    def stats(self) -> dict:
        sources = {g.manifest.source_id: g.stats() for g in self.graphs}
        missing = [key for key in SOURCE_KEYS if key not in sources or not sources[key]["authentic_ready"]]
        return {"digest": self.digest(), "topology_digest": self.topology_digest(), "sources": sources,
                "total_nodes": sum(s["node_count"] for s in sources.values()),
                "total_edges": sum(s["edge_count"] for s in sources.values()),
                "computational_edges": sum(s["computational_edges"] for s in sources.values()),
                "authentic_four_source_ready": not missing, "missing_authentic_sources": missing,
                "rights_four_source_ready": not missing and all(s["rights_ready"] for s in sources.values())}


def rewire_graph(graph: SourceGraph, seed: int, swaps_per_edge: int = 5) -> SourceGraph:
    """Bounded double-edge swaps, partitioned by relation and direction.

    Weights/confidence stay with edge records; their per-relation multisets are
    preserved, not each node's weighted strength. Autapses stay fixed. Attempt
    counts include rejected proposals and successful counts may be zero.
    """
    if isinstance(swaps_per_edge, bool) or not isinstance(swaps_per_edge, int) or swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be a nonnegative integer")
    rng = random.Random(seed)
    edges = list(graph.edges)
    partitions = defaultdict(list)
    for index, edge in enumerate(edges):
        if edge.src != edge.dst:
            partitions[(edge.relation, edge.directed)].append(index)
    attempted, successful = 0, 0
    per_relation = []
    for (relation, directed), indices in sorted(partitions.items()):
        keys = {_edge_key(edges[index]) for index in indices}
        relation_success, relation_attempts = 0, 0
        if len(indices) >= 2:
            for _ in range(len(indices) * swaps_per_edge):
                attempted += 1
                relation_attempts += 1
                i, j = rng.sample(indices, 2)
                first, second = edges[i], edges[j]
                a, b, c, d = first.src, first.dst, second.src, second.dst
                if not directed and rng.getrandbits(1):
                    c, d = d, c
                candidate_a = replace(first, src=a, dst=d)
                candidate_b = replace(second, src=c, dst=b)
                new_a, new_b = _edge_key(candidate_a), _edge_key(candidate_b)
                old_a, old_b = _edge_key(first), _edge_key(second)
                if a == d or c == b or new_a == new_b:
                    continue
                if {new_a, new_b} == {old_a, old_b}:
                    continue
                if (new_a in keys and new_a not in (old_a, old_b)) or (new_b in keys and new_b not in (old_a, old_b)):
                    continue
                keys.remove(old_a)
                keys.remove(old_b)
                keys.update((new_a, new_b))
                edges[i], edges[j] = candidate_a, candidate_b
                successful += 1
                relation_success += 1
        per_relation.append(f"{relation}:attempted={relation_attempts},successful={relation_success}")
    receipt = (f"degree_rewire: seed={seed}; swaps_per_edge={swaps_per_edge}; attempted_swaps={attempted}; "
               f"successful_swaps={successful}; loops_fixed={sum(e.src == e.dst for e in graph.edges)}; "
               f"weight_policy=edge-record-multiset; partitions=[{'|'.join(per_relation)}]; parent_digest={graph.digest()}")
    manifest = replace(graph.manifest, processing=graph.manifest.processing + (receipt,))
    return SourceGraph(manifest, graph.nodes, tuple(edges))
