"""Explicitly synthetic sparse circuits for deterministic offline development."""
from __future__ import annotations

import random

from .graphs import Edge, GraphBundle, Node, SOURCE_KEYS, SourceGraph, SourceManifest


def make_fixture_bundle(seed: int = 0, nodes_per_source: int = 8) -> GraphBundle:
    if isinstance(nodes_per_source, bool) or not isinstance(nodes_per_source, int) or not 2 <= nodes_per_source <= 10000:
        raise ValueError("nodes_per_source must be an integer from 2 to 10000")
    rng = random.Random(seed)
    graphs = []
    for source in SOURCE_KEYS:
        human = source == "human"
        manifest = SourceManifest(source, "synthetic human regions" if human else "synthetic neurons",
            "fixture-v1", "local://neurograft/synthetic-fixture", "NEUROGRAFT synthetic software fixture",
            "synthetic-test-data", "brain_region" if human else "single_neuron", "fixture", "",
            "2026-10-17T00:00:00Z", (f"synthetic_fixture: seed={seed}; nodes_per_source={nodes_per_source}; no-biological-input",))
        nodes = tuple(Node(f"{source}:{index}", source, "region" if human else "neuron", str(index))
                      for index in range(nodes_per_source))
        # A ring plus bounded random edges gives sparse connected fixtures while
        # still leaving room for degree-preserving swaps in normal small runs.
        n = nodes_per_source
        edges = []
        primary = "macro_structural" if human else "chemical"
        primary_pairs = {(i, (i + 1) % n) if not human else tuple(sorted((i, (i + 1) % n))) for i in range(n)}
        for _ in range(n):
            a, b = rng.sample(range(n), 2)
            primary_pairs.add(tuple(sorted((a, b))) if human else (a, b))
        for a, b in sorted(primary_pairs):
            edges.append(Edge(nodes[a].id, nodes[b].id, primary, rng.randint(1, 5), source, not human))
        secondary = "macro_functional" if human else "electrical"
        secondary_pairs = {tuple(sorted(rng.sample(range(n), 2))) for _ in range(n)}
        for a, b in sorted(secondary_pairs):
            weight = rng.choice((-0.8, -0.4, 0.3, 0.7)) if human else rng.randint(1, 3)
            edges.append(Edge(nodes[a].id, nodes[b].id, secondary, weight, source, False))
        graphs.append(SourceGraph(manifest, nodes, tuple(edges)))
    return GraphBundle(tuple(graphs))
