"""Integrity, sparse statistics, storage, and control graph contracts."""
import importlib.util
import json
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest


def test_graph_layer_exists_before_use():
    assert importlib.util.find_spec("neurograft.graphs") is not None, "typed graph layer is missing"


def _manifest(source="worm", **updates):
    from neurograft.graphs import SourceManifest
    values = dict(source_id=source, species="synthetic", version="test-1",
                  url="https://example.test/source", citation="Test fixture",
                  license="synthetic", scale="brain_region" if source == "human" else "single_neuron",
                  status="fixture", sha256="", retrieved_at="2026-10-17T00:00:00Z")
    values.update(updates)
    return SourceManifest(**values)


def _small_graph():
    from neurograft.graphs import Edge, Node, SourceGraph
    nodes = tuple(Node(f"worm:{i}", "worm", "neuron", str(i)) for i in range(5))
    edges = (Edge("worm:0", "worm:1", "chemical", 2, "worm"),
             Edge("worm:1", "worm:2", "chemical", 3, "worm"),
             Edge("worm:2", "worm:0", "electrical", 1, "worm", False))
    return SourceGraph(_manifest(), nodes, edges)


def test_fixture_bundle_is_reproducible_immutable_and_never_authentic():
    from neurograft.fixtures import make_fixture_bundle
    left = make_fixture_bundle(seed=12, nodes_per_source=8)
    right = make_fixture_bundle(seed=12, nodes_per_source=8)
    assert left.digest() == right.digest()
    assert left.digest() != make_fixture_bundle(seed=13, nodes_per_source=8).digest()
    assert isinstance(left.graphs, tuple)
    assert all(isinstance(g.nodes, tuple) and isinstance(g.edges, tuple) for g in left.graphs)
    assert not left.stats()["authentic_four_source_ready"]
    assert left.stats()["missing_authentic_sources"] == ["worm", "fly_f", "fly_m", "human"]
    with pytest.raises(FrozenInstanceError):
        left.graphs[0].nodes = ()


@pytest.mark.parametrize("updates", [
    {"source_id": "other"}, {"status": "available"}, {"scale": "brain_region"},
    {"status": "imported"}, {"status": "imported", "sha256": "x" * 64},
    {"status": "imported", "sha256": "a" * 64, "citation": ""},
    {"status": "imported", "sha256": "a" * 64, "version": ""},
    {"status": "imported", "sha256": "a" * 64, "retrieved_at": ""},
])
def test_manifest_rejects_untyped_sources_and_incomplete_import_provenance(updates):
    with pytest.raises((ValueError, TypeError)):
        _manifest(**updates)


def test_graph_rejects_cross_source_endpoints_duplicates_and_wrong_relations():
    from neurograft.graphs import Edge, Node, SourceGraph
    graph = _small_graph()
    invalid = [
        (graph.nodes, graph.edges + (Edge("worm:0", "worm:99", "chemical", 1, "worm"),)),
        (graph.nodes + (graph.nodes[0],), graph.edges),
        (graph.nodes, graph.edges + (graph.edges[0],)),
        (graph.nodes, (Edge("worm:0", "worm:1", "macro_structural", 1, "worm"),)),
        (graph.nodes, (Edge("worm:0", "worm:1", "chemical", 1, "fly_m"),)),
        (graph.nodes, (Edge("worm:0", "worm:1", "chemical", 1, "worm", False),)),
    ]
    for nodes, edges in invalid:
        with pytest.raises(ValueError):
            SourceGraph(graph.manifest, nodes, edges)
    with pytest.raises(ValueError):
        Node("fly_f:1", "worm", "neuron", "1")
    with pytest.raises(ValueError):
        SourceGraph(graph.manifest, (Node("worm:1", "worm", "region", "1"),), ())


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_count_edges_reject_nonfinite_and_negative_weights(weight):
    from neurograft.graphs import Edge
    with pytest.raises(ValueError):
        Edge("worm:0", "worm:1", "chemical", weight, "worm")


def test_functional_correlations_keep_signed_values_and_bound_range():
    from neurograft.graphs import Edge
    edge = Edge("human:A", "human:B", "macro_functional", -0.7, "human", False)
    assert edge.weight == -0.7
    for weight in (-1.01, 1.01):
        with pytest.raises(ValueError):
            replace(edge, weight=weight)
    for confidence in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            replace(edge, confidence=confidence)


def test_undirected_reverse_duplicates_are_rejected():
    from neurograft.graphs import Edge, SourceGraph
    graph = _small_graph()
    with pytest.raises(ValueError):
        SourceGraph(graph.manifest, graph.nodes,
                    (Edge("worm:0", "worm:1", "electrical", 1, "worm", False),
                     Edge("worm:1", "worm:0", "electrical", 1, "worm", False)))


def test_sparse_statistics_count_observations_and_computational_arcs_separately():
    from neurograft.graphs import GraphBundle
    stats = GraphBundle((_small_graph(),)).stats()
    worm = stats["sources"]["worm"]
    assert worm["node_count"] == 5
    assert worm["edge_count"] == 3
    assert worm["computational_edges"] == 4
    assert worm["weak_components"] == 3
    assert worm["isolated_nodes"] == 2
    assert worm["largest_component_nodes"] == 3
    assert worm["density"] == pytest.approx(0.2)
    assert worm["relation_counts"] == {"chemical": 2, "electrical": 1}


def test_bundle_round_trip_and_tampering_fail_closed(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.graphs import GraphBundle, SourceGraph
    bundle = make_fixture_bundle()
    assert SourceGraph.from_dict(bundle.graphs[0].to_dict()) == bundle.graphs[0]
    assert GraphBundle.from_dict(bundle.to_dict()) == bundle
    path = tmp_path / "bundle.json"
    bundle.save(path)
    assert GraphBundle.load(path) == bundle
    document = json.loads(path.read_text())
    # Corrupt actual graph data while retaining the stored integrity checksum.
    document["graphs"][0]["edges"][0]["weight"] = 99
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="(?i)(checksum|digest|integrity)"):
        GraphBundle.load(path)


def test_duplicate_source_graphs_are_rejected_and_partial_bundles_are_not_ready():
    from neurograft.graphs import GraphBundle
    graph = _small_graph()
    with pytest.raises(ValueError):
        GraphBundle((graph, graph))
    imported = replace(graph, manifest=replace(graph.manifest, status="imported", sha256="a" * 64, authorized=True))
    stats = GraphBundle((imported,)).stats()
    assert stats["sources"]["worm"]["authentic_ready"]
    assert not stats["authentic_four_source_ready"]
    assert stats["missing_authentic_sources"] == ["fly_f", "fly_m", "human"]


def _degrees(graph):
    degree = Counter()
    for edge in graph.edges:
        if edge.directed:
            degree[(edge.relation, edge.src, "out")] += 1
            degree[(edge.relation, edge.dst, "in")] += 1
        else:
            degree[(edge.relation, edge.src, "undirected")] += 1
            degree[(edge.relation, edge.dst, "undirected")] += 1
    return degree


def test_rewire_preserves_per_relation_degrees_weights_and_original():
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.graphs import rewire_graph
    bundle = make_fixture_bundle(seed=7, nodes_per_source=12)
    original_digest = bundle.digest()
    changed = 0
    for graph in bundle.graphs:
        control = rewire_graph(graph, seed=7, swaps_per_edge=8)
        assert _degrees(control) == _degrees(graph)
        assert Counter((e.relation, e.weight, e.confidence) for e in control.edges) == Counter((e.relation, e.weight, e.confidence) for e in graph.edges)
        assert all(e.src != e.dst for e in control.edges)
        keys = [(e.relation, e.src, e.dst) if e.directed else (e.relation, *sorted((e.src, e.dst))) for e in control.edges]
        assert len(keys) == len(set(keys))
        assert any("attempted_swaps=" in receipt and "successful_swaps=" in receipt for receipt in control.manifest.processing)
        assert control.digest() == rewire_graph(graph, seed=7, swaps_per_edge=8).digest()
        changed += set((e.src, e.dst, e.relation) for e in graph.edges) != set((e.src, e.dst, e.relation) for e in control.edges)
    assert changed > 0
    assert bundle.digest() == original_digest


def test_unswappable_control_records_zero_success_without_looping():
    from neurograft.graphs import rewire_graph
    graph = _small_graph()
    control = rewire_graph(graph, seed=1, swaps_per_edge=2)
    assert _degrees(control) == _degrees(graph)
    assert "successful_swaps=0" in control.manifest.processing[-1]
    with pytest.raises(ValueError):
        rewire_graph(graph, seed=0, swaps_per_edge=-1)


def test_rewiring_preserves_observed_autapses_and_marks_control_as_non_authentic():
    from neurograft.graphs import Edge, rewire_graph
    graph = _small_graph()
    graph = replace(graph, manifest=replace(graph.manifest, status="imported", sha256="b" * 64),
                    edges=graph.edges + (Edge("worm:0", "worm:0", "chemical", 4, "worm"),))
    control = rewire_graph(graph, seed=2)
    assert _degrees(control) == _degrees(graph)
    assert [e for e in control.edges if e.src == e.dst] == [graph.edges[-1]]
    assert graph.stats()["authentic_ready"]
    assert not control.stats()["authentic_ready"]


def test_readiness_does_not_conflate_authorization_with_technical_import():
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.graphs import GraphBundle
    fixtures = make_fixture_bundle()
    imported = GraphBundle(tuple(replace(g, manifest=replace(g.manifest, status="imported", sha256="b" * 64))
                                 for g in fixtures.graphs))
    assert imported.stats()["authentic_four_source_ready"]
    assert not imported.stats()["rights_four_source_ready"]
    assert not fixtures.stats()["authentic_four_source_ready"]


def test_serialization_rejects_missing_checksum_and_duplicate_json_keys(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.graphs import GraphBundle
    document = make_fixture_bundle().to_dict()
    del document["sha256"]
    with pytest.raises(ValueError):
        GraphBundle.from_dict(document)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":2,"graphs":[],"sha256":"x"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        GraphBundle.load(path)


def test_topology_digest_separates_observation_changes_from_receipts():
    from neurograft.graphs import rewire_graph
    graph = _small_graph()
    control = rewire_graph(graph, seed=1, swaps_per_edge=0)
    assert control.digest() != graph.digest()
    assert control.stats()["topology_digest"] == graph.stats()["topology_digest"]


def test_topology_digest_is_invariant_to_node_and_edge_permutations_and_undirected_reversals():
    graph = _small_graph()
    reordered = replace(graph, nodes=tuple(reversed(graph.nodes)),
                        edges=tuple(replace(edge, src=edge.dst, dst=edge.src) if not edge.directed else edge
                                    for edge in reversed(graph.edges)))
    assert reordered.digest() != graph.digest()
    assert reordered.topology_digest() == graph.topology_digest()


@pytest.mark.parametrize("updates", [
    {"src": "worm:1", "dst": "worm:0"},  # reverse a directed edge
    {"dst": "worm:3"},                    # change an endpoint
    {"weight": 2.5},                       # change an observed weight
    {"confidence": 0.6},                   # change observation confidence
])
def test_topology_digest_changes_when_weighted_directed_observations_change(updates):
    graph = _small_graph()
    changed = replace(graph, edges=(replace(graph.edges[0], **updates), *graph.edges[1:]))
    assert changed.topology_digest() != graph.topology_digest()


def test_known_rewire_with_unchanged_weighted_observations_has_same_topology_hash():
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.graphs import rewire_graph
    graph = make_fixture_bundle(0, 4).graphs[0]
    control = rewire_graph(graph, 6)
    # Independent observation comparison: orientation is immaterial only for
    # undirected edges; values/confidence and directed orientation stay relevant.
    def observations(value):
        return Counter((edge.relation, edge.directed,
                        (edge.src, edge.dst) if edge.directed else tuple(sorted((edge.src, edge.dst))),
                        edge.weight, edge.confidence) for edge in value.edges)
    assert observations(control) == observations(graph)
    assert control.digest() != graph.digest()
    assert control.topology_digest() == graph.topology_digest()
