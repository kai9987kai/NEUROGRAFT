"""Local observation import validates semantics and computes provenance."""
from dataclasses import replace
import hashlib

import pytest


def _manifest(source="worm", **updates):
    from neurograft.graphs import SourceManifest
    data = dict(source_id=source, species="human" if source == "human" else "worm",
                version="release-1", url="https://example.test/data", citation="Example source",
                license="declared research terms", scale="brain_region" if source == "human" else "single_neuron",
                status="imported", sha256="0" * 64, retrieved_at="2026-10-17T00:00:00Z", authorized=True)
    data.update(updates)
    return SourceManifest(**data)


def _files(tmp_path, nodes="id,kind\nA,neuron\nB,neuron\nC,neuron\n", edges="src,dst,relation,weight,directed,confidence\nA,B,chemical,2,true,\nB,C,electrical,1,false,0.8\n"):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    nodes_path.write_text(nodes)
    edges_path.write_text(edges)
    return nodes_path, edges_path


def test_csv_namespaces_endpoints_and_replaces_user_hash_with_raw_checksums(tmp_path):
    from neurograft.ingest import import_edge_csv
    nodes_path, edges_path = _files(tmp_path)
    graph = import_edge_csv(nodes_path, edges_path, _manifest())
    assert [node.id for node in graph.nodes] == ["worm:A", "worm:B", "worm:C"]
    assert graph.nodes[0].original_id == "A"
    assert graph.edges[0].src == "worm:A" and graph.edges[0].confidence is None
    assert not graph.edges[1].directed and graph.edges[1].confidence == 0.8
    assert graph.manifest.sha256 != "0" * 64
    assert hashlib.sha256(nodes_path.read_bytes()).hexdigest() in " ".join(graph.manifest.processing)
    assert hashlib.sha256(edges_path.read_bytes()).hexdigest() in " ".join(graph.manifest.processing)
    same = import_edge_csv(nodes_path, edges_path, _manifest())
    assert graph.digest() == same.digest()


@pytest.mark.parametrize("nodes,edges", [
    ("id,kind\nA,neuron\nA,neuron\n", "src,dst,relation,weight,directed,confidence\nA,A,chemical,1,true,\n"),
    ("id,kind\nA,neuron\n", "src,dst,relation,weight,directed,confidence\nA,B,chemical,1,true,\n"),
    ("id,kind\nA,neuron\nB,neuron\n", "src,dst,relation,weight,directed,confidence\nA,B,chemical,nan,true,\n"),
    ("id,kind\nA,neuron\nB,neuron\n", "src,dst,relation,weight,directed,confidence\nA,B,chemical,1,perhaps,\n"),
    ("node,type\nA,neuron\n", "src,dst,relation,weight,directed,confidence\nA,B,chemical,1,true,\n"),
    ("id,kind\nA,neuron\nB,neuron\n", "src,dst,relation,weight,directed,confidence\nA,B,chemical,1,true,,extra\n"),
])
def test_csv_rejects_malformed_observations(tmp_path, nodes, edges):
    from neurograft.ingest import import_edge_csv
    paths = _files(tmp_path, nodes, edges)
    with pytest.raises(ValueError):
        import_edge_csv(*paths, _manifest())


def test_importer_refuses_to_launder_a_fixture_into_authentic_data(tmp_path):
    from neurograft.ingest import import_edge_csv
    paths = _files(tmp_path)
    graph = import_edge_csv(*paths, _manifest(status="fixture", authorized=True))
    assert graph.manifest.status == "fixture"


def test_human_matrix_keeps_signed_upper_triangle_and_records_processing(tmp_path):
    from neurograft.ingest import import_human_matrix
    path = tmp_path / "human.csv"
    path.write_text(",A,B,C\nA,1,-0.6,0.1\nB,-0.6,1,0.8\nC,0.1,0.8,1\n")
    graph = import_human_matrix(path, _manifest("human"), threshold=0.5)
    assert [n.kind for n in graph.nodes] == ["region"] * 3
    assert [(e.src, e.dst, e.weight, e.directed) for e in graph.edges] == [
        ("human:A", "human:B", -0.6, False), ("human:B", "human:C", 0.8, False)]
    assert graph.manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert any("threshold" in step and "diagonal" in step for step in graph.manifest.processing)


@pytest.mark.parametrize("data", [
    ",A,B\nA,1,0.2\nB,0.3,1\n",  # asymmetric
    ",A,B\nA,1,0.2\nC,0.2,1\n",  # mismatched labels
    ",A,A\nA,1,0.2\nA,0.2,1\n",  # duplicate labels
    ",A,B\nA,1,nan\nB,nan,1\n",  # nonfinite
    ",A,B\nA,1,1.2\nB,1.2,1\n",  # invalid correlation
    ",A,B\nA,1,0.2\n",  # not square
    ",A,B\nA,1,0.2,0.3\nB,0.2,1\n",  # ragged
])
def test_human_matrix_rejects_invalid_labels_shape_or_values(tmp_path, data):
    from neurograft.ingest import import_human_matrix
    path = tmp_path / "human.csv"
    path.write_text(data)
    with pytest.raises(ValueError):
        import_human_matrix(path, _manifest("human"))


def test_human_structural_matrix_allows_counts_but_rejects_wrong_source_and_threshold(tmp_path):
    from neurograft.ingest import import_human_matrix
    path = tmp_path / "matrix.csv"
    path.write_text(",A,B\nA,0,6\nB,6,0\n")
    graph = import_human_matrix(path, _manifest("human"), relation="macro_structural")
    assert len(graph.edges) == 1 and graph.edges[0].weight == 6
    with pytest.raises(ValueError):
        import_human_matrix(path, _manifest("worm"), relation="macro_structural")
    for threshold in (-0.1, float("nan")):
        with pytest.raises(ValueError):
            import_human_matrix(path, _manifest("human"), threshold=threshold)


def test_oversized_local_input_is_rejected_before_reading(tmp_path):
    from neurograft.ingest import import_edge_csv
    paths = _files(tmp_path)
    with paths[0].open("r+b") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="(?i)(size|large|limit)"):
        import_edge_csv(*paths, _manifest())
