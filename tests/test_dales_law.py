from neurograft.dales_law import audit_dales_law, audit_bundle_dales_law, project_dales_law, infer_node_polarity
from neurograft.fixtures import make_fixture_bundle
from neurograft.graphs import SourceGraph, Node, Edge, SourceManifest


def test_dales_law_audit_on_fixtures():
    bundle = make_fixture_bundle()
    reports = audit_bundle_dales_law(bundle)
    assert set(reports) == {"worm", "fly_f", "fly_m", "human"}
    for src, rep in reports.items():
        assert rep["total_nodes"] > 0
        assert 0.0 <= rep["compliance_ratio"] <= 1.0
        assert rep["spectral_radius_bound"] >= 0.0
        assert rep["is_contractive"] is True


def test_dales_law_polarity_inference():
    assert infer_node_polarity("VD1") == -1
    assert infer_node_polarity("DD2") == -1
    assert infer_node_polarity("AVL") == -1
    assert infer_node_polarity("VA1") == 1
    assert infer_node_polarity("DA1") == 1
    assert infer_node_polarity("AVAL") == 1


def test_dales_law_projection_contraction():
    bundle = make_fixture_bundle()
    worm = [g for g in bundle.graphs if g.manifest.source_id == "worm"][0]
    proj = project_dales_law(worm, radius=0.85)
    rep = audit_dales_law(proj, radius=0.85)
    assert rep.is_contractive is True
    assert rep.spectral_radius_bound <= 0.85 + 1e-4
