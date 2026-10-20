import pytest


def test_controller_report_labels_assistance_and_preserves_all_identities(tmp_path):
    from neurograft.navigation_report import write_navigation_evaluation_report
    report = {'schema_version': 2, 'model_hash': 'm' * 64, 'graph_hash': 'g' * 64,
        'protocol_id': 'p' * 64, 'config': {'size': 11, 'preset': 'labyrinth', 'seeds': [41000]},
        'evidence_scope': 'Synthetic observation-only controller; no raw neural gain established.',
        'variants': {'policy': {'summary': {'episodes': 1, 'success_rate': 0, 'spl': 0, 'collision_rate': .9, 'mean_steps': 32}},
                     'observed_map': {'summary': {'episodes': 1, 'success_rate': 1, 'spl': .75, 'collision_rate': 0, 'mean_steps': 16}}}}
    path = write_navigation_evaluation_report(report, tmp_path / 'report.html')
    html = path.read_text(encoding='utf-8')
    assert 'Raw policy' in html and 'Observed map + policy' in html
    assert '100.0%' in html and '0.750' in html
    assert 'engineered assistance' in html and 'm' * 64 in html and 'p' * 64 in html
    assert 'src="https://' not in html
    with pytest.raises(ValueError, match='exists'):
        write_navigation_evaluation_report(report, path)


def test_report_escapes_untrusted_metadata(tmp_path):
    from neurograft.navigation_report import write_navigation_evaluation_report
    report = {'variants': {}, 'evidence_scope': '<script>alert(1)</script>', 'config': {}}
    path = write_navigation_evaluation_report(report, tmp_path / 'report.html')
    assert '<script>' not in path.read_text()
