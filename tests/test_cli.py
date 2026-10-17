import json


def test_live_command_exposes_persistent_service_defaults():
    from neurograft.cli import parser
    args = parser().parse_args(['live'])
    assert args.port == 8768
    assert args.state == 'data/live/session.pt'
    assert args.interval == .25 and args.autosave == 15


def test_fixture_inspection_round_trip(tmp_path, capsys):
    from neurograft.cli import main
    path = tmp_path / 'fixture.json'
    assert main(['fixtures', '--out', str(path), '--nodes', '6']) == 0
    capsys.readouterr()
    assert main(['inspect', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['authentic_four_source_ready'] is False
    assert len(result['sources']) == 4


def test_doctor_does_not_need_to_import_torch(capsys):
    from neurograft.cli import main
    assert main(['doctor']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['packages']['torch']['installed'] is True
    assert 'available_ram_gb' in result


def test_cli_reports_missing_input_without_traceback(capsys, tmp_path):
    from neurograft.cli import main
    assert main(['inspect', str(tmp_path / 'missing.json')]) == 2
    assert 'error:' in capsys.readouterr().err


def test_fixture_output_is_not_silently_overwritten(tmp_path, capsys):
    from neurograft.cli import main
    path = tmp_path / 'fixture.json'
    assert main(['fixtures', '--out', str(path)]) == 0
    before = path.read_bytes()
    assert main(['fixtures', '--out', str(path), '--seed', '5']) == 2
    assert path.read_bytes() == before


def test_report_escapes_labels_and_shows_evidence_scope(tmp_path):
    from neurograft.report import write_html_report
    report = {'evidence_scope': 'synthetic_fixture_only',
              'summary': {'grafted': {'mean_accuracy': 0.5}},
              'runs': [{'variant': '<script>alert(1)</script>', 'seed': 0, 'test_accuracy': 0.5}],
              'source_readiness': {'authentic_four_source_ready': False}}
    out = tmp_path / 'report.html'
    write_html_report(report, out)
    text = out.read_text(encoding='utf-8')
    assert '<script>alert(1)</script>' not in text
    assert 'synthetic_fixture_only' in text
    assert 'NEUROGRAFT' in text


def test_cli_checkpoint_replay_matches_saved_test_logits(tmp_path, capsys):
    from neurograft.cli import main
    bundle = tmp_path / 'bundle.json'
    out = tmp_path / 'run'
    assert main(['fixtures', '--out', str(bundle), '--nodes', '4']) == 0
    capsys.readouterr()
    assert main(['benchmark', '--bundle', str(bundle), '--out', str(out), '--seeds', '0',
                 '--epochs', '1', '--train-size', '8', '--val-size', '8', '--test-size', '8',
                 '--batch-size', '4', '--hidden-dim', '4', '--variants', 'grafted']) == 0
    report = json.loads(capsys.readouterr().out)
    checkpoint = out / report['runs'][0]['checkpoint_path']
    assert main(['replay', '--bundle', str(bundle), '--checkpoint', str(checkpoint)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay['matches_checkpoint'] is True
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    saved['test_logits_hash'] = 'f' * 64
    tampered = tmp_path / 'tampered.pt'
    torch.save(saved, tampered)
    assert main(['replay', '--bundle', str(bundle), '--checkpoint', str(tampered)]) == 2
    assert json.loads(capsys.readouterr().out)['matches_checkpoint'] is False


def test_cli_checks_input_cap_before_any_source_read(tmp_path, monkeypatch, capsys):
    from pathlib import Path
    from types import SimpleNamespace
    from neurograft.cli import main
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_id': 'human', 'species': 'synthetic', 'version': 'v1',
        'url': 'https://example.invalid', 'citation': 'fixture', 'license': 'fixture',
        'scale': 'brain_region', 'status': 'fixture', 'sha256': '0'*64, 'retrieved_at': ''}))
    large = tmp_path / 'large.csv'
    original_stat, original_read = Path.stat, Path.read_bytes
    def stat(path, *a, **kw):
        return SimpleNamespace(st_size=65*1024*1024) if path == large else original_stat(path, *a, **kw)
    def read(path):
        if path == large:
            raise AssertionError('oversized source read before size guard')
        return original_read(path)
    monkeypatch.setattr(Path, 'stat', stat)
    monkeypatch.setattr(Path, 'read_bytes', read)
    assert main(['import-human', '--matrix', str(large), '--manifest', str(manifest), '--out', str(tmp_path/'out')]) == 2
    assert '64 MiB' in capsys.readouterr().err
