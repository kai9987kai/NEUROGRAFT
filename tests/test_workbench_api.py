"""Workbench API isolation, admission limits and launch-path regression tests."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import urllib.request

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, _checkpoint_digest
from neurograft.service import LiveApp
from test_server import request


@contextmanager
def running_server(app):
    from neurograft.server import create_server
    server = create_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_fixture_workbench_routes_and_live_capture_are_isolated(tmp_path):
    app = LiveApp(make_fixture_bundle(nodes_per_source=3), tmp_path / 'live.pt', background=False,
                  config=LiveConfig(hidden_dim=4, unroll=2))
    try:
        app.session.running = False
        app.session.tick(steps=4)
        before = _checkpoint_digest(app.session._payload())
        with running_server(app) as url:
            status, fixture = request(url + '/api/workbench')
            assert status == 200
            assert fixture['provenance']['kind'] == 'synthetic-fixture'
            status, captured = request(url + '/api/workbench', {'action': 'capture', 'window': 8})
            assert status == 200
            assert captured['provenance']['kind'] == 'live_capture'
            assert captured['provenance']['source_bundle_sha256'] == app.session.bundle.digest()
            assert 'source_graph_sha256' not in captured['provenance']
            assert captured['models']['cumulative']['total_transitions'] == 4
            assert request(url + '/api/workbench', {'action': 'replay', 'budget': 4})[0] == 200
            status, receipt = request(url + '/api/workbench/receipt')
            assert status == 200
            assert request(url + '/api/workbench', {'action': 'import', 'receipt': receipt})[0] == 200
            with urllib.request.urlopen(url + '/workbench') as response:
                assert response.status == 200 and b'workbench' in response.read().lower()
        assert _checkpoint_digest(app.session._payload()) == before
        assert not app.state_path.exists()
    finally:
        app.close(save=False)


def test_invalid_import_and_commands_preserve_workbench(tmp_path):
    app = LiveApp(make_fixture_bundle(nodes_per_source=3), tmp_path / 'live.pt', background=False)
    try:
        with running_server(app) as url:
            original = request(url + '/api/workbench')[1]
            assert request(url + '/api/workbench', {'action': 'replay', 'budget': True})[0] == 400
            assert request(url + '/api/workbench', {'action': 'fixture', 'seed': True})[0] == 400
            receipt = request(url + '/api/workbench/receipt')[1]
            receipt['final_sha256'] = '0' * 64
            assert request(url + '/api/workbench', {'action': 'import', 'receipt': receipt})[0] == 400
            assert request(url + '/api/workbench')[1] == original
            assert request(url + '/api/control', {'action': 'pause', 'padding': 'x' * 66000})[0] == 413
            assert request(url + '/api/workbench', {'action': 'reset', 'padding': 'x' * 66000})[0] == 413
    finally:
        app.close(save=False)


def test_standalone_workbench_never_creates_a_live_session():
    from neurograft.workbench_service import FixtureWorkbenchApp
    app = FixtureWorkbenchApp(seed=17)
    with running_server(app) as url:
        assert request(url + '/api/health')[1]['mode'] == 'fixture_workbench'
        assert request(url + '/api/workbench')[0] == 200
        assert request(url + '/api/workbench', {'action': 'capture'})[0] == 400
        assert request(url + '/api/control', {'action': 'resume'})[0] == 404
        with urllib.request.urlopen(url + '/') as response:
            assert response.status == 200
    assert not hasattr(app, 'session')


def test_workbench_and_revision_cli_entry_points(tmp_path, capsys):
    from neurograft.cli import main, parser
    assert parser().parse_args(['workbench', '--port', '8770']).command == 'workbench'
    destination = tmp_path / 'revision'
    assert main(['study-revision', '--out', str(destination), '--seeds', '17', '--budget', '16']) == 0
    assert isinstance(json.loads(capsys.readouterr().out)['summary'], list)
    assert main(['verify-revision-study', str(destination), '--replay']) == 0
    assert 'true' in capsys.readouterr().out.lower()
