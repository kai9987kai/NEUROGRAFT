import contextlib
import json
import threading
import urllib.error
import urllib.request

import pytest


@contextlib.contextmanager
def service(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.server import LiveApp, create_server
    app = LiveApp(make_fixture_bundle(nodes_per_source=4), tmp_path / 'session.pt',
                  background=False)
    server = create_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield app, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        app.close()


def request(url, payload=None, headers=None, raw=None):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    values = {'Content-Type': 'application/json'} if data is not None else {}
    values.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=values)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_live_http_learning_memory_and_named_state_recovery(tmp_path):
    with service(tmp_path) as (app, url):
        assert request(url + '/api/health') == (200, {'ok': True})
        assert request(url + '/api/control', {'action': 'pause'})[1]['running'] is False
        status, state = request(url + '/api/control', {'action': 'step', 'steps': 8})
        assert status == 200 and state['steps'] == 8 and state['updates'] > 0
        status, chat = request(url + '/api/chat', {'message': 'remember name = Atlas'})
        assert status == 200 and chat['state']['facts']['name'] == 'Atlas'
        status, saved = request(url + '/api/save', {'name': 'test-slot'})
        assert status == 200 and saved['saved']['name'] == 'test-slot'
        baseline = saved['state']
        request(url + '/api/control', {'action': 'step', 'steps': 5})
        status, restored = request(url + '/api/load', {'name': 'test-slot'})
        assert status == 200
        assert restored['steps'] == baseline['steps']
        assert restored['model_hash'] == baseline['model_hash']
        assert restored['world'] == baseline['world']
        assert restored['facts']['name'] == 'Atlas'


def test_live_api_rejects_paths_invalid_actions_and_cross_origin(tmp_path):
    with service(tmp_path) as (app, url):
        assert request(url + '/api/save', {'name': '../outside'})[0] == 400
        assert request(url + '/api/control', {'action': 'step', 'steps': 999})[0] == 400
        assert request(url + '/api/control', {'action': 'learning', 'enabled': 1})[0] == 400
        assert request(url + '/api/control', {'action': 'pause'}, {'Origin': 'https://unrelated.example'})[0] == 403
        assert request(url + '/api/control', raw=b'{}', headers={'Content-Type': 'text/plain'})[0] == 415
        assert request(url + '/api/chat', {'message': 'x' * 3000})[0] == 400
        assert request(url + '/api/control', raw=b'{broken json')[0] == 400
        assert request(url + '/api/control', {'action': 'move', 'direction': 0})[0] == 409


def test_close_saves_and_restart_resumes_without_browser(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.server import LiveApp
    bundle = make_fixture_bundle(nodes_per_source=4)
    path = tmp_path / 'session.pt'
    app = LiveApp(bundle, path, background=False)
    app.control({'action': 'pause'})
    app.control({'action': 'step', 'steps': 8})
    before = app.state()
    app.close()
    resumed = LiveApp(bundle, path, background=False)
    try:
        assert resumed.state()['steps'] == before['steps']
        assert resumed.state()['model_hash'] == before['model_hash']
    finally:
        resumed.close()


def test_state_writer_has_exclusive_lease(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.server import LiveApp
    bundle = make_fixture_bundle(nodes_per_source=4)
    app = LiveApp(bundle, tmp_path / 'session.pt', background=False)
    try:
        with pytest.raises(RuntimeError, match='already|writer|locked'):
            LiveApp(bundle, tmp_path / 'session.pt', background=False)
    finally:
        app.close()


def test_background_learning_ticks_and_autosaves_without_a_page(tmp_path):
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.server import LiveApp
    import time
    app = LiveApp(make_fixture_bundle(nodes_per_source=4), tmp_path / 'session.pt',
                  interval=0.05, autosave=1, background=True)
    try:
        deadline = time.monotonic() + 8
        while (app.state()['steps'] < 8 or not (tmp_path / 'session.pt').exists()) and time.monotonic() < deadline:
            time.sleep(.05)
        assert app.state()['steps'] >= 8
        assert app.state()['updates'] > 0
        assert (tmp_path / 'session.pt').exists()
    finally:
        app.close()


def test_static_assets_and_manual_goal_controls(tmp_path):
    from neurograft.server import ASSETS
    with service(tmp_path) as (app, url):
        for route in ASSETS:
            with urllib.request.urlopen(url + route, timeout=10) as response:
                assert response.status == 200
                assert len(response.read()) > 0
                assert "default-src 'self'" in response.headers.get('Content-Security-Policy', '')
        with urllib.request.urlopen(url + '/favicon.ico', timeout=10) as response:
            assert response.status == 204
        assert request(url + '/missing-asset')[0] == 404
        assert request(url + '/api/control', {'action': 'pause'})[0] == 200
        status, state = request(url + '/api/control', {'action': 'learning', 'enabled': False})
        assert status == 200 and state['learning'] is False
        status, state = request(url + '/api/control', {'action': 'goal', 'color': 'rose'})
        assert status == 200 and state['world']['goal']['color'] == 'rose'
        status, state = request(url + '/api/control', {'action': 'move', 'direction': 1})
        assert status == 200 and state['last_action'] == 1 and state['last_action_source'] == 'manual'
        status, checkpoints = request(url + '/api/checkpoints')
        assert status == 200 and isinstance(checkpoints, list)
