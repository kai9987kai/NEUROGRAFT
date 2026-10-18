import contextlib
import json
from pathlib import Path
import shutil
import subprocess
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
        health_status, health = request(url + '/api/health')
        assert health_status == 200 and health['ok'] is True
        assert health['worker'] == {'enabled': False, 'alive': False}
        assert health['running'] is True and health['last_error'] is None
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
        assert 'telemetry' in state and len(state['telemetry']['action_probs']) == 4
        before_steps = state['steps']
        status, boosted = request(url + '/api/control', {'action': 'boost', 'steps': 5})
        assert status == 200 and boosted['steps'] == before_steps + 5
        status, resized = request(url + '/api/control', {'action': 'world', 'size': 17, 'preset': 'chambers'})
        assert status == 200 and resized['world']['size'] == 17
        assert resized['telemetry']['environment']['preset'] == 'chambers'
        status, lesioned = request(url + '/api/control', {'action': 'memory_lesion', 'mode': 'shifted'})
        assert status == 200 and lesioned['telemetry']['memory']['mode'] == 'shifted'
        assert 'compass' in lesioned['telemetry'] and lesioned['telemetry']['compass']['num_wedges'] == 16
        status, checkpoints = request(url + '/api/checkpoints')
        assert status == 200 and isinstance(checkpoints, list)


@pytest.mark.parametrize('preset', ['labyrinth', 'chambers', 'corridors', 'pillars', 'concentric', 'hierarchical'])
def test_every_advertised_world_preset_is_usable_through_http(tmp_path, preset):
    with service(tmp_path) as (app, url):
        status, state = request(url + '/api/control', {'action': 'world', 'size': 11, 'preset': preset})
        assert status == 200
        assert state['telemetry']['environment']['preset'] == preset


def test_health_exposes_failed_worker_instead_of_reporting_success(tmp_path):
    with service(tmp_path) as (app, url):
        app.last_error = 'Injected checkpoint write failure'
        app.session.running = False
        status, health = request(url + '/api/health')
        assert status == 503 and health['ok'] is False
        assert health['running'] is False
        assert health['last_error'] == 'Injected checkpoint write failure'


def test_legacy_restore_exposes_auxiliary_reset_notice_in_runtime(tmp_path):
    import copy
    import torch
    from neurograft.live import _checkpoint_digest
    with service(tmp_path) as (app, url):
        payload = copy.deepcopy(app.session._payload())
        payload.pop('auxiliary')
        payload['version'] = 1
        app.slots.mkdir(parents=True)
        torch.save({'payload': payload, 'sha256': _checkpoint_digest(payload)}, app.slots / 'legacy.pt')
        with pytest.warns(RuntimeWarning, match='auxiliary'):
            status, restored = request(url + '/api/load', {'name': 'legacy'})
        assert status == 200
        assert restored['runtime']['checkpoint_compatibility'] == {'loaded_format': 1, 'auxiliary_state_restored': False}


def test_frozen_navigation_evaluation_api_retains_report_without_changing_live_state(tmp_path):
    with service(tmp_path) as (app, url):
        assert request(url + '/api/evaluation') == (200, None)
        assert request(url + '/api/control', {'action': 'pause'})[0] == 200
        assert request(url + '/api/control', {'action': 'step', 'steps': 2})[0] == 200
        before = app.session.snapshot()
        status, report = request(url + '/api/evaluate', {'seeds': [10000], 'size': 5, 'preset': 'concentric', 'max_steps': 2})
        assert status == 200
        assert report['learning'] is False and report['teacher_assistance'] == 0
        assert report['model_hash'] == before['model_hash']
        assert report['checkpoint_steps'] == before['steps']
        assert set(report['variants']) == {'policy', 'no_graft', 'no_recurrence', 'random', 'shortest_path'}
        assert report['variants']['policy']['summary']['episodes'] == 1
        assert request(url + '/api/evaluation') == (200, report)
        assert app.session.snapshot() == before
        assert not (tmp_path / 'session.pt').exists()


@pytest.mark.parametrize('payload', [
    {'seeds': []}, {'seeds': list(range(17))}, {'seeds': '0,1'}, {'max_steps': 129},
    {'max_steps': True}, {'size': 4}, {'preset': 'missing'}, {'unexpected': 1},
])
def test_navigation_evaluation_api_rejects_invalid_budgets(tmp_path, payload):
    with service(tmp_path) as (app, url):
        assert request(url + '/api/evaluate', payload)[0] == 400
        assert request(url + '/api/evaluation') == (200, None)


def test_navigation_evaluation_rejects_a_second_inflight_request(tmp_path):
    with service(tmp_path) as (app, url):
        app.evaluation_guard.acquire()
        try:
            assert request(url + '/api/evaluate', {})[0] == 409
            assert request(url + '/api/health')[1]['evaluation_running'] is True
        finally:
            app.evaluation_guard.release()


def test_browser_application_javascript_parses():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is unavailable for the browser syntax check')
    script = Path(__file__).resolve().parents[1] / 'neurograft' / 'web' / 'app.js'
    result = subprocess.run([node, '--check', str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr

