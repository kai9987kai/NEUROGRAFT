import json

import pytest


def test_navigation_mode_and_extended_trials_are_available_through_http(tmp_path):
    from test_server import service, request
    with service(tmp_path) as (app, url):
        request(url + '/api/control', {'action': 'pause'})
        status, state = request(url + '/api/control', {'action': 'navigation', 'mode': 'observed_map'})
        assert status == 200 and state['navigation']['mode'] == 'observed_map'
        assert state['navigation']['teacher_probability'] == 0
        status, state = request(url + '/api/control', {'action': 'step', 'steps': 3})
        assert status == 200 and state['last_action_source'] == 'observed_map'
        assert state['navigation']['map']['visited'] >= 1
        before = app.session._payload()
        assert request(url + '/api/control', {'action': 'navigation', 'mode': 'oracle'})[0] == 400
        from neurograft.live import _checkpoint_digest
        assert _checkpoint_digest(app.session._payload()) == _checkpoint_digest(before)
        status, report = request(url + '/api/evaluate', {'seeds': [10000], 'size': 5,
            'max_steps': 16, 'include_controllers': True})
        assert status == 200 and len(report['variants']) == 8
        assert report['variants']['observed_map']['summary']['success_rate'] == 1
        assert report['variants']['observed_map']['summary']['collision_rate'] == 0
        assert _checkpoint_digest(app.session._payload()) == _checkpoint_digest(before)


def test_cli_exposes_extended_controller_trials():
    from neurograft.cli import parser
    args = parser().parse_args(['evaluate-live', '--bundle', 'b.json', '--checkpoint', 'c.pt',
                               '--out', 'o.json', '--include-controllers'])
    assert args.include_controllers is True
