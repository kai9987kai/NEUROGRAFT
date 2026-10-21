"""New modes and optional architecture retain old archive identities."""
from pathlib import Path
import importlib.util
import sys

from neurograft.fixtures import make_fixture_bundle
from neurograft.live import LiveConfig, LiveSession, _checkpoint_digest


def test_steering_checkpoint_continues_exactly_after_a_controller_action(tmp_path):
    bundle = make_fixture_bundle(nodes_per_source=3)
    session = LiveSession(bundle, LiveConfig(hidden_dim=4, predictive_head=True))
    session.learning = False
    session.set_navigation_mode('cx_steer')
    session.tick(2)
    assert session.last_action_source == 'cx_steer'
    path = tmp_path / 'steering.pt'
    session.save(path)
    restored = LiveSession.load(path, bundle)
    assert restored.navigation_mode == 'cx_steer'
    assert restored.snapshot()['navigation']['teacher_probability'] == 0
    assert restored.snapshot()['navigation']['predictive_head'] is True
    assert 'synthetic' in restored.snapshot()['navigation']['reason'].lower()
    session.tick(1)
    restored.tick(1)
    assert _checkpoint_digest(session._payload()) == _checkpoint_digest(restored._payload())


def test_existing_v10_study_still_verifies_without_rewriting_its_plan():
    root = Path(__file__).resolve().parents[1]
    path = root / 'tools' / 'study_navigation_readouts.py'
    spec = importlib.util.spec_from_file_location('v10_compatibility_study', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    directory = root / 'reports' / 'runs' / 'navigation-v10-readout-study-20261020-final'
    before = (directory / 'plan.json').read_bytes()
    assert module.verify_study(directory)['ok']
    assert (directory / 'plan.json').read_bytes() == before
