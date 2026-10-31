"""V2 adaptive workbench continuation without rewriting the v1 protocol."""
import copy

import pytest

from neurograft.workbench import ReplayWorkbench
from neurograft.workbench_service import WorkbenchStore


def workbench(**kwargs):
    from neurograft.adaptive_workbench import AdaptiveReplayWorkbench
    return AdaptiveReplayWorkbench(**kwargs)


def observation(count=1):
    return {'action': 'observe', 'source': [3, 1], 'action_index': 1,
            'destination': [3, 1], 'count': count}


def test_v2_nine_arms_and_batch_matches_exact_single_event_scoring():
    batch, singles = workbench(seed=42), workbench(seed=42)
    initial = batch.snapshot()
    assert initial['version'] == 2
    assert set(initial['models']) == {'cumulative', 'recent', 'adaptive'}
    assert initial['adaptive_evidence']['events'] == 0
    result = batch.apply(observation(8))
    for _ in range(8):
        separate = singles.apply(observation())
    assert result['adaptive_evidence'] == separate['adaptive_evidence']
    assert result['models'] == separate['models']
    assert result['arms'] == separate['arms']
    assert all(arm['stale'] for model in result['arms'].values() for arm in model.values())
    assert result['models']['adaptive']['sha256'] == result['models']['recent']['sha256']


def test_replay_goal_and_snapshot_do_not_score_outcomes():
    bench = workbench(seed=17)
    bench.apply(observation(4))
    evidence = bench.snapshot()['adaptive_evidence']
    snapshot = bench.apply({'action': 'replay', 'budget': 8})
    assert snapshot['adaptive_evidence'] == evidence
    for model in snapshot['arms'].values():
        assert model['none']['total_backups'] == 0
        assert model['uniform']['total_backups'] == 8
    bench.apply({'action': 'goal', 'x': 1, 'z': 1})
    assert bench.snapshot()['adaptive_evidence'] == evidence
    receipt = bench.export_receipt()
    assert bench.snapshot() == bench.snapshot()
    assert bench.export_receipt() == receipt


def test_v2_receipt_reconstructs_scores_and_exact_next_event():
    bench = workbench(seed=17)
    bench.apply(observation(4))
    bench.apply({'action': 'replay', 'budget': 8})
    restored = type(bench).from_receipt(bench.export_receipt())
    assert restored.snapshot() == bench.snapshot()
    assert restored.apply(observation()) == bench.apply(observation())
    corrupted = copy.deepcopy(bench.export_receipt())
    corrupted['operations'][0]['count'] = 3
    with pytest.raises(ValueError, match='digest'):
        type(bench).from_receipt(corrupted)


def test_default_store_v2_and_legacy_import_remains_byte_identical():
    store = WorkbenchStore(seed=17)
    assert store.snapshot()['version'] == 2
    legacy = ReplayWorkbench(seed=42)
    legacy.apply(observation(4))
    legacy.apply({'action': 'replay', 'budget': 8})
    result = store.control({'action': 'import', 'receipt': legacy.export_receipt()})
    assert result == legacy.snapshot()
    assert store.receipt() == legacy.export_receipt()
    assert set(result['models']) == {'cumulative', 'recent'}
    assert store.control(observation()) == legacy.apply(observation())
    assert store.control({'action': 'fixture', 'seed': 42})['version'] == 2


@pytest.mark.parametrize('version', [True, 0, 3, 2.0, '2', None])
def test_import_version_rejection_is_atomic(version):
    store = WorkbenchStore(seed=17)
    before = store.snapshot()
    receipt = store.receipt()
    receipt['version'] = version
    with pytest.raises(ValueError):
        store.control({'action': 'import', 'receipt': receipt})
    assert store.snapshot() == before


def test_v2_command_limit_atomicity_and_reset():
    bench = workbench(seed=17)
    before = bench.snapshot()
    with pytest.raises(ValueError):
        bench.apply({**observation(16), 'destination': [30, 30]})
    assert bench.snapshot() == before
    bench.apply(observation(16))
    reset = bench.apply({'action': 'reset'})
    assert reset == before
