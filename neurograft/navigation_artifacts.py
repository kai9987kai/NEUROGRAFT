"""Self-contained navigation experiment evidence and read-only verification.

Hashes detect changed files, not authenticity. Optional replay verifies the
recorded frozen trials in the current runtime; it does not retrain a candidate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARTIFACTS = frozenset({
    'frozen-input.pt', 'source.bundle.json', 'candidate.pt', 'config.json',
    'progress.json', 'report.json', 'report.html', 'status.json',
})
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


def _read(path: Path, limit=MAX_ARTIFACT_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'artifact must be a regular file: {path.name}')
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'artifact exceeds size limit: {path.name}')
    return data


def _json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'nonfinite JSON number: {value}')

    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def _write_json(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(raw)


def capture_input(checkpoint, directory):
    """Freeze bytes before loading; concurrent source autosaves cannot change this input."""
    destination = Path(directory) / 'frozen-input.pt'
    data = _read(Path(checkpoint))
    with destination.open('xb') as handle:
        handle.write(data)
    return destination


def write_run_evidence(directory, result, bundle):
    directory = Path(directory)
    _write_json(directory / 'source.bundle.json', bundle.to_dict())
    _write_json(directory / 'config.json', result['config'])
    _write_json(directory / 'progress.json', result['progress'])
    _write_json(directory / 'report.json', result)


def seal_run(directory):
    directory = Path(directory)
    files = {name: hashlib.sha256(_read(directory / name)).hexdigest()
             for name in sorted(ARTIFACTS)}
    _write_json(directory / 'manifest.json', {'schema_version': 1, 'files': files})


def verify_navigation_run(directory, replay=False):
    """Validate all artifacts and identities; optionally rerun both frozen evaluations."""
    from .graphs import GraphBundle
    from .live import LiveSession, _checkpoint_digest
    from .navigation_evaluation import (NavigationEvalConfig, frozen_navigation_evaluation,
                                         navigation_protocol, _json_hash)

    directory = Path(directory).resolve()
    manifest = _json(_read(directory / 'manifest.json', 65536))
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or not isinstance(manifest.get('files'), dict)
            or set(manifest['files']) != ARTIFACTS):
        raise ValueError('manifest must list exactly the required experiment artifacts')
    blobs = {}
    for name in sorted(ARTIFACTS):
        data = _read(directory / name)
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest['files'][name]:
            raise ValueError(f'artifact hash mismatch: {name}')
        blobs[name] = data
    try:
        result = _json(blobs['report.json'])
        config = _json(blobs['config.json'])
        progress = _json(blobs['progress.json'])
        status = _json(blobs['status.json'])
        if (status != {'status': 'COMPLETE', 'selected_round': result['selected_round']}
                or config != result['config'] or progress != result['progress']
                or result['promotion_performed'] is not False):
            raise ValueError('experiment status/config/progress identity mismatch')
        from .navigation_training import CurriculumConfig, _selection_key
        settings = dict(config)
        for name in ('sizes', 'presets', 'validation_seeds', 'test_seeds'):
            settings[name] = tuple(settings[name])
        curriculum = CurriculumConfig(**settings)
        if ([entry['round'] for entry in progress] != list(range(curriculum.rounds + 1))
                or result['selected_round'] != max(progress, key=lambda entry: _selection_key(entry['validation']))['round']):
            raise ValueError('selected round does not match recorded validation selection')
        bundle = GraphBundle.from_dict(_json(blobs['source.bundle.json']))
        baseline = LiveSession.load(directory / 'frozen-input.pt', bundle)
        candidate = LiveSession.load(directory / 'candidate.pt', bundle)
        baseline_hash = _checkpoint_digest(baseline.model.state_dict())
        candidate_hash = _checkpoint_digest(candidate.model.state_dict())
        if (result['baseline_model_hash'] != baseline_hash
                or result['baseline']['model_hash'] != baseline_hash
                or result['selected_model_hash'] != candidate_hash
                or result['candidate']['model_hash'] != candidate_hash
                or result['baseline']['graph_hash'] != bundle.digest()
                or result['candidate']['graph_hash'] != bundle.digest()):
            raise ValueError('checkpoint model/graph identity mismatch')
        if candidate.running or candidate.learning:
            raise ValueError('candidate inference export must be paused with learning disabled')
        if candidate.config.readout != curriculum.readout:
            raise ValueError('candidate readout does not match declared curriculum')
        if candidate.config.predictive_head != curriculum.predictive_head:
            raise ValueError('candidate predictive_head does not match declared curriculum')
        eval_config = NavigationEvalConfig(seeds=curriculum.test_seeds, size=curriculum.eval_size,
                                          preset=curriculum.eval_preset, max_steps=curriculum.max_steps,
                                          include_readout_controls=curriculum.include_readout_controls)
        expected_config = navigation_protocol(eval_config, candidate.config.unroll)['config']
        if (result['baseline']['config'] != expected_config
                or result['candidate']['config'] != expected_config
                or result['baseline']['protocol_id'] != result['candidate']['protocol_id']
                or result['protocol_id'] != result['candidate']['protocol_id']):
            raise ValueError('frozen evaluation protocol identity mismatch')
        for name, session in (('baseline', baseline), ('candidate', candidate)):
            report = result[name]
            expected_protocol = navigation_protocol(eval_config, session.config.unroll)
            expected_id = _json_hash(expected_protocol)
            if list(report['variants']) != expected_protocol['variants']:
                raise ValueError(f'{name} evaluation variants do not match declared controls')
            if (report['protocol_id'] != expected_id
                    or report['inference_unroll'] != session.config.unroll
                    or report['checkpoint_steps'] != session.steps
                    or report['checkpoint_updates'] != session.updates
                    or report['learning'] is not False or report['teacher_assistance'] != 0):
                raise ValueError(f'{name} evaluation protocol/checkpoint identity mismatch')
        if replay:
            for name, session in (('baseline', baseline), ('candidate', candidate)):
                if frozen_navigation_evaluation(session, eval_config) != result[name]:
                    raise ValueError(f'{name} frozen trial replay differs in this runtime')
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f'invalid experiment evidence: {exc}') from exc
    return {'ok': True, 'artifacts_verified': len(ARTIFACTS), 'replay_verified': bool(replay),
            'selected_model_hash': candidate_hash, 'graph_hash': bundle.digest(),
            'protocol_id': result['protocol_id'], 'promotion_performed': False,
            'evidence_scope': 'Local file integrity and checkpoint identities; hashes are not authenticity signatures. '
                              'Replay, when requested, covers frozen trials in the current runtime, not training.'}
