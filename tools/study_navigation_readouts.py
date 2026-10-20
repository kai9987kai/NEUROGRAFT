"""Run and verify a bounded, predeclared four-arm navigation learning study.

This tool keeps every arm and initialization, with validation-only checkpoint
selection inside each run. File hashes are local integrity receipts, not
authenticity signatures. Replay covers frozen evaluations, never training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields, replace
import hashlib
from html import escape
import json
import math
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurograft.environment import NavigationWorld
from neurograft.dynamics import GraphCircuit
from neurograft.graphs import GraphBundle
from neurograft.live import LiveSession, _checkpoint_digest
from neurograft.navigation_artifacts import _json, _read, _write_json, capture_input, verify_navigation_run
from neurograft.navigation_evaluation import _summary
from neurograft.navigation_training import CurriculumConfig, _world_schedule, run_navigation_training


SCHEMA = 'navigation-readout-study-v1'
ARMS = (
    {'id': 'human_plain', 'readout': 'human', 'rotation_augmentation': False},
    {'id': 'human_rotated', 'readout': 'human', 'rotation_augmentation': True},
    {'id': 'context_residual_plain', 'readout': 'context_residual', 'rotation_augmentation': False},
    {'id': 'context_residual_rotated', 'readout': 'context_residual', 'rotation_augmentation': True},
)
_FILES = {'frozen-input.pt', 'source.bundle.json', 'plan.json', 'aggregate.json', 'report.html', 'status.json'}
_SCOPE = (
    'Synthetic navigation learning study. All four arms and all initialization seeds are reported; '
    'there is no test-based selection among arms or seeds. Each run selects its checkpoint using '
    'validation success, SPL and collisions only, retaining the earlier checkpoint on ties. '
    'Each seed sets both model initialization and the training-world seed offset; arms share that seed\'s corpus. '
    'Raw policy is unassisted during frozen evaluation. Expert collection uses privileged full-map '
    'shortest-path labels with fixed N/E/S/W tie-breaking; the student sees only its observation window. '
    'Rotation augmentation transforms observations and labels, without providing architectural equivariance. '
    'No-graft, no-recurrence and no-context are inference interventions, not separately trained ablations. '
    'Context residual increases the parameter count while optimizer updates and raw sample budgets are matched. '
    'Random and shortest-path controls are reported separately. Episodes pooled across initialization '
    'seeds are correlated by scenario and are not independent replications; no episode-level significance '
    'test or confidence interval is claimed. Declared test seeds do not certify absence from earlier '
    'source training. No checkpoint promotion or biological advantage is established. '
    'Hashes are local integrity receipts, not authenticity signatures; training is not replayed.'
)
# The first v1 run was already predeclared when seed-coupling prose and table
# scrolling were clarified. Preserve that exact declaration and rendering;
# every numerical/configuration check remains identical for both versions.
_LEGACY_SCOPE = (
    'Synthetic navigation learning study. All four arms and all initialization seeds are reported; '
    'there is no test-based selection among arms or seeds. Each run selects its checkpoint using '
    'validation success, SPL and collisions only, retaining the earlier checkpoint on ties. '
    'Raw policy is unassisted during frozen evaluation. Expert collection uses privileged full-map '
    'shortest-path labels with fixed N/E/S/W tie-breaking; the student sees only its observation window. '
    'Rotation augmentation transforms observations and labels, without providing architectural equivariance. '
    'No-graft, no-recurrence and no-context are inference interventions, not separately trained ablations. '
    'Context residual increases the parameter count while optimizer updates and raw sample budgets are matched. '
    'Random and shortest-path controls are reported separately. Episodes pooled across initialization '
    'seeds are correlated by scenario and are not independent replications; no episode-level significance '
    'test or confidence interval is claimed. Declared test seeds do not certify absence from earlier '
    'source training. No checkpoint promotion or biological advantage is established. '
    'Hashes are local integrity receipts, not authenticity signatures; training is not replayed.'
)


def _values(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(path):
    return hashlib.sha256(_read(Path(path))).hexdigest()


def _seeds(values):
    if (type(values) is not tuple or not 1 <= len(values) <= 3
            or any(type(seed) is not int or not 0 <= seed <= 2**63 - 1 for seed in values)
            or len(set(values)) != len(values)):
        raise ValueError('initialization seeds must be a tuple of 1-3 distinct integers in [0, 2**63-1]')
    return values


def _curriculum(values):
    if type(values) is not dict or set(values) != {field.name for field in fields(CurriculumConfig)}:
        raise ValueError('study plan must contain the complete curriculum config')
    copied = dict(values)
    for name in ('sizes', 'presets', 'validation_seeds', 'test_seeds'):
        if type(copied[name]) is not list:
            raise ValueError('study curriculum tuple fields require JSON arrays')
        copied[name] = tuple(copied[name])
    return CurriculumConfig(**copied)


def _base_config(config, seeds):
    if config is None:
        config = CurriculumConfig(worlds_per_round=24, updates_per_round=128, sizes=(7, 11),
                                  validation_seeds=tuple(range(50000, 50006)),
                                  test_seeds=tuple(range(60000, 60008)), max_steps=48)
    if not isinstance(config, CurriculumConfig):
        raise ValueError('config must be CurriculumConfig')
    return replace(config, init='fresh', collection_policy='expert', readout='human',
                   rotation_augmentation=False, include_readout_controls=True,
                   seed=seeds[0], train_seed_start=seeds[0])


def _plan(bundle, frozen, config, seeds):
    session = LiveSession.load(frozen, bundle)
    parameter_counts = {}
    for readout in ('human', 'context_residual'):
        with torch.random.fork_rng(devices=[]):
            model = GraphCircuit(bundle, replace(session.model.config, readout=readout))
        context = sum(value.numel() for name, value in model.named_parameters()
                      if name.startswith('context_policy.'))
        total = sum(value.numel() for value in model.parameters())
        parameter_counts[readout] = {'total': total, 'common': total - context, 'context_readout': context}
    runs = []
    for arm in ARMS:
        for seed in seeds:
            settings = replace(config, readout=arm['readout'],
                               rotation_augmentation=arm['rotation_augmentation'],
                               seed=seed, train_seed_start=seed)
            runs.append({'arm': arm['id'], 'seed': seed,
                         'directory': f"runs/{arm['id']}/seed-{seed}",
                         'config': _values(asdict(settings)),
                         'training_world_seeds': [condition['seed']
                             for round_idx in range(1, settings.rounds + 1)
                             for condition in _world_schedule(settings, round_idx)]})
    return {'schema': SCHEMA, 'initialization_seeds': list(seeds), 'arms': list(ARMS),
            'curriculum': _values(asdict(config)), 'runs': runs,
            'frozen_input_sha256': _digest(frozen),
            'frozen_model_hash': _checkpoint_digest(session.model.state_dict()),
            'graph_hash': bundle.digest(), 'frozen_config': _values(asdict(session.config)),
            'parameter_counts': parameter_counts,
            'frozen_checkpoint_steps': session.steps, 'frozen_checkpoint_updates': session.updates,
            'checkpoint_selection': 'validation success_rate, then spl, then lower collision_rate; earlier tie',
            'test_selection_performed': False, 'promotion_performed': False,
            'evidence_scope': _SCOPE}


def _hash_value(value, label):
    if (type(value) is not str or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'invalid matched {label} hash')
    return value


def _recognized_scope(scope):
    if type(scope) is not str or scope not in (_SCOPE, _LEGACY_SCOPE):
        raise ValueError('study plan has an unrecognized evidence scope')
    return scope


def _round_receipts(report):
    config = _curriculum(report['config'])
    if (any(type(entry['round']) is not int for entry in report['progress'])
            or [entry['round'] for entry in report['progress']] != list(range(config.rounds + 1))
            or report['progress'][0]['updates'] != 0):
        raise ValueError('training receipt rounds differ from the declared budget')
    receipts = []
    cumulative_actions = Counter()
    cumulative_transitions = 0
    names = ('north', 'east', 'south', 'west')

    def integer(value, minimum=0):
        return type(value) is int and value >= minimum

    def action_counts(values):
        return (type(values) is dict and set(values) == set(names)
                and all(integer(value) for value in values.values()))

    for round_idx, entry in enumerate(report['progress'][1:], 1):
        collection, training = entry['collection'], entry['training']
        _hash_value(collection['data_sha256'], 'collection data')
        _hash_value(training['batch_sha256'], 'raw training batch')
        schedule = _world_schedule(config, round_idx)
        conditions = Counter((item['size'], item['preset']) for item in schedule)
        expected_conditions = [{'size': size, 'preset': preset, 'worlds': count}
                               for (size, preset), count in sorted(conditions.items())]
        if (not integer(collection['worlds'], 1) or collection['worlds'] != len(schedule)
                or type(collection['world_seeds']) is not list
                or any(type(seed) is not int for seed in collection['world_seeds'])
                or collection['world_seeds'] != [item['seed'] for item in schedule]
                or type(collection['condition_counts']) is not list
                or any(type(condition) is not dict or type(condition.get('size')) is not int
                       or type(condition.get('worlds')) is not int
                       for condition in collection['condition_counts'])
                or collection['condition_counts'] != expected_conditions):
            raise ValueError('collection receipt differs from the declared training world schedule')
        transitions = collection['transitions']
        if (not integer(transitions, 1) or not len(schedule) <= transitions <= len(schedule) * config.max_steps
                or not action_counts(collection['action_counts'])
                or sum(collection['action_counts'].values()) != transitions
                or type(collection['teacher_probability']) not in (int, float)
                or collection['teacher_probability'] != 1.0
                or not integer(collection['teacher_steps']) or collection['teacher_steps'] != transitions
                or type(collection['learner_steps']) is not int or collection['learner_steps'] != 0):
            raise ValueError('expert collection receipt has invalid transition or teacher accounting')
        cumulative_transitions += transitions
        cumulative_actions.update(collection['action_counts'])
        replay = entry['replay']
        quotas = {name: config.replay_capacity // 4 + int(index < config.replay_capacity % 4)
                  for index, name in enumerate(names)}
        retained_counts = {name: min(quotas[name], cumulative_actions[name]) for name in names}
        if (replay['strategy'] != 'per-action-algorithm-r'
                or type(replay['capacity']) is not int or replay['capacity'] != config.replay_capacity
                or type(replay['seen_total']) is not int or replay['seen_total'] != cumulative_transitions
                or not action_counts(replay['seen_action_counts'])
                or replay['seen_action_counts'] != dict(cumulative_actions)
                or not action_counts(replay['action_quotas']) or replay['action_quotas'] != quotas
                or not action_counts(replay['action_counts']) or replay['action_counts'] != retained_counts
                or type(replay['retained_total']) is not int
                or replay['retained_total'] != sum(retained_counts.values())
                or not integer(replay['retained_worlds'], 1)
                or replay['retained_worlds'] > min(replay['retained_total'], round_idx * config.worlds_per_round)
                or type(replay['retained_rounds']) is not list or not replay['retained_rounds']
                or any(not integer(value, 1) or value > round_idx for value in replay['retained_rounds'])
                or replay['retained_rounds'] != sorted(set(replay['retained_rounds']))
                or len(replay['retained_rounds']) > replay['retained_worlds']
                or replay['retained_worlds'] > len(replay['retained_rounds']) * config.worlds_per_round):
            raise ValueError('replay receipt violates the declared reservoir capacity or accounting')
        if (type(entry['updates']) is not int or entry['updates'] != round_idx * config.updates_per_round
                or type(training['updates']) is not int or training['updates'] != config.updates_per_round
                or type(training['batch_size']) is not int or training['batch_size'] != config.batch_size
                or type(training['losses']) is not list or len(training['losses']) != config.updates_per_round
                or any(type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                       for loss in training['losses'])):
            raise ValueError('training receipt differs from the declared update or batch budget')
        rotations = training['rotation_counts']
        sample_count = config.updates_per_round * config.batch_size
        expected_rotations = [0] * 4
        augmentation_rng = random.Random(config.seed ^ (round_idx << 32) ^ 0x524F54415445)
        for _ in range(sample_count):
            expected_rotations[augmentation_rng.randrange(4) if config.rotation_augmentation else 0] += 1
        if (type(rotations) is not list or len(rotations) != 4
                or any(not integer(count) for count in rotations)
                or sum(rotations) != sample_count or rotations != expected_rotations):
            raise ValueError('rotation receipt differs from the declared augmentation schedule')
        receipts.append({'round': entry['round'], 'updates': entry['updates'],
                         'collection': collection, 'replay': entry['replay'],
                         'training': {key: training[key] for key in ('updates', 'batch_size', 'batch_sha256')}})
    return receipts


def _aggregate(plan, reports, plan_hash):
    """Derive every aggregate value and matching claim from the child reports."""
    scope = _recognized_scope(plan['evidence_scope'])
    matched = []
    scenario_identity = None
    variant_names = None
    for run in plan['runs']:
        report = reports[run['directory']]
        if report['config'] != run['config']:
            raise ValueError('child curriculum config differs from the predeclared study plan')
        actual_training_seeds = [seed for entry in report['progress'][1:]
                                 for seed in entry['collection']['world_seeds']]
        if actual_training_seeds != run['training_world_seeds']:
            raise ValueError('collection receipt differs from predeclared full training seeds')
        _round_receipts(report)
        for name in ('baseline', 'candidate'):
            evaluation = report[name]
            if evaluation['graph_hash'] != plan['graph_hash']:
                raise ValueError('child graph identity differs from the study plan')
            if name == 'baseline' and evaluation['model_hash'] != plan['frozen_model_hash']:
                raise ValueError('child baseline differs from the shared frozen input')
            current_names = tuple(evaluation['variants'])
            if variant_names is None:
                variant_names = current_names
            if current_names != variant_names:
                raise ValueError('study arms must expose identical evaluation variants')
            for variant in evaluation['variants'].values():
                episodes = variant['episodes']
                if (not episodes or [episode['seed'] for episode in episodes] != run['config']['test_seeds']
                        or variant['summary'] != _summary(episodes)):
                    raise ValueError('child evaluation summary or test seed identity mismatch')
                identity = [{key: episode[key] for key in
                             ('seed', 'initial_world_hash', 'start', 'goal', 'shortest_distance')}
                            for episode in episodes]
                if scenario_identity is None:
                    scenario_identity = identity
                if identity != scenario_identity:
                    raise ValueError('study evaluation scenarios are not matched across arms and variants')
    for seed in plan['initialization_seeds']:
        group = [reports[run['directory']] for run in plan['runs'] if run['seed'] == seed]
        reference = group[0]
        common_hash = _hash_value(reference['initial_common_model_hash'], 'initial common model')
        rounds = _round_receipts(reference)
        for report in group[1:]:
            if (report['initial_common_model_hash'] != common_hash
                    or report['protocol_id'] != reference['protocol_id']
                    or _round_receipts(report) != rounds):
                raise ValueError('matched arm receipt mismatch: common initialization, data, replay or batches')
        matched.append({'seed': seed, 'initial_common_model_hash': common_hash,
                        'protocol_id': reference['protocol_id'], 'rounds': rounds})
    arms = []
    for arm in plan['arms']:
        runs = [run for run in plan['runs'] if run['arm'] == arm['id']]
        variants = {}
        for variant in variant_names:
            episodes = [{**episode, 'initialization_seed': run['seed']}
                        for run in runs
                        for episode in reports[run['directory']]['candidate']['variants'][variant]['episodes']]
            variants[variant] = {'episodes': episodes, 'summary': _summary(episodes)}
        arms.append({**arm, 'variants': variants, 'parameter_counts': plan['parameter_counts'][arm['readout']],
                     'runs': [{'seed': run['seed'], 'directory': run['directory'],
                               'selected_round': reports[run['directory']]['selected_round'],
                               'selected_model_hash': reports[run['directory']]['selected_model_hash']}
                              for run in runs]})
    return {'schema': SCHEMA, 'plan_sha256': plan_hash, 'arms': arms,
            'initialization_seeds': plan['initialization_seeds'], 'matched_receipts': matched,
            'test_selection_performed': False, 'promotion_performed': False, 'evidence_scope': scope}


def _html(result):
    scope = _recognized_scope(result['evidence_scope'])
    table_style = ('border-collapse:collapse;width:100%;margin:1rem 0' if scope == _LEGACY_SCOPE else
                   'display:block;overflow-x:auto;border-collapse:collapse;width:100%;margin:1rem 0')
    def metrics(summary):
        return ''.join(f'<td>{escape(f"{summary[name]:.4f}")}</td>'
                       for name in ('success_rate', 'spl', 'collision_rate'))
    primary, controls = [], []
    for arm in result['arms']:
        links = ', '.join(f'<a href="{escape(run["directory"], quote=True)}/report.html">'
                          f'seed {run["seed"]}, selected round {run["selected_round"]}</a>' for run in arm['runs'])
        primary.append(f'<tr><th>{escape(arm["id"])}</th><td>{arm["parameter_counts"]["total"]}</td>'
                       f'{metrics(arm["variants"]["policy"]["summary"])}'
                       f'<td>{links}</td></tr>')
        for name, variant in arm['variants'].items():
            if name != 'policy':
                controls.append(f'<tr><th>{escape(arm["id"])}</th><td>{escape(name)}</td>'
                                f'{metrics(variant["summary"])}</tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Matched navigation readout study</title><style>'
            'body{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;'
            'color:#172733;background:#f7fafc}table{' + table_style + '}'
            'td,th{padding:.65rem;text-align:left;border:1px solid #b7cad6}th{background:#e4edf3}'
            'a{color:#075b88}p{line-height:1.55}td{font-variant-numeric:tabular-nums}</style>'
            '<h1>Matched navigation readout study</h1><p>' + escape(scope) + '</p>'
            '<p><a href="plan.json">Predeclared plan</a> · <a href="aggregate.json">Full aggregate receipts</a>'
            ' · <a href="manifest.json">Integrity manifest</a></p><h2>Raw unassisted candidate policy</h2>'
            '<p>Descriptive episode pooling across all declared initialization seeds. Each row retains every seed.</p>'
            '<table><thead><tr><th>Arm</th><th>Parameters</th><th>Success rate</th><th>SPL</th><th>Collision rate</th>'
            '<th>Individual archived runs</th></tr></thead><tbody>' + ''.join(primary) + '</tbody></table>'
            '<h2>Inference interventions and reference controls</h2><table><thead><tr><th>Arm</th>'
            '<th>Variant</th><th>Success rate</th><th>SPL</th><th>Collision rate</th></tr></thead><tbody>'
            + ''.join(controls) + '</tbody></table></html>\n')


def _manifest_files(plan):
    return _FILES | {run['directory'] + '/manifest.json' for run in plan['runs']}


def _safe_file(directory, relative):
    directory = Path(directory).resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or relative_path.drive or '..' in relative_path.parts:
        raise ValueError('study artifact path must be relative to its archive')
    path = directory / relative_path
    if not path.resolve().is_relative_to(directory):
        raise ValueError('study artifact path escapes its archive')
    for part in (path, *path.parents):
        if part == directory:
            return path
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('study artifact paths must not contain symlinks or junctions')
    raise ValueError('study artifact path escapes its archive')


def run_study(bundle, checkpoint, out_dir, config=None, seeds=(0, 1000, 2000)):
    """Freeze once, predeclare all runs, then retain all four matched arms."""
    directory = Path(out_dir)
    if directory.exists():
        raise ValueError(f'output already exists: {directory}; choose a new path')
    seeds = _seeds(seeds)
    config = _base_config(config, seeds)
    # Validate every seed-specific budget and overlap before creating artifacts.
    for seed in seeds:
        replace(config, seed=seed, train_seed_start=seed)
    directory.mkdir(parents=True, exist_ok=False)
    status_path = directory / 'status.json'
    status_path.write_text(json.dumps({'status': 'RUNNING', 'completed_runs': 0}), encoding='utf-8')
    try:
        frozen = capture_input(checkpoint, directory)
        _write_json(directory / 'source.bundle.json', bundle.to_dict())
        plan = _plan(bundle, frozen, config, seeds)
        _write_json(directory / 'plan.json', plan)
        reports = {}
        for run in plan['runs']:
            reports[run['directory']] = _values(run_navigation_training(
                bundle, frozen, directory / run['directory'], _curriculum(run['config'])))
            status_path.write_text(json.dumps({'status': 'RUNNING', 'completed_runs': len(reports)}), encoding='utf-8')
        result = _aggregate(plan, reports, _digest(directory / 'plan.json'))
        _write_json(directory / 'aggregate.json', result)
        (directory / 'report.html').write_bytes(_html(result).encode('utf-8'))
        status_path.write_text(json.dumps({'status': 'COMPLETE', 'runs': len(reports)}), encoding='utf-8')
        _write_json(directory / 'manifest.json', {'schema': SCHEMA,
                    'files': {name: _digest(directory / name) for name in sorted(_manifest_files(plan))}})
        verify_study(directory)
        return result
    except Exception as exc:
        status_path.write_text(json.dumps({'status': 'FAILED', 'error': str(exc)}), encoding='utf-8')
        raise


def verify_study(directory, replay=False):
    """Verify sealed children, reconstruct the declaration and derive all aggregates."""
    directory = Path(directory).resolve()
    try:
        plan = _json(_read(_safe_file(directory, 'plan.json'), 2 * 1024 * 1024))
        bundle = GraphBundle.from_dict(_json(_read(_safe_file(directory, 'source.bundle.json'))))
        seeds = _seeds(tuple(plan['initialization_seeds']))
        config = _base_config(_curriculum(plan['curriculum']), seeds)
        expected_plan = _plan(bundle, _safe_file(directory, 'frozen-input.pt'), config, seeds)
        expected_plan['evidence_scope'] = _recognized_scope(plan['evidence_scope'])
        if plan != expected_plan:
            raise ValueError('study plan config/frozen input identity mismatch')
        manifest = _json(_read(_safe_file(directory, 'manifest.json'), 65536))
        if (type(manifest) is not dict or set(manifest) != {'schema', 'files'}
                or manifest['schema'] != SCHEMA or type(manifest['files']) is not dict
                or set(manifest['files']) != _manifest_files(plan)):
            raise ValueError('study manifest must list exactly the declared artifacts')
        for relative, digest in manifest['files'].items():
            if _digest(_safe_file(directory, relative)) != digest:
                raise ValueError(f'study artifact hash mismatch: {relative}')
        status = _json(_read(directory / 'status.json'))
        if status != {'status': 'COMPLETE', 'runs': len(plan['runs'])}:
            raise ValueError('study status is not complete')
        reports = {}
        for run in plan['runs']:
            child = directory / run['directory']
            verify_navigation_run(child, replay=replay)
            if _digest(child / 'frozen-input.pt') != plan['frozen_input_sha256']:
                raise ValueError('child frozen input is not the shared study input')
            reports[run['directory']] = _json(_read(child / 'report.json'))
        expected = _aggregate(plan, reports, _digest(directory / 'plan.json'))
        actual = _json(_read(directory / 'aggregate.json'))
        if actual != expected:
            raise ValueError('study aggregate differs from its declared child archives')
        if _read(directory / 'report.html') != _html(expected).encode('utf-8'):
            raise ValueError('study HTML report differs from its derived aggregate')
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f'invalid study evidence: {exc}') from exc
    return {'ok': True, 'schema': SCHEMA, 'runs_verified': len(plan['runs']),
            'artifacts_verified': len(manifest['files']), 'replay_verified': bool(replay),
            'training_replayed': False, 'promotion_performed': False, 'evidence_scope': _SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--worlds', type=int, default=24)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--seeds', default='0,1000,2000')
    parser.add_argument('--verify', type=Path, help='verify an existing study without training')
    parser.add_argument('--replay', action='store_true', help='also replay every child frozen evaluation')
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            print(json.dumps(verify_study(args.verify, replay=args.replay), indent=2))
            return 0
        if args.bundle is None or args.checkpoint is None or args.out is None:
            parser.error('--bundle, --checkpoint and --out are required when running a study')
        seeds = tuple(int(value.strip()) for value in args.seeds.split(','))
        config = CurriculumConfig(rounds=args.rounds, worlds_per_round=args.worlds,
                                  updates_per_round=args.updates, sizes=(7, 11),
                                  presets=NavigationWorld.PRESETS, max_steps=48,
                                  validation_seeds=tuple(range(50000, 50006)),
                                  test_seeds=tuple(range(60000, 60008)))
        result = run_study(GraphBundle.load(args.bundle), args.checkpoint, args.out, config, seeds)
        print(json.dumps({'out': str(args.out.resolve()), 'arms': len(result['arms']),
                          'initialization_seeds': result['initialization_seeds'],
                          'promotion_performed': False}, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
