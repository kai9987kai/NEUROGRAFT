"""Predeclare, run and verify a bounded predictive/recovery navigation study.

This is a source-checkout research driver. Replay checks frozen evaluation and
prediction records; it does not reproduce training or authenticate local files.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
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

from neurograft.dynamics import DynamicsConfig, GraphCircuit
from neurograft.environment import NavigationWorld
from neurograft.graphs import GraphBundle
from neurograft.live import LiveSession, _checkpoint_digest
from neurograft.navigation_artifacts import _json, _read, _write_json, capture_input, verify_navigation_run
from neurograft.navigation_evaluation import _summary
from neurograft.navigation_training import CurriculumConfig, _world_schedule, run_navigation_training


SCHEMA = 'predictive-recovery-study-v1'
ARMS = (
    {'id': 'expert_aux0', 'collection_policy': 'expert', 'predictive_weight': 0.0},
    {'id': 'expert_aux05', 'collection_policy': 'expert', 'predictive_weight': 0.5},
    {'id': 'recovery_aux0', 'collection_policy': 'recovery', 'predictive_weight': 0.0},
    {'id': 'recovery_aux05', 'collection_policy': 'recovery', 'predictive_weight': 0.5},
)
VARIANTS = ('policy', 'no_graft', 'no_recurrence', 'random', 'shortest_path', 'no_context')
GROUPS = {'position': (0, 1), 'heading': (2, 3), 'goal': (4, 5, 6, 7),
          'walls': (8, 9, 10, 11), 'reward_clock': (12, 13, 14, 15)}
FILES = {'frozen-input.pt', 'source.bundle.json', 'plan.json', 'configs.json',
         'probe-transitions.json', 'prediction-probes.json', 'aggregate.json', 'report.html', 'status.json'}
BASELINE = (
    'copy_current predicts the unchanged latest observation. kinematic uses the executed cardinal '
    'action, adjacent wall bits and declared grid size: move by 1/(size-1) only when unblocked, '
    'set heading to that action, keep goal coordinates and recompute goal displacement, '
    'advance episode clock by 1/256, predict nonterminal reward -0.1 for collision or -0.01 '
    'otherwise, and retain progress/constant features. Adjacent walls are copied because newly '
    'visible walls are unknown. Terminal transitions, including goal changes and episode resets, '
    'are excluded from every predictor and baseline score.'
)
SCOPE = (
    'Synthetic CPU navigation experiment with identical context_residual/predictive-head architecture '
    'in all four arms. Auxiliary weights are 0 and 0.5; the zero-weight predictor remains untrained. '
    'One collection round precedes all optimizer updates, giving exactly matched trajectories and '
    'minibatches within each collection-policy/initialization pair. Expert and recovery exposures '
    'intentionally differ. Recovery is a 50/50 teacher/learner mixture, not DART noise injection. '
    'All arms and model seeds are retained; checkpoint selection uses validation policy success, '
    'SPL and collisions only, retaining round zero on ties. Neither held-out test outcomes nor '
    'prediction probes select a candidate or arm. Raw policy tests disable teachers and learning. '
    'Full-map expert labels are privileged; the student receives local observation windows. '
    'Held-out prediction probes use one common frozen random-action corpus, report all feature '
    'groups and compare learned prediction with persistence and an explicit kinematic baseline. '
    'Prediction is an auxiliary representation objective, not rollout planning. Eight default test '
    'scenarios repeated across two model seeds are correlated, not sixteen independent episodes; '
    'no episode-level significance or confidence interval is claimed. Test seeds are disjoint '
    'from this study training/validation schedule, not certified absent from earlier source training. '
    'No checkpoint promotion, biological validation or connectome advantage is established. '
    'Hashes are local integrity receipts, not authenticity signatures. Optional replay checks '
    'frozen evaluation and prediction records in this CPU/runtime; training is not replayed.'
)


def _values(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _digest(path):
    return hashlib.sha256(_read(Path(path))).hexdigest()


def _value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _safe_file(directory, relative):
    directory = Path(directory).resolve()
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or '..' in relative.parts:
        raise ValueError('study archive path must be relative and remain inside its directory')
    path = directory / relative
    if not path.resolve().is_relative_to(directory):
        raise ValueError('study archive path escapes its directory')
    for parent in (path, *path.parents):
        if parent == directory:
            return path
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ValueError('study archive path contains a symlink or junction')
    raise ValueError('invalid study archive path')


def _seeds(seeds):
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= 2
            or any(type(seed) is not int or not 0 <= seed <= 2**63 - 1 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('initialization seeds must be 1-2 distinct integer seeds')
    return seeds


def _curriculum(values):
    if type(values) is not dict or set(values) != {field.name for field in fields(CurriculumConfig)}:
        raise ValueError('study requires the complete curriculum config')
    values = dict(values)
    for name in ('sizes', 'presets', 'validation_seeds', 'test_seeds'):
        if type(values[name]) is not list:
            raise ValueError('curriculum tuple fields must be JSON arrays')
        values[name] = tuple(values[name])
    return CurriculumConfig(**values)


def _base_config(config, seeds):
    if config is None:
        config = CurriculumConfig(rounds=1, worlds_per_round=24, updates_per_round=256,
                                  sizes=(7, 11), presets=NavigationWorld.PRESETS, max_steps=48,
                                  validation_seeds=tuple(range(55000, 55004)),
                                  test_seeds=tuple(range(65000, 65008)), eval_size=11)
    if (not isinstance(config, CurriculumConfig) or config.rounds != 1
            or config.worlds_per_round > 24 or config.updates_per_round > 256 or config.max_steps > 48):
        raise ValueError('study requires one collection round, at most 24 worlds, 256 updates and 48 steps')
    return replace(config, init='fresh', readout='context_residual', predictive_head=True,
                   predictive_weight=0.0, collection_policy='expert', rotation_augmentation=False,
                   include_readout_controls=True, seed=seeds[0], train_seed_start=seeds[0])


def _probe_config(transitions):
    if type(transitions) is not int or not 1 <= transitions <= 64:
        raise ValueError('probe transitions per condition must be in [1,64]')
    return {'seeds': [75000, 75001], 'size': 11, 'presets': ['labyrinth', 'corridors'],
            'transitions_per_condition': transitions, 'unroll': 4, 'attempt_multiplier': 4}


def probe_corpus(config):
    """Generate common private-RNG transitions, clearing history at discontinuities."""
    if config != _probe_config(config.get('transitions_per_condition')):
        raise ValueError('invalid prediction probe config')
    transitions, conditions = [], []
    for preset_index, preset in enumerate(config['presets']):
        for seed in config['seeds']:
            world = NavigationWorld(seed=seed, size=config['size'], preset=preset)
            rng = random.Random(seed ^ 0x50524544494354 ^ (preset_index << 32))
            history = deque(maxlen=4)
            valid = attempted = excluded = 0
            target = config['transitions_per_condition']
            while valid < target and attempted < config['attempt_multiplier'] * target:
                observation = world.observe()
                history.append(observation)
                padded = [[[0.0] * 4 for _ in range(4)] for _ in range(4 - len(history))] + list(history)
                action = rng.randrange(4)
                outcome = world.step(action)
                attempted += 1
                if outcome['episode_done']:
                    excluded += 1
                    history.clear()
                    continue
                transitions.append({'seed': seed, 'preset': preset, 'size': world.size,
                                    'observations': padded, 'action': action,
                                    'next_observation': world.observe()})
                valid += 1
            if not valid:
                raise ValueError('prediction probe condition produced no valid transitions')
            conditions.append({'seed': seed, 'preset': preset, 'valid_transitions': valid,
                               'attempted_steps': attempted, 'terminal_excluded': excluded})
    return {'schema': SCHEMA + '-transitions', 'config': config,
            'conditions': conditions, 'transitions': transitions}


def kinematic_prediction(observation, action, size):
    """Local nonterminal kinematics; only the newly visible wall row is approximated."""
    predicted = [list(row) for row in observation]
    dx, dz = NavigationWorld.ACTIONS[action]
    collision = observation[2][action] > 0.5
    if not collision:
        predicted[0][0] += dx / (size - 1)
        predicted[0][1] += dz / (size - 1)
    predicted[0][2:] = [float(dx), float(dz)]
    predicted[1][2:] = [predicted[1][0] - predicted[0][0], predicted[1][1] - predicted[0][1]]
    predicted[3][0] = -0.1 if collision else -0.01
    predicted[3][1] += 1 / NavigationWorld.MAX_EPISODE_STEPS
    return predicted


def _metrics(predictions, corpus):
    count = len(corpus['transitions'])
    try:
        actual = torch.tensor(predictions, dtype=torch.float64)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError('invalid prediction probe values') from exc
    if actual.shape != (count, 4, 4) or not torch.isfinite(actual).all():
        raise ValueError('prediction probe values must be finite with shape [transitions,4,4]')
    target = torch.tensor([item['next_observation'] for item in corpus['transitions']], dtype=torch.float64)
    errors = (actual - target).square().reshape(count, 16)
    return {'overall_mse': float(errors.mean()),
            'groups': {name: float(errors[:, indices].mean()) for name, indices in GROUPS.items()}}


def _make_plan(bundle, frozen, config, seeds, corpus):
    session = LiveSession.load(frozen, bundle)
    with torch.random.fork_rng(devices=[]):
        model = GraphCircuit(bundle, DynamicsConfig(hidden_dim=session.config.hidden_dim, output_dim=4,
                                                   readout='context_residual', predictive_head=True))
    counts = {'total': sum(parameter.numel() for parameter in model.parameters()),
              'context_residual': sum(parameter.numel() for name, parameter in model.named_parameters()
                                      if name.startswith('context_policy.')),
              'predictor': sum(parameter.numel() for name, parameter in model.named_parameters()
                               if name.startswith('predictor.'))}
    runs = []
    probe_seeds = set(corpus['config']['seeds'])
    for arm in ARMS:
        for seed in seeds:
            settings = replace(config, seed=seed, train_seed_start=seed,
                               collection_policy=arm['collection_policy'], predictive_weight=arm['predictive_weight'])
            schedule = _world_schedule(settings, 1)
            if probe_seeds & (set(settings.validation_seeds) | set(settings.test_seeds)
                              | {condition['seed'] for condition in schedule}):
                raise ValueError('prediction probe seeds must be disjoint from training/validation/test seeds')
            runs.append({'arm': arm['id'], 'seed': seed, 'directory': f"runs/{arm['id']}/seed-{seed}",
                         'config': _values(asdict(settings)), 'training_worlds': schedule})
    return {'schema': SCHEMA, 'arms': list(ARMS), 'initialization_seeds': list(seeds),
            'curriculum': _values(asdict(config)), 'runs': runs, 'parameter_counts': counts,
            'graph_hash': bundle.digest(), 'frozen_input_sha256': _digest(frozen),
            'frozen_model_hash': _checkpoint_digest(session.model.state_dict()),
            'frozen_config': _values(asdict(session.config)), 'probe_config': corpus['config'],
            'probe_corpus_sha256': _value_digest(corpus), 'prediction_baselines': BASELINE,
            'checkpoint_selection': 'validation success_rate, spl, lower collision_rate; earlier round on ties',
            'test_selection_performed': False, 'probe_selection_performed': False,
            'promotion_performed': False, 'evidence_scope': SCOPE}


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid study receipt hash')
    return value


def _receipt(report, run, plan):
    if report['config'] != run['config'] or report['parameter_counts'] != plan['parameter_counts']:
        raise ValueError('child config or parameter counts differ from the study plan')
    cfg = _curriculum(run['config'])
    progress = report['progress']
    if len(progress) != 2 or [entry['round'] for entry in progress] != [0, 1]:
        raise ValueError('training receipt must contain rounds zero and one')
    entry = progress[1]
    collection, training = entry['collection'], entry['training']
    schedule = run['training_worlds']
    conditions = Counter((item['size'], item['preset']) for item in schedule)
    expected = [{'size': size, 'preset': preset, 'worlds': count}
                for (size, preset), count in sorted(conditions.items())]
    transitions = collection['transitions']
    if (collection['worlds'] != cfg.worlds_per_round or collection['condition_counts'] != expected
            or collection['world_seeds'] != [item['seed'] for item in schedule]
            or type(transitions) is not int or not len(schedule) <= transitions <= len(schedule) * cfg.max_steps
            or sum(collection['action_counts'].values()) != transitions
            or collection['teacher_steps'] + collection['learner_steps'] != transitions
            or collection['teacher_probability'] != (1.0 if cfg.collection_policy == 'expert' else 0.5)
            or (cfg.collection_policy == 'expert' and collection['learner_steps'] != 0)
            or collection['prediction_valid_transitions'] + collection['prediction_excluded_transitions'] != transitions):
        raise ValueError('collection receipt violates the declared schedule or transition budget')
    if (entry['updates'] != cfg.updates_per_round or training['updates'] != cfg.updates_per_round
            or training['batch_size'] != cfg.batch_size or len(training['losses']) != cfg.updates_per_round
            or training['rotation_counts'] != [cfg.updates_per_round * cfg.batch_size, 0, 0, 0]
            or any(type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0 for loss in training['losses'])):
        raise ValueError('training receipt violates the declared update or batch budget')
    prediction = training['prediction']
    counts, losses, totals = prediction['valid_counts'], prediction['losses'], prediction['total_losses']
    if (prediction['weight'] != cfg.predictive_weight or len(counts) != cfg.updates_per_round
            or len(losses) != len(counts) or len(totals) != len(counts)
            or any(type(count) is not int or not 0 <= count <= cfg.batch_size for count in counts)
            or prediction['valid_examples'] != sum(counts)):
        raise ValueError('prediction training receipt has invalid accounting')
    for ce, loss, total, count in zip(training['losses'], losses, totals, counts):
        if ((count == 0) != (loss is None) or (loss is not None and
                (type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0))
                or type(total) not in (int, float) or not math.isfinite(total)
                or not math.isclose(total, ce + cfg.predictive_weight * (loss or 0), rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError('prediction training loss receipt is inconsistent')
    mean = sum((loss or 0) * count for loss, count in zip(losses, counts)) / sum(counts) if sum(counts) else None
    if prediction['mean_mse'] != mean:
        raise ValueError('prediction training mean differs from raw losses')
    return {'initial_model_hash': _hash(report['initial_model_hash']),
            'data_sha256': _hash(collection['data_sha256']), 'batch_sha256': _hash(training['batch_sha256']),
            'base_data_sha256': _hash(collection['base_data_sha256']),
            'base_batch_sha256': _hash(training['base_batch_sha256'])}


def _aggregate(plan, reports, plan_hash, probes):
    receipts, identity = {}, None
    for run in plan['runs']:
        report = reports[run['directory']]
        receipts[run['directory']] = _receipt(report, run, plan)
        for kind in ('baseline', 'candidate'):
            evaluation = report[kind]
            if tuple(evaluation['variants']) != VARIANTS or evaluation['graph_hash'] != plan['graph_hash']:
                raise ValueError('child evaluation variants or graph identity differ from the plan')
            if kind == 'baseline' and evaluation['model_hash'] != plan['frozen_model_hash']:
                raise ValueError('child baseline differs from the frozen source')
            for variant in evaluation['variants'].values():
                episodes = variant['episodes']
                if ([item['seed'] for item in episodes] != run['config']['test_seeds']
                        or variant['summary'] != _summary(episodes)):
                    raise ValueError('child test summary differs from raw episodes')
                current = [{key: item[key] for key in ('seed', 'initial_world_hash', 'start', 'goal', 'shortest_distance')}
                           for item in episodes]
                identity = current if identity is None else identity
                if current != identity:
                    raise ValueError('child frozen test scenarios are not matched')
    pairs = []
    for seed in plan['initialization_seeds']:
        seed_runs = [run for run in plan['runs'] if run['seed'] == seed]
        if len({receipts[run['directory']]['initial_model_hash'] for run in seed_runs}) != 1:
            raise ValueError('matched architecture initialization hashes differ')
        for policy in ('expert', 'recovery'):
            pair = [run for run in seed_runs if run['config']['collection_policy'] == policy]
            left, right = (receipts[run['directory']] for run in pair)
            if left != right:
                raise ValueError('matched auxiliary pair initialization/data/batch receipt mismatch')
            pairs.append({'seed': seed, 'collection_policy': policy, **left})
    arms = []
    for arm in ARMS:
        runs = [run for run in plan['runs'] if run['arm'] == arm['id']]
        variants = {}
        for variant in VARIANTS:
            episodes = [{**episode, 'initialization_seed': run['seed']}
                        for run in runs for episode in reports[run['directory']]['candidate']['variants'][variant]['episodes']]
            variants[variant] = {'episodes': episodes, 'summary': _summary(episodes)}
        arms.append({**arm, 'parameter_counts': plan['parameter_counts'], 'variants': variants,
                     'runs': [{'seed': run['seed'], 'directory': run['directory'],
                               'selected_round': reports[run['directory']]['selected_round'],
                               'selected_model_hash': reports[run['directory']]['selected_model_hash'],
                               'prediction_mse': next(item['metrics'] for item in probes['runs']
                                                      if item['directory'] == run['directory'])}
                              for run in runs]})
    return {'schema': SCHEMA, 'plan_sha256': plan_hash, 'arms': arms, 'matched_pairs': pairs,
            'initialization_seeds': plan['initialization_seeds'],
            'test_selection_performed': False, 'probe_selection_performed': False,
            'promotion_performed': False, 'evidence_scope': SCOPE}


def _probe_report(plan, corpus, run_predictions):
    records = []
    for run, record in zip(plan['runs'], run_predictions):
        if record['directory'] != run['directory']:
            raise ValueError('prediction probe run identity differs from the plan')
        records.append({'directory': run['directory'], 'model_hash': _hash(record['model_hash']),
                        'predictions': record['predictions'], 'metrics': _metrics(record['predictions'], corpus)})
    if len(records) != len(plan['runs']) or len(run_predictions) != len(records):
        raise ValueError('prediction probes must retain every declared run')
    persistence = [item['observations'][-1] for item in corpus['transitions']]
    kinematic = [kinematic_prediction(item['observations'][-1], item['action'], item['size'])
                 for item in corpus['transitions']]
    return {'schema': SCHEMA + '-probes', 'probe_corpus_sha256': _value_digest(corpus),
            'valid_transitions': len(corpus['transitions']),
            'terminal_excluded': sum(item['terminal_excluded'] for item in corpus['conditions']),
            'conditions': corpus['conditions'], 'baseline_definition': BASELINE,
            'baselines': {'copy_current': _metrics(persistence, corpus), 'kinematic': _metrics(kinematic, corpus)},
            'runs': records, 'selection_performed': False}


def prediction_probes(directory, plan, bundle, corpus):
    """Run selected checkpoints frozen, without consuming global RNG or changing files."""
    records = []
    with torch.inference_mode():
        for run in plan['runs']:
            path = _safe_file(directory, run['directory'] + '/candidate.pt')
            session = LiveSession.load(path, bundle)
            session.model.eval()
            if not session.config.predictive_head:
                raise ValueError('probe candidate does not have a predictive head')
            sequence = torch.tensor([item['observations'] for item in corpus['transitions']], dtype=session.model.policy.weight.dtype)
            actions = torch.tensor([item['action'] for item in corpus['transitions']], dtype=torch.long)
            prediction = session.model(sequence, action=actions).predicted_obs
            records.append({'directory': run['directory'], 'model_hash': _checkpoint_digest(session.model.state_dict()),
                            'predictions': prediction.tolist()})
    return _probe_report(plan, corpus, records)


def _html(result, probes):
    rows, controls, prediction_rows = [], [], []
    def metrics(summary):
        return ''.join(f'<td>{summary[key]:.4f}</td>' for key in ('success_rate', 'spl', 'collision_rate'))
    for arm in result['arms']:
        links = ', '.join(f'<a href="{escape(run["directory"], quote=True)}/report.html">seed {run["seed"]}, '
                          f'round {run["selected_round"]}</a>' for run in arm['runs'])
        rows.append(f'<tr><th>{arm["id"]}</th><td>{arm["parameter_counts"]["total"]}</td>'
                    + metrics(arm['variants']['policy']['summary']) + f'<td>{links}</td></tr>')
        for variant in VARIANTS[1:]:
            controls.append(f'<tr><th>{arm["id"]}</th><td>{variant}</td>' + metrics(arm['variants'][variant]['summary']) + '</tr>')
    for name, value in list(probes['baselines'].items()) + [(item['directory'], item['metrics']) for item in probes['runs']]:
        prediction_rows.append(f'<tr><th>{escape(name)}</th><td>{value["overall_mse"]:.6f}</td>'
                               + ''.join(f'<td>{value["groups"][group]:.6f}</td>' for group in GROUPS) + '</tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Predictive recovery study</title><style>body{font:16px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem;'
            'background:#f7fafc;color:#172733}table{display:block;overflow-x:auto;border-collapse:collapse;margin:1rem 0;max-width:100%}'
            'td,th{padding:.6rem;border:1px solid #b7cad6;text-align:left}th{background:#e4edf3}p{line-height:1.55}</style>'
            '<h1>Predictive recovery study</h1><p>' + escape(SCOPE) + '</p><p><a href="plan.json">Predeclared plan</a> · '
            '<a href="aggregate.json">Full policy records</a> · <a href="prediction-probes.json">Raw prediction records</a> · '
            '<a href="probe-transitions.json">Common probe transitions</a> · <a href="manifest.json">Manifest</a></p>'
            '<h2>Raw unassisted policy</h2><table><tr><th>Arm</th><th>Parameters</th><th>Success</th><th>SPL</th><th>Collisions</th>'
            '<th>Every initialization</th></tr>' + ''.join(rows) + '</table><h2>Inference interventions and references</h2>'
            '<table><tr><th>Arm</th><th>Variant</th><th>Success</th><th>SPL</th><th>Collisions</th></tr>' + ''.join(controls)
            + '</table><h2>Held-out one-step prediction</h2><p>' + escape(BASELINE) + '</p><p>'
            + str(probes['valid_transitions']) + ' shared nonterminal transitions; ' + str(probes['terminal_excluded'])
            + ' terminal transitions excluded. Overall MSE weights all sixteen scalars equally. Group MSE averages that group.</p>'
            '<table><tr><th>Predictor</th><th>Overall</th>' + ''.join('<th>' + group + '</th>' for group in GROUPS)
            + '</tr>' + ''.join(prediction_rows) + '</table></html>\n')


def _manifest_files(plan):
    return FILES | {run['directory'] + '/manifest.json' for run in plan['runs']}


def run_study(bundle, checkpoint, out_dir, config=None, seeds=(0, 1000), probe_transitions=64):
    """Freeze source and declare every config/corpus before collecting any outcomes."""
    directory = Path(out_dir)
    if directory.exists():
        raise ValueError(f'output already exists: {directory}; choose a new path')
    seeds = _seeds(seeds)
    config = _base_config(config, seeds)
    probe_config = _probe_config(probe_transitions)
    for seed in seeds:
        replace(config, seed=seed, train_seed_start=seed)
    directory.mkdir(parents=True, exist_ok=False)
    status = directory / 'status.json'
    status.write_text(json.dumps({'status': 'RUNNING', 'completed_runs': 0}), encoding='utf-8')
    try:
        frozen = capture_input(checkpoint, directory)
        _write_json(directory / 'source.bundle.json', bundle.to_dict())
        corpus = probe_corpus(probe_config)
        plan = _make_plan(bundle, frozen, config, seeds, corpus)
        _write_json(directory / 'plan.json', plan)
        _write_json(directory / 'configs.json', {run['directory']: run['config'] for run in plan['runs']})
        _write_json(directory / 'probe-transitions.json', corpus)
        reports = {}
        for run in plan['runs']:
            reports[run['directory']] = _values(run_navigation_training(bundle, frozen,
                _safe_file(directory, run['directory']), _curriculum(run['config'])))
            status.write_text(json.dumps({'status': 'RUNNING', 'completed_runs': len(reports),
                                         'total_runs': len(plan['runs'])}), encoding='utf-8')
        probes = prediction_probes(directory, plan, bundle, corpus)
        result = _aggregate(plan, reports, _digest(directory / 'plan.json'), probes)
        _write_json(directory / 'prediction-probes.json', probes)
        _write_json(directory / 'aggregate.json', result)
        (directory / 'report.html').write_bytes(_html(result, probes).encode('utf-8'))
        status.write_text(json.dumps({'status': 'COMPLETE', 'runs': len(reports)}), encoding='utf-8')
        _write_json(directory / 'manifest.json', {'schema': SCHEMA,
                    'files': {name: _digest(_safe_file(directory, name)) for name in sorted(_manifest_files(plan))}})
        verify_study(directory)
        return result
    except Exception as exc:
        status.write_text(json.dumps({'status': 'FAILED', 'error': str(exc)}), encoding='utf-8')
        raise


def verify_study(directory, replay=False):
    """Derive summaries from sealed child records; optional replay also reruns inference."""
    directory = Path(directory).resolve()
    try:
        plan = _json(_read(_safe_file(directory, 'plan.json'), 2 * 1024 * 1024))
        bundle = GraphBundle.from_dict(_json(_read(_safe_file(directory, 'source.bundle.json'))))
        seeds = _seeds(tuple(plan['initialization_seeds']))
        config = _base_config(_curriculum(plan['curriculum']), seeds)
        corpus = _json(_read(_safe_file(directory, 'probe-transitions.json')))
        if corpus != probe_corpus(plan['probe_config']):
            raise ValueError('prediction probe corpus differs from its declared generator')
        expected_plan = _make_plan(bundle, _safe_file(directory, 'frozen-input.pt'), config, seeds, corpus)
        if plan != expected_plan:
            raise ValueError('study plan differs from its source/config identity')
        manifest = _json(_read(_safe_file(directory, 'manifest.json'), 65536))
        if (type(manifest) is not dict or set(manifest) != {'schema', 'files'} or manifest['schema'] != SCHEMA
                or type(manifest['files']) is not dict or set(manifest['files']) != _manifest_files(plan)):
            raise ValueError('study manifest must contain exactly the declared artifacts')
        for name, digest in manifest['files'].items():
            if _digest(_safe_file(directory, name)) != digest:
                raise ValueError(f'study artifact hash mismatch: {name}')
        if _json(_read(directory / 'configs.json')) != {run['directory']: run['config'] for run in plan['runs']}:
            raise ValueError('predeclared configs differ from the plan')
        if _json(_read(directory / 'status.json')) != {'status': 'COMPLETE', 'runs': len(plan['runs'])}:
            raise ValueError('study status is not complete')
        reports = {}
        for run in plan['runs']:
            child = _safe_file(directory, run['directory'])
            # Validate every child path before delegating to the existing archive reader.
            from neurograft.navigation_artifacts import ARTIFACTS
            for name in ARTIFACTS | {'manifest.json'}:
                _safe_file(directory, run['directory'] + '/' + name)
            verify_navigation_run(child, replay=replay)
            if _digest(child / 'frozen-input.pt') != plan['frozen_input_sha256']:
                raise ValueError('child frozen source differs from the shared study input')
            reports[run['directory']] = _json(_read(child / 'report.json'))
        probes = _json(_read(directory / 'prediction-probes.json'))
        expected_probes = _probe_report(plan, corpus, probes['runs'])
        if probes != expected_probes:
            raise ValueError('prediction probe metrics differ from their raw prediction records')
        for run in plan['runs']:
            record = next(item for item in probes['runs'] if item['directory'] == run['directory'])
            if record['model_hash'] != reports[run['directory']]['selected_model_hash']:
                raise ValueError('prediction probe checkpoint identity differs from selected candidate')
        if replay and prediction_probes(directory, plan, bundle, corpus) != probes:
            raise ValueError('prediction probe replay differs in this runtime')
        aggregate = _aggregate(plan, reports, _digest(directory / 'plan.json'), probes)
        if _json(_read(directory / 'aggregate.json')) != aggregate:
            raise ValueError('study aggregate differs from declared child archives and probes')
        if _read(directory / 'report.html') != _html(aggregate, probes).encode('utf-8'):
            raise ValueError('study HTML differs from its derived aggregate and probe records')
    except (KeyError, TypeError, AttributeError, IndexError, StopIteration) as exc:
        raise ValueError(f'invalid study evidence: {exc}') from exc
    return {'ok': True, 'schema': SCHEMA, 'runs_verified': len(plan['runs']),
            'replay_verified': bool(replay), 'training_replayed': False,
            'prediction_records_verified': True, 'promotion_performed': False, 'evidence_scope': SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seeds', default='0,1000')
    parser.add_argument('--updates', type=int, default=256)
    parser.add_argument('--probe-transitions', type=int, default=64)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            print(json.dumps(verify_study(args.verify, replay=args.replay), indent=2))
        else:
            if args.bundle is None or args.checkpoint is None or args.out is None:
                parser.error('--bundle, --checkpoint and --out are required for a study run')
            seeds = _seeds(tuple(int(seed) for seed in args.seeds.split(',')))
            config = replace(_base_config(None, seeds), updates_per_round=args.updates)
            result = run_study(GraphBundle.load(args.bundle), args.checkpoint, args.out, config,
                               seeds=seeds, probe_transitions=args.probe_transitions)
            print(json.dumps({'out': str(args.out.resolve()), 'arms': len(result['arms']),
                              'initialization_seeds': result['initialization_seeds'],
                              'promotion_performed': False}, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
