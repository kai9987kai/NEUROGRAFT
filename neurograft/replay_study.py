"""Matched empirical-model replay assays; no neural training or live checkpoint access.

The reference below independently solves the observed empirical MDP. Generated
experience is a synthetic test fixture, not evidence of autonomous exploration.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
import math
from pathlib import Path
import random

from .memory import TopologicalMemoryGraph

SCHEMA = 'empirical-replay-study-v1'
DEFAULT_SEEDS = tuple(range(13000, 13008))
DEFAULT_BUDGETS = (0, 4, 16, 64)
STRATEGIES = ('prioritized', 'uniform', 'none')
SCOPE = (
    'Empirical tabular value learning on independent seeded synthetic grid/branch graphs. '
    'Schedules share identical experience and backup caps; actual backups may differ. '
    'The oracle solves only experienced transition counts, not unseen true geometry. '
    'Partial exposure is a shared shuffled history prefix, not an autonomous exploration policy. '
    'Goal relocation starts from identical warmed state; the planner should invalidate its old goal values. '
    'No neural weights are trained or promoted; these assays establish no biological equivalence '
    'or general navigation improvement. Timings are not included in deterministic evidence.'
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n'


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _config(seeds, budgets, discount):
    if (not isinstance(seeds, (tuple, list)) or not 1 <= len(seeds) <= 64
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be 1-64 distinct integers in [0, 2**31)')
    if (not isinstance(budgets, (tuple, list)) or not 1 <= len(budgets) <= 16
            or any(type(b) is not int or not 0 <= b <= 4096 for b in budgets)
            or list(budgets) != sorted(set(budgets))):
        raise ValueError('budgets must be distinct ascending integers in [0, 4096]')
    try:
        valid_discount = (type(discount) in (int, float) and math.isfinite(discount)
                          and 0 <= discount <= .99)
    except OverflowError:
        valid_discount = False
    if not valid_discount:
        raise ValueError('study discount must be finite in [0, 0.99]')
    return {'seeds': list(seeds), 'budgets': list(budgets), 'discount': float(discount),
            'strategies': list(STRATEGIES), 'partial_history_fraction': .35,
            'reference_tolerance': 1e-13, 'warmup_budget': 4096}


def _history(seed):
    """Random spanning tree plus independently drawn extra directed experiences."""
    rng = random.Random(seed)
    size = rng.choice((4, 5))
    nodes = [(x, z) for x in range(1, size + 1) for z in range(1, size + 1)]
    allowed = set(nodes)
    kind = 'branch' if seed % 2 else 'grid'
    links = set()
    visited, stack = {nodes[0]}, [nodes[0]]
    while stack:
        u = stack[-1]
        candidates = [(u[0] + dx, u[1] + dz) for dx, dz in TopologicalMemoryGraph.ACTIONS]
        candidates = sorted(v for v in candidates if v in allowed and v not in visited)
        if candidates:
            v = rng.choice(candidates)
            links.add(tuple(sorted((u, v))))
            visited.add(v)
            stack.append(v)
        else:
            stack.pop()
    for u in nodes:
        for dx, dz in ((1, 0), (0, 1)):
            v = (u[0] + dx, u[1] + dz)
            if v in allowed and rng.random() < (.08 if kind == 'branch' else .65):
                links.add(tuple(sorted((u, v))))
    history = []
    for index, (u, v) in enumerate(sorted(links)):
        for source, target in ((u, v), (v, u)):
            # Some reverse actions have no experience: directed observation support.
            if source == v and index != 0 and rng.random() < .12:
                continue
            action = TopologicalMemoryGraph.ACTIONS.index((target[0] - source[0], target[1] - source[1]))
            for _ in range(rng.randint(2, 6)):
                history.append([list(source), action, list(target)])
            for _ in range(rng.randint(1 if index == 0 else 0, 3)):
                history.append([list(source), action, list(source)])
    rng.shuffle(history)
    return kind, nodes, history


def _graph(history):
    graph = TopologicalMemoryGraph()
    for source, action, target in history:
        graph.record_transition(tuple(source), action, tuple(target), 0.)
    return graph


def _q(graph, values, source, goal, discount):
    """Independent reference expectation; never calls ReplayPlanner."""
    result = {action: 0. for action in range(4)}
    if source == goal:
        return result
    for action in range(4):
        counts = graph.transitions.get((source, action), {})
        count = sum(counts.values())
        if count:
            result[action] = sum(n * (1. if target == goal else discount * values.get(target, 0.))
                                 for target, n in sorted(counts.items())) / count
    return result


def _residual(graph, goal, values, discount):
    return max((abs(values.get(source, 0.) - max(_q(graph, values, source, goal, discount).values()))
                for source in sorted(graph.nodes)), default=0.)


def _reference(graph, goal, discount):
    values = {source: 0. for source in sorted(graph.nodes)}
    for iteration in range(10000):
        updated = {source: max(_q(graph, values, source, goal, discount).values())
                   for source in sorted(graph.nodes)}
        delta = max((abs(updated[s] - values[s]) for s in values), default=0.)
        values = updated
        if delta < 1e-13:
            return values, iteration + 1, _residual(graph, goal, values, discount)
    raise RuntimeError('independent empirical-model reference did not converge')


def _metric(graph, goal, discount, values, reference):
    nodes = sorted(graph.nodes)
    decisive, matches = 0, 0
    decisions = []
    for source in nodes:
        expected = _q(graph, reference, source, goal, discount)
        actual = _q(graph, values, source, goal, discount)
        ordering = sorted(expected, key=lambda a: (-expected[a], a))
        if source != goal and expected[ordering[0]] - expected[ordering[1]] > 1e-9:
            decisive += 1
            selected = max(actual, key=lambda a: (actual[a], -a))
            matches += selected == ordering[0]
            decisions.append({'source': list(source), 'action': selected,
                              'reference_action': ordering[0],
                              'action_values': [actual[a] for a in range(4)]})
    return {'value_mae': sum(abs(values.get(n, 0.) - reference[n]) for n in nodes) / max(1, len(nodes)),
            'action_agreement': matches / decisive if decisive else None,
            'decisive_states': decisive, 'decisions': decisions,
            'bellman_residual': _residual(graph, goal, values, discount)}


def _rows(values, nodes):
    return [{'node': list(node), 'value': values.get(node, 0.)} for node in sorted(nodes)]


def _phase(graph, goal, warm_goal, config):
    from .replay import ReplayPlanner
    discount = config['discount']
    reference, iterations, residual = _reference(graph, goal, discount)
    warm = ReplayPlanner(discount=discount)
    warm_summary = (warm.replay(graph, warm_goal, budget=config['warmup_budget'], strategy='prioritized')
                    if warm_goal is not None else {'backups': 0})
    phase = {'id': 'fresh_goal' if warm_goal is None else 'relocated_goal', 'goal': list(goal),
             'warm_goal': list(warm_goal) if warm_goal is not None else None,
             'warm_backups': warm_summary['backups'], 'warm_values': _rows(warm.values, graph.nodes),
             'reference_iterations': iterations, 'reference_residual': residual,
             'reference_values': _rows(reference, graph.nodes), 'arms': []}
    for strategy in STRATEGIES:
        curve = []
        for budget in config['budgets']:
            planner = ReplayPlanner.from_state(warm.to_state())
            summary = planner.replay(graph, goal, budget=budget, strategy=strategy)
            metrics = _metric(graph, goal, discount, planner.values, reference)
            # Assert the tested planner's read surface agrees with the independent expectation.
            for source in sorted(graph.nodes):
                observed = planner.action_values(graph, source, goal)
                expected = _q(graph, planner.values, source, goal, discount)
                if any(abs(observed[a] - expected[a]) > 1e-11 for a in range(4)):
                    raise ValueError('planner action values disagree with independent empirical expectation')
            curve.append({'budget': budget, 'backups': summary['backups'],
                          'residual_before': summary['residual_before'],
                          'residual_after': summary['residual_after'],
                          'values': _rows(planner.values, graph.nodes), **metrics})
        phase['arms'].append({'strategy': strategy, 'curve': curve})
    return phase


def _aggregates(instances, config):
    result = []
    for exposure in ('partial', 'full'):
        for phase in ('fresh_goal', 'relocated_goal'):
            for strategy in STRATEGIES:
                for budget in config['budgets']:
                    points = [p for i in instances for e in i['exposures'] if e['id'] == exposure
                              for f in e['phases'] if f['id'] == phase
                              for a in f['arms'] if a['strategy'] == strategy
                              for p in a['curve'] if p['budget'] == budget]
                    agreements = [p['action_agreement'] for p in points if p['action_agreement'] is not None]
                    result.append({'exposure': exposure, 'phase': phase, 'strategy': strategy, 'budget': budget,
                                   'instances': len(points),
                                   'mean_value_mae': sum(p['value_mae'] for p in points) / len(points),
                                   'mean_action_agreement': sum(agreements) / len(agreements) if agreements else None,
                                   'mean_backups': sum(p['backups'] for p in points) / len(points),
                                   'max_residual': max(p['bellman_residual'] for p in points)})
    return result


def _compute(config):
    from .replay import ReplayPlanner
    instances = []
    for seed in config['seeds']:
        kind, generated_nodes, history = _history(seed)
        partial_count = max(1, int(len(history) * config['partial_history_fraction']))
        shared_nodes = sorted(_graph(history[:partial_count]).nodes)
        first_goal, second_goal = shared_nodes[-1], shared_nodes[0]
        exposures = []
        for name, experiences in (('partial', history[:partial_count]), ('full', history)):
            graph = _graph(experiences)
            unknown = ReplayPlanner(discount=config['discount'])
            unknown_summary = unknown.replay(graph, (30, 30), budget=max(config['budgets']), strategy='prioritized')
            exposures.append({'id': name, 'experience_fraction': len(experiences) / len(history),
                              'experienced_events': len(experiences), 'observed_nodes': len(graph.nodes),
                              'generated_nodes': len(generated_nodes), 'observed_state_actions': len(graph.transitions),
                              'history_sha256': _digest(experiences), 'history': experiences,
                              'graph_sha256': _digest(graph.to_state()),
                              'phases': [_phase(graph, first_goal, None, config),
                                         _phase(graph, second_goal, first_goal, config)],
                              'unknown_goal': {'goal': [30, 30], 'backups': unknown_summary['backups'],
                                               'max_value': max(unknown.values.values(), default=0.)}})
        instances.append({'seed': seed, 'kind': kind, 'full_history_sha256': _digest(history),
                          'stochastic_outcomes': sum(source == target for source, _, target in history),
                          'exposures': exposures})
    return {'schema': SCHEMA, 'configuration': config, 'evidence_scope': SCOPE,
            'promotion_performed': False, 'instances': instances,
            'aggregate': _aggregates(instances, config)}


def _render(report):
    rows = []
    for row in report['aggregate']:
        agreement = '-' if row['mean_action_agreement'] is None else f"{row['mean_action_agreement']:.3f}"
        rows.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in (
            row['exposure'], row['phase'], row['strategy'], row['budget'], f"{row['mean_backups']:.2f}",
            f"{row['mean_value_mae']:.6f}", agreement, f"{row['max_residual']:.3g}")) + '</tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>NEUROGRAFT empirical replay study</title><style>body{font:16px system-ui;max-width:1150px;margin:32px auto;'
            'padding:0 20px;background:#101722;color:#e8eff8}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}'
            'th,td{padding:9px;border-bottom:1px solid #3a4658;text-align:left}th{position:sticky;top:0;background:#1f2c40}'
            'p{line-height:1.6}code{color:#9bd4ff}</style><h1>Empirical replay learning study</h1><p>'
            + escape(report['evidence_scope']) + '</p><p><strong>Protocol:</strong> '
            + str(len(report['instances'])) + ' independent graph seeds; budgets '
            + escape(str(report['configuration']['budgets']))
            + '. Agreement is measured only on states with a unique reference-best action. '
            'Curves start from the same state at each budget; they are not cumulative. '
            'Goal changes reset stale values, so identical fresh/relocated results can be expected.</p>'
            '<p>Compare prioritized and uniform schedules at the same cap and inspect actual backups. '
            'Neither schedule is guaranteed to dominate. Raw histories, per-state values and decisive action traces '
            'are retained in <code>report.json</code>. Independent reference residuals must be below 1e-10.</p>'
            '<table><thead><tr><th>Exposure</th><th>Phase</th><th>Schedule</th><th>Cap</th><th>Backups</th>'
            '<th>Value MAE</th><th>Action agreement</th><th>Max residual</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></html>\n')


def run_replay_study(out, seeds=DEFAULT_SEEDS, budgets=DEFAULT_BUDGETS, discount=.9):
    """Write a new deterministic assay archive; never reads a live checkpoint."""
    config = _config(seeds, budgets, discount)
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'plan.json').write_text(_json({'schema': SCHEMA, 'configuration': config,
                                             'evidence_scope': SCOPE}), encoding='utf-8')
    report = _compute(config)
    (directory / 'report.json').write_text(_json(report), encoding='utf-8')
    (directory / 'report.html').write_text(_render(report), encoding='utf-8')
    manifest = {'schema': SCHEMA, 'files': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                          for name in ('plan.json', 'report.json', 'report.html')}}
    (directory / 'manifest.json').write_text(_json(manifest), encoding='utf-8')
    return report


def _reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate study JSON key')
        value[key] = item
    return value


def _read_json(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('study artifact exceeds size limit')
    def bad_constant(value):
        raise ValueError('nonfinite study JSON value')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_reject_duplicates,
                      parse_constant=bad_constant)


def _validate_content(report):
    config = report['configuration']
    if set(report) != {'schema', 'configuration', 'evidence_scope', 'promotion_performed', 'instances', 'aggregate'}:
        raise ValueError('invalid study content fields')
    if report['schema'] != SCHEMA or report['evidence_scope'] != SCOPE or report['promotion_performed'] is not False:
        raise ValueError('invalid study content scope')
    if len(report['instances']) != len(config['seeds']):
        raise ValueError('study instance count mismatch')
    for seed, instance in zip(config['seeds'], report['instances']):
        kind, generated, history = _history(seed)
        if (instance['seed'] != seed or instance['kind'] != kind or instance['full_history_sha256'] != _digest(history)
                or instance['stochastic_outcomes'] != sum(u == v for u, _, v in history)):
            raise ValueError('study generated history content mismatch')
        prefixes = (history[:max(1, int(len(history) * config['partial_history_fraction']))], history)
        if [e['id'] for e in instance['exposures']] != ['partial', 'full']:
            raise ValueError('study exposure content mismatch')
        shared_nodes = sorted(_graph(prefixes[0]).nodes)
        for stage, experiences in zip(instance['exposures'], prefixes):
            graph = _graph(experiences)
            expected_exposure = {'experience_fraction': len(experiences) / len(history),
                                 'experienced_events': len(experiences), 'observed_nodes': len(graph.nodes),
                                 'generated_nodes': len(generated), 'observed_state_actions': len(graph.transitions),
                                 'history': experiences, 'history_sha256': _digest(experiences),
                                 'graph_sha256': _digest(graph.to_state())}
            if any(stage.get(key) != value for key, value in expected_exposure.items()):
                raise ValueError('study exposure derived content mismatch')
            if stage['unknown_goal'] != {'goal': [30, 30], 'backups': 0, 'max_value': 0.}:
                raise ValueError('unknown-goal negative control mismatch')
            if [p['id'] for p in stage['phases']] != ['fresh_goal', 'relocated_goal']:
                raise ValueError('study phase content mismatch')
            for index, phase in enumerate(stage['phases']):
                goal = shared_nodes[-1] if index == 0 else shared_nodes[0]
                warm_goal = None if index == 0 else list(shared_nodes[-1])
                if phase['warm_goal'] != warm_goal:
                    raise ValueError('study relocation goal content mismatch')
                reference, iterations, residual = _reference(graph, goal, config['discount'])
                if (phase['goal'] != list(goal) or phase['reference_values'] != _rows(reference, graph.nodes)
                        or phase['reference_iterations'] != iterations or phase['reference_residual'] != residual):
                    raise ValueError('study reference derived content mismatch')
                if [a['strategy'] for a in phase['arms']] != list(STRATEGIES):
                    raise ValueError('invalid study schedules')
                for arm in phase['arms']:
                    if [p['budget'] for p in arm['curve']] != config['budgets']:
                        raise ValueError('invalid study backup caps')
                    for point in arm['curve']:
                        if (type(point['backups']) is not int or not 0 <= point['backups'] <= point['budget']
                                or (arm['strategy'] == 'none' and point['backups'] != 0)):
                            raise ValueError('invalid study backup accounting')
                        values = {tuple(row['node']): row['value'] for row in point['values']}
                        if (set(values) != graph.nodes or len(point['values']) != len(graph.nodes)
                                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1.
                                       for v in values.values())):
                            raise ValueError('invalid study value vector')
                        expected = _metric(graph, goal, config['discount'], values, reference)
                        if any(point.get(key) != value for key, value in expected.items()):
                            raise ValueError('study derived metric content mismatch')
                        if (type(point['residual_after']) not in (int, float)
                                or not math.isfinite(point['residual_after'])
                                or abs(point['residual_after'] - expected['bellman_residual']) > 1e-11):
                            raise ValueError('study residual derived metric mismatch')
    if report['aggregate'] != _aggregates(report['instances'], config):
        raise ValueError('study aggregate derived content mismatch')


def verify_replay_study(path, replay=False):
    """Verify seals and independent derived metrics; optionally rerun schedules exactly.

    Hashes are corruption checks, not signatures or authenticity proofs. Full replay
    additionally rejects coordinated resealing of value vectors and their metrics.
    """
    if type(replay) is not bool:
        raise ValueError('replay must be boolean')
    directory = Path(path)
    try:
        manifest = _read_json(directory / 'manifest.json')
        if (set(manifest) != {'schema', 'files'} or manifest['schema'] != SCHEMA
                or set(manifest['files']) != {'plan.json', 'report.json', 'report.html'}):
            raise ValueError('invalid study manifest')
        for name, expected in manifest['files'].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'study artifact hash mismatch: {name}')
        report = _read_json(directory / 'report.json')
        declared = report['configuration']
        config = _config(declared['seeds'], declared['budgets'], declared['discount'])
        if config != declared:
            raise ValueError('invalid study protocol content')
        plan = _read_json(directory / 'plan.json')
        if plan != {'schema': SCHEMA, 'configuration': config, 'evidence_scope': SCOPE}:
            raise ValueError('study plan content mismatch')
        _validate_content(report)
        if (directory / 'report.html').read_text(encoding='utf-8') != _render(report):
            raise ValueError('study HTML render content mismatch')
        if replay and _json(_compute(config)) != _json(report):
            raise ValueError('study deterministic replay content mismatch')
    except (KeyError, TypeError, IndexError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid replay study archive: {exc}') from exc
    return {'ok': True, 'schema': SCHEMA, 'instances': len(report['instances']),
            'deterministic_replay': replay, 'derived_metrics_verified': True,
            'report_sha256': manifest['files']['report.json'], 'evidence_scope': SCOPE}
