"""Controlled transition revision with an independent current-dynamics oracle.

The workbench receives supplied transition outcomes, never the held-out dynamics.
These finite synthetic assays do not open checkpoints, train a policy or promote it.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
from html import escape
import json
import math
from pathlib import Path
import random

from .memory import TopologicalMemoryGraph

SCHEMA = 'transition-revision-study-v1'
DEFAULT_SEEDS = tuple(range(14000, 14008))
MODELS = ('cumulative', 'recent')
STRATEGIES = ('prioritized', 'uniform', 'none')
DISCOUNT = .9
SCOPE = (
    'Exploratory finite synthetic two-route assays with supplied state/action outcomes. '
    'The six arms share each ordered evidence stream and requested backup cap. '
    'Current dynamics are independently specified and withheld from the workbench; '
    'empirical-model planning error and current-dynamics error are separate measures. '
    'No neural policy, autonomous exploration, biological equivalence or checkpoint promotion is tested. '
    'Equal backup caps do not imply equal computation. Seeds draw geometry, dynamics and counts; '
    'cases and phases within a seed are correlated, not additional independent replicates.'
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n'


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _config(seeds, window, budget):
    if (type(seeds) not in (list, tuple) or not 1 <= len(seeds) <= 32
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be 1-32 distinct integers in [0, 2**31)')
    if type(window) is not int or window not in (4, 8, 16):
        raise ValueError('window must be 4, 8 or 16')
    if type(budget) is not int or not 0 <= budget <= 256:
        raise ValueError('budget must be an integer from 0 through 256')
    return {'seeds': list(seeds), 'window': window, 'budget': budget,
            'warmup_budget': 256, 'discount': DISCOUNT, 'models': list(MODELS),
            'strategies': list(STRATEGIES), 'stationary_sample_per_action': 16,
            'policy_tie_break': 'smallest action index N,E,S,W',
            'reference_tolerance': 1e-13}


def _protocol(config):
    return {
        'schema': SCHEMA, 'config': config,
        'hypotheses': [
            'Recent evidence can reduce stale-model prediction and decision error after closure.',
            'The same recency rule can increase prediction error in stationary stochastic dynamics.',
            'Replay schedule changes empirical planning error without adding outcome information.',
        ],
        'model_rule': ('Cumulative retains baseline counts plus every new outcome. Recent retains '
                       'baseline counts for untouched state/actions; the first new outcome replaces '
                       'that pair with its last W explicitly supplied new outcomes. Baseline '
                       'aggregate counts have no reconstructed chronology.'),
        'sampling': ('Each seed independently draws rectangle width 3..7, height 2..5, '
                     'closure edge, transition probabilities and acquisition counts. Graphs have '
                     'a direct corridor and a longer perimeter detour. Acquisition supplies '
                     'independent state/action outcome samples, not autonomous trajectories. '
                     'Even seeds use deterministic revision dynamics; odd seeds use stochastic '
                     'revision dynamics. The separate stationary case is always stochastic.'),
        'stationary_adverse_control': ('Conditional adverse-run stress test: after the declared '
                                      'unbiased stationary sample, supply W collisions at each '
                                      'target action without changing true dynamics. This '
                                      'conditions on an adverse run and is not an estimate '
                                      'of its frequency or unconditional expected performance.'),
        'evidence': ('Barrier closure/reopening first changes held-out truth with no commands. '
                     'Only later explicit observe commands change empirical models. All arms '
                     'receive the same stream; replay follows the observation batch. Goal '
                     'relocation supplies a public already observed goal and no outcomes.'),
        'metrics': ('Empirical values are compared to an independent synchronous value-iteration '
                    'oracle on each arm model. Current-dynamics values and deterministic action '
                    'choices are compared to a distinct oracle on held-out true probabilities. '
                    'Brier scores compare predicted outcome probabilities to the true '
                    'distribution (excess Brier risk, excluding irreducible outcome noise). '
                    'Frozen policy start value is evaluated analytically with no feedback, '
                    'rather than counted as a sampled success rate.'),
        'analysis': ('All declared cases and arms are retained. Means weight seeds equally; '
                     'there are no significance tests, tuning, selection or promotion gates. '
                     'The fixed default eight seeds are a local descriptive pilot.'),
        'sources': [
            {'url': 'https://arxiv.org/abs/2509.15032', 'status': '2025 preprint',
             'role': 'Separate environment change from value-learning prioritization; not DEER replication.'},
            {'url': 'https://proceedings.iclr.cc/paper_files/paper/2026/hash/23c2c77c921111ec8902ac482d92b289-Abstract-Conference.html',
             'status': 'ICLR 2026', 'role': 'Updateable episodic spatial models; no learned ESWM implementation.'},
            {'url': 'https://www.nature.com/articles/s41467-025-56731-y',
             'status': '2025 journal article', 'role': 'Replay and uncertainty motivation; no Bayesian or biological equivalence.'},
        ],
        'scope': SCOPE,
    }


def _geometry(seed):
    rng = random.Random(seed ^ 0x47454F4D)
    width, height = rng.randint(3, 7), rng.randint(2, 5)
    top = [(x, 1) for x in range(1, width + 1)]
    detour = ([(1, z) for z in range(1, height + 1)]
              + [(x, height) for x in range(2, width + 1)]
              + [(width, z) for z in range(height - 1, 0, -1)])
    links = sorted({tuple(sorted((a, b))) for path in (top, detour)
                    for a, b in zip(path, path[1:])})
    edge_x = rng.randint(1, width - 1)
    left, right = (edge_x, 1), (edge_x + 1, 1)
    targets = ((left, 1), (right, 3))
    return {'width': width, 'height': height,
            'closure_edge': [list(left), list(right)]}, links, targets


def _truth(links, seed, stationary=False):
    rng = random.Random(seed ^ (0x53544154 if stationary else 0x44594E41))
    model = {}
    for u, v in links:
        for source, destination in ((u, v), (v, u)):
            action = TopologicalMemoryGraph.ACTIONS.index((destination[0] - source[0],
                                                         destination[1] - source[1]))
            probability = rng.choice((.75, .8, .9)) if stationary else (
                rng.choice((.8, .9, 1.)) if seed % 2 else 1.)
            model.setdefault(source, {})[action] = {destination: probability}
            model.setdefault(destination, {})
            if probability < 1:
                model[source][action][source] = 1. - probability
    return model


def _blocked(model, targets):
    result = copy.deepcopy(model)
    for source, action in targets:
        result[source][action] = {source: 1.}
    return result


def _truth_rows(model):
    return [{'source': list(source), 'action': action,
             'outcomes': [{'destination': list(destination), 'probability': probability}
                          for destination, probability in sorted(outcomes.items())]}
            for source, actions in sorted(model.items())
            for action, outcomes in sorted(actions.items())]


def _from_truth_rows(rows):
    model = {}
    for row in rows:
        source, action = tuple(row['source']), row['action']
        outcomes = {tuple(item['destination']): item['probability'] for item in row['outcomes']}
        if (source in model and action in model[source]) or not outcomes:
            raise ValueError('invalid current-dynamics model')
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1
               for p in outcomes.values()) or abs(math.fsum(outcomes.values()) - 1) > 1e-12:
            raise ValueError('invalid current-dynamics probabilities')
        for destination in outcomes:
            TopologicalMemoryGraph._transition(source, action, destination)
            model.setdefault(destination, {})
        model.setdefault(source, {})[action] = outcomes
    return model


def _acquire(model, seed, stationary=False):
    rng = random.Random(seed ^ (0x434F554E if stationary else 0x41435155))
    graph = TopologicalMemoryGraph()
    for source in sorted(model):
        graph.add_node(source)
        for action, outcomes in sorted(model[source].items()):
            moved = next(destination for destination in outcomes if destination != source)
            for _ in range(rng.randint(32, 64)):
                destination = moved if rng.random() < outcomes[moved] else source
                graph.record_transition(source, action, destination)
    return graph


def _edge_model(edges, nodes):
    counts = defaultdict(dict)
    for row in edges:
        source, destination, action = tuple(row['source']), tuple(row['destination']), row['action']
        TopologicalMemoryGraph._transition(source, action, destination)
        if type(row['count']) is not int or row['count'] < 1:
            raise ValueError('invalid empirical counts')
        if destination in counts[source, action]:
            raise ValueError('duplicate empirical outcome')
        counts[source, action][destination] = row['count']
    model = {tuple(position): {} for position in nodes}
    for (source, action), outcomes in sorted(counts.items()):
        total = sum(outcomes.values())
        model.setdefault(source, {})[action] = {v: count / total for v, count in sorted(outcomes.items())}
    return model


def _q(model, values, source, goal):
    if source == goal:
        return dict.fromkeys(range(4), 0.)
    return {action: math.fsum(p * (1. if destination == goal else DISCOUNT * values.get(destination, 0.))
                             for destination, p in model.get(source, {}).get(action, {}).items())
            for action in range(4)}


def _reference(model, goal):
    """Independent synchronous solver; never invokes ReplayPlanner or its helpers."""
    values = dict.fromkeys(sorted(model), 0.)
    for _ in range(10000):
        updated = {source: max(_q(model, values, source, goal).values()) for source in values}
        delta = max((abs(updated[source] - values[source]) for source in values), default=0.)
        values = updated
        if delta <= 1e-13:
            break
    residual = max((abs(max(_q(model, values, source, goal).values()) - values[source])
                    for source in values), default=0.)
    if residual > 1e-12:
        raise ValueError('independent reference failed to converge')
    return values, residual


def _choice(scores):
    return max(range(4), key=lambda action: (scores[action], -action))


def _agreement(model, values, reference_model, reference, goal):
    decisive, correct = 0, 0
    for source in sorted(reference_model):
        if source == goal:
            continue
        scores = _q(reference_model, reference, source, goal)
        ordered = sorted(scores.values(), reverse=True)
        if ordered[0] - ordered[1] > 1e-10:
            decisive += 1
            correct += _choice(_q(model, values, source, goal)) == _choice(scores)
    return (correct / decisive if decisive else None), decisive


def _policy_value(empirical, values, truth, goal, start):
    actions = {source: _choice(_q(empirical, values, source, goal)) for source in truth}
    result = dict.fromkeys(sorted(truth), 0.)
    for _ in range(10000):
        updated = {source: (0. if source == goal else _q(truth, result, source, goal)[actions[source]])
                   for source in result}
        delta = max((abs(updated[source] - result[source]) for source in result), default=0.)
        result = updated
        if delta <= 1e-13:
            return result.get(start, 0.)
    raise ValueError('frozen policy evaluation failed to converge')


def _brier(empirical, truth, keys=None):
    errors = []
    pairs = keys if keys is not None else [(source, action) for source in sorted(truth)
                                           for action in sorted(truth[source])]
    for source, action in pairs:
        expected = truth[source][action]
        predicted = empirical.get(source, {}).get(action, {})
        errors.append(math.fsum((predicted.get(destination, 0.) - expected.get(destination, 0.)) ** 2
                                for destination in sorted(set(expected) | set(predicted))))
    return math.fsum(errors) / len(errors) if errors else 0.


def _metrics(empirical, values, truth, goal, start, targets, true_values):
    reference, _ = _reference(empirical, goal)
    nodes = [source for source in reference if source != goal]
    empirical_agreement, empirical_decisive = _agreement(empirical, values, empirical, reference, goal)
    current_agreement, current_decisive = _agreement(empirical, values, truth, true_values, goal)
    policy_value = _policy_value(empirical, values, truth, goal, start)
    empirical_metrics = {
        'value_mae': math.fsum(abs(values.get(source, 0.) - reference[source]) for source in nodes) / max(1, len(nodes)),
        'bellman_residual': max((abs(max(_q(empirical, values, source, goal).values()) - values.get(source, 0.))
                                 for source in empirical), default=0.),
        'action_agreement': empirical_agreement, 'decisive_states': empirical_decisive,
    }
    current_metrics = {
        'value_mae': math.fsum(abs(values.get(source, 0.) - true_values[source]) for source in truth if source != goal) / max(1, len(truth) - 1),
        'action_agreement': current_agreement, 'decisive_states': current_decisive,
        'transition_brier': _brier(empirical, truth), 'target_action_brier': _brier(empirical, truth, targets),
        'start_policy_value': policy_value, 'optimal_start_value': true_values.get(start, 0.),
        'start_regret': true_values.get(start, 0.) - policy_value,
    }
    return empirical_metrics, current_metrics


def _stage(bench, stage_id, truth, start, targets, commands):
    before = bench.snapshot()['arms']
    for command in commands:
        bench.apply(command)
    snapshot = bench.snapshot()
    goal = tuple(snapshot['goal'])
    true_values, residual = _reference(truth, goal)
    arms = []
    for model in MODELS:
        graph = snapshot['models'][model]
        empirical = _edge_model(graph['edges'], graph['nodes'])
        for strategy in STRATEGIES:
            arm = snapshot['arms'][model][strategy]
            values = {tuple(row['position']): row['value'] for row in arm['values']}
            empirical_metrics, current_metrics = _metrics(empirical, values, truth, goal, start, targets, true_values)
            arms.append({'model': model, 'strategy': strategy, 'values': arm['values'],
                         'total_backups': arm['total_backups'], 'summary': arm['summary'],
                         'stage_backups': arm['total_backups'] - before[model][strategy]['total_backups'],
                         'stale': arm['stale'], 'empirical': empirical_metrics,
                         'current_dynamics': current_metrics})
    return {'id': stage_id, 'goal': list(goal), 'commands': copy.deepcopy(commands),
            'truth': _truth_rows(truth), 'reference_residual': residual,
            'models': snapshot['models'], 'arms': arms}


def _observation(source, action, destination, count=1):
    return {'action': 'observe', 'source': list(source), 'action_index': action,
            'destination': list(destination), 'count': count}


def _case(seed, geometry, links, targets, config, stationary=False):
    from .workbench import ReplayWorkbench
    truth = _truth(links, seed, stationary)
    graph = _acquire(truth, seed, stationary)
    start, goal = (1, 1), (geometry['width'], 1)
    case_id = 'stationary' if stationary else 'revision'
    provenance = {'source': 'seeded-controlled-study', 'seed': seed, 'case': case_id}
    bench = ReplayWorkbench(graph=graph, goal=goal, provenance=provenance, window=config['window'], seed=seed)
    initial = {'graph': graph.to_state(), 'goal': list(goal), 'provenance': provenance}
    stages = [_stage(bench, 'baseline', truth, start, targets,
                     [{'action': 'replay', 'budget': config['warmup_budget']}])]
    replay = {'action': 'replay', 'budget': config['budget']}
    if stationary:
        rng = random.Random(seed ^ 0x46524553)
        fresh = []
        for source, action in targets:
            moved = next(destination for destination in truth[source][action] if destination != source)
            for _ in range(config['stationary_sample_per_action']):
                destination = moved if rng.random() < truth[source][action][moved] else source
                fresh.append(_observation(source, action, destination))
        stages.append(_stage(bench, 'stationary_sample', truth, start, targets, fresh + [replay]))
        adverse = [_observation(source, action, source, config['window']) for source, action in targets]
        stages.append(_stage(bench, 'stationary_adverse_run', truth, start, targets, adverse + [replay]))
    else:
        closed = _blocked(truth, targets)
        stages.append(_stage(bench, 'closed_unobserved', closed, start, targets, []))
        observations = [_observation(source, action, source, config['window']) for source, action in targets]
        stages.append(_stage(bench, 'closed_observed', closed, start, targets, observations + [replay]))
        stages.append(_stage(bench, 'reopened_unobserved', truth, start, targets, []))
        opened = []
        for source, action in targets:
            destination = next(v for v in truth[source][action] if v != source)
            # A supplied successful batch is an intervention, not a sampled estimate of p.
            opened.append(_observation(source, action, destination, config['window']))
        stages.append(_stage(bench, 'reopened_observed', truth, start, targets, opened + [replay]))
        stages.append(_stage(bench, 'goal_relocated', truth, start, targets,
                             [{'action': 'goal', 'x': 1, 'z': geometry['height']}, replay]))
    return {'id': case_id, 'start': list(start),
            'targets': [{'source': list(source), 'action': action} for source, action in targets],
            'initial': initial, 'initial_sha256': _digest(initial), 'stages': stages}


def _aggregate(instances):
    groups = defaultdict(list)
    for instance in instances:
        for case in instance['cases']:
            for stage in case['stages']:
                for arm in stage['arms']:
                    groups[case['id'], stage['id'], arm['model'], arm['strategy']].append(arm)
    results = []
    for (case, stage, model, strategy), arms in sorted(groups.items()):
        def mean(section, key):
            values = [arm[section][key] for arm in arms if arm[section][key] is not None]
            return math.fsum(values) / len(values) if values else None
        results.append({'case': case, 'stage': stage, 'model': model, 'strategy': strategy,
                        'seed_count': len(arms),
                        'mean_backups': math.fsum(arm['stage_backups'] for arm in arms) / len(arms),
                        'mean_total_backups': math.fsum(arm['total_backups'] for arm in arms) / len(arms),
                        'empirical': {key: mean('empirical', key) for key in
                                      ('value_mae', 'bellman_residual', 'action_agreement')},
                        'current_dynamics': {key: mean('current_dynamics', key) for key in
                                             ('value_mae', 'action_agreement', 'transition_brier',
                                              'target_action_brier', 'start_policy_value', 'start_regret')}})
    return results


def _compute(config):
    instances = []
    for seed in config['seeds']:
        geometry, links, targets = _geometry(seed)
        instances.append({'seed': seed, 'geometry': geometry,
                          'revision_dynamics': 'stochastic' if seed % 2 else 'deterministic',
                          'cases': [_case(seed, geometry, links, targets, config),
                                    _case(seed, geometry, links, targets, config, stationary=True)]})
    protocol = _protocol(config)
    return {'schema': SCHEMA, 'protocol': protocol, 'protocol_sha256': _digest(protocol),
            'instances': instances, 'aggregates': _aggregate(instances), 'evidence_scope': SCOPE}


def _render(report):
    rows = []
    for row in report['aggregates']:
        def fmt(number):
            return 'n/a' if number is None else f'{number:.5g}'
        current, empirical = row['current_dynamics'], row['empirical']
        rows.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in (
            row['case'], row['stage'], row['model'], row['strategy'], fmt(row['mean_backups']),
            fmt(empirical['value_mae']), fmt(empirical['bellman_residual']),
            fmt(current['target_action_brier']), fmt(current['action_agreement']), fmt(current['start_regret']))) + '</tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>NEUROGRAFT controlled transition revision</title><style>body{font:16px system-ui;'
            'max-width:1400px;margin:30px auto;padding:0 20px;background:#101923;color:#e2edf4}'
            'p{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;'
            'text-align:left;border-bottom:1px solid #344353}.scroll{overflow:auto}a{color:#8ecfff}</style>'
            '<h1>Controlled transition revision</h1><p>' + escape(SCOPE) + '</p>'
            '<p><b>Current dynamics and empirical-model consistency are different evidence.</b> '
            'A low Bellman residual can describe a perfectly solved stale model. Prediction scores '
            'are excess distribution Brier risk; start regret evaluates a frozen deterministic policy '
            'analytically, without feedback. These are not sampled goal-reaching rates.</p>'
            '<p>' + escape(report['protocol']['model_rule']) + '</p><p>'
            + escape(report['protocol']['stationary_adverse_control']) + '</p><p>'
            + escape(report['protocol']['analysis']) + '</p><div class="scroll"><table><thead><tr>'
            '<th>Case</th><th>Stage</th><th>Memory</th><th>Replay</th><th>Actual backups</th>'
            '<th>Empirical value MAE</th><th>Empirical residual</th><th>Current target Brier</th>'
            '<th>Current action agreement</th><th>Current start regret</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div><p>Hash checks detect corruption, not authenticity. '
            'Plain verification rederives metrics and the rendered report; --replay also regenerates '
            'all seeded inputs, commands, model updates and schedules.</p></html>')


def run_revision_study(out, seeds=DEFAULT_SEEDS, window=8, budget=64):
    config = _config(seeds, window, budget)
    out = Path(out)
    if out.exists():
        raise FileExistsError('revision study output already exists')
    report = _compute(config)
    out.mkdir(parents=True, exist_ok=False)
    contents = {'plan.json': _json(report['protocol']), 'report.json': _json(report),
                'report.html': _render(report)}
    for name, content in contents.items():
        (out / name).write_bytes(content.encode('utf-8'))
    manifest = {'schema': SCHEMA, 'files': {name: hashlib.sha256(content.encode('utf-8')).hexdigest()
                                          for name, content in contents.items()}}
    (out / 'manifest.json').write_text(_json(manifest), encoding='utf-8')
    return report


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON object key')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('nonfinite JSON number')
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('study JSON exceeds size limit')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def _validate_metrics(report):
    for instance in report['instances']:
        for case in instance['cases']:
            if _digest(case['initial']) != case['initial_sha256']:
                raise ValueError('initial evidence content mismatch')
            start = tuple(case['start'])
            targets = [(tuple(row['source']), row['action']) for row in case['targets']]
            previous_totals = {(model, strategy): 0 for model in MODELS for strategy in STRATEGIES}
            for stage in case['stages']:
                for graph in stage['models'].values():
                    if type(graph) is not dict or set(graph) != {'nodes', 'edges', 'total_transitions', 'sha256'}:
                        raise ValueError('invalid empirical graph fields')
                    nodes = graph['nodes']
                    if type(nodes) is not list or len(nodes) > 256:
                        raise ValueError('invalid empirical graph nodes')
                    parsed = [TopologicalMemoryGraph._position(node) for node in nodes]
                    if len(set(parsed)) != len(parsed):
                        raise ValueError('invalid duplicate empirical graph node')
                    state = {'version': 1, 'nodes': nodes,
                             'transitions': graph['edges'], 'total_transitions': graph['total_transitions'],
                             'reward_totals': [{'position': node, 'total_reward': 0.0} for node in nodes]}
                    TopologicalMemoryGraph.from_state(state)
                    raw = json.dumps({k: v for k, v in graph.items() if k != 'sha256'},
                                     sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')
                    if hashlib.sha256(raw).hexdigest() != graph['sha256']:
                        raise ValueError('invalid empirical graph digest')
                truth = _from_truth_rows(stage['truth'])
                goal = tuple(stage['goal'])
                reference, residual = _reference(truth, goal)
                if residual != stage['reference_residual']:
                    raise ValueError('derived reference residual mismatch')
                if {(arm['model'], arm['strategy']) for arm in stage['arms']} != {
                        (model, strategy) for model in MODELS for strategy in STRATEGIES} or len(stage['arms']) != 6:
                    raise ValueError('invalid crossed arms')
                for arm in stage['arms']:
                    identity = (arm['model'], arm['strategy'])
                    spent = arm['stage_backups']
                    if (type(spent) is not int or spent < 0
                            or spent != arm['total_backups'] - previous_totals[identity]):
                        raise ValueError('invalid stage backup accounting')
                    previous_totals[identity] = arm['total_backups']
                    graph = stage['models'][arm['model']]
                    model = _edge_model(graph['edges'], graph['nodes'])
                    rows = arm['values']
                    if type(rows) is not list or len(rows) != len(graph['nodes']):
                        raise ValueError('invalid empirical value coverage')
                    positions = []
                    for row in rows:
                        if type(row) is not dict or set(row) != {'position', 'value'}:
                            raise ValueError('invalid empirical value fields')
                        positions.append(TopologicalMemoryGraph._position(row['position']))
                        if type(row['value']) not in (int, float) or not math.isfinite(row['value']) or not 0 <= row['value'] <= 1:
                            raise ValueError('invalid empirical value')
                    if len(set(positions)) != len(positions) or set(positions) != {tuple(n) for n in graph['nodes']}:
                        raise ValueError('invalid empirical value positions')
                    values = {tuple(row['position']): row['value'] for row in arm['values']}
                    empirical, current = _metrics(model, values, truth, goal, start, targets, reference)
                    if empirical != arm['empirical'] or current != arm['current_dynamics']:
                        raise ValueError('independently derived metric mismatch')
    if _aggregate(report['instances']) != report['aggregates']:
        raise ValueError('derived aggregate mismatch')


def verify_revision_study(path, replay=False):
    """Verify seals and derived results; optional replay regenerates the entire assay."""
    if type(replay) is not bool:
        raise ValueError('replay must be boolean')
    path = Path(path)
    try:
        manifest = _read(path / 'manifest.json')
        if (set(manifest) != {'schema', 'files'} or manifest['schema'] != SCHEMA
                or set(manifest['files']) != {'plan.json', 'report.json', 'report.html'}):
            raise ValueError('invalid study manifest')
        for name, expected in manifest['files'].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
                raise ValueError('study artifact hash mismatch: ' + name)
        report = _read(path / 'report.json')
        config = report['protocol']['config']
        canonical_config = _config(config['seeds'], config['window'], config['budget'])
        protocol = _protocol(canonical_config)
        if (report['schema'] != SCHEMA or report['protocol'] != protocol
                or report['protocol_sha256'] != _digest(protocol)
                or _read(path / 'plan.json') != protocol or report['evidence_scope'] != SCOPE):
            raise ValueError('declared protocol content mismatch')
        if [instance['seed'] for instance in report['instances']] != canonical_config['seeds']:
            raise ValueError('declared seed coverage mismatch')
        _validate_metrics(report)
        if (path / 'report.html').read_text(encoding='utf-8') != _render(report):
            raise ValueError('rendered report content mismatch')
        if replay and _json(_compute(canonical_config)) != _json(report):
            raise ValueError('deterministic replay content mismatch')
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError(f'invalid revision study archive: {exc}') from exc
    return {'ok': True, 'schema': SCHEMA, 'deterministic_replay': replay,
            'derived_metrics_verified': True, 'seed_count': len(canonical_config['seeds']),
            'report_sha256': manifest['files']['report.json'], 'promotion_performed': False}
