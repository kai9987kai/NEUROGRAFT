"""Causal adaptive evidence assay; truth never enters the memory/planner interface.

The independent synchronous solver from revision_study is evaluation-only. The
assay collects supplied per-action samples, not autonomous navigation trajectories.
"""
from __future__ import annotations

import copy
import hashlib
from html import escape
import json
import math
from pathlib import Path
import random

from .memory import TopologicalMemoryGraph
from .replay import ReplayPlanner
from .revision_study import (
    _acquire, _edge_model, _from_truth_rows, _geometry, _metrics,
    _reference, _truth, _truth_rows,
)

SCHEMA = 'adaptive-evidence-study-v1'
DEFAULT_SEEDS = tuple(range(15000, 15008))
MODELS = ('cumulative', 'recent', 'adaptive')
FORECAST_MODELS = ('cumulative', 'recent', 'guarded_recent', 'adaptive')
STRATEGIES = ('prioritized', 'uniform', 'none')
CASE_IDS = ('stationary', 'reversals', 'gradual', 'conditional_adverse', 'sparse')
PARAMETERS = {'decay': .95, 'threshold': math.log(4), 'minimum_fresh': 4, 'prior': .5}
MAX_JSON_BYTES = 256 * 1024 * 1024
SCOPE = (
    'Exploratory finite synthetic supplied-outcome streams, not autonomous exploration, '
    'neural learning, biological equivalence or checkpoint promotion. Parameters are fixed '
    'before the eight-seed pilot and are not tuned on its outcomes. Truth is withheld '
    'from memory and planners. Smoothed forecasts and integer-count planning are distinct. '
    'A discounted predictive score is a heuristic, not a posterior or calibrated detector. '
    'Equal backup caps do not imply equal computation. SHA256 checks consistency, not authenticity.'
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n'


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _config(seeds, window, budget):
    if (type(seeds) not in (list, tuple) or not 1 <= len(seeds) <= 32
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be 1-32 distinct integers in [0, 2**31)')
    if type(window) is not int or window not in (4, 8, 16):
        raise ValueError('window must be 4, 8 or 16')
    if type(budget) is not int or not 0 <= budget <= 256:
        raise ValueError('budget must be an integer from 0 through 256')
    return {'seeds': list(seeds), 'window': window, 'budget': budget,
            'warmup_budget': 256, 'discount': .9, 'parameters': dict(PARAMETERS),
            'forecast_models': list(FORECAST_MODELS), 'models': list(MODELS),
            'strategies': list(STRATEGIES), 'cases': list(CASE_IDS)}


def _protocol(config):
    return {
        'schema': SCHEMA, 'config': config,
        'rule': ('Score Beta(.5,.5) binary move predictions before updates; local evidence = '
                 '.95 * evidence + log(p_recent(outcome)/p_cumulative(outcome)). Select recent '
                 'iff fresh_count >= 4 and evidence > log(4), otherwise cumulative. '
                 'Guarded recent forecasts cumulative until four previous fresh events, then recent. '
                 'Planning uses unsmoothed positive integer counts; no edges are fabricated.'),
        'acquisition': ('Seeded two-route geometries and independent per-action baseline samples '
                        'from stationary move probabilities .75/.8/.9, with 32..64 samples per action. '
                        'Captured aggregate counts are unordered; no history is reconstructed.'),
        'streams': {
            'stationary': '96 unconditional independent outcomes, alternating target actions; checkpoints every 32.',
            'reversals': ('Four 32-outcome blocks: closure, reopening, closure, reopening. '
                          'Each truth change has a zero-observation/zero-backup checkpoint before sampling.'),
            'gradual': ('96 outcomes alternating targets; each pair probability decreases linearly '
                        'to .1 over 48 events then returns over 48; checkpoints every 24.'),
            'conditional_adverse': ('W forced collisions per target under unchanged truth, followed by '
                                   '48 unconditional recovery samples. Conditional stress, not its frequency.'),
            'sparse': ('Close the first target without observations; sample the second target 24 times, '
                       'revisit closed target twice, sample second 24 more times, revisit closed target '
                       '16 times. A third baseline action is never observed.'),
            'ambiguity': ('Identical baselines and W collisions per target are presented to two memories '
                          'with different evaluation-only truths. Their forecasts, selected graphs '
                          'and diagnostic state must be identical.'),
        },
        'metrics': ('Record per-event pre-update binary Brier and natural log loss, absolute probability '
                    'error against held-out truth, local selections, switches and evidence. '
                    'First recent-selection delay counts target visits and stream events, not calibrated '
                    'detection delay. Stage selection summaries use stage-relative visits and event offsets; '
                    'first_recent_event is the global stream index, first_recent_event_relative subtracts '
                    'the stage offset, and recent_at_entry distinguishes already-active recency. '
                    'Whole-case summaries span that case lifetime. Stationary recent selections/switches are descriptive false-switch '
                    'proxies with no statistical error-rate guarantee. Independent synchronous contraction '
                    'iteration evaluates empirical residual/MAE and current-dynamics frozen-policy '
                    'discounted start-value regret; evaluation never updates a model.'),
        'analysis': ('Retain every case, stage and seed. Equal-weight seed means and paired seed '
                     'differences are descriptive; stages/cases within a seed are correlated, '
                     'not independent replicates. No significance test, tuning, dominance gate or promotion.'),
        'limits': {'max_seeds': 32, 'max_events_per_case': 2048, 'max_nodes': 256,
                   'json_bytes': MAX_JSON_BYTES},
        'scope': SCOPE,
    }


def _move_destination(source, action):
    dx, dz = TopologicalMemoryGraph.ACTIONS[action]
    return source[0] + dx, source[1] + dz


def _p_move(truth, source, action):
    return truth[source][action].get(_move_destination(source, action), 0.)


def _change(truth, keys, probabilities):
    changed = copy.deepcopy(truth)
    for (source, action), probability in zip(keys, probabilities):
        outcomes = {}
        if probability > 0:
            outcomes[_move_destination(source, action)] = probability
        if probability < 1:
            outcomes[source] = 1. - probability
        changed[source][action] = outcomes
    return changed


def _sample(rng, truth, source, action, forced=None):
    probability = _p_move(truth, source, action)
    destination = (source if forced == 'collision' else
                   (_move_destination(source, action) if rng.random() < probability else source))
    return {'source': list(source), 'action': action, 'destination': list(destination),
            'truth_p_move': probability}


def _blocks(seed, case_id, truth, targets, window):
    rng = random.Random(seed ^ (0x53545200 + CASE_IDS.index(case_id)))
    baseline = copy.deepcopy(truth)
    blocks = []

    def block(name, current, n, keys=targets, forced=None):
        events = [_sample(rng, current, *keys[i % len(keys)], forced=forced) for i in range(n)]
        blocks.append({'id': name, 'truth': copy.deepcopy(current), 'events': events})

    if case_id == 'stationary':
        for i in range(3):
            block(f'stationary_{i + 1}', truth, 32)
    elif case_id == 'reversals':
        for name, probabilities in (
                ('closed', (0., 0.)), ('reopened', tuple(_p_move(baseline, *k) for k in targets)),
                ('closed_again', (0., 0.)), ('reopened_again', tuple(_p_move(baseline, *k) for k in targets))):
            truth = _change(baseline, targets, probabilities)
            block(name + '_unobserved', truth, 0)
            block(name, truth, 32)
    elif case_id == 'gradual':
        events = []
        for i in range(96):
            fraction = (i + 1) / 48 if i < 48 else (96 - i - 1) / 48
            probabilities = [p + fraction * (.1 - p) for p in (_p_move(baseline, *k) for k in targets)]
            truth = _change(baseline, targets, probabilities)
            events.append(_sample(rng, truth, *targets[i % 2]))
            if len(events) == 24:
                blocks.append({'id': f'drift_{i + 1}', 'truth': copy.deepcopy(truth), 'events': events})
                events = []
    elif case_id == 'conditional_adverse':
        block('forced_adverse', truth, window * 2, forced='collision')
        block('unbiased_recovery', truth, 48)
    elif case_id == 'sparse':
        truth = _change(truth, (targets[0],), (0.,))
        block('closed_unobserved', truth, 0)
        block('common_visits', truth, 24, (targets[1],))
        block('rare_two_visits', truth, 2, (targets[0],))
        block('common_more_visits', truth, 24, (targets[1],))
        block('rare_revisited', truth, 16, (targets[0],))
    return blocks


def _model(graph):
    state = graph.to_state()
    return {'state': state, 'sha256': _digest(state)}


def _models(memory):
    return {'cumulative': memory.cumulative, 'recent': memory.recent, 'adaptive': memory.graph}


def _totals(events):
    return {model: {'count': len(events),
                    **{metric: math.fsum(event[metric][model] for event in events)
                       for metric in ('brier', 'log_loss', 'absolute_error')}}
            for model in FORECAST_MODELS}


def _selection_summary(events, event_offset=0):
    by_pair = {}
    for event in events:
        key = (tuple(event['source']), event['action'])
        by_pair.setdefault(key, []).append(event)
    return [{'source': list(source), 'action': action, 'visits': len(rows),
             'event_offset': event_offset,
             'recent_at_entry': rows[0]['selected_before'] == 'recent',
             'recent_selections': sum(row['selected_after'] == 'recent' for row in rows),
             'switches': sum(row['switch'] for row in rows),
             'first_recent_visit': next((i + 1 for i, row in enumerate(rows)
                                         if row['selected_after'] == 'recent'), None),
             'first_recent_event': next((row['index'] for row in rows
                                         if row['selected_after'] == 'recent'), None),
             'first_recent_event_relative': next((row['index'] - event_offset for row in rows
                                                  if row['selected_after'] == 'recent'), None),
             'first_recent_switch_visit': next((i + 1 for i, row in enumerate(rows)
                                                if row['switch'] and row['selected_after'] == 'recent'), None),
             'first_recent_switch_event_relative': next((row['index'] - event_offset for row in rows
                                                         if row['switch'] and row['selected_after'] == 'recent'), None)}
            for (source, action), rows in sorted(by_pair.items())]


def _stage(memory, planners, stage_id, truth, goal, start, targets, events, budget):
    true_values, residual = _reference(truth, goal)
    graphs = _models(memory)
    arms = []
    for model in MODELS:
        graph = graphs[model]
        state = graph.to_state()
        empirical = _edge_model(state['transitions'], state['nodes'])
        for strategy in STRATEGIES:
            planner = planners[model][strategy]
            before = planner.total_backups
            # Hidden changes are not even presented as replay calls.
            if stage_id == 'baseline' or events:
                planner.replay(graph, goal, budget=budget, strategy=strategy)
            empirical_metrics, current = _metrics(empirical, planner.values, truth, goal,
                                                   start, targets, true_values)
            arms.append({'model': model, 'strategy': strategy, 'planner': planner.to_state(),
                         'stage_backups': planner.total_backups - before,
                         'total_backups': planner.total_backups,
                         'empirical': empirical_metrics, 'current_dynamics': current})
    return {'id': stage_id, 'new_events': len(events), 'event_end': memory.diagnostics()['events'],
            'requested_backups': budget, 'truth': _truth_rows(truth),
            'reference_residual': residual, 'models': {name: _model(g) for name, g in graphs.items()},
            'forecast_totals': _totals(events),
            'selections': _selection_summary(events, event_offset=memory.diagnostics()['events'] - len(events)),
            'diagnostics': memory.diagnostics(), 'arms': arms}


def _case(seed, case_id, graph, truth, targets, geometry, config):
    from .adaptive_memory import AdaptiveTransitionMemory
    memory = AdaptiveTransitionMemory(graph, window=config['window'])
    planners = {model: {strategy: ReplayPlanner() for strategy in STRATEGIES} for model in MODELS}
    start, goal = (1, 1), (geometry['width'], 1)
    stages = [_stage(memory, planners, 'baseline', truth, goal, start, targets, [], config['warmup_budget'])]
    events = []
    for block in _blocks(seed, case_id, truth, targets, config['window']):
        fresh = []
        for observation in block['events']:
            source, action, destination = tuple(observation['source']), observation['action'], tuple(observation['destination'])
            result = memory.observe(source, action, destination)
            row = {**copy.deepcopy(observation), **result,
                   'index': len(events) + 1, 'stage': block['id'],
                   'absolute_error': {model: abs(result['predictions'][model] - observation['truth_p_move'])
                                      for model in FORECAST_MODELS}}
            events.append(row)
            fresh.append(row)
        stages.append(_stage(memory, planners, block['id'], block['truth'], goal, start, targets,
                             fresh, config['budget'] if fresh else 0))
    other = next(key for key in sorted(graph.transitions) if key not in targets)
    preserved = all(g.transitions[other] == graph.transitions[other] for g in _models(memory).values())
    return {'id': case_id, 'start': list(start), 'goal': list(goal),
            'targets': [{'source': list(s), 'action': a} for s, a in targets],
            'initial': graph.to_state(), 'initial_sha256': _digest(graph.to_state()),
            'events': events, 'event_count': len(events), 'stages': stages,
            'forecast_totals': _totals(events), 'selections': _selection_summary(events),
            'untouched_pair': {'source': list(other[0]), 'action': other[1]},
            'untouched_pair_preserved': preserved}


def _ambiguity(graph, truth, targets, window):
    from .adaptive_memory import AdaptiveTransitionMemory
    left = AdaptiveTransitionMemory(graph, window=window)
    right = AdaptiveTransitionMemory(graph, window=window)
    observations, left_predictions, right_predictions = [], [], []
    for i in range(window * 2):
        source, action = targets[i % 2]
        observations.append({'source': list(source), 'action': action, 'destination': list(source)})
        left_predictions.append(left.observe(source, action, source))
        right_predictions.append(right.observe(source, action, source))
    closed = _change(truth, targets, (0., 0.))
    return {'observations': observations, 'left_truth': _truth_rows(closed), 'right_truth': _truth_rows(truth),
            'truths_differ': closed != truth, 'observations_equal': True,
            'left_predictions': left_predictions, 'right_predictions': right_predictions,
            'memory_states_equal': left.diagnostics() == right.diagnostics()
                                   and left.graph.to_state() == right.graph.to_state(),
            'left_diagnostics': left.diagnostics(), 'right_diagnostics': right.diagnostics()}


def _aggregate(instances):
    forecasts, planning = [], []
    for case_id in CASE_IDS:
        cases = [next(case for case in item['cases'] if case['id'] == case_id) for item in instances]
        for model in FORECAST_MODELS:
            row = {'case': case_id, 'model': model, 'seed_count': len(cases)}
            for metric in ('brier', 'log_loss', 'absolute_error'):
                values = [case['forecast_totals'][model][metric] / case['event_count'] for case in cases]
                row['mean_' + metric] = math.fsum(values) / len(values)
                row['seed_values_' + metric] = values
            forecasts.append(row)
        for index, stage in enumerate(cases[0]['stages']):
            for model in MODELS:
                for strategy in STRATEGIES:
                    arms = [next(a for a in case['stages'][index]['arms']
                                 if a['model'] == model and a['strategy'] == strategy) for case in cases]
                    planning.append({'case': case_id, 'stage': stage['id'], 'model': model,
                                     'strategy': strategy, 'seed_count': len(cases),
                                     'mean_backups': math.fsum(a['stage_backups'] for a in arms) / len(arms),
                                     'mean_start_regret': math.fsum(a['current_dynamics']['start_regret'] for a in arms) / len(arms),
                                     'mean_target_brier': math.fsum(a['current_dynamics']['target_action_brier'] for a in arms) / len(arms),
                                     'mean_empirical_residual': math.fsum(a['empirical']['bellman_residual'] for a in arms) / len(arms)})
    paired = []
    for case_id in CASE_IDS:
        rows = {row['model']: row for row in forecasts if row['case'] == case_id}
        for comparator in ('cumulative', 'recent', 'guarded_recent'):
            differences = [a - b for a, b in zip(rows['adaptive']['seed_values_brier'], rows[comparator]['seed_values_brier'])]
            paired.append({'case': case_id, 'metric': 'mean_brier', 'difference': 'adaptive-minus-' + comparator,
                           'seed_differences': differences, 'mean': math.fsum(differences) / len(differences),
                           'minimum': min(differences), 'maximum': max(differences)})
    return {'forecasts': forecasts, 'planning': planning, 'paired': paired}


def _compute(config):
    instances = []
    for seed in config['seeds']:
        geometry, links, targets = _geometry(seed)
        truth = _truth(links, seed, stationary=True)
        graph = _acquire(truth, seed, stationary=True)
        instances.append({'seed': seed, 'stream_seed': seed, 'geometry': geometry,
                          'cases': [_case(seed, case_id, graph, truth, targets, geometry, config) for case_id in CASE_IDS],
                          'ambiguity': _ambiguity(graph, truth, targets, config['window'])})
    protocol = _protocol(config)
    return {'schema': SCHEMA, 'protocol': protocol, 'protocol_sha256': _digest(protocol),
            'evidence_scope': SCOPE, 'instances': instances, 'aggregates': _aggregate(instances)}


def _render(report):
    rows = ''.join('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in (
        r['case'], r['model'], f"{r['mean_brier']:.6f}", f"{r['mean_log_loss']:.6f}",
        f"{r['mean_absolute_error']:.6f}")) + '</tr>' for r in report['aggregates']['forecasts'])
    planning = ''.join('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in (
        r['case'], r['stage'], r['model'], r['strategy'], f"{r['mean_backups']:.2f}",
        f"{r['mean_start_regret']:.6f}", f"{r['mean_empirical_residual']:.3g}")) + '</tr>'
        for r in report['aggregates']['planning'])
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>NEUROGRAFT adaptive evidence study</title><style>body{font:16px system-ui;max-width:1200px;'
            'margin:40px auto;padding:0 20px;color:#172339;background:#f4f7fa}table{border-collapse:collapse;'
            'font-size:14px;width:100%;background:white}td,th{padding:8px;border:1px solid #c7d1dc;text-align:left}'
            '.scroll{overflow:auto}p{line-height:1.6}h1,h2{color:#163e67}</style>'
            '<h1>Adaptive transition evidence</h1><p>' + escape(SCOPE) + '</p><p>'
            + str(len(report['instances'])) + ' independent geometry/dynamics seeds; all declared cases retained. '
            'Lower losses and policy regret are better. Negative findings remain in the report.</p>'
            '<h2>Pre-update forecasts</h2><p>Beta(.5,.5) smoothed binary move probabilities; Brier is (p-y)²; '
            'log loss uses natural logarithms. Absolute error compares to held-out true move probability. '
            'Each seed has equal weight. Guarded recent isolates the four-event warmup.</p>'
            '<div class="scroll"><table><tr><th>Case</th><th>Model</th><th>Brier</th><th>Log loss</th>'
            '<th>Absolute probability error</th></tr>' + rows + '</table></div>'
            '<h2>Independent current-dynamics policy evaluation</h2><p>Integer observed counts drive nine '
            'planning arms. The evaluator freezes each policy and solves its discounted start value under '
            'held-out current dynamics. This is not sampled navigation success. Small empirical residual '
            'does not establish a correct world model.</p><div class="scroll"><table><tr><th>Case</th><th>Stage</th>'
            '<th>Evidence</th><th>Schedule</th><th>Actual backups</th><th>Start regret</th><th>Empirical residual</th>'
            '</tr>' + planning + '</table></div><h2>Interpretation and reproducibility</h2><p>'
            + escape(report['protocol']['rule']) + '</p><p>'
            + escape(report['protocol']['streams']['conditional_adverse']) + '</p><p>'
            + escape(report['protocol']['streams']['ambiguity']) + '</p><p>'
            + escape(report['protocol']['analysis']) + '</p><p>Complete ordered outcomes, forecasts, switches, '
            'selection delays, seed differences, model states, values and truth are in report.json. '
            'Plain verification rederives causal forecast scores and independent metrics. Strict replay '
            'also regenerates seeded streams and planning. No checkpoint is loaded or promoted.</p></html>')


def run_adaptation_study(out, seeds=DEFAULT_SEEDS, window=8, budget=64):
    """Write a new deterministic archive without opening any live checkpoint."""
    config = _config(seeds, window, budget)
    out = Path(out)
    if out.exists():
        raise FileExistsError('adaptation study output already exists')
    # Retain the validated declaration even when execution is interrupted. Such
    # an incomplete directory has no manifest and cannot verify as a result.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_bytes(_json(_protocol(config)).encode('utf-8'))
    report = _compute(config)
    contents = {'plan.json': _json(report['protocol']), 'report.json': _json(report), 'report.html': _render(report)}
    if any(len(content.encode('utf-8')) > MAX_JSON_BYTES for content in contents.values()):
        raise ValueError('study artifact exceeds size limit')
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

    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('study JSON exceeds size limit')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def _validate(report, config):
    from .adaptive_memory import AdaptiveTransitionMemory
    for instance in report['instances']:
        if type(instance['stream_seed']) is not int or instance['stream_seed'] != instance['seed']:
            raise ValueError('stream seed content mismatch')
        if [case['id'] for case in instance['cases']] != list(CASE_IDS):
            raise ValueError('case coverage mismatch')
        for case in instance['cases']:
            graph = TopologicalMemoryGraph.from_state(case['initial'])
            if _digest(case['initial']) != case['initial_sha256']:
                raise ValueError('initial model digest mismatch')
            if type(case['events']) is not list or not 0 < len(case['events']) <= 2048:
                raise ValueError('invalid event coverage')
            memory = AdaptiveTransitionMemory(graph, window=config['window'])
            previous_backups = {(m, s): 0 for m in MODELS for s in STRATEGIES}
            cursor = 0
            for stage in case['stages']:
                n = stage['new_events']
                if type(n) is not int or n < 0 or cursor + n > len(case['events']):
                    raise ValueError('invalid stage event accounting')
                events = case['events'][cursor:cursor + n]
                for event in events:
                    source, action = tuple(event['source']), event['action']
                    destination = tuple(event['destination'])
                    p = event['truth_p_move']
                    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                        raise ValueError('invalid held-out forecast truth')
                    if event['index'] != cursor + 1 or event['stage'] != stage['id']:
                        raise ValueError('forecast event ordering mismatch')
                    expected = memory.observe(source, action, destination)
                    if any(event[key] != value for key, value in expected.items()):
                        raise ValueError('derived causal forecast loss/state mismatch')
                    absolute = {m: abs(expected['predictions'][m] - p) for m in FORECAST_MODELS}
                    if absolute != event['absolute_error']:
                        raise ValueError('derived forecast absolute error mismatch')
                    cursor += 1
                if (stage['event_end'] != cursor or stage['diagnostics'] != memory.diagnostics()
                        or stage['forecast_totals'] != _totals(events)
                        or stage['selections'] != _selection_summary(events, event_offset=cursor - n)):
                    raise ValueError('derived stage forecast mismatch')
                expected_budget = config['warmup_budget'] if stage['id'] == 'baseline' else (config['budget'] if n else 0)
                if type(stage['requested_backups']) is not int or stage['requested_backups'] != expected_budget:
                    raise ValueError('invalid requested backup accounting')
                models = {name: _model(g) for name, g in _models(memory).items()}
                if stage['models'] != models:
                    raise ValueError('derived empirical model content mismatch')
                truth = _from_truth_rows(stage['truth'])
                goal, start = tuple(case['goal']), tuple(case['start'])
                reference, residual = _reference(truth, goal)
                if residual != stage['reference_residual']:
                    raise ValueError('derived reference residual mismatch')
                if (len(stage['arms']) != 9 or {(a['model'], a['strategy']) for a in stage['arms']}
                        != {(m, s) for m in MODELS for s in STRATEGIES}):
                    raise ValueError('invalid crossed planning arms')
                targets = [(tuple(k['source']), k['action']) for k in case['targets']]
                for arm in stage['arms']:
                    identity = arm['model'], arm['strategy']
                    planner = ReplayPlanner.from_state(arm['planner'])
                    spent = arm['stage_backups']
                    if (type(spent) is not int or not 0 <= spent <= expected_budget
                            or spent != planner.total_backups - previous_backups[identity]
                            or arm['total_backups'] != planner.total_backups
                            or (arm['strategy'] == 'none' and spent)):
                        raise ValueError('invalid actual backup accounting')
                    previous_backups[identity] = planner.total_backups
                    state = models[arm['model']]['state']
                    empirical = _edge_model(state['transitions'], state['nodes'])
                    expected_empirical, current = _metrics(empirical, planner.values, truth, goal,
                                                          start, targets, reference)
                    if arm['empirical'] != expected_empirical or arm['current_dynamics'] != current:
                        raise ValueError('independently derived planning metric mismatch')
            if (cursor != len(case['events']) or case['event_count'] != cursor
                    or case['forecast_totals'] != _totals(case['events'])
                    or case['selections'] != _selection_summary(case['events'])):
                raise ValueError('derived total forecast mismatch')
            key = tuple(case['untouched_pair']['source']), case['untouched_pair']['action']
            preserved = all(g.transitions[key] == graph.transitions[key] for g in _models(memory).values())
            if case['untouched_pair_preserved'] != preserved or not preserved:
                raise ValueError('untouched action content mismatch')
        # Reconstruct the ambiguity control using declared baseline and evaluation truth.
        first = instance['cases'][0]
        graph = TopologicalMemoryGraph.from_state(first['initial'])
        truth = _from_truth_rows(first['stages'][0]['truth'])
        targets = [(tuple(k['source']), k['action']) for k in first['targets']]
        if instance['ambiguity'] != _ambiguity(graph, truth, targets, config['window']):
            raise ValueError('matched-prefix ambiguity content mismatch')
    if report['aggregates'] != _aggregate(report['instances']):
        raise ValueError('derived aggregate content mismatch')


def verify_adaptation_study(path, replay=False):
    """Check seals, causal scores and independent metrics; optionally regenerate streams."""
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
        declared = report['protocol']['config']
        config = _config(declared['seeds'], declared['window'], declared['budget'])
        protocol = _protocol(config)
        if (report['schema'] != SCHEMA or report['protocol'] != protocol
                or report['protocol_sha256'] != _digest(protocol)
                or _read(path / 'plan.json') != protocol or report['evidence_scope'] != SCOPE):
            raise ValueError('declared protocol content mismatch')
        if [i['seed'] for i in report['instances']] != config['seeds']:
            raise ValueError('declared seed coverage mismatch')
        _validate(report, config)
        if (path / 'report.html').read_text(encoding='utf-8') != _render(report):
            raise ValueError('rendered report content mismatch')
        if replay and _json(_compute(config)) != _json(report):
            raise ValueError('deterministic replay content mismatch')
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError(f'invalid adaptation study archive: {exc}') from exc
    return {'ok': True, 'schema': SCHEMA, 'deterministic_replay': replay,
            'derived_metrics_verified': True, 'seed_count': len(config['seeds']),
            'report_sha256': manifest['files']['report.json'], 'promotion_performed': False}
