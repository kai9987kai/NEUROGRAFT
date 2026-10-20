"""Self-contained, fixed-weight comparisons of navigation assistance.

Every size/topology condition uses the same captured checkpoint and declared
seeds. Local hashes are integrity receipts, not authenticity signatures or proof
that worlds were absent from earlier training. No model is trained or promoted.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random

from .environment import NavigationWorld
from .navigation_artifacts import _read, _json, _write_json, capture_input
from .navigation_evaluation import (
    NavigationEvalConfig, _json_hash, _summary, frozen_navigation_evaluation, navigation_protocol,
)


SCHEMA = 'navigation-controller-study-v1'
_FIXED_FILES = frozenset({'frozen-input.pt', 'source.bundle.json', 'config.json',
                          'protocol.json', 'report.json', 'report.html', 'status.json'})
_SCOPE = (
    'Fixed-weight synthetic navigation across declared size/topology conditions. '
    'Raw policy, legal-action masking and observed-map planning are separate controls. '
    'Observed-map planning is engineered episodic assistance from local observations and public grid size; '
    'map_without_policy removes the learned tie-break prior. Zero masked/planned collisions follow legal filtering. '
    'Each variant and world starts with empty controller memory; no teacher actions or learning are used. '
    'The shortest-path oracle has privileged full-map access. All controls share one frozen model, '
    'and no-graft/no-recurrence are inference interventions rather than separately trained models. '
    'Seeds are paired across conditions and variants; conditions are correlated, not independent replications. '
    'Small arenas may share geometry across topology labels. Declared seeds do not prove unseen training worlds. '
    'No biological advantage, raw neural capability gain or pretrained language improvement is established. '
    'No checkpoint promotion is performed.'
)


@dataclass(frozen=True)
class StudyConfig:
    sizes: tuple[int, ...] = (7, 11)
    presets: tuple[str, ...] = NavigationWorld.PRESETS
    seeds: tuple[int, ...] = (41000, 41001)
    max_steps: int = 96

    def __post_init__(self):
        if (type(self.sizes) is not tuple or not 1 <= len(self.sizes) <= 27
                or any(type(size) is not int or not 5 <= size <= 31 for size in self.sizes)
                or len(set(self.sizes)) != len(self.sizes)):
            raise ValueError('sizes must be a nonempty tuple of distinct integers from 5 through 31')
        if (type(self.presets) is not tuple or not 1 <= len(self.presets) <= len(NavigationWorld.PRESETS)
                or any(type(preset) is not str or preset not in NavigationWorld.PRESETS for preset in self.presets)
                or len(set(self.presets)) != len(self.presets)):
            raise ValueError('presets must be a nonempty tuple of distinct supported topology names')
        if (type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= 16
                or any(type(seed) is not int or not 0 <= seed <= 2**63 - 1 for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('seeds must be 1-16 distinct integers from 0 through 2**63-1')
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 128:
            raise ValueError('max_steps must be an integer from 1 through 128')
        if len(self.sizes) * len(self.presets) * len(self.seeds) > 64:
            raise ValueError('study exceeds 64 size/topology/seed worlds')


def _settings(config):
    return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(config).items()}


def _config(values):
    if not isinstance(values, dict) or set(values) != {'sizes', 'presets', 'seeds', 'max_steps'}:
        raise ValueError('invalid study config fields')
    copied = dict(values)
    for key in ('sizes', 'presets', 'seeds'):
        if not isinstance(copied[key], list):
            raise ValueError('serialized study config requires JSON arrays')
        copied[key] = tuple(copied[key])
    return StudyConfig(**copied)


def _cap(bundle):
    if (sum(len(graph.nodes) for graph in bundle.graphs) > 2000
            or sum(len(graph.edges) for graph in bundle.graphs) > 50000):
        raise ValueError('navigation study cap is 2,000 nodes and 50,000 observations')


def _declaration(session, config):
    from .live import _checkpoint_digest
    conditions = []
    for size in config.sizes:
        for preset in config.presets:
            index = len(conditions)
            evaluation = NavigationEvalConfig(seeds=config.seeds, size=size, preset=preset,
                                              max_steps=config.max_steps, include_controllers=True)
            protocol = navigation_protocol(evaluation, session.config.unroll)
            conditions.append({'index': index, 'file': f'trial_{index:03d}.json',
                               'size': size, 'preset': preset, 'config': protocol['config'],
                               'protocol_id': _json_hash(protocol)})
    identity = {'schema': SCHEMA, 'config': _settings(config),
                'inference_unroll': session.config.unroll,
                'condition_protocol_ids': [condition['protocol_id'] for condition in conditions]}
    return {
        **identity, 'study_id': _json_hash(identity), 'conditions': conditions,
        'model_hash': _checkpoint_digest(session.model.state_dict()), 'graph_hash': session.bundle.digest(),
        'checkpoint_steps': session.steps, 'checkpoint_updates': session.updates,
        'learning': False, 'teacher_assistance': 0, 'promotion_performed': False,
    }


def _validate_episode_geometry(episode, seed, config, variant, initial):
    """Check recorded actions and metrics without running the learned model.

    Replaying the deterministic grid is much cheaper than neural inference. It
    checks internal physical and accounting consistency; only ``--replay``
    establishes that the captured neural model reproduces the recorded actions.
    """
    def number(value, low=None, high=None):
        if type(value) not in (int, float):
            raise ValueError('invalid study metric numeric type')
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (low is not None and value < low) or (high is not None and value > high):
            raise ValueError('study metric is outside its finite domain')

    def position(value, heading=False):
        keys = {'x', 'z', 'direction'} if heading else {'x', 'z', 'color'}
        if (type(value) is not dict or set(value) != keys
                or any(type(value[key]) is not int or not 1 <= value[key] <= config.size - 2 for key in ('x', 'z'))
                or (heading and (type(value['direction']) is not int or not 0 <= value['direction'] <= 3))
                or (not heading and type(value['color']) is not str)):
            raise ValueError('invalid study geometry position')

    for key in ('spl', 'teacher_agreement'):
        number(episode[key], 0, 1)
    number(episode['total_reward'])
    for key in ('shortest_distance', 'terminal_distance'):
        if type(episode[key]) is not int or not 0 <= episode[key] <= config.size * config.size:
            raise ValueError('invalid study distance metric')
    position(episode['start'], heading=True)
    position(episode['goal'])
    initial_snapshot = initial.snapshot()
    if (episode['start'] != initial_snapshot['agent'] or episode['goal'] != initial_snapshot['goal']
            or episode['initial_world_hash'] != _json_hash(initial.to_dict())):
        raise ValueError('study initial geometry identity mismatch')
    trace = episode['controller_trace']
    if type(trace) is not list or len(trace) != episode['steps']:
        raise ValueError('controller trace length disagrees with recorded actions')
    neural = variant not in ('random', 'shortest_path')
    expected_prior = neural and variant != 'map_without_policy'
    if type(episode['policy_prior_used']) is not bool or episode['policy_prior_used'] != expected_prior:
        raise ValueError('controller policy prior metadata disagrees with its variant')
    for key in ('policy_proposals', 'policy_overrides', 'proposed_wall_actions'):
        if type(episode[key]) is not int or not 0 <= episode[key] <= episode['steps']:
            raise ValueError('invalid controller proposal or override count')

    world = copy.deepcopy(initial)
    shortest = len(world.shortest_path()) - 1
    rng = random.Random(seed ^ 0x4E41564556414C)
    known_free, visited = set(), set()
    navigator = None
    if variant == 'map_without_policy':
        from .navigation_controller import ObservedMapNavigator
        navigator = ObservedMapNavigator(config.size)
    collisions = moves = matches = proposals = overrides = walls = 0
    reward = 0.0
    success = False
    for index, (action, frame) in enumerate(zip(episode['actions'], trace)):
        if (type(frame) is not dict or set(frame) != {'position', 'action', 'policy_action', 'reason',
                                                    'known_free', 'visited', 'frontiers'}
                or type(frame['action']) is not int or frame['action'] != action):
            raise ValueError('controller trace action disagrees with recorded actions')
        position(frame['position'], heading=True)
        if frame['position'] != world.snapshot()['agent']:
            raise ValueError('controller trace pose disagrees with deterministic geometry')
        raw = frame['policy_action']
        if ((neural and (type(raw) is not int or not 0 <= raw <= 3))
                or (not neural and raw is not None)):
            raise ValueError('invalid controller trace policy action')
        for key in ('known_free', 'visited', 'frontiers'):
            if type(frame[key]) is not int or not 0 <= frame[key] <= config.size * config.size:
                raise ValueError('invalid controller mapped-cell count')
        observation = world.observe()
        expert = world.expert_action()
        if neural:
            proposals += 1
            walls += bool(observation[2][raw])
            overrides += raw != action
        if variant in ('observed_map', 'map_without_policy'):
            current = world._agent
            visited.add(current)
            known_free.add(current)
            for direction, wall in enumerate(observation[2]):
                if not wall:
                    dx, dz = world.ACTIONS[direction]
                    known_free.add((current[0] + dx, current[1] + dz))
            expected_counts = (len(known_free), len(visited), len(known_free - visited))
            reason = 'known_goal' if world._goal in known_free else 'frontier'
            if (tuple(frame[key] for key in ('known_free', 'visited', 'frontiers')) != expected_counts
                    or frame['reason'] != reason or observation[2][action]):
                raise ValueError('controller map counts/reason disagree with observed geometry')
            if navigator is not None and navigator.choose(observation, [0, 0, 0, 0], use_policy=False) != action:
                raise ValueError('map-only controller actions disagree with deterministic planning')
        else:
            if any(frame[key] for key in ('known_free', 'visited', 'frontiers')):
                raise ValueError('non-map controller claims mapped cells')
            expected_reason = 'legal_action_mask' if variant == 'masked_policy' else variant
            if frame['reason'] != expected_reason:
                raise ValueError('controller reason disagrees with its variant')
            if variant == 'masked_policy':
                if observation[2][action] or (not observation[2][raw] and action != raw):
                    raise ValueError('masked policy action contradicts its local legal mask')
            elif neural and action != raw:
                raise ValueError('raw policy action differs from its neural proposal')
            elif variant == 'shortest_path' and action != expert:
                raise ValueError('oracle actions disagree with shortest-path geometry')
            elif variant == 'random' and action != rng.randrange(4):
                raise ValueError('random actions disagree with their declared seed')
        matches += action == expert
        previous = world._agent
        result = world.step(action)
        collisions += bool(result['collision'])
        moves += world._agent != previous
        reward += result['reward']
        if result['goal_reached']:
            success = True
            if index != episode['steps'] - 1:
                raise ValueError('study episode continued after first goal arrival')
    if not success and episode['steps'] != config.max_steps:
        raise ValueError('failed study episode stopped before its declared horizon')
    # Constructed worlds always have distinct initial agent and goal cells.
    # Consequently successful trials have a positive shortest path denominator.
    expected_metrics = {
        'shortest_distance': shortest, 'success': success, 'collisions': collisions,
        'path_length': moves, 'total_reward': reward, 'teacher_agreement': matches / episode['steps'],
        'terminal_distance': 0 if success else len(world.shortest_path()) - 1,
        'spl': shortest / max(shortest, moves) if success else 0.0,
        'policy_proposals': proposals, 'policy_overrides': overrides, 'proposed_wall_actions': walls,
    }
    if any(episode[key] != value for key, value in expected_metrics.items()):
        raise ValueError('study metrics disagree with recorded-action geometry')


def _validate_trial(report, declaration, condition):
    if not isinstance(report, dict):
        raise ValueError('invalid study condition report')
    for name in ('model_hash', 'graph_hash', 'inference_unroll', 'checkpoint_steps', 'checkpoint_updates'):
        if report.get(name) != declaration[name]:
            raise ValueError(f'condition checkpoint/model/graph identity mismatch: {name}')
    if (report.get('schema_version') != 2 or report.get('protocol_id') != condition['protocol_id']
            or report.get('config') != condition['config'] or report.get('learning') is not False
            or type(report.get('teacher_assistance')) is not int or report['teacher_assistance'] != 0):
        raise ValueError('condition evaluation protocol identity mismatch')
    settings = condition['config']
    config = NavigationEvalConfig(seeds=tuple(settings['seeds']), size=settings['size'],
                                  preset=settings['preset'], max_steps=settings['max_steps'], include_controllers=True)
    expected = navigation_protocol(config, declaration['inference_unroll'])['variants']
    if not isinstance(report.get('variants'), dict) or set(report['variants']) != set(expected):
        raise ValueError('condition variants disagree with the declared protocol')
    world_hashes = None
    initial_worlds = {seed: NavigationWorld(seed=seed, size=config.size, preset=config.preset) for seed in config.seeds}
    for name, variant in report['variants'].items():
        episodes = variant['episodes']
        if not isinstance(episodes, list) or len(episodes) != len(config.seeds):
            raise ValueError('condition episode count disagrees with the protocol')
        for seed, episode in zip(config.seeds, episodes):
            if (type(episode['seed']) is not int or episode['seed'] != seed or type(episode['steps']) is not int
                    or not 1 <= episode['steps'] <= config.max_steps
                    or type(episode['success']) is not bool
                    or type(episode['collisions']) is not int
                    or not 0 <= episode['collisions'] <= episode['steps']
                    or type(episode['path_length']) is not int
                    or not 0 <= episode['path_length'] <= episode['steps']
                    or not isinstance(episode['actions'], list)
                    or len(episode['actions']) != episode['steps']
                    or any(type(action) is not int or not 0 <= action <= 3 for action in episode['actions'])):
                raise ValueError('invalid study episode/action counts')
            _validate_episode_geometry(episode, seed, config, name, initial_worlds[seed])
        hashes = [episode['initial_world_hash'] for episode in episodes]
        if world_hashes is not None and hashes != world_hashes:
            raise ValueError('condition variants do not share initial worlds')
        world_hashes = hashes
        if variant['summary'] != _summary(episodes):
            raise ValueError('condition aggregate does not match its episodes')


def _aggregate(declaration, trials):
    variants = {}
    conditions = []
    for condition, trial in zip(declaration['conditions'], trials):
        _validate_trial(trial, declaration, condition)
        conditions.append({**condition, 'summaries': {name: value['summary'] for name, value in trial['variants'].items()}})
        for name, values in trial['variants'].items():
            bucket = variants.setdefault(name, {'episodes': []})
            bucket['episodes'].extend({**episode, 'size': condition['size'], 'preset': condition['preset'],
                                       'condition_index': condition['index']} for episode in values['episodes'])
    for variant in variants.values():
        variant['summary'] = _summary(variant['episodes'])
    return {
        'schema_version': 1, 'schema': SCHEMA, 'study_id': declaration['study_id'],
        'config': declaration['config'], 'model_hash': declaration['model_hash'],
        'graph_hash': declaration['graph_hash'], 'inference_unroll': declaration['inference_unroll'],
        'checkpoint_steps': declaration['checkpoint_steps'], 'checkpoint_updates': declaration['checkpoint_updates'],
        'learning': False, 'teacher_assistance': 0, 'promotion_performed': False,
        'conditions': conditions, 'variants': variants, 'evidence_scope': _SCOPE,
        'metric_reference': 'https://arxiv.org/abs/1807.06757',
    }


def _status(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_navigation_study(bundle, checkpoint, out_dir, config=None):
    """Freeze one input before loading, then archive every declared condition.

    The output must be a new directory. The supplied checkpoint and live session
    are never saved, stepped, installed or otherwise modified.
    """
    from .graphs import GraphBundle
    from .live import LiveSession
    from .navigation_report import write_navigation_evaluation_report

    directory = Path(out_dir).resolve()
    if directory.exists():
        raise ValueError(f'output already exists: {directory}; choose a new path')
    config = StudyConfig() if config is None else config
    if not isinstance(config, StudyConfig):
        raise ValueError('config must be StudyConfig')
    _cap(bundle)
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as exc:
        raise ValueError(f'output already exists: {directory}; choose a new path') from exc
    _write_json(directory / 'status.json', {'status': 'RUNNING'})
    try:
        frozen = capture_input(checkpoint, directory)
        _write_json(directory / 'source.bundle.json', bundle.to_dict())
        archived_bundle = GraphBundle.from_dict(_json(_read(directory / 'source.bundle.json')))
        session = LiveSession.load(frozen, archived_bundle)
        declaration = _declaration(session, config)
        # Declaration is on disk before any rollout can inspect outcomes.
        _write_json(directory / 'config.json', _settings(config))
        _write_json(directory / 'protocol.json', declaration)
        trials = []
        for condition in declaration['conditions']:
            trial = frozen_navigation_evaluation(session, NavigationEvalConfig(
                seeds=config.seeds, size=condition['size'], preset=condition['preset'],
                max_steps=config.max_steps, include_controllers=True))
            _write_json(directory / condition['file'], trial)
            trials.append(trial)
        result = _aggregate(declaration, trials)
        _write_json(directory / 'report.json', result)
        write_navigation_evaluation_report(result, directory / 'report.html')
        _status(directory / 'status.json', {'status': 'COMPLETE', 'study_id': result['study_id']})
        names = _FIXED_FILES | {condition['file'] for condition in declaration['conditions']}
        _write_json(directory / 'manifest.json', {'schema_version': 1,
                    'files': {name: hashlib.sha256(_read(directory / name)).hexdigest() for name in sorted(names)}})
        return result
    except BaseException as exc:
        _status(directory / 'status.json', {'status': 'FAILED', 'error': str(exc)[:2000]})
        raise


def verify_navigation_study(directory, replay=False):
    """Check artifact/identity/aggregate integrity and optionally rerun all trials."""
    from .graphs import GraphBundle
    from .live import LiveSession

    if type(replay) is not bool:
        raise ValueError('replay must be boolean')
    directory = Path(directory).resolve()
    try:
        settings = _json(_read(directory / 'config.json', 65536))
        config = _config(settings)
        names = _FIXED_FILES | {f'trial_{index:03d}.json' for index in range(len(config.sizes) * len(config.presets))}
        manifest = _json(_read(directory / 'manifest.json', 65536))
        if (not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'files'}
                or manifest['schema_version'] != 1 or not isinstance(manifest['files'], dict)
                or set(manifest['files']) != names):
            raise ValueError('study manifest must list exactly the declared artifacts')
        blobs = {}
        for name in sorted(names):
            blobs[name] = _read(directory / name)
            if hashlib.sha256(blobs[name]).hexdigest() != manifest['files'][name]:
                raise ValueError(f'study artifact hash mismatch: {name}')
        if _json(blobs['config.json']) != settings:
            raise ValueError('study config changed during verification')
        bundle = GraphBundle.from_dict(_json(blobs['source.bundle.json']))
        _cap(bundle)
        session = LiveSession.load(directory / 'frozen-input.pt', bundle)
        declaration = _declaration(session, config)
        if _json(blobs['protocol.json']) != declaration:
            raise ValueError('study protocol/model/graph identity mismatch')
        if _json(blobs['status.json']) != {'status': 'COMPLETE', 'study_id': declaration['study_id']}:
            raise ValueError('study completion status identity mismatch')
        trials = [_json(blobs[condition['file']]) for condition in declaration['conditions']]
        expected = _aggregate(declaration, trials)
        if _json(blobs['report.json']) != expected:
            raise ValueError('study aggregate or model/graph identity mismatch')
        if replay:
            replayed = []
            for condition, recorded in zip(declaration['conditions'], trials):
                actual = frozen_navigation_evaluation(session, NavigationEvalConfig(
                    seeds=config.seeds, size=condition['size'], preset=condition['preset'],
                    max_steps=config.max_steps, include_controllers=True))
                if actual != recorded:
                    raise ValueError(f'condition frozen replay differs in this runtime: {condition["file"]}')
                replayed.append(actual)
            if _aggregate(declaration, replayed) != expected:
                raise ValueError('replayed study aggregate differs in this runtime')
    except (KeyError, TypeError, AttributeError, IndexError, ZeroDivisionError) as exc:
        raise ValueError(f'invalid study evidence: {exc}') from exc
    return {
        'ok': True, 'schema': SCHEMA, 'study_id': declaration['study_id'],
        'model_hash': declaration['model_hash'], 'graph_hash': declaration['graph_hash'],
        'conditions_verified': len(trials), 'artifacts_verified': len(names), 'replay_verified': replay,
        'promotion_performed': False,
        'evidence_scope': 'Local artifact integrity, declared protocol and checkpoint identities. '
                          'Hashes are not authenticity signatures or proof of unseen worlds. '
                          'Replay checks frozen trials and their aggregate in the current runtime; no training occurs.',
    }
