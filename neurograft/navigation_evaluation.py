"""Bounded, frozen navigation rollouts with matched inference-time controls.

GraphCircuit supplies raw policy actions. Optional controllers add explicitly
labelled local masking/mapping. Diagnostic substrates remain excluded. No
training or teacher mixing occurs, and live state and global RNGs do not advance.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import random

import torch

from .environment import NavigationWorld


@dataclass(frozen=True)
class NavigationEvalConfig:
    seeds: tuple[int, ...] = (10000, 10001, 10002)
    size: int = 11
    preset: str = 'labyrinth'
    max_steps: int = 48
    include_controllers: bool = False
    include_readout_controls: bool = False
    include_steering_controls: bool = False

    def __post_init__(self):
        if type(self.include_controllers) is not bool:
            raise ValueError('include_controllers must be boolean')
        if type(self.include_readout_controls) is not bool:
            raise ValueError('include_readout_controls must be boolean')
        if type(self.include_steering_controls) is not bool:
            raise ValueError('include_steering_controls must be boolean')
        if (not isinstance(self.seeds, tuple) or not 1 <= len(self.seeds) <= 16
                or any(type(seed) is not int or not 0 <= seed <= 2**63 - 1 for seed in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError('seeds must be 1-16 distinct integers in [0, 2**63-1]')
        if type(self.size) is not int or not 5 <= self.size <= 31:
            raise ValueError('size must be an integer from 5 through 31')
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 128:
            raise ValueError('max_steps must be an integer from 1 through 128')
        if type(self.preset) is not str or self.preset not in NavigationWorld.PRESETS:
            raise ValueError(f'preset must be one of {NavigationWorld.PRESETS}')


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def navigation_protocol(config, unroll):
    """Canonical identity shared by evaluation and archived-evidence validation."""
    settings = asdict(config)
    settings['seeds'] = list(config.seeds)
    if not config.include_controllers:
        settings.pop('include_controllers')
    if not config.include_readout_controls:
        settings.pop('include_readout_controls')
    if not config.include_steering_controls:
        settings.pop('include_steering_controls')
    protocol = {'schema_version': 4 if config.include_steering_controls else 3 if config.include_readout_controls else 2 if config.include_controllers else 1,
                'config': settings, 'unroll': unroll,
                'variants': ['policy', 'no_graft', 'no_recurrence', 'random', 'shortest_path'],
                'success_rule': 'first arrival at initial goal', 'path_length': 'successful cell moves'}
    if config.include_controllers:
        protocol['variants'] += ['masked_policy', 'observed_map', 'map_without_policy']
        protocol['controller'] = 'observed-frontier-v1; local walls; policy prior only breaks ties'
    if config.include_readout_controls:
        protocol['variants'].append('no_context')
    if config.include_steering_controls:
        protocol['variants'] += ['cx_steer', 'cx_without_policy']
        protocol['steering'] = ('synthetic-goal-heading-v1; observed heading and goal displacement; '
                                'legal walls; weights 1.5/1.0; cosine mixture 0.65/0.35; '
                                'cx_without_policy uses zero logits')
    return protocol


def _episode(model, unroll, config, seed, variant):
    world = NavigationWorld(seed=seed, size=config.size, preset=config.preset)
    initial_hash = _json_hash(world.to_dict())
    start = dict(world.snapshot()['agent'])
    goal = dict(world.snapshot()['goal'])
    shortest = len(world.shortest_path()) - 1
    history = deque(maxlen=unroll)
    rng = random.Random(seed ^ 0x4E41564556414C)
    actions = []
    collisions = moved = matches = 0
    success = False
    reward = 0.0
    trace = []
    proposed_walls = overrides = 0
    navigator = None
    if variant in ('observed_map', 'map_without_policy'):
        from .navigation_controller import ObservedMapNavigator
        navigator = ObservedMapNavigator(config.size)
    for _ in range(config.max_steps):
        expert = world.expert_action()
        observation = world.observe()
        history.append(observation)
        raw_action = None
        reason = variant
        if variant == 'shortest_path':
            action = expert
        elif variant == 'random':
            action = rng.randrange(4)
        else:
            padding = [[[0.0] * 4 for _ in range(4)] for _ in range(unroll - len(history))]
            sequence = torch.tensor([padding + list(history)], dtype=model.policy.weight.dtype)
            model_controls = {'graft_enabled': variant != 'no_graft',
                              'recurrence': variant != 'no_recurrence'}
            if variant == 'no_context':
                model_controls['context_enabled'] = False
            output = model(sequence, **model_controls)
            if not torch.isfinite(output.logits).all():
                raise ValueError('frozen policy produced nonfinite logits')
            raw_action = int(output.logits.argmax(-1))
            action = raw_action
            logits = output.logits[0].tolist()
            if variant == 'masked_policy':
                from .navigation_controller import legal_action
                action = legal_action(observation, logits)
                reason = 'legal_action_mask'
            elif variant in ('cx_steer', 'cx_without_policy'):
                from .navigation_controller import cx_steering_action
                action = cx_steering_action(observation, logits if variant == 'cx_steer' else [0.] * 4)
                reason = 'cx_goal_heading_prior' if variant == 'cx_steer' else 'cx_goal_heading_without_policy'
            elif navigator is not None:
                action = navigator.choose(observation, logits, use_policy=variant == 'observed_map')
                reason = navigator.snapshot()['reason']
        if config.include_controllers or config.include_steering_controls:
            proposed_walls += raw_action is not None and bool(observation[2][raw_action])
            overrides += raw_action is not None and action != raw_action
            info = navigator.snapshot() if navigator is not None else {}
            trace.append({'position': dict(world.snapshot()['agent']), 'action': action,
                          'policy_action': raw_action, 'reason': reason,
                          'known_free': info.get('known_free', 0),
                          'visited': info.get('visited', 0), 'frontiers': info.get('frontiers', 0)})
        matches += action == expert
        actions.append(action)
        previous = world._agent
        result = world.step(action)
        collisions += bool(result['collision'])
        moved += world._agent != previous
        reward += result['reward']
        if result['goal_reached']:
            success = True
            break
    episode = {
        'seed': seed, 'initial_world_hash': initial_hash, 'start': start, 'goal': goal,
        'shortest_distance': shortest, 'steps': len(actions), 'path_length': moved,
        'success': success, 'spl': shortest / max(shortest, moved) if success else 0.0,
        'collisions': collisions, 'teacher_agreement': matches / len(actions),
        'terminal_distance': 0 if success else len(world.shortest_path()) - 1,
        'total_reward': reward, 'actions': actions,
    }
    if config.include_controllers or config.include_steering_controls:
        episode.update({'controller_trace': trace, 'proposed_wall_actions': proposed_walls,
                        'policy_overrides': overrides, 'policy_proposals': sum(t['policy_action'] is not None for t in trace),
                        'policy_prior_used': variant not in ('map_without_policy', 'cx_without_policy') and raw_action is not None})
    return episode


def _summary(episodes):
    count = len(episodes)
    steps = sum(item['steps'] for item in episodes)
    summary = {
        'episodes': count,
        'success_rate': sum(item['success'] for item in episodes) / count,
        'spl': sum(item['spl'] for item in episodes) / count,
        'collision_rate': sum(item['collisions'] for item in episodes) / steps,
        'mean_steps': steps / count,
        'teacher_agreement': sum(item['teacher_agreement'] * item['steps'] for item in episodes) / steps,
        'mean_terminal_distance': sum(item['terminal_distance'] for item in episodes) / count,
    }
    if 'controller_trace' in episodes[0]:
        proposals = sum(item['policy_proposals'] for item in episodes)
        summary.update({'proposed_wall_rate': sum(item['proposed_wall_actions'] for item in episodes) / proposals if proposals else None,
                        'policy_override_rate': sum(item['policy_overrides'] for item in episodes) / proposals if proposals else None})
    return summary


def frozen_navigation_evaluation(session, config=None):
    """Copy under the session lock, then evaluate outside it; returns JSON values.

    A report identifies the captured policy even if training continues during the
    trials. Seeds specify reproducible scenarios; they are not certified absent
    from earlier training. Repeatability is scoped to the same CPU/runtime.
    """
    from .live import _checkpoint_digest
    if config is None:
        config = NavigationEvalConfig()
    if not isinstance(config, NavigationEvalConfig):
        raise ValueError('config must be NavigationEvalConfig')
    with session.lock:
        if (sum(len(g.nodes) for g in session.bundle.graphs) > 2000
                or sum(len(g.edges) for g in session.bundle.graphs) > 50000):
            raise ValueError('navigation evaluation cap is 2,000 nodes and 50,000 observations')
        model = copy.deepcopy(session.model).eval()
        model.requires_grad_(False)
        model_hash = _checkpoint_digest(model.state_dict())
        graph_hash = session.bundle.digest()
        unroll = session.config.unroll
        checkpoint_steps, checkpoint_updates = session.steps, session.updates
    protocol = navigation_protocol(config, unroll)
    settings = protocol['config']
    variants = {}
    with torch.inference_mode():
        for variant in protocol['variants']:
            episodes = [_episode(model, unroll, config, seed, variant) for seed in config.seeds]
            variants[variant] = {'summary': _summary(episodes), 'episodes': episodes}
    baseline = variants['policy']['summary']
    effects = {
        name: {metric + '_delta': baseline[metric] - variants[name]['summary'][metric]
               for metric in ('success_rate', 'spl')}
        for name in ('no_graft', 'no_recurrence', 'random') + (('no_context',) if config.include_readout_controls else ())
    }
    return {
        'schema_version': protocol['schema_version'], 'protocol_id': _json_hash(protocol), 'config': settings,
        'inference_unroll': unroll, 'graph_hash': graph_hash, 'model_hash': model_hash,
        'checkpoint_steps': checkpoint_steps, 'checkpoint_updates': checkpoint_updates,
        'teacher_assistance': 0, 'learning': False, 'variants': variants, 'paired_effects': effects,
        'evidence_scope': (('Extended controller comparison: masked_policy filters adjacent walls; observed_map adds symbolic '
                           'episodic mapping/planning from the same local sensors and known grid size. map_without_policy '
                           'removes the learned tie-break prior. These are engineered assistance, not raw neural gains. '
                           'Zero collisions follow legal filtering. No controller receives the full obstacle map or teacher action. '
                           if config.include_controllers else '') + 'Frozen synthetic grid navigation; no teacher assistance or training. '
                           'No-graft and no-recurrence are inference-time interventions on identical weights, '
                           'not separately trained baselines. Random is seeded; shortest_path has full-map access. '
                           'Declared seeds are not proof of unseen training worlds. SPL counts successful cell moves; '
                           'collisions and attempted steps are separate. Teacher agreement uses one deterministic '
                           'shortest-path tie-break. No biological advantage or pretrained language capability established.'
                           + (' Readout control: no_graft retains the direct all-source readout in context_residual mode. '
                              'no_context bypasses only the context residual readout when present; legacy human readout '
                              'is unchanged. This is an inference-time intervention on identical weights, not a training '
                              'comparison or evidence of learning.' if config.include_readout_controls else '')
                           + (' Steering controls add engineered assistance: cx_steer adds a hand-selected goal/heading '
                              'cosine prior and legal wall mask to learned logits; cx_without_policy uses zero learned '
                              'logits for action selection. Heading is read directly from observations, not the diagnostic '
                              'ring attractor. These are inference-time interventions, not a cell-specific PFL circuit '
                              'reconstruction, learned biological mechanism or separately trained comparison.'
                              if config.include_steering_controls else '')),
        'metric_reference': 'https://arxiv.org/abs/1807.06757',
    }
