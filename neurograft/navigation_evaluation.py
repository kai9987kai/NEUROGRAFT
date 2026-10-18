"""Bounded, frozen navigation rollouts with matched inference-time controls.

Only GraphCircuit supplies policy actions. Auxiliary diagnostic substrates do not
drive that policy and are deliberately excluded. No training or teacher mixing
occurs here, and neither the live session nor global RNGs are advanced.
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

    def __post_init__(self):
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
    for _ in range(config.max_steps):
        expert = world.expert_action()
        history.append(world.observe())
        if variant == 'shortest_path':
            action = expert
        elif variant == 'random':
            action = rng.randrange(4)
        else:
            padding = [[[0.0] * 4 for _ in range(4)] for _ in range(unroll - len(history))]
            sequence = torch.tensor([padding + list(history)], dtype=model.policy.weight.dtype)
            output = model(sequence, graft_enabled=variant != 'no_graft',
                           recurrence=variant != 'no_recurrence')
            if not torch.isfinite(output.logits).all():
                raise ValueError('frozen policy produced nonfinite logits')
            action = int(output.logits.argmax(-1))
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
    return {
        'seed': seed, 'initial_world_hash': initial_hash, 'start': start, 'goal': goal,
        'shortest_distance': shortest, 'steps': len(actions), 'path_length': moved,
        'success': success, 'spl': shortest / max(shortest, moved) if success else 0.0,
        'collisions': collisions, 'teacher_agreement': matches / len(actions),
        'terminal_distance': 0 if success else len(world.shortest_path()) - 1,
        'total_reward': reward, 'actions': actions,
    }


def _summary(episodes):
    count = len(episodes)
    steps = sum(item['steps'] for item in episodes)
    return {
        'episodes': count,
        'success_rate': sum(item['success'] for item in episodes) / count,
        'spl': sum(item['spl'] for item in episodes) / count,
        'collision_rate': sum(item['collisions'] for item in episodes) / steps,
        'mean_steps': steps / count,
        'teacher_agreement': sum(item['teacher_agreement'] * item['steps'] for item in episodes) / steps,
        'mean_terminal_distance': sum(item['terminal_distance'] for item in episodes) / count,
    }


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
    settings = asdict(config)
    settings['seeds'] = list(config.seeds)
    protocol = {'schema_version': 1, 'config': settings, 'unroll': unroll,
                'variants': ['policy', 'no_graft', 'no_recurrence', 'random', 'shortest_path'],
                'success_rule': 'first arrival at initial goal', 'path_length': 'successful cell moves'}
    variants = {}
    with torch.inference_mode():
        for variant in protocol['variants']:
            episodes = [_episode(model, unroll, config, seed, variant) for seed in config.seeds]
            variants[variant] = {'summary': _summary(episodes), 'episodes': episodes}
    baseline = variants['policy']['summary']
    effects = {
        name: {metric + '_delta': baseline[metric] - variants[name]['summary'][metric]
               for metric in ('success_rate', 'spl')}
        for name in ('no_graft', 'no_recurrence', 'random')
    }
    return {
        'schema_version': 1, 'protocol_id': _json_hash(protocol), 'config': settings,
        'inference_unroll': unroll, 'graph_hash': graph_hash, 'model_hash': model_hash,
        'checkpoint_steps': checkpoint_steps, 'checkpoint_updates': checkpoint_updates,
        'teacher_assistance': 0, 'learning': False, 'variants': variants, 'paired_effects': effects,
        'evidence_scope': ('Frozen synthetic grid navigation; no teacher assistance or training. '
                           'No-graft and no-recurrence are inference-time interventions on identical weights, '
                           'not separately trained baselines. Random is seeded; shortest_path has full-map access. '
                           'Declared seeds are not proof of unseen training worlds. SPL counts successful cell moves; '
                           'collisions and attempted steps are separate. Teacher agreement uses one deterministic '
                           'shortest-path tie-break. No biological advantage or pretrained language capability established.'),
        'metric_reference': 'https://arxiv.org/abs/1807.06757',
    }
