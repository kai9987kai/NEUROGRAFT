"""Curriculum training for isolated navigation candidates with frozen evaluations."""
from __future__ import annotations

from collections import Counter, deque
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from .dynamics import GraphCircuit
from .environment import NavigationWorld
from .graphs import GraphBundle
from .live import LiveConfig, LiveSession, _checkpoint_digest
from .navigation_evaluation import (
    NavigationEvalConfig,
    _episode,
    _summary,
    frozen_navigation_evaluation,
)
from .navigation_report import write_navigation_report
from .navigation_artifacts import capture_input, write_run_evidence, seal_run


@dataclass(frozen=True)
class CurriculumConfig:
    rounds: int = 4
    worlds_per_round: int = 32
    updates_per_round: int = 64
    batch_size: int = 16
    sizes: tuple[int, ...] = (5, 7, 9, 11)
    presets: tuple[str, ...] = NavigationWorld.PRESETS
    init: str = 'fresh'
    learning_rate: float = 0.005
    validation_seeds: tuple[int, ...] = (20000, 20001, 20002, 20003)
    test_seeds: tuple[int, ...] = (10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007)
    dense_warmup_rounds: int = 1
    train_seed_start: int = 0
    seed: int = 0
    replay_capacity: int = 256
    max_steps: int = 48
    eval_size: int = 11
    eval_preset: str = 'labyrinth'
    readout: str = 'human'
    rotation_augmentation: bool = False
    collection_policy: str = 'mixed'
    include_readout_controls: bool = False
    predictive_head: bool = False
    predictive_weight: float = 0.0

    def __post_init__(self) -> None:
        if type(self.readout) is not str or self.readout not in ('human', 'context_residual'):
            raise ValueError("readout must be 'human' or 'context_residual'")
        if type(self.rotation_augmentation) is not bool:
            raise ValueError('rotation_augmentation must be boolean')
        if type(self.include_readout_controls) is not bool:
            raise ValueError('include_readout_controls must be boolean')
        if type(self.collection_policy) is not str or self.collection_policy not in ('mixed', 'expert', 'recovery'):
            raise ValueError("collection_policy must be 'mixed', 'expert' or 'recovery'")
        if type(self.predictive_head) is not bool:
            raise ValueError('predictive_head must be boolean')
        if (type(self.predictive_weight) not in (int, float)
                or not math.isfinite(self.predictive_weight) or not 0.0 <= self.predictive_weight <= 10.0):
            raise ValueError('predictive_weight must be a finite float in [0.0, 10.0]')
        if not self.predictive_head and self.predictive_weight != 0:
            raise ValueError('predictive_weight requires predictive_head=True')
        if self.train_seed_start == 0 and self.seed != 0:
            object.__setattr__(self, 'train_seed_start', self.seed)
        elif self.seed == 0 and self.train_seed_start != 0:
            object.__setattr__(self, 'seed', self.train_seed_start)

        if type(self.rounds) is not int or not 1 <= self.rounds <= 20:
            raise ValueError('rounds must be an integer from 1 through 20')
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 16:
            raise ValueError('batch_size must be an integer from 1 through 16')
        if type(self.worlds_per_round) is not int or not 1 <= self.worlds_per_round <= 256:
            raise ValueError('worlds_per_round must be an integer from 1 through 256')
        if type(self.updates_per_round) is not int or not 1 <= self.updates_per_round <= 256:
            raise ValueError('updates_per_round must be an integer from 1 through 256')
        if self.rounds * self.updates_per_round > 4096:
            raise ValueError('curriculum exceeds 4096 total optimizer updates')

        if not isinstance(self.sizes, tuple) or not self.sizes or any(type(s) is not int or not 5 <= s <= 31 for s in self.sizes):
            raise ValueError('sizes must be a non-empty tuple of integers from 5 through 31')
        if not isinstance(self.presets, tuple) or not self.presets or any(type(p) is not str or p not in NavigationWorld.PRESETS for p in self.presets):
            raise ValueError(f'presets must be a tuple of preset names in {NavigationWorld.PRESETS}')
        if len(set(self.sizes)) != len(self.sizes) or len(set(self.presets)) != len(self.presets):
            raise ValueError('sizes and presets must contain distinct conditions')

        if self.init not in ('fresh', 'checkpoint'):
            raise ValueError("init must be 'fresh' or 'checkpoint'")

        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= 0.1:
            raise ValueError('learning_rate must be finite and in [1e-6, 0.1]')

        if type(self.dense_warmup_rounds) is not int or not 0 <= self.dense_warmup_rounds <= self.rounds:
            raise ValueError('dense_warmup_rounds must be an integer from 0 through rounds')

        if type(self.train_seed_start) is not int or not 0 <= self.train_seed_start <= 2**63 - 1:
            raise ValueError('train_seed_start must be an integer in [0, 2**63-1]')
        if self.train_seed_start + self.rounds * self.worlds_per_round > 2**63 - 1:
            raise ValueError('train seeds exceed 2**63 - 1')

        if type(self.seed) is not int or not 0 <= self.seed <= 2**63 - 1:
            raise ValueError('seed must be an integer in [0, 2**63-1]')

        if type(self.replay_capacity) is not int or not 4 <= self.replay_capacity <= 4096 or self.replay_capacity < self.batch_size:
            raise ValueError('replay_capacity must be in [4, 4096] and at least batch_size')

        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 128:
            raise ValueError('max_steps must be an integer from 1 through 128')
        if self.rounds * self.worlds_per_round * self.max_steps > 65536:
            raise ValueError('curriculum exceeds 65536 potential rollout transitions')
        if type(self.eval_size) is not int or not 5 <= self.eval_size <= 31:
            raise ValueError('eval_size must be an integer from 5 through 31')
        if type(self.eval_preset) is not str or self.eval_preset not in NavigationWorld.PRESETS:
            raise ValueError(f'eval_preset must be one of {NavigationWorld.PRESETS}')

        if not isinstance(self.validation_seeds, tuple) or not 1 <= len(self.validation_seeds) <= 16 or any(type(s) is not int or not 0 <= s <= 2**63 - 1 for s in self.validation_seeds):
            raise ValueError('validation_seeds must be 1-16 integers in [0, 2**63-1]')
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError('validation_seeds must be distinct')

        if not isinstance(self.test_seeds, tuple) or not 1 <= len(self.test_seeds) <= 16 or any(type(s) is not int or not 0 <= s <= 2**63 - 1 for s in self.test_seeds):
            raise ValueError('test_seeds must be 1-16 integers in [0, 2**63-1]')
        if len(set(self.test_seeds)) != len(self.test_seeds):
            raise ValueError('test_seeds must be distinct')

        val_set = set(self.validation_seeds)
        test_set = set(self.test_seeds)
        if val_set & test_set:
            raise ValueError('validation_seeds and test_seeds must be disjoint')

        train_seeds = set(range(self.train_seed_start, self.train_seed_start + self.rounds * self.worlds_per_round))
        if val_set & train_seeds:
            raise ValueError('validation_seeds must not overlap with training seeds')
        if test_set & train_seeds:
            raise ValueError('test_seeds must not overlap with training seeds')


def _selection_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    """Evaluation key using validation metrics only: (success_rate, spl, -collision_rate)."""
    return (
        float(summary.get('success_rate', 0.0)),
        float(summary.get('spl', 0.0)),
        -float(summary.get('collision_rate', 1.0)),
    )


_ACTION_NAMES = ('north', 'east', 'south', 'west')


def _rotate_navigation_batch(sequence, labels, turns):
    """Rotate normalized x/z observations and action labels clockwise.

    All-zero frames are left padding, not observations. The transformation
    rotates a labelled route; it does not query the fixed-tie full-map expert.
    This is data augmentation, not an equivariant architecture guarantee.
    """
    rotated = sequence.clone()
    for turn in range(1, 4):
        indices = (turns == turn).nonzero(as_tuple=True)[0]
        if not indices.numel():
            continue
        source = sequence[indices]
        result = source.clone()
        for _ in range(turn):
            previous = result.clone()
            for row in (0, 1):
                result[:, :, row, 0] = 1 - previous[:, :, row, 1]
                result[:, :, row, 1] = previous[:, :, row, 0]
                result[:, :, row, 2] = -previous[:, :, row, 3]
                result[:, :, row, 3] = previous[:, :, row, 2]
            result[:, :, 2] = previous[:, :, 2].roll(1, dims=-1)
        result[(source == 0).all(dim=-1).all(dim=-1)] = 0
        rotated[indices] = result
    return rotated, (labels + turns) % 4


def _hash_sample(digest, value):
    digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8') + b'\n')


def _prepare_navigation_batch(batch, turns, dtype, predictive):
    """Apply one common frame transform to observations, actions and targets."""
    sequence = torch.tensor([sample['observations'] for sample in batch], dtype=dtype)
    labels = torch.tensor([sample['action'] for sample in batch], dtype=torch.long)
    sequence, labels = _rotate_navigation_batch(sequence, labels, turns)
    if not predictive:
        return sequence, labels, None, None, None
    executed = torch.tensor([sample['executed_action'] for sample in batch], dtype=torch.long)
    target = torch.tensor([sample['next_observation'] for sample in batch], dtype=dtype)
    target, executed = _rotate_navigation_batch(target[:, None], executed, turns)
    valid = torch.tensor([sample['transition_valid'] for sample in batch], dtype=torch.bool)
    return sequence, labels, executed, target[:, 0], valid


def _masked_prediction_loss(prediction, targets, valid):
    """Omit boundary transitions rather than train on resets or report zero error."""
    return F.mse_loss(prediction[valid], targets[valid]) if bool(valid.any()) else None


def _world_schedule(config: CurriculumConfig, round_idx: int) -> list[dict[str, Any]]:
    """Cycle the Cartesian product, sizes fastest, independently of seed offset.

    A prefix shorter than len(sizes)*len(presets) cannot cover all conditions;
    each complete cycle does. Round boundaries continue the same ordinal stream.
    """
    start = (round_idx - 1) * config.worlds_per_round
    return [
        {'seed': config.train_seed_start + ordinal,
         'size': config.sizes[ordinal % len(config.sizes)],
         'preset': config.presets[(ordinal // len(config.sizes)) % len(config.presets)]}
        for ordinal in range(start, start + config.worlds_per_round)
    ]


class _ActionReservoir:
    """Independent Algorithm R reservoirs for each expert-action class.

    Quotas split capacity equally, assigning any remainder in N/E/S/W order.
    Missing classes leave their slots unused, so rare actions remain protected
    when observed later. Within each class, all historical transitions have the
    same inclusion probability. Sampling with replacement balances present
    classes and shuffles their cycle to avoid always favoring north in small batches.
    """

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity, self.rng = capacity, rng
        self.quotas = [capacity // 4 + int(action < capacity % 4) for action in range(4)]
        self.buckets: list[list[dict[str, Any]]] = [[] for _ in range(4)]
        self.seen = [0] * 4

    @property
    def samples(self) -> list[dict[str, Any]]:
        return [sample for bucket in self.buckets for sample in bucket]

    def add(self, sample: dict[str, Any]) -> None:
        action = sample['action']
        self.seen[action] += 1
        bucket, quota = self.buckets[action], self.quotas[action]
        if len(bucket) < quota:
            bucket.append(sample)
        else:
            index = self.rng.randrange(self.seen[action])
            if index < quota:
                bucket[index] = sample

    def balanced_batch(self, size: int, rng: random.Random) -> list[dict[str, Any]]:
        actions = [action for action, bucket in enumerate(self.buckets) if bucket]
        if not actions:
            raise ValueError('cannot sample an empty navigation replay')
        rng.shuffle(actions)
        return [rng.choice(self.buckets[actions[index % len(actions)]]) for index in range(size)]

    def summary(self) -> dict[str, Any]:
        samples = self.samples
        return {
            'strategy': 'per-action-algorithm-r', 'capacity': self.capacity,
            'seen_total': sum(self.seen), 'retained_total': len(samples),
            'action_quotas': dict(zip(_ACTION_NAMES, self.quotas)),
            'action_counts': {name: len(bucket) for name, bucket in zip(_ACTION_NAMES, self.buckets)},
            'seen_action_counts': dict(zip(_ACTION_NAMES, self.seen)),
            'retained_worlds': len({sample['world_seed'] for sample in samples}),
            'retained_rounds': sorted({sample['round'] for sample in samples}),
        }


def _collect_round(
    bundle: GraphBundle,
    model: GraphCircuit,
    config: CurriculumConfig,
    round_idx: int,
    replay_buffer: _ActionReservoir,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    unroll: int,
) -> dict[str, Any]:
    """Collect rollouts for one round and perform optimizer updates."""
    warmup = round_idx <= config.dense_warmup_rounds
    model.config = replace(model.config, top_k=3 if warmup else 2)

    teacher_prob = (1.0 if config.collection_policy == 'expert' else
                    0.5 if config.collection_policy == 'recovery' else
                    max(0.15, 1.0 - (round_idx - 1) / max(1, config.rounds)))
    collection_hash = hashlib.sha256()
    batch_hash = hashlib.sha256()
    base_collection_hash = hashlib.sha256()
    base_batch_hash = hashlib.sha256()
    prediction_valid = prediction_excluded = 0
    augmentation_rng = random.Random(config.seed ^ (round_idx << 32) ^ 0x524F54415445)
    rotation_counts = [0] * 4

    schedule = _world_schedule(config, round_idx)
    counts = [0] * 4
    teacher_steps = learner_steps = 0
    for condition in schedule:
        world_seed, size, preset = condition['seed'], condition['size'], condition['preset']
        world = NavigationWorld(seed=world_seed, size=size, preset=preset)
        world.goals_reached = (world_seed * 7) % 6
        world.steps = (world_seed * 13) % 40

        history = deque(maxlen=unroll)
        for _ in range(config.max_steps):
            obs = world.observe()
            expert = world.expert_action()
            history.append(obs)
            padding = [[[0.0] * 4 for _ in range(4)] for _ in range(unroll - len(history))]
            padded = padding + list(history)

            # Preserve the historical draw even when probability is exactly 1.
            # Its position determines all subsequent raw minibatch identities.
            if rng.random() < teacher_prob:
                action = expert
                teacher_steps += 1
            else:
                learner_steps += 1
                with torch.no_grad():
                    model.eval()
                    seq_t = torch.tensor([padded], dtype=model.policy.weight.dtype)
                    action = int(model(seq_t).logits.argmax(-1))
            res = world.step(action)
            next_obs = world.observe()

            sample = {'observations': padded, 'action': expert,
                      'world_seed': world_seed, 'size': size,
                      'preset': preset, 'round': round_idx}
            _hash_sample(base_collection_hash, sample)
            if config.predictive_head:
                sample['next_observation'] = next_obs
                sample['executed_action'] = action
                # Success changes the goal; timeout can also teleport the agent.
                # Neither is a deterministic physical transition target.
                sample['transition_valid'] = not res['episode_done']
                prediction_valid += int(sample['transition_valid'])
                prediction_excluded += int(not sample['transition_valid'])
            replay_buffer.add(sample)
            _hash_sample(collection_hash, sample)
            counts[expert] += 1
            if res['goal_reached']:
                break

    losses, prediction_losses, total_losses = [], [], []
    valid_counts = []
    for _ in range(config.updates_per_round):
        batch = replay_buffer.balanced_batch(config.batch_size, rng)
        _hash_sample(batch_hash, batch)
        _hash_sample(base_batch_hash, [{name: sample[name] for name in
            ('observations', 'action', 'world_seed', 'size', 'preset', 'round')} for sample in batch])
        turns = [augmentation_rng.randrange(4) if config.rotation_augmentation else 0 for _ in batch]
        for turn in turns:
            rotation_counts[turn] += 1
        seq, labels, executed_actions, targets, valid = _prepare_navigation_batch(
            batch, torch.tensor(turns, dtype=torch.long), model.policy.weight.dtype, config.predictive_head)

        optimizer.zero_grad(set_to_none=True)
        model.train()
        out = model(seq, action=executed_actions) if config.predictive_head else model(seq)
        loss = F.cross_entropy(out.logits, labels)
        total_loss = loss
        pred_loss = None
        count = int(valid.sum()) if valid is not None else 0
        if config.predictive_head and count:
            pred_loss = _masked_prediction_loss(out.predicted_obs, targets, valid)
            if not torch.isfinite(pred_loss):
                raise ValueError('prediction loss became nonfinite')
            if config.predictive_weight > 0:
                total_loss = loss + float(config.predictive_weight) * pred_loss
        if not torch.isfinite(total_loss):
            model.eval()
            raise ValueError('training loss became nonfinite')
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        losses.append(float(loss.detach()))
        prediction_losses.append(float(pred_loss.detach()) if pred_loss is not None else None)
        total_losses.append(float(total_loss.detach()))
        valid_counts.append(count)

    condition_counts = Counter((item['size'], item['preset']) for item in schedule)
    receipt = {
        'collection': {
            'worlds': len(schedule), 'world_seeds': [item['seed'] for item in schedule],
            'condition_counts': [{'size': size, 'preset': preset, 'worlds': count}
                                 for (size, preset), count in sorted(condition_counts.items())],
            'transitions': sum(counts), 'action_counts': dict(zip(_ACTION_NAMES, counts)),
            'teacher_probability': teacher_prob, 'teacher_steps': teacher_steps,
            'learner_steps': learner_steps,
            'data_sha256': collection_hash.hexdigest(),
        },
        'replay': replay_buffer.summary(),
        'training': {
            'updates': len(losses), 'batch_size': config.batch_size, 'losses': losses,
            'mean_loss': sum(losses) / len(losses), 'min_loss': min(losses),
            'max_loss': max(losses), 'final_loss': losses[-1],
            'batch_sha256': batch_hash.hexdigest(), 'rotation_counts': rotation_counts,
        },
    }
    if config.predictive_head:
        receipt['collection'].update({
            'base_data_sha256': base_collection_hash.hexdigest(),
            'prediction_valid_transitions': prediction_valid,
            'prediction_excluded_transitions': prediction_excluded,
        })
        receipt['training'].update({'base_batch_sha256': base_batch_hash.hexdigest(),
            'prediction': {'weight': config.predictive_weight, 'losses': prediction_losses,
                           'total_losses': total_losses, 'valid_examples': sum(valid_counts),
                           'valid_counts': valid_counts,
                           'mean_mse': sum((loss or 0.0) * n for loss, n in zip(prediction_losses, valid_counts))
                                       / sum(valid_counts) if sum(valid_counts) else None}})
    return receipt


def run_navigation_training(
    bundle: GraphBundle,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    config: CurriculumConfig | None = None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    if out_path.exists():
        raise ValueError(f'output already exists: {out_path}; choose a new path')

    if config is None:
        config = CurriculumConfig()
    if not isinstance(config, CurriculumConfig):
        raise ValueError('config must be CurriculumConfig')

    test_eval_config = NavigationEvalConfig(
        seeds=config.test_seeds,
        size=config.eval_size,
        preset=config.eval_preset,
        max_steps=config.max_steps,
        include_readout_controls=config.include_readout_controls,
    )
    val_eval_config = NavigationEvalConfig(
        seeds=config.validation_seeds,
        size=config.eval_size,
        preset=config.eval_preset,
        max_steps=config.max_steps,
        include_readout_controls=config.include_readout_controls,
    )

    saved_py_rng = random.getstate()
    saved_torch_rng = torch.get_rng_state().clone()

    out_path.mkdir(parents=True, exist_ok=False)
    status_file = out_path / 'status.json'
    status_file.write_text(json.dumps({'status': 'RUNNING', 'round': 0}, indent=2), encoding='utf-8')

    try:
        baseline_path = capture_input(checkpoint_path, out_path)
        baseline_session = LiveSession.load(baseline_path, bundle)
        unroll = baseline_session.config.unroll
        hidden_dim = baseline_session.config.hidden_dim
        baseline_eval = frozen_navigation_evaluation(baseline_session, test_eval_config)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        training_rng = random.Random(config.seed ^ 0x4E4156545241494E)

        if config.init == 'checkpoint':
            if baseline_session.config.readout != config.readout:
                raise ValueError('checkpoint readout must match curriculum readout; use fresh initialization to compare architectures')
            if baseline_session.config.predictive_head != config.predictive_head:
                raise ValueError('checkpoint predictive_head must match curriculum predictive_head')
            model = copy.deepcopy(baseline_session.model)
            model.config = replace(model.config, top_k=2)
        else:
            fresh_session = LiveSession(
                bundle,
                LiveConfig(seed=config.seed, hidden_dim=hidden_dim, batch_size=config.batch_size, unroll=unroll,
                           readout=config.readout, predictive_head=config.predictive_head)
            )
            model = copy.deepcopy(fresh_session.model)

        initial_model_hash = _checkpoint_digest(model.state_dict())
        initial_common_model_hash = _checkpoint_digest({name: value for name, value in model.state_dict().items()
                                                       if not name.startswith('context_policy.') and not name.startswith('predictor.')})
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        replay_buffer = _ActionReservoir(config.replay_capacity, random.Random(config.seed ^ 0x5245504C4159))

        # Round 0: Untrained validation with top_k=2
        model.config = replace(model.config, top_k=2)
        with torch.inference_mode():
            episodes_0 = [_episode(model, unroll, val_eval_config, s, 'policy') for s in val_eval_config.seeds]
            val_summary_0 = _summary(episodes_0)

        progress = [{'round': 0, 'validation': val_summary_0, 'updates': 0}]
        best_round = 0
        best_key = _selection_key(val_summary_0)
        best_state_dict = copy.deepcopy(model.state_dict())

        # Training rounds 1..rounds
        for round_idx in range(1, config.rounds + 1):
            receipt = _collect_round(bundle, model, config, round_idx, replay_buffer, optimizer, training_rng, unroll)
            total_updates = round_idx * config.updates_per_round

            saved_top_k = model.config.top_k
            model.config = replace(model.config, top_k=2)
            with torch.inference_mode():
                episodes_r = [_episode(model, unroll, val_eval_config, s, 'policy') for s in val_eval_config.seeds]
                val_summary_r = _summary(episodes_r)
            model.config = replace(model.config, top_k=saved_top_k)

            progress.append({'round': round_idx, 'validation': val_summary_r,
                             'updates': total_updates, **receipt})
            status_file.write_text(json.dumps({'status': 'RUNNING', 'round': round_idx,
                                               'updates': total_updates, 'validation': val_summary_r},
                                              indent=2), encoding='utf-8')
            key_r = _selection_key(val_summary_r)
            if key_r > best_key:
                best_key = key_r
                best_round = round_idx
                best_state_dict = copy.deepcopy(model.state_dict())

        # Export candidate checkpoint using best model
        candidate_session = LiveSession(
            bundle,
            LiveConfig(seed=config.seed, hidden_dim=hidden_dim, batch_size=config.batch_size, unroll=unroll,
                       readout=config.readout, predictive_head=config.predictive_head)
        )
        candidate_session.model.load_state_dict(best_state_dict)
        candidate_session.anchor_model.load_state_dict(best_state_dict)
        candidate_session.model.config = replace(candidate_session.model.config, top_k=2)
        candidate_session.anchor_model.config = replace(candidate_session.anchor_model.config, top_k=2)
        candidate_session.steps = 0
        candidate_session.updates = 0
        candidate_session.running = False
        candidate_session.learning = False

        candidate_path = out_path / 'candidate.pt'
        candidate_session.save(candidate_path)

        candidate_eval = frozen_navigation_evaluation(candidate_session, test_eval_config)
        selected_model_hash = _checkpoint_digest(best_state_dict)

        result = {
            'schema_version': 1,
            'protocol_id': baseline_eval['protocol_id'],
            'selected_round': best_round,
            'selected_model_hash': selected_model_hash,
            'initial_model_hash': initial_model_hash,
            'initial_common_model_hash': initial_common_model_hash,
            'parameter_counts': {
                'total': sum(parameter.numel() for parameter in model.parameters()),
                'context_residual': sum(parameter.numel() for name, parameter in model.named_parameters()
                                        if name.startswith('context_policy.')),
                'predictor': sum(parameter.numel() for name, parameter in model.named_parameters()
                                 if name.startswith('predictor.')),
            },
            'baseline_model_hash': baseline_eval['model_hash'],
            'baseline': baseline_eval,
            'candidate': candidate_eval,
            'progress': progress,
            'config': asdict(config),
            'promotion_performed': False,
            'evidence_scope': (
                'Synthetic navigation experiment; inspect raw receipts. '
                'Teacher assistance and learning are disabled during frozen trials. '
                'No biological-connectome advantage or pretrained language ability established.'
            ),
        }

        write_run_evidence(out_path, result, bundle)
        # Write report.html
        write_navigation_report(result, out_path / 'report.html')

        # Write status.json
        status_file.write_text(json.dumps({'status': 'COMPLETE', 'selected_round': best_round}, indent=2), encoding='utf-8')

        # Write manifest.json
        seal_run(out_path)

        return result

    except Exception as exc:
        status_file.write_text(json.dumps({'status': 'FAILED', 'error': str(exc)}, indent=2), encoding='utf-8')
        if (out_path / 'candidate.pt').exists():
            (out_path / 'candidate.pt').unlink()
        raise
    finally:
        random.setstate(saved_py_rng)
        torch.set_rng_state(saved_torch_rng)
