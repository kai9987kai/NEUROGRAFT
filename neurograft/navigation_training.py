"""Curriculum training for isolated navigation candidates with frozen evaluations."""
from __future__ import annotations

from collections import deque
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

    def __post_init__(self) -> None:
        if self.train_seed_start == 0 and self.seed != 0:
            object.__setattr__(self, 'train_seed_start', self.seed)
        elif self.seed == 0 and self.train_seed_start != 0:
            object.__setattr__(self, 'seed', self.train_seed_start)

        if type(self.rounds) is not int or not 1 <= self.rounds <= 20:
            raise ValueError('rounds must be an integer from 1 through 20')
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if type(self.worlds_per_round) is not int or not 1 <= self.worlds_per_round <= 256:
            raise ValueError('worlds_per_round must be an integer from 1 through 256')
        if type(self.updates_per_round) is not int or self.updates_per_round < 1:
            raise ValueError('updates_per_round must be a positive integer')

        if not isinstance(self.sizes, tuple) or not self.sizes or any(type(s) is not int or not 5 <= s <= 31 for s in self.sizes):
            raise ValueError('sizes must be a non-empty tuple of integers from 5 through 31')
        if not isinstance(self.presets, tuple) or not self.presets or any(type(p) is not str or p not in NavigationWorld.PRESETS for p in self.presets):
            raise ValueError(f'presets must be a tuple of preset names in {NavigationWorld.PRESETS}')

        if self.init not in ('fresh', 'checkpoint'):
            raise ValueError("init must be 'fresh' or 'checkpoint'")

        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be a finite positive number')

        if type(self.dense_warmup_rounds) is not int or not 0 <= self.dense_warmup_rounds <= self.rounds:
            raise ValueError('dense_warmup_rounds must be an integer from 0 through rounds')

        if type(self.train_seed_start) is not int or not 0 <= self.train_seed_start <= 2**63 - 1:
            raise ValueError('train_seed_start must be an integer in [0, 2**63-1]')
        if self.train_seed_start + self.rounds * self.worlds_per_round > 2**63 - 1:
            raise ValueError('train seeds exceed 2**63 - 1')

        if type(self.seed) is not int or not 0 <= self.seed <= 2**63 - 1:
            raise ValueError('seed must be an integer in [0, 2**63-1]')

        if type(self.replay_capacity) is not int or self.replay_capacity < 4 or self.replay_capacity < self.batch_size:
            raise ValueError('replay_capacity must be at least 4 and at least batch_size')

        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 128:
            raise ValueError('max_steps must be an integer from 1 through 128')
        if type(self.eval_size) is not int or not 5 <= self.eval_size <= 31:
            raise ValueError('eval_size must be an integer from 5 through 31')
        if type(self.eval_preset) is not str or self.eval_preset not in NavigationWorld.PRESETS:
            raise ValueError(f'eval_preset must be one of {NavigationWorld.PRESETS}')

        if not isinstance(self.validation_seeds, tuple) or not 1 <= len(self.validation_seeds) <= 64 or any(type(s) is not int or not 0 <= s <= 2**63 - 1 for s in self.validation_seeds):
            raise ValueError('validation_seeds must be a tuple of integers in [0, 2**63-1]')
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError('validation_seeds must be distinct')

        if not isinstance(self.test_seeds, tuple) or not 1 <= len(self.test_seeds) <= 64 or any(type(s) is not int or not 0 <= s <= 2**63 - 1 for s in self.test_seeds):
            raise ValueError('test_seeds must be a tuple of integers in [0, 2**63-1]')
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


def _collect_round(
    bundle: GraphBundle,
    model: GraphCircuit,
    config: CurriculumConfig,
    round_idx: int,
    replay_buffer: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    unroll: int,
) -> None:
    """Collect rollouts for one round and perform optimizer updates."""
    warmup = round_idx <= config.dense_warmup_rounds
    model.config = replace(model.config, top_k=3 if warmup else 2)

    teacher_prob = max(0.15, 1.0 - (round_idx - 1) / max(1, config.rounds))

    for w_idx in range(config.worlds_per_round):
        world_seed = config.train_seed_start + (round_idx - 1) * config.worlds_per_round + w_idx
        size = config.sizes[world_seed % len(config.sizes)]
        preset = config.presets[world_seed % len(config.presets)]
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
            replay_buffer.append({'observations': padded, 'action': expert})

            if rng.random() < teacher_prob:
                action = expert
            else:
                with torch.no_grad():
                    model.eval()
                    seq_t = torch.tensor([padded], dtype=model.policy.weight.dtype)
                    action = int(model(seq_t).logits.argmax(-1))
            res = world.step(action)
            if res['goal_reached']:
                break

    if len(replay_buffer) > config.replay_capacity:
        del replay_buffer[:-config.replay_capacity]

    for _ in range(config.updates_per_round):
        by_action: dict[int, list[dict[str, Any]]] = {}
        for item in replay_buffer:
            by_action.setdefault(item['action'], []).append(item)
        actions_present = sorted(by_action.keys())
        batch = []
        for i in range(config.batch_size):
            act = actions_present[i % len(actions_present)]
            batch.append(rng.choice(by_action[act]))

        seq = torch.tensor([s['observations'] for s in batch], dtype=model.policy.weight.dtype)
        labels = torch.tensor([s['action'] for s in batch], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        model.train()
        out = model(seq)
        loss = F.cross_entropy(out.logits, labels)
        if not torch.isfinite(loss):
            model.eval()
            raise ValueError('training loss became nonfinite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.eval()


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

    baseline_path = Path(checkpoint_path)
    baseline_session = LiveSession.load(baseline_path, bundle)
    unroll = baseline_session.config.unroll
    hidden_dim = baseline_session.config.hidden_dim

    test_eval_config = NavigationEvalConfig(
        seeds=config.test_seeds,
        size=config.eval_size,
        preset=config.eval_preset,
        max_steps=config.max_steps,
    )
    val_eval_config = NavigationEvalConfig(
        seeds=config.validation_seeds,
        size=config.eval_size,
        preset=config.eval_preset,
        max_steps=config.max_steps,
    )

    baseline_eval = frozen_navigation_evaluation(baseline_session, test_eval_config)

    saved_py_rng = random.getstate()
    saved_torch_rng = torch.get_rng_state().clone()

    out_path.mkdir(parents=True, exist_ok=False)
    status_file = out_path / 'status.json'
    status_file.write_text(json.dumps({'status': 'RUNNING', 'round': 0}, indent=2), encoding='utf-8')

    try:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        training_rng = random.Random(config.seed ^ 0x4E4156545241494E)

        if config.init == 'checkpoint':
            model = copy.deepcopy(baseline_session.model)
            model.config = replace(model.config, top_k=2)
        else:
            fresh_session = LiveSession(
                bundle,
                LiveConfig(seed=config.seed, hidden_dim=hidden_dim, batch_size=config.batch_size, unroll=unroll)
            )
            model = copy.deepcopy(fresh_session.model)

        initial_model_hash = _checkpoint_digest(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        replay_buffer: list[dict[str, Any]] = []

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
            _collect_round(bundle, model, config, round_idx, replay_buffer, optimizer, training_rng, unroll)
            total_updates = round_idx * config.updates_per_round

            saved_top_k = model.config.top_k
            model.config = replace(model.config, top_k=2)
            with torch.inference_mode():
                episodes_r = [_episode(model, unroll, val_eval_config, s, 'policy') for s in val_eval_config.seeds]
                val_summary_r = _summary(episodes_r)
            model.config = replace(model.config, top_k=saved_top_k)

            progress.append({'round': round_idx, 'validation': val_summary_r, 'updates': total_updates})
            key_r = _selection_key(val_summary_r)
            if key_r > best_key:
                best_key = key_r
                best_round = round_idx
                best_state_dict = copy.deepcopy(model.state_dict())

        # Export candidate checkpoint using best model
        candidate_session = LiveSession(
            bundle,
            LiveConfig(seed=config.seed, hidden_dim=hidden_dim, batch_size=config.batch_size, unroll=unroll)
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

        # Write report.html
        write_navigation_report(result, out_path / 'report.html')

        # Write status.json
        status_file.write_text(json.dumps({'status': 'COMPLETE', 'selected_round': best_round}, indent=2), encoding='utf-8')

        # Write manifest.json
        files_manifest: dict[str, str] = {}
        for p in sorted(out_path.iterdir(), key=lambda x: x.name):
            if p.is_file() and p.name != 'manifest.json':
                files_manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        (out_path / 'manifest.json').write_text(json.dumps({'files': files_manifest}, indent=2), encoding='utf-8')

        return result

    except Exception as exc:
        status_file.write_text(json.dumps({'status': 'FAILED', 'error': str(exc)}, indent=2), encoding='utf-8')
        if (out_path / 'candidate.pt').exists():
            (out_path / 'candidate.pt').unlink()
        raise
    finally:
        random.setstate(saved_py_rng)
        torch.set_rng_state(saved_torch_rng)
