"""Persistent bounded online imitation in a synthetic navigation task.

The command/memory chat is deliberately local and is not a pretrained language
model. Source observations remain immutable; only circuit parameters adapt.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import random
import re
import struct
import tempfile
import threading
from typing import Any
import zipfile
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F

from .active_inference import ActiveInferenceArbiter
from .archaeology import ArchaeologyReport, MemoryArchaeologyAssay, ResectionSpec
from .bioelectric import BioelectricConfig, BioelectricMorphogeneticSubstrate
from .byte_bridge import ByteBridgeConfig, ConnectomeByteBridge
from .compass import CXRingCompass
from .consolidation import ConnectomeConsolidationStack, ConsolidationConfig
from .dales_law import audit_bundle_dales_law
from .dynamics import DynamicsConfig, GraphCircuit
from .environment import NavigationWorld
from .graphs import GraphBundle
from .memory import TriTierMemory
from .navigation_controller import MODES, ObservedMapNavigator, legal_action
from . import live_state
from .phase_plane import PhasePlaneEngine
from .self_model import ConnectomeForwardSelfModel, SelfModelConfig
from .spectral_diffusion import ChebyshevSpectralDiffusion
from .stp import STPConfig, TsodyksMarkramSubstrate


MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_VERSION = 1
_CHECKPOINT_VERSION = 3
_BACKEND = "local-memory-and-control"
_TORCH_INIT_LOCK = threading.RLock()
_SCOPE = ("Online supervised shortest-path imitation in a synthetic navigation world; "
          "recent policy accuracy is not held-out generalization. Cross-source routes are "
          "synthetic latent edges. Chat stores explicit facts and controls; no pretrained language model.")


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return value


def _number(value: Any, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric) or not low <= numeric <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return numeric


def _text(value: Any, maximum: int, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} characters")
    return value


def _checkpoint_digest(value: Any) -> str:
    """Canonical integrity hash over supported primitives and exact tensor bytes."""
    digest = hashlib.sha256()
    def visit(item, depth=0):
        if depth > 32:
            raise ValueError("checkpoint nesting is too deep")
        if item is None:
            digest.update(b"n")
        elif type(item) is bool:
            digest.update(b"t" if item else b"f")
        elif type(item) is int:
            digest.update(b"i" + str(item).encode() + b";")
        elif type(item) is float:
            digest.update(b"d" + struct.pack("!d", item))
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"s" + str(len(encoded)).encode() + b":" + encoded)
        elif isinstance(item, Tensor):
            if item.layout != torch.strided or item.device.type != "cpu":
                raise ValueError("checkpoint tensors must be dense CPU tensors")
            digest.update(b"T" + str(item.dtype).encode() + str(tuple(item.shape)).encode())
            digest.update(item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                if type(key) not in (str, int):
                    raise ValueError("checkpoint mapping keys must be strings or integers")
                visit(key, depth + 1)
                visit(item[key], depth + 1)
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"(" if isinstance(item, tuple) else b"[")
            for child in item:
                visit(child, depth + 1)
            digest.update(b")" if isinstance(item, tuple) else b"]")
        else:
            raise ValueError("unsupported checkpoint value")
    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class LiveConfig:
    seed: int = 0
    hidden_dim: int = 8
    learning_rate: float = 0.008
    batch_size: int = 8
    train_every: int = 1
    unroll: int = 4
    replay_capacity: int = 128
    teacher_probability: float = 0.45
    threads: int = 2
    readout: str = "human"

    def __post_init__(self):
        for name, low, high in (("seed", 0, 2**63 - 1), ("hidden_dim", 2, 32), ("batch_size", 1, 16),
                                ("train_every", 1, 128), ("unroll", 1, 16),
                                ("replay_capacity", 2, 512), ("threads", 1, 8)):
            _integer(getattr(self, name), low, high, name)
        _number(self.learning_rate, 1e-6, 0.1, "learning_rate")
        _number(self.teacher_probability, 0, 1, "teacher_probability")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must cover batch_size")
        if type(self.readout) is not str or self.readout not in ("human", "context_residual"):
            raise ValueError("readout must be 'human' or 'context_residual'")


class LiveSession:
    """One lock-protected CPU model, world, optimizer and local chat memory."""

    def __init__(self, bundle: GraphBundle, config: LiveConfig | None = None):
        self.lock = threading.RLock()
        self.bundle = bundle
        self.config = config if config is not None else LiveConfig()
        if not isinstance(self.config, LiveConfig):
            raise ValueError("config must be LiveConfig")
        with _TORCH_INIT_LOCK:
            # Initialization is isolated; ticks consume no global Python/torch RNG.
            if torch.get_num_threads() != self.config.threads:
                torch.set_num_threads(self.config.threads)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.config.seed)
                self.model = GraphCircuit(bundle, DynamicsConfig(hidden_dim=self.config.hidden_dim, output_dim=4,
                                                                 readout=self.config.readout))
                self.consolidation = ConnectomeConsolidationStack(
                    ConsolidationConfig(hidden_size=self.config.hidden_dim, shared_dim=32, latent_experts=4, latent_top_k=2)
                )
                self.anchor_model = GraphCircuit(bundle, DynamicsConfig(hidden_dim=self.config.hidden_dim, output_dim=4,
                                                                        readout=self.config.readout))
                self.anchor_model.load_state_dict(self.model.state_dict())
                self.anchor_model.eval()
                self.stp = TsodyksMarkramSubstrate([len(g.nodes) for g in bundle.graphs], STPConfig(dt=0.20))
                self.stp_state = self.stp.initial_state(1)
                self.spectral_diffusers = {g.manifest.source_id: ChebyshevSpectralDiffusion(g) for g in bundle.graphs}
                self.byte_bridge = ConnectomeByteBridge(ByteBridgeConfig(context_dim=4 * self.config.hidden_dim))
                self.bioelectric = BioelectricMorphogeneticSubstrate(bundle, BioelectricConfig(dt=0.20))
                self.bioelectric_state = self.bioelectric.initial_voltage(1)
                self.self_model = ConnectomeForwardSelfModel(SelfModelConfig(latent_dim=4 * self.config.hidden_dim))
                self.archaeology = MemoryArchaeologyAssay(bundle)
        self.model.eval()
        self._stp_summary = {
            "mean_release_prob": 0.35,
            "mean_vesicle_pool": 1.0,
            "mean_synaptic_efficacy": 1.0,
            "depression_fraction": 0.0,
            "facilitation_fraction": 0.0,
        }
        self._bioelectric_summary = {
            "global_mean_mv": -65.0,
            "resting_baseline_mv": -65.0,
            "excitability_gain": 1.0,
            "depolarized_ratio": 0.0,
        }
        self._self_model_summary = {
            "prediction_error": 0.0,
            "counterfactual_null_error": 0.0,
            "agency_score": 0.0,
            "reality_drift": 0.0,
            "agency_regime": "passive-concordance",
            "self_alignment": 1.0,
            "total_audits": 0,
        }
        self._prev_latent = None
        self._prev_action = None
        self._phase_summary = None
        self._telemetry_archaeology_cache = None
        self._checkpoint_compatibility = {"loaded_format": _CHECKPOINT_VERSION, "auxiliary_state_restored": True}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.world = NavigationWorld(seed=self.config.seed)
        self.navigation_mode = 'policy'
        self.navigator = ObservedMapNavigator(self.world.size)
        self.rng = random.Random(self.config.seed ^ 0x4E47524F)
        self.recurrent_state = self.model.initial_state(1)
        self.replay = deque(maxlen=self.config.replay_capacity)
        self.observation_history = deque(maxlen=self.config.unroll)
        self.loss_history = deque(maxlen=64)
        self.reward_history = deque(maxlen=64)
        self.policy_matches = deque(maxlen=128)
        self.steps = 0
        self.updates = 0
        self.last_loss = None
        self.last_action = None
        self.last_action_source = None
        self.last_logits = torch.zeros(1, 4)
        self.routes = torch.zeros(4, 5)
        self.routes[:, 4] = 1
        self.facts = {}
        self.chat_history = deque(maxlen=64)
        self.last_saved = None
        self._running = True
        self._learning = True
        self._model_hash = None
        self._sources = bundle.stats()["sources"]
        self._initial_weights = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        self._connectome_motifs = self._compute_connectome_motifs(bundle)
        self.compass = CXRingCompass(num_wedges=16)
        self.active_inference = ActiveInferenceArbiter()
        self.memory = TriTierMemory(capacity=128, lesion_mode="intact")
        self.phase_plane = PhasePlaneEngine(dt=0.20)
        self._dales_law_reports = audit_bundle_dales_law(bundle)
        self._anchor_kl_drift = 0.0

    @staticmethod
    def _compute_connectome_motifs(bundle: GraphBundle) -> dict[str, dict[str, Any]]:
        motifs: dict[str, dict[str, Any]] = {}
        for graph in bundle.graphs:
            src_id = graph.manifest.source_id
            deg = {node.id: 0 for node in graph.nodes}
            chem_pairs: set[tuple[str, str]] = set()
            chem_count = elec_count = macro_count = 0
            pos_w = neg_w = 0.0
            for edge in graph.edges:
                deg[edge.src] = deg.get(edge.src, 0) + 1
                if edge.dst != edge.src:
                    deg[edge.dst] = deg.get(edge.dst, 0) + 1
                if edge.relation == "chemical":
                    chem_count += 1
                    if edge.src != edge.dst:
                        chem_pairs.add((edge.src, edge.dst))
                elif edge.relation == "electrical":
                    elec_count += 1
                else:
                    macro_count += 1
                if edge.weight >= 0:
                    pos_w += abs(edge.weight)
                else:
                    neg_w += abs(edge.weight)
            reciprocal = sum(1 for u, v in chem_pairs if u < v and (v, u) in chem_pairs)
            ranked = sorted(
                enumerate(graph.nodes),
                key=lambda item: (-deg.get(item[1].id, 0), item[1].original_id),
            )[:4]
            motifs[src_id] = {
                "chemical_edges": chem_count,
                "electrical_edges": elec_count,
                "macro_edges": macro_count,
                "reciprocal_dyads": reciprocal,
                "excitatory_fraction": pos_w / max(1e-9, pos_w + neg_w),
                "rich_club_hubs": [
                    {"index": idx, "id": node.original_id, "degree": deg.get(node.id, 0)}
                    for idx, node in ranked
                ],
            }
        return motifs

    @property
    def running(self):
        with self.lock:
            return self._running

    @running.setter
    def running(self, enabled):
        with self.lock:
            if type(enabled) is not bool:
                raise ValueError("running must be boolean")
            self._running = enabled

    @property
    def learning(self):
        with self.lock:
            return self._learning

    @learning.setter
    def learning(self, enabled):
        with self.lock:
            if type(enabled) is not bool:
                raise ValueError("learning must be boolean")
            self._learning = enabled

    def _reset_inference(self):
        self.recurrent_state = self.model.initial_state(1)
        self.stp_state = self.stp.initial_state(1)
        self.observation_history.clear()
        self.compass.reset(float(self.world.direction) * (math.pi / 2.0))
        self._prev_latent = None
        self._prev_action = None
        self.phase_plane.reset()
        self._phase_summary = None
        self._telemetry_archaeology_cache = None
        self.navigator = ObservedMapNavigator(self.world.size)

    def set_navigation_mode(self, mode):
        """Select explicit control assistance; learned-policy behavior is default.

        Opt-in controllers read observations and never mix shortest-path teacher
        actions. Supervised replay labels can still be collected while learning.
        """
        if type(mode) is not str or mode not in MODES:
            raise ValueError('navigation mode must be one of: ' + ', '.join(MODES))
        with self.lock:
            self.navigation_mode = mode
            self._reset_inference()
            return self.snapshot()

    @property
    def checkpoint_compatibility(self):
        """Legacy v1 artifacts restore policy state but lack observer/anchor history."""
        with self.lock:
            return dict(self._checkpoint_compatibility)

    def set_goal(self, color=None, x=None, z=None):
        with self.lock:
            self.world.set_goal(color=color, x=x, z=z)
            self._reset_inference()
            return self.snapshot()

    def resize_world(self, size: int = 15, preset: str = "labyrinth"):
        with self.lock:
            new_world = NavigationWorld(seed=self.world.seed, size=size, preset=preset)
            new_world.steps = self.world.steps
            new_world.goals_reached = self.world.goals_reached
            new_world.total_reward = self.world.total_reward
            new_world.episode = max(self.world.episode, self.world.goals_reached, 1 if self.world.steps > 0 else 0)
            new_world.episode_steps = 0
            new_world.last_reward = 0.0
            self.world = new_world
            self._reset_inference()
            return self.snapshot()

    def _train(self):
        samples = self.rng.sample(list(self.replay), self.config.batch_size)
        sequence = torch.tensor([sample["observations"] for sample in samples], dtype=torch.float32)
        labels = torch.tensor([sample["action"] for sample in samples], dtype=torch.long)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        student_out = self.model(sequence)
        imitation_loss = F.cross_entropy(student_out.logits, labels)

        # Soft-KL teacher distillation from anchor checkpoint to prevent catastrophic forgetting
        with torch.no_grad():
            anchor_out = self.anchor_model(sequence)
            anchor_probs = F.softmax(anchor_out.logits, dim=-1)
        student_log_probs = F.log_softmax(student_out.logits, dim=-1)
        distill_loss = F.kl_div(student_log_probs, anchor_probs, reduction="batchmean")
        # Roundoff can put an otherwise zero KL a few ulps below zero.
        self._anchor_kl_drift = max(0.0, float(distill_loss.detach().item()))

        # Consolidation is an observer in this live policy. Its detached balance
        # diagnostic was a constant, not a trainable policy regularizer.
        loss = imitation_loss + 0.15 * distill_loss
        if not torch.isfinite(loss):
            self.model.eval()
            raise ValueError("online loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.model.eval()
        self.last_loss = float(loss.detach())
        self.loss_history.append(self.last_loss)
        self.updates += 1
        self._model_hash = None

    def tick(self, steps=1, manual_action=None):
        _integer(steps, 1, 256, "steps")
        if manual_action is not None:
            _integer(manual_action, 0, 3, "manual_action")
        with self.lock:
            for _ in range(steps):
                observation = self.world.observe()
                expert = self.world.expert_action()
                self.observation_history.append(observation)
                window = list(self.observation_history)
                padded = [[[0.0] * 4 for _ in range(4)] for _ in range(self.config.unroll - len(window))] + window
                with torch.no_grad():
                    output = self.model(torch.tensor([padded], dtype=torch.float32))
                self.recurrent_state = tuple(state.detach() for state in output.state)
                _, self.stp_state, self._stp_summary = self.stp.step(self.recurrent_state, self.stp_state)
                self.last_logits = output.logits.detach().clone()
                self.routes = output.routes[0].detach().clone()
                self.replay.append({"observations": padded, "action": expert})
                policy_action = int(output.logits.argmax(-1))
                controlled_action = None
                if self.navigation_mode == 'observed_map':
                    controlled_action = self.navigator.choose(observation, output.logits[0].tolist())
                elif self.navigation_mode == 'masked_policy':
                    controlled_action = legal_action(observation, output.logits[0].tolist())
                if manual_action is not None:
                    action, source = manual_action, "manual"
                elif controlled_action is not None:
                    self.policy_matches.append(policy_action == expert)
                    action, source = controlled_action, self.navigation_mode
                else:
                    self.policy_matches.append(policy_action == expert)
                    assisted = self.rng.random() < self.config.teacher_probability
                    action, source = (expert, "teacher") if assisted else (policy_action, "policy")
                previous_direction = self.world.direction
                result = self.world.step(action)
                self.steps += 1
                self.last_action, self.last_action_source = action, source
                self.reward_history.append(float(result["reward"]))

                # Biological Central Complex ring attractor compass update
                cur_dir = self.world.direction
                turn = (cur_dir - previous_direction + 2) % 4 - 2
                ang_vel = float(turn) * (math.pi / 2.0) / 0.20
                env_t = self.world.environment_telemetry()
                self.compass.step(angular_velocity=ang_vel, sensory_input=env_t["lidar_8"], dt=0.20)

                # Tri-tier cognitive memory recording
                flat_obs = [v for row in observation for v in row]
                surprise = abs(float(result["reward"]) - (self.reward_history[-2] if len(self.reward_history) >= 2 else 0.0))
                self.memory.record_step(
                    step=self.steps,
                    agent_pos=(self.world._agent[0], self.world._agent[1]),
                    action=action,
                    reward=float(result["reward"]),
                    state_features=flat_obs,
                    surprise=surprise,
                )

                # Nonlinear dynamical systems phase-plane audit
                dist_to_goal = math.hypot(self.world._goal[0] - self.world._agent[0], self.world._goal[1] - self.world._agent[1])
                max_dist = max(1.0, math.hypot(self.world.size, self.world.size))
                self._phase_summary = self.phase_plane.audit(
                    energy_rate=self.world.goals_reached / max(1.0, self.steps / 50.0),
                    goal_proximity=1.0 - (dist_to_goal / max_dist),
                    clearance_margin=min(env_t["lidar_8"]) if env_t["lidar_8"] else 0.5,
                    heading_alignment=1.0 if action == expert else 0.2,
                )

                # Bioelectric Morphogenetic Substrate (BMS) field update
                neural_acts = {
                    g.manifest.source_id: st
                    for g, st in zip(self.bundle.graphs, self.recurrent_state)
                }
                self.bioelectric_state, self._bioelectric_summary = self.bioelectric.step(
                    self.bioelectric_state, neural_acts
                )

                # Connectome Forward Self-Model & Agency Audit
                cur_latent = torch.cat([st.mean(dim=1) for st in self.recurrent_state], dim=-1).detach()
                if self._prev_latent is not None and self._prev_action is not None:
                    self._self_model_summary = self.self_model.step_audit(
                        self._prev_latent, self._prev_action, cur_latent, train=self._learning
                    )
                self._prev_latent = cur_latent
                self._prev_action = action

                if result["episode_done"]:
                    self._reset_inference()
                    self._prev_latent = None
                    self._prev_action = None
                if (self._learning and self.steps % self.config.train_every == 0
                        and len(self.replay) >= self.config.batch_size):
                    self._train()
            return self.snapshot()

    def telemetry(self) -> dict[str, Any]:
        """Compute live circuit introspection, virtual lesions, LTC tau, neuromodulators, and environment telemetry."""
        with self.lock:
            expert = self.world.expert_action()
            path = self.world.shortest_path()
            env_info = self.world.environment_telemetry()
            gates = [float(g.detach().sigmoid().mean()) for g in self.model.graft_gates]
            delta_sq = sum(
                float((param.detach() - self._initial_weights[name]).pow(2).sum())
                for name, param in self.model.named_parameters()
            )
            weight_delta = math.sqrt(max(0.0, delta_sq))
            graft_impact = 0.0
            source_keys = ("worm", "fly_f", "fly_m", "human")
            source_ablation = {k: 0.0 for k in source_keys}
            # Preview the next decision on the current world with current weights.
            # Appending to a local window leaves the actual rollout history intact.
            window = (list(self.observation_history) + [self.world.observe()])[-self.config.unroll:]
            padded = [[[0.0] * 4 for _ in range(4)] for _ in range(self.config.unroll - len(window))] + window
            seq = torch.tensor([padded], dtype=torch.float32)
            with torch.no_grad():
                intact = self.model(seq)
                probe_logits, probe_context = intact.logits, intact.context
                ungrafted = self.model(seq, graft_enabled=False).logits
                graft_impact = float((probe_logits - ungrafted).norm())
                for idx, skey in enumerate(source_keys):
                    ablated = self.model(seq, ablate_source=idx).logits
                    source_ablation[skey] = float((probe_logits - ablated).norm())
            routes = intact.routes.detach().clamp(min=1e-12)
            entropy = max(0.0, min(1.0, float(-(routes * routes.log()).sum(dim=-1).mean() / math.log(5.0))))
            probs = torch.softmax(probe_logits[0], dim=-1).tolist()
            env_info["policy_probe"] = {"step": self.steps, "updates": self.updates,
                                        "input": "current world observation with a copied history window",
                                        "effect_scope": "paired logit sensitivity; not behavioral or biological causality"}
            top_nodes = {}
            liquid_tau = []
            connectome_motifs = {}
            graph_map = {graph.manifest.source_id: graph for graph in self.bundle.graphs}
            for idx, (source_id, state) in enumerate(zip(source_keys, self.recurrent_state)):
                graph = graph_map[source_id]
                mags = state[0].detach().abs().mean(dim=-1)
                mean_act = float(mags.mean())
                tau_s = 1.0 / (0.35 + 0.65 * mean_act + 0.30 * gates[idx])
                liquid_tau.append(round(float(tau_s), 4))
                k = min(4, len(graph.nodes))
                vals, indices = torch.topk(mags, k=k)
                top_nodes[source_id] = [
                    {
                        "id": graph.nodes[int(n_idx)].original_id,
                        "full_id": graph.nodes[int(n_idx)].id,
                        "kind": graph.nodes[int(n_idx)].kind,
                        "activity": float(val),
                    }
                    for val, n_idx in zip(vals, indices)
                ]
                base_motif = self._connectome_motifs[source_id]
                connectome_motifs[source_id] = {
                    **base_motif,
                    "rich_club_hubs": [
                        {
                            "id": h["id"],
                            "degree": h["degree"],
                            "activity": float(mags[h["index"]]),
                        }
                        for h in base_motif["rich_club_hubs"]
                    ],
                }
            recent_r = (
                sum(list(self.reward_history)[-8:]) / max(1, len(list(self.reward_history)[-8:]))
                if self.reward_history else 0.0
            )
            acc = sum(self.policy_matches) / len(self.policy_matches) if self.policy_matches else 0.5
            dopamine = max(0.0, min(1.0, 0.45 * env_info["chemoattractant"] + 0.35 * max(0.0, recent_r + 0.1) + 0.20 * float(max(probs))))
            serotonin = max(0.0, min(1.0, 0.60 * acc + 0.40 * (self.world.goals_reached / (self.world.goals_reached + 2.0))))
            wall_prox = 1.0 - (sum(env_info["lidar_8"]) / 8.0)
            octopamine = max(0.0, min(1.0, 0.45 * wall_prox + 0.30 * min(1.0, env_info["detour_index"] - 1.0) + 0.25 * entropy))
            plasticity_gain = round(0.5 + 0.75 * dopamine + 0.35 * octopamine - 0.20 * serotonin, 3)
            if env_info["choke_points"] and wall_prox > 0.55:
                regime = "choke-point-maneuver"
            elif dopamine >= 0.62 and acc >= 0.75:
                regime = "goal-exploitation"
            elif octopamine >= 0.55:
                regime = "arousal-exploration"
            else:
                regime = "foraging-chemotaxis"
            obstacles_set = {(item["x"], item["z"]) for item in self.world.snapshot()["obstacles"]}
            agent_pt = (self.world._agent[0], self.world._agent[1])
            goal_pt = (self.world._goal[0], self.world._goal[1])
            visits_dict = {
                (int(k.split(",")[0]), int(k.split(",")[1])): v["visits"]
                for k, v in self.memory.spatial_map.items()
            }
            efe_eval = self.active_inference.evaluate_actions(
                agent_pos=agent_pt,
                goal_pos=goal_pt,
                obstacles=obstacles_set,
                world_size=self.world.size,
                lidar_8=env_info["lidar_8"],
                visit_counts=visits_dict,
                dopamine=dopamine,
                octopamine=octopamine,
            )
            flat_obs = [v for row in self.world.observe() for v in row]
            memory_probe = copy.deepcopy(self.memory)
            if memory_probe.lesion_mode == "random_retrieval" and memory_probe.episodes:
                # A stable observer-local draw neither consumes global/session RNG
                # nor changes what the next training sample will be.
                selected = random.Random(self.config.seed ^ self.steps ^ 0x4D454D).choice(list(memory_probe.episodes))
                memory_probe.episodes = deque([selected], maxlen=memory_probe.capacity)
                memory_probe.lesion_mode = "intact"
            mem_context = memory_probe.retrieve_context(flat_obs)
            mem_context["mode"] = self.memory.lesion_mode
            mem_context["role"] = "observer_only; retrieval does not drive the navigation policy"
            dist_to_goal = math.hypot(goal_pt[0] - agent_pt[0], goal_pt[1] - agent_pt[1])
            max_dist = max(1.0, math.hypot(self.world.size, self.world.size))
            phase_audit = copy.deepcopy(self._phase_summary) if self._phase_summary is not None else copy.deepcopy(self.phase_plane).audit(
                energy_rate=self.world.goals_reached / max(1.0, self.steps / 50.0),
                goal_proximity=1.0 - (dist_to_goal / max_dist),
                clearance_margin=min(env_info["lidar_8"]) if env_info["lidar_8"] else 0.5,
                heading_alignment=1.0 if self.last_action == expert else 0.2,
            )
            spectral_summary = {}
            for src_key, state in zip(source_keys, self.recurrent_state):
                if src_key in self.spectral_diffusers:
                    spectral_summary[src_key] = self.spectral_diffusers[src_key].multi_scale_energies(state)

            flat_context = torch.cat([st.mean(dim=1) for st in self.recurrent_state], dim=-1)
            byte_summary = self.byte_bridge.decode_semantic_intent(flat_context)
            byte_summary["role"] = "untrained observer projection; labels are not language understanding"

            with torch.no_grad():
                _, consolidation_aux = self.consolidation(probe_context)
            route_mean = consolidation_aux["route_probs"].reshape(-1, 4).mean(dim=0).tolist()

            gain = float(self._bioelectric_summary.get("excitability_gain", 1.0))
            cache_key = (self.steps, self.updates, _checkpoint_digest(padded), gain)
            if self._telemetry_archaeology_cache is None or self._telemetry_archaeology_cache[0] != cache_key:
                report = copy.copy(self.archaeology).run_resection_assay(self.model, seq, bioelectric_gain=gain).to_dict()
                self._telemetry_archaeology_cache = (cache_key, report)
            arch_summary = copy.deepcopy(self._telemetry_archaeology_cache[1])
            arch_summary["role"] = "observer sensitivity assay; rescue rescales logits and cannot change the chosen action"

            return {
                "action_probs": [float(p) for p in probs],
                "policy_confidence": float(max(probs)),
                "expert_action": int(expert),
                "expert_path": path,
                "graft_gates": gates,
                "route_entropy": entropy,
                "weight_delta_norm": weight_delta,
                "graft_impact": graft_impact,
                "top_nodes": top_nodes,
                "source_ablation": source_ablation,
                "liquid_tau": liquid_tau,
                "neuromodulators": {
                    "dopamine": round(dopamine, 4),
                    "serotonin": round(serotonin, 4),
                    "octopamine": round(octopamine, 4),
                    "plasticity_gain": plasticity_gain,
                    "regime": regime,
                },
                "environment": env_info,
                "connectome_motifs": connectome_motifs,
                "compass": self.compass.to_dict(),
                "active_inference": efe_eval,
                "memory": mem_context,
                "phase_plane": phase_audit,
                "dales_law": self._dales_law_reports,
                "consolidation": {
                    "gate_abs_mean": consolidation_aux["gate_abs_mean"],
                    "router_entropy": consolidation_aux["router_entropy"],
                    "router_balance": consolidation_aux["router_balance"],
                    "route_probs": route_mean,
                    "memory_entropy": consolidation_aux.get("memory_entropy", 0.0),
                    "anchor_kl_drift": round(self._anchor_kl_drift, 4),
                    "role": "observer_only; consolidation is not an optimized policy input",
                },
                "stp": copy.deepcopy(getattr(self, "_stp_summary", {
                    "mean_release_prob": 0.35,
                    "mean_vesicle_pool": 1.0,
                    "mean_synaptic_efficacy": 1.0,
                    "depression_fraction": 0.0,
                    "facilitation_fraction": 0.0,
                })),
                "spectral_diffusion": spectral_summary,
                "byte_bridge": byte_summary,
                "bioelectric": {**copy.deepcopy(self._bioelectric_summary), "role": "observer_only; voltage does not modulate policy"},
                "self_model": {**copy.deepcopy(self._self_model_summary), "role": "online predictive observer; agency score is not causal evidence"},
                "archaeology": arch_summary,
            }

    def snapshot(self):
        with self.lock:
            # New primitive containers prevent callers from mutating live memory.
            world = self.world.snapshot()
            if self._model_hash is None:
                self._model_hash = _checkpoint_digest(self.model.state_dict())
            return {"version": _VERSION, "running": self._running, "learning": self._learning,
                    "steps": self.steps, "updates": self.updates, "episodes": world["episode"],
                    "goals_reached": world["goals_reached"], "last_loss": self.last_loss,
                    "loss_history": list(self.loss_history), "reward_history": list(self.reward_history),
                    "policy_accuracy": sum(self.policy_matches) / len(self.policy_matches) if self.policy_matches else 0.0,
                    "last_action": self.last_action, "last_action_source": self.last_action_source,
                    "teacher_probability": self.config.teacher_probability, "world": world,
                    "navigation": {"mode": self.navigation_mode,
                                   "readout": self.config.readout,
                                   "teacher_probability": self.config.teacher_probability if self.navigation_mode == 'policy' else 0.0,
                                   "map": self.navigator.snapshot()},
                    "sources": {key: dict(value) for key, value in self._sources.items()},
                    "routes": self.routes.tolist(),
                    "activity": [float(state.abs().mean()) for state in self.recurrent_state],
                    "facts": dict(self.facts), "chat": [dict(item) for item in self.chat_history],
                    "backend": _BACKEND, "graph_hash": self.bundle.digest(), "model_hash": self._model_hash,
                    "last_saved": dict(self.last_saved) if self.last_saved else None, "evidence_scope": _SCOPE}

    def chat(self, message):
        message = _text(message, 2000, "message").strip()
        with self.lock:
            lower = message.lower().rstrip(".!?")
            action = None
            if lower in ("pause", "stop", "pause learning"):
                self._running = False
                reply = "Session paused. Explicit steps and manual moves remain available."
            elif lower in ("resume", "start", "continue"):
                self._running = True
                reply = "Session resumed."
            elif lower in ("learning on", "learn on", "enable learning", "learning off", "learn off", "disable learning"):
                self._learning = lower in ("learning on", "learn on", "enable learning")
                reply = f"Online supervised learning is {'on' if self._learning else 'off'}."
            elif lower in ("save", "save state", "save session"):
                reply, action = "Saving the complete session to the default checkpoint.", "save"
            elif lower in ("facts", "list facts", "stored facts"):
                reply = ("Remembered facts: " + ", ".join(f"{k} = {v}" for k, v in self.facts.items())
                         if self.facts else "No facts stored yet. Try: remember key = value.")
            elif match := re.fullmatch(r"forget\s+(.+)", lower):
                key = match[1].strip()
                reply = f"Forgot {key}." if self.facts.pop(key, None) is not None else f"No fact stored for {key}."
            elif match := re.fullmatch(r"step(?:\s+(\d+))?", lower):
                count = max(1, min(32, int(match[1] or 1)))
                self.tick(steps=count)
                reply = f"Advanced {count} step{'s' if count != 1 else ''} (now at step {self.steps}, {self.updates} updates)."
            elif match := re.fullmatch(r"(?:evolve|boost|train)(?:\s+(\d+))?", lower):
                count = max(1, min(64, int(match[1] or 16)))
                self.tick(steps=count)
                info = self.telemetry()
                loss_str = f"{self.last_loss:.4f}" if self.last_loss is not None else "n/a"
                reply = (f"Executed {count}-step evolution burst: step {self.steps}, {self.updates} updates, "
                         f"loss {loss_str}, ||Δθ||₂ = {info['weight_delta_norm']:.3f}, "
                         f"graft impact Δlogits = {info['graft_impact']:.3f}.")
            elif match := re.fullmatch(r"(?:world|resize|arena)\s+(\d+)(?:\s+(" + "|".join(map(re.escape, NavigationWorld.PRESETS)) + r"))?", lower):
                new_size = max(5, min(31, int(match[1])))
                preset = match[2] or getattr(self.world, "preset", "labyrinth")
                self.resize_world(size=new_size, preset=preset)
                env_info = self.world.environment_telemetry()
                reply = (f"Resized 3D arena to {new_size}×{new_size} ({preset} topology, "
                         f"{env_info['free_cells']} free cells, {env_info['obstacle_cells']} interior obstacles, "
                         f"{env_info['corridor_cells']} corridors).")
            elif match := re.fullmatch(r"(?:preset|topology)\s+(" + "|".join(map(re.escape, NavigationWorld.PRESETS)) + r")", lower):
                preset = match[1]
                self.resize_world(size=self.world.size, preset=preset)
                env_info = self.world.environment_telemetry()
                reply = (f"Switched {self.world.size}×{self.world.size} arena to {preset} topology "
                         f"(detour index {env_info['detour_index']:.2f}×, {len(env_info['choke_points'])} path choke points).")
            elif lower in ("neurons", "top neurons", "what neurons are firing", "firing"):
                info = self.telemetry()
                parts = []
                for src, items in info["top_nodes"].items():
                    top_str = ", ".join(f"{item['id']} ({item['activity']:.3f})" for item in items[:3])
                    parts.append(f"{src}: {top_str}")
                reply = "Top firing nodes by source — " + " | ".join(parts) + "."
            elif lower in ("lesion", "ablation", "silencing", "virtual lesion"):
                info = self.telemetry()
                abl = info["source_ablation"]
                parts = ", ".join(f"{k}: Δlogits={v:.3f}" for k, v in abl.items())
                dominant = max(abl, key=abl.get)
                reply = f"Virtual connectome lesion attribution — {parts} (dominant driver: {dominant})."
            elif lower in ("neuromodulators", "modulators", "liquid", "tau"):
                info = self.telemetry()
                nm = info["neuromodulators"]
                taus = ", ".join(f"{s}={t:.2f}s" for s, t in zip(("worm", "fly_f", "fly_m", "human"), info["liquid_tau"]))
                reply = (f"Endogenous neuromodulation [{nm['regime']}]: Dopamine={nm['dopamine']:.2f}, "
                         f"Serotonin={nm['serotonin']:.2f}, Octopamine={nm['octopamine']:.2f}, "
                         f"plasticity gain={nm['plasticity_gain']:.2f}× | Liquid τ [{taus}].")
            elif lower in ("motifs", "hubs", "rich club"):
                info = self.telemetry()
                worm_m = info["connectome_motifs"]["worm"]
                hubs = ", ".join(f"{h['id']} (deg={h['degree']}, act={h['activity']:.2f})" for h in worm_m["rich_club_hubs"][:4])
                reply = (f"Worm connectome motifs: {worm_m['reciprocal_dyads']} reciprocal chemical dyads, "
                         f"{worm_m['electrical_edges']} gap junctions, Rich-Club hubs [{hubs}].")
            elif lower in ("compass", "pva", "central complex", "heading"):
                info = self.telemetry()
                cmp = info["compass"]
                status = "disoriented (optomotor reflex active)" if cmp["disoriented"] else "coherent bump"
                reply = (f"Central Complex (CX) Ring Attractor: PVA heading {cmp['pva_heading']:.2f} rad, "
                         f"coherence R = {cmp['pva_coherence']:.2f} ({status}, 16 wedges).")
            elif lower in ("active inference", "efe", "free energy", "curiosity"):
                info = self.telemetry()
                efe = info["active_inference"]
                names = ("north", "east", "south", "west")
                dist = ", ".join(f"{n}: {p:.0%}" for n, p in zip(names, efe["efe_probs"]))
                best = names[efe["best_action"]]
                reply = (f"Active Inference EFE Arbiter: policy [{dist}], best {best} "
                         f"(epistemic weight {efe['epistemic_weight']:.2f}, "
                         f"pragmatic {efe['pragmatic'][efe['best_action']]:.2f}, "
                         f"epistemic {efe['epistemic'][efe['best_action']]:.2f}, "
                         f"empowerment {efe['empowerment'][efe['best_action']]:.2f}).")
            elif lower in ("memory", "mnemosyne", "spatial map"):
                info = self.telemetry()
                mem = info["memory"]
                reply = (f"Tri-Tier Cognitive Memory [{mem['mode']}]: {mem['mapped_cells']} mapped cells, "
                         f"spatial familiarity {mem['spatial_familiarity']:.1%}, "
                         f"working context norm {math.sqrt(sum(v*v for v in mem['working_context'])):.3f}.")
            elif match := re.fullmatch(r"(?:lesion memory|memory mode)\s+(intact|erased|shifted|random_retrieval|memory_free)", lower):
                mode = match[1]
                self.memory.set_lesion_mode(mode)
                reply = f"Tri-tier memory lesion assay mode set to {mode}."
            elif lower in ("phase", "lyapunov", "stability", "separatrix"):
                info = self.telemetry()
                pp = info["phase_plane"]
                reply = (f"Lyapunov Separatrix Phase Plane [{pp['stability_regime']}]: energy L = {pp['lyapunov_energy']:.4f}, "
                         f"drift L_dot = {pp['lyapunov_drift']:.4f}, separatrix margin = {pp['separatrix_distance']:.2f} "
                         f"({'CRITICAL BASIN' if pp['in_critical_basin'] else 'safe orbit'}).")
            elif lower in ("maze", "environment", "lidar"):
                env_info = self.world.environment_telemetry()
                reply = (f"Arena {env_info['size']}×{env_info['size']} ({env_info['preset']}): "
                         f"geodesic={env_info['geodesic_distance']} steps, detour={env_info['detour_index']:.2f}×, "
                         f"choke points={len(env_info['choke_points'])}, plume={env_info['chemoattractant']:.2f}, "
                         f"junctions={env_info['branch_junctions']}, corridors={env_info['corridor_cells']}.")
            elif lower in ("routes", "bridges", "graft", "routing"):
                info = self.telemetry()
                gates = ", ".join(f"{s}={g:.2f}" for s, g in zip(("worm", "fly_f", "fly_m", "human"), info["graft_gates"]))
                reply = (f"Synthetic routing telemetry: route entropy {info['route_entropy']:.2f}, "
                         f"counterfactual graft impact Δlogits = {info['graft_impact']:.3f}, "
                         f"graft gate openness [{gates}], weight evolution ||Δθ||₂ = {info['weight_delta_norm']:.3f}.")
            elif lower in ("dale", "dales law", "polarity", "spectral radius"):
                info = self.telemetry()
                dl = info["dales_law"]
                parts = []
                for src in ("worm", "fly_f", "fly_m", "human"):
                    rep = dl[src]
                    status = "contractive" if rep["is_contractive"] else "expansive"
                    parts.append(f"{src}: comp={rep['compliance_ratio']:.0%}, ρ≤{rep['spectral_radius_bound']:.2f} ({status})")
                reply = "Dale's law & contractive spectral bounds — " + " | ".join(parts) + "."
            elif lower in ("consolidation", "clcs", "latent experts", "anchor"):
                info = self.telemetry()
                cons = info["consolidation"]
                probs_str = ", ".join(f"E{i}={p:.0%}" for i, p in enumerate(cons["route_probs"]))
                reply = (f"Connectome Latent Consolidation Substrate (CLCS): Top-2 SwiGLU routing [{probs_str}], "
                         f"entropy H = {cons['router_entropy']:.2f}, balance = {cons['router_balance']:.2f}, "
                         f"memory H = {cons['memory_entropy']:.2f}, anchor KL drift = {cons['anchor_kl_drift']:.4f}.")
            elif lower in ("anchor sync", "sync anchor"):
                with torch.no_grad():
                    self.anchor_model.load_state_dict(self.model.state_dict())
                self._anchor_kl_drift = 0.0
                reply = "Anchor distillation model synchronized with current policy weights."
            elif lower in ("stp", "synapses", "synaptic plasticity", "plasticity"):
                info = self.telemetry()
                s = info["stp"]
                reply = (f"Tsodyks-Markram Short-Term Plasticity: release prob u={s['mean_release_prob']:.2f}, "
                         f"vesicle pool R={s['mean_vesicle_pool']:.2f}, synaptic efficacy E={s['mean_synaptic_efficacy']:.2f}× "
                         f"(depression={s['depression_fraction']:.0%}, facilitation={s['facilitation_fraction']:.0%}).")
            elif lower in ("spectral", "spectral diffusion", "wavelets", "diffusion"):
                info = self.telemetry()
                sd = info["spectral_diffusion"]
                parts = []
                for src in ("worm", "fly_f", "fly_m", "human"):
                    if src in sd:
                        energies = ", ".join(f"{k.split('_')[1]}={v:.2f}" for k, v in sd[src].items())
                        parts.append(f"{src}: [{energies}]")
                reply = "Chebyshev Spectral Graph Diffusion multi-scale energy — " + " | ".join(parts) + "."
            elif lower in ("byte", "bytes", "byte bridge", "introspect", "semantic"):
                info = self.telemetry()
                bb = info["byte_bridge"]
                chars = ", ".join(f"'{c}'" for c in bb["top_byte_chars"])
                reply = (f"Connectome Byte-Bridge Attractor [{bb['semantic_intent']}]: confidence={bb['confidence']:.2f}, "
                         f"byte entropy={bb['byte_entropy']:.2f} bits, top byte tokens [{chars}], "
                         f"active vocab spread={bb['vocab_spread']:.1%}.")
            elif lower in ("plume", "odor", "chemoattractant"):
                env_info = self.world.environment_telemetry()
                grad = env_info["plume_gradient"]
                gate_str = "OPEN" if env_info["gate_open"] else "CLOSED"
                reply = (f"Advection-Diffusion Odor Plume: intensity={env_info['plume_intensity']:.3f}, "
                         f"width={env_info['plume_width']:.2f}, gradient ∇=[{grad[0]:.2f}, {grad[1]:.2f}], "
                         f"dynamic gate={gate_str}, topological entropy={env_info['topological_entropy']:.2f}.")
            elif lower in ("terrain", "substrate", "substrates", "agar", "zones"):
                env_info = self.world.environment_telemetry()
                sub = env_info["substrate"]
                counts = env_info["substrates"]["counts"]
                counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
                reply = (f"Biophysical Substrate [{sub['type'].upper()}]: drag={sub['drag_factor']:.2f}× (mobility {sub['mobility']:.2f}×), "
                         f"gap conductance={sub['conductance']:.2f}×, temp={sub['temperature_celsius']:.1f}°C, "
                         f"V_mem={sub['bioelectric_potential_mv']:.1f}mV | Grid zones: [{counts_str}].")
            elif lower in ("sensilla", "tropotaxis", "antennae", "antenna", "klinotaxis"):
                env_info = self.world.environment_telemetry()
                ant = env_info["antennal_tropotaxis"]
                plume = env_info["plume_dynamics"]
                w = plume["wind_vector"]
                reply = (f"Bilateral Antennal Sensilla: S_L={ant['left_sensilla']:.4f}, S_R={ant['right_sensilla']:.4f}, "
                         f"bilateral contrast ΔS={ant['bilateral_contrast']:+.4f} (steer={ant['tropotaxis_steer'].upper()}), "
                         f"klinotaxis dS/dt={ant['klinotaxis_derivative']:+.4f} | Advection wind=[{w[0]:.2f}, {w[1]:.2f}], Pe={plume['peclet_number']:.2f}.")
            elif lower in ("optic", "compound eye", "ommatidia", "looming", "ttc"):
                env_info = self.world.environment_telemetry()
                eye = env_info["compound_eye"]
                alert_str = "⚠️ CRITICAL LOOMING ALERT" if eye["looming_alert"] else "Clear trajectory"
                reply = (f"Drosophila 16-Ommatidia Optic Flow: time-to-collision τ_TTC={eye['looming_tau_ttc']:.2f} ticks, "
                         f"retinal expansion rate={eye['retinal_expansion_rate']:.3f}/s, min clearance={eye['min_clearance']:.2f} "
                         f"[{alert_str}] | Sectors: 16 radial ommatidia spanning 360°.")
            elif lower in ("gates", "osmotic", "choke", "portals"):
                env_info = self.world.environment_telemetry()
                dg = env_info["dynamic_gates"]
                parts = [f"({g['x']},{g['z']}): {g['permeability']:.2f} [{g['status']}]" for g in dg["choke_permeabilities"][:4]]
                choke_str = " | ".join(parts) if parts else "No interior choke bottlenecks"
                reply = (f"Dynamic Osmotic Choke Gates: {dg['count']} choke bottlenecks detected, primary portal="
                         f"{'OPEN' if dg['primary_gate_open'] else 'RESTRICTED'} | Permeabilities: {choke_str}.")
            elif lower in ("circadian", "lux", "diurnal", "day", "night", "clock"):
                env_info = self.world.environment_telemetry()
                c = env_info["circadian"]
                reply = (f"128-Step Circadian Oscillator: step {c['cycle_step']}/128 (phase φ={c['phase']:.3f}), "
                         f"diurnal state={c['diurnal_state'].upper()}, ambient lux={c['ambient_lux']:.2f}, "
                         f"melatonin={c['melatonin_level']:.2f}.")
            elif lower in ("bioelectric", "voltage", "v_mem", "membrane", "gap junctions"):
                info = self.telemetry()
                bio = info["bioelectric"]
                sp = bio.get("species_potentials", {})
                parts = []
                for src in ("worm", "fly_f", "fly_m", "human"):
                    if src in sp:
                        parts.append(f"{src}: {sp[src]['mean_v_mv']:.1f}mV")
                reply = (f"Bioelectric Morphogenetic Substrate (BMS): global V_mem={bio['global_mean_mv']:.1f}mV "
                         f"(resting={bio['resting_baseline_mv']:.1f}mV, excitability gain={bio['excitability_gain']:.2f}×, "
                         f"depolarized={bio['depolarized_ratio']:.0%}) | Potentials: [{' | '.join(parts)}].")
            elif lower in ("self model", "agency", "reality drift", "drift", "forward model"):
                info = self.telemetry()
                sm = info["self_model"]
                reply = (f"Connectome Forward Self-Model [{sm['agency_regime']}]: active agency={sm['agency_score']:+.3f}, "
                         f"prediction error={sm['prediction_error']:.4f}, null error={sm['counterfactual_null_error']:.4f}, "
                         f"reality drift EMA={sm['reality_drift']:.4f}, self-alignment={sm['self_alignment']:.1%}.")
            elif lower in ("archaeology", "resection", "mtpr", "resection assay", "engram"):
                info = self.telemetry()
                arch = info["archaeology"]
                reply = (f"Adaptive Memory Archaeology [{arch['regime']}]: target={arch['target_species']} ({arch['resection_mode']}), "
                         f"MTPR={arch['mtpr']:.3f}, intact entropy={arch['intact_entropy']:.2f}, "
                         f"lesioned fidelity={arch['lesioned_fidelity']:.1%}, rescued fidelity={arch['rescued_fidelity']:.1%} "
                         f"(bioelectric rescue Δ={arch['rescue_delta']:+.3f}).")
            elif lower in ('navigation', 'navigator'):
                effective_teacher = self.config.teacher_probability if self.navigation_mode == 'policy' else 0.0
                observed = self.navigator.snapshot()
                reply = (f"Navigation controller: {self.navigation_mode}; effective teacher actions {effective_teacher:.0%}. "
                         f"Observed map: {observed['known_free']} free cells, {observed['known_blocked']} blocked cells, "
                         f"{observed['visited']} visited; last decision reason: {observed['reason']}. "
                         "Observed-map routing and legal-action masking are engineered assistance; raw policy accuracy is separate.")
            elif lower in ("why", "explain", "decision", "probe"):
                info = self.telemetry()
                names = ("north", "east", "south", "west")
                dist = ", ".join(f"{n}: {p:.0%}" for n, p in zip(names, info["action_probs"]))
                teacher = names[info["expert_action"]]
                goal = self.world.snapshot()["goal"]
                reply = (f"Decision probe at ({self.world._agent[0]}, {self.world._agent[1]}) toward {goal['color']} "
                         f"({goal['x']}, {goal['z']}): policy [{dist}], shortest-path teacher recommends {teacher} "
                         f"(remaining path {max(0, len(info['expert_path']) - 1)} cells, graft Δlogits = {info['graft_impact']:.3f}).")
                effective_teacher = self.config.teacher_probability if self.navigation_mode == 'policy' else 0.0
                assistance = 'configured teacher mixing' if self.navigation_mode == 'policy' else 'teacher actions bypassed'
                reply += (f" Active controller: {self.navigation_mode}; effective teacher actions {effective_teacher:.0%} "
                          f"({assistance}); observed-map last decision reason: {self.navigator.snapshot()['reason']}.")
            elif match := re.fullmatch(r"move\s+(north|east|south|west)", lower):
                direction = {"north": 0, "east": 1, "south": 2, "west": 3}[match[1]]
                self.tick(steps=1, manual_action=direction)
                reply = f"Moved {match[1]} to ({self.world._agent[0]}, {self.world._agent[1]})."
            elif lower in ("reset", "reset world"):
                self.world.reset()
                self._reset_inference()
                reply = "Arena episode reset; circuit weights and learned memory retained."
            elif lower in ("status", "help", "commands"):
                acc = sum(self.policy_matches) / len(self.policy_matches) if self.policy_matches else 0.0
                info = self.telemetry()
                reply = (f"Local command/memory backend. Step {self.steps}; {self.updates} supervised updates; "
                         f"{self.world.goals_reached} goals reached; recent policy match {acc:.0%}; "
                         f"||Δθ||₂ = {info['weight_delta_norm']:.3f}; {'running' if self._running else 'paused'}. "
                         "Commands: pause, resume, learn on/off, evolve [1-64], world [5-31] [preset], preset [labyrinth|chambers|corridors|pillars|concentric|hierarchical], "
                         "neurons, lesion, neuromodulators, motifs, maze, routes, dale, consolidation, anchor sync, stp, spectral, byte, plume, bioelectric, agency, archaeology, why, navigation, "
                         "step [1-32], move north/east/south/west, status, save, go to blue/amber/rose or go to x,z, "
                         "remember key = value, recall key, forget key, facts, reset world.")
            elif match := re.fullmatch(r"remember\s+(.+?)\s*(?:=|\bis\b)\s*(.+)", message, re.IGNORECASE):
                key = _text(match[1].strip(), 80, "fact key").lower()
                value = _text(match[2].strip(), 1000, "fact value")
                if key not in self.facts and len(self.facts) >= 64:
                    reply = "Fact memory is full (64 keys). Replace an existing key."
                else:
                    self.facts[key] = value
                    reply = f"Remembered {key} = {value}."
            elif match := re.fullmatch(r"(?:recall|what is)\s+(.+?)[?]?", message, re.IGNORECASE):
                key = match[1].strip().lower().rstrip("?").strip()
                reply = f"{key} = {self.facts[key]}" if key in self.facts else f"No explicit fact stored for {key}."
            elif match := re.fullmatch(r"go to\s+(blue|amber|rose)", lower):
                self.set_goal(color=match[1])
                reply = f"Goal set to the {match[1]} beacon; recurrent navigation memory reset."
            elif match := re.fullmatch(r"go to\s+(-?\d+)\s*,\s*(-?\d+)", lower):
                self.set_goal(x=int(match[1]), z=int(match[2]))
                reply = f"Goal set to {match[1]}, {match[2]}; recurrent navigation memory reset."
            else:
                reply = ("I use a local command and explicit fact-memory backend, without a pretrained language model. "
                         "Try help, status, evolve 16, world 19 chambers, lesion, neuromodulators, motifs, neurons, routes, why, or go to blue.")
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": reply})
            return {"reply": reply, "backend": _BACKEND, **({"action": action} if action else {})}

    def _payload(self):
        return {"version": _CHECKPOINT_VERSION, "graph_hash": self.bundle.digest(), "topology_hash": self.bundle.topology_digest(),
                "config": asdict(self.config), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "world": self.world.to_dict(), "rng": self.rng.getstate(), "recurrent_state": list(self.recurrent_state),
                "replay": list(self.replay), "observation_history": list(self.observation_history),
                "steps": self.steps, "updates": self.updates, "running": self._running, "learning": self._learning,
                "last_loss": self.last_loss, "loss_history": list(self.loss_history),
                "reward_history": list(self.reward_history), "policy_matches": list(self.policy_matches),
                "last_action": self.last_action, "last_action_source": self.last_action_source,
                "last_logits": self.last_logits, "routes": self.routes, "facts": dict(self.facts),
                "chat": list(self.chat_history), "last_saved": self.last_saved,
                "auxiliary": live_state.capture(self),
                "navigation": {"mode": self.navigation_mode, "navigator": self.navigator.to_dict()}}

    def save(self, path):
        path = Path(path).resolve()
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = {"path": str(path), "time": datetime.now(timezone.utc).isoformat()}
            payload = self._payload()
            payload["last_saved"] = saved
            integrity = _checkpoint_digest(payload)
            descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    torch.save({"payload": payload, "sha256": integrity}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                size = os.path.getsize(temporary)
                if size > MAX_CHECKPOINT_BYTES:
                    raise ValueError("checkpoint exceeds size cap")
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.last_saved = saved
            return {**saved, "bytes": size, "sha256": integrity}

    @classmethod
    def load(cls, path, bundle):
        path = Path(path)
        try:
            with path.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                if not 1 <= size <= MAX_CHECKPOINT_BYTES:
                    raise ValueError("checkpoint size exceeds allowed bounds")
                # Reject compressed expansion and header/data corruption before the restricted torch reader.
                with zipfile.ZipFile(handle) as archive:
                    entries = archive.infolist()
                    if len(entries) > 4096 or sum(item.file_size for item in entries) > MAX_CHECKPOINT_BYTES:
                        raise ValueError("checkpoint expanded size exceeds allowed bounds")
                    if archive.testzip() is not None:
                        raise ValueError("corrupted checkpoint archive")
                    for item in entries:
                        handle.seek(item.header_offset)
                        header = handle.read(30)
                        name_bytes = item.filename.encode("utf-8")
                        if len(header) != 30 or header[:4] != b"PK\x03\x04":
                            raise ValueError("corrupted checkpoint local header")
                        version, flags, compress, mod_time, mod_date, crc, comp_size, file_size, name_len, extra_len = struct.unpack("<5H3I2H", header[4:30])
                        dt = item.date_time
                        expected_time = (dt[3] << 11) | (dt[4] << 5) | (dt[5] // 2)
                        expected_date = ((dt[0] - 1980) << 9) | (dt[1] << 5) | dt[2]
                        expected_sizes = (0, 0, 0) if (flags & 0x08) else (item.CRC, item.compress_size, item.file_size)
                        if ((version, flags, compress, mod_time, mod_date) != (item.extract_version, item.flag_bits, item.compress_type, expected_time, expected_date)
                                or (crc, comp_size, file_size) != expected_sizes or name_len != len(name_bytes)):
                            raise ValueError("corrupted checkpoint local header fields")
                        if handle.read(name_len) != name_bytes:
                            raise ValueError("corrupted checkpoint entry name")
                        extra = handle.read(extra_len)
                        if extra and (len(extra) < 4 or extra != b"FB" + struct.pack("<H", len(extra) - 4) + b"Z" * (len(extra) - 4)):
                            raise ValueError("corrupted checkpoint alignment padding")
                handle.seek(0)
                checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or set(checkpoint) != {"payload", "sha256"}:
                raise ValueError("invalid checkpoint envelope")
            payload = checkpoint["payload"]
            if _checkpoint_digest(payload) != checkpoint["sha256"]:
                raise ValueError("checkpoint integrity failure")
            if not isinstance(payload, dict) or payload.get("graph_hash") != bundle.digest():
                raise ValueError("checkpoint graph does not match supplied source bundle")
            if payload.get("topology_hash") != bundle.topology_digest():
                raise ValueError("checkpoint topology does not match source graph")
            config = LiveConfig(**payload["config"])
            candidate = cls(bundle, config)
            candidate._restore_validated(payload)
            return candidate
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"unable to read checkpoint: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"invalid checkpoint: {exc}") from exc

    def _restore_validated(self, payload):
        """Validate the entire candidate before it can replace a live session."""
        with self.lock:
            version = payload.get("version")
            expected_fields = set(self._payload())
            if version == 1:
                expected_fields.remove("auxiliary")
            if version in (1, 2):
                expected_fields.remove("navigation")
            if set(payload) != expected_fields or type(version) is not int or version not in (1, 2, _CHECKPOINT_VERSION):
                raise ValueError("unsupported checkpoint fields or version")
            _integer(payload["steps"], 0, 2**53 - 1, "steps")
            _integer(payload["updates"], 0, payload["steps"] // self.config.train_every, "updates")
            if version >= 2:
                live_state.validate(self, payload["auxiliary"], payload["steps"])
            if type(payload["running"]) is not bool or type(payload["learning"]) is not bool:
                raise ValueError("checkpoint control flags must be booleans")
            expected = self.model.state_dict()
            model = payload["model"]
            if not isinstance(model, dict) or model.keys() != expected.keys():
                raise ValueError("checkpoint model fields do not match")
            buffers = dict(self.model.named_buffers())
            for name, expected_tensor in expected.items():
                tensor = model[name]
                self._tensor(tensor, expected_tensor.shape, expected_tensor.dtype, name)
                if name in buffers and not torch.equal(tensor, expected_tensor):
                    raise ValueError(f"checkpoint topology buffer disagrees with source graph: {name}")
            world = NavigationWorld.from_dict(payload["world"])
            if world.steps != payload["steps"]:
                raise ValueError("world/session step counters disagree")
            navigation_mode = 'policy'
            navigator = ObservedMapNavigator(world.size)
            if version >= 3:
                navigation = payload['navigation']
                if (type(navigation) is not dict or set(navigation) != {'mode', 'navigator'}
                        or type(navigation['mode']) is not str or navigation['mode'] not in MODES):
                    raise ValueError('invalid checkpoint navigation fields or mode')
                navigation_mode = navigation['mode']
                navigator = ObservedMapNavigator.from_dict(navigation['navigator'])
                if navigator.size != world.size:
                    raise ValueError('checkpoint navigation map size disagrees with world')
                if sum(navigator.visits.values()) > payload['steps']:
                    raise ValueError('checkpoint navigation visits exceed session steps')
                if (any(not world._free(cell) for cell in navigator.known_free)
                        or any(world._free(cell) for cell in navigator.known_blocked)):
                    raise ValueError('checkpoint navigation observations disagree with world occupancy')
                if navigator.goal is not None and navigator.goal != world._goal:
                    raise ValueError('checkpoint navigation goal disagrees with world')
            rng = random.Random()
            try:
                rng.setstate(payload["rng"])
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("invalid checkpoint private RNG") from exc
            initial = self.model.initial_state(1)
            state = payload["recurrent_state"]
            if not isinstance(state, list) or len(state) != 4:
                raise ValueError("invalid recurrent state")
            for index, reference in enumerate(initial):
                self._tensor(state[index], reference.shape, reference.dtype, "recurrent_state")
            self._tensor(payload["last_logits"], (1, 4), torch.float32, "last_logits")
            self._tensor(payload["routes"], (4, 5), torch.float32, "routes")
            routes = payload["routes"]
            if ((routes < 0).any() or (routes > 1).any() or not torch.allclose(routes.sum(-1), torch.ones(4), atol=1e-6)
                    or torch.count_nonzero(routes[range(4), range(4)])):
                raise ValueError("invalid synthetic route probabilities")
            self._validate_optimizer(payload["optimizer"], payload["updates"])
            replay = payload["replay"]
            if not isinstance(replay, list) or len(replay) > self.config.replay_capacity:
                raise ValueError("invalid replay size")
            for sample in replay:
                if not isinstance(sample, dict) or set(sample) != {"observations", "action"}:
                    raise ValueError("invalid replay sample")
                self._observations(sample["observations"], self.config.unroll)
                _integer(sample["action"], 0, 3, "expert action")
            history = payload["observation_history"]
            if not isinstance(history, list) or len(history) > self.config.unroll:
                raise ValueError("invalid observation history")
            self._observations(history, len(history))
            for name in ("loss_history", "reward_history", "policy_matches"):
                values = payload[name]
                if not isinstance(values, list) or len(values) > (128 if name == "policy_matches" else 64):
                    raise ValueError(f"invalid {name}")
                for value in values:
                    if name == "policy_matches":
                        if type(value) is not bool:
                            raise ValueError("policy matches must be booleans")
                    else:
                        _number(value, 0 if name == "loss_history" else -1e9, 1e9, name)
            if payload["last_loss"] is not None:
                _number(payload["last_loss"], 0, 1e9, "last_loss")
            if payload["last_action"] is not None:
                _integer(payload["last_action"], 0, 3, "last_action")
            if navigator.current is not None:
                # A controller records the sensory pose before its last step.
                # The saved world records the pose after the executed action,
                # which may differ from the controller's proposal after a
                # manual override, or stay put after a wall collision.
                if payload['last_action'] is None:
                    raise ValueError('checkpoint navigation transition lacks an executed action')
                dx, dz = world.ACTIONS[payload['last_action']]
                expected_position = (navigator.current[0] + dx, navigator.current[1] + dz)
                if not world._free(expected_position):
                    expected_position = navigator.current
                if world._agent not in navigator.known_free or world._agent != expected_position:
                    raise ValueError('checkpoint navigation position disagrees with the executed transition')
            action_sources = (None, "manual", "policy", "teacher")
            if version >= 3:
                action_sources += ('masked_policy', 'observed_map')
            if payload["last_action_source"] not in action_sources:
                raise ValueError("invalid action source")
            facts = payload["facts"]
            if not isinstance(facts, dict) or len(facts) > 64:
                raise ValueError("invalid fact memory")
            for key, value in facts.items():
                _text(key, 80, "fact key")
                _text(value, 1000, "fact value")
            chat = payload["chat"]
            if not isinstance(chat, list) or len(chat) > 64:
                raise ValueError("invalid chat memory")
            for message in chat:
                if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] not in ("user", "assistant"):
                    raise ValueError("invalid chat message")
                _text(message["content"], 2400, "chat content")
            saved = payload["last_saved"]
            if saved is not None:
                if not isinstance(saved, dict) or set(saved) != {"path", "time"}:
                    raise ValueError("invalid save receipt")
                _text(saved["path"], 4096, "save path")
                _text(saved["time"], 80, "save time")
                datetime.fromisoformat(saved["time"])
            # All validation precedes installation even inside this new candidate.
            self.model.load_state_dict(model, strict=True)
            self._anchor_kl_drift = 0.0
            self.optimizer.load_state_dict(payload["optimizer"])
            self.world, self.rng = world, rng
            self.navigation_mode, self.navigator = navigation_mode, navigator
            self.recurrent_state = tuple(tensor.detach().clone() for tensor in state)
            self.last_logits, self.routes = payload["last_logits"].clone(), routes.clone()
            self.replay.extend(replay)
            self.observation_history.extend(history)
            for name in ("steps", "updates", "last_loss", "last_action", "last_action_source", "last_saved"):
                setattr(self, name, payload[name])
            self._running, self._learning = payload["running"], payload["learning"]
            self.loss_history.extend(payload["loss_history"])
            self.reward_history.extend(payload["reward_history"])
            self.policy_matches.extend(payload["policy_matches"])
            self.facts = dict(facts)
            self.chat_history.extend(chat)
            self._model_hash = None
            if version >= 2:
                live_state.install(self, payload["auxiliary"])
                self._checkpoint_compatibility = {'loaded_format': version, 'auxiliary_state_restored': True}
            else:
                self._checkpoint_compatibility = {"loaded_format": 1, "auxiliary_state_restored": False}
                warnings.warn("Legacy live checkpoint lacks auxiliary state; anchor and observers were reinitialized. Exact full-session continuation is unavailable.", RuntimeWarning, stacklevel=2)

    @staticmethod
    def _tensor(tensor, shape, dtype, name):
        if (not isinstance(tensor, Tensor) or tensor.layout != torch.strided or tuple(tensor.shape) != tuple(shape)
                or tensor.dtype != dtype or tensor.device.type != "cpu" or tensor.requires_grad
                or not torch.isfinite(tensor).all() or (tensor.dtype != torch.bool and (tensor.abs() > 1e9).any())):
            raise ValueError(f"invalid checkpoint tensor: {name}")

    @staticmethod
    def _observations(observations, length):
        if not isinstance(observations, list) or len(observations) != length:
            raise ValueError("invalid observation window")
        for observation in observations:
            if not isinstance(observation, list) or len(observation) != 4:
                raise ValueError("invalid observation rows")
            for row in observation:
                if not isinstance(row, list) or len(row) != 4:
                    raise ValueError("invalid observation features")
                for value in row:
                    _number(value, -10, 10, "observation")

    def _validate_optimizer(self, state, updates):
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise ValueError("invalid optimizer state")
        expected = self.optimizer.state_dict()
        if state["param_groups"] != expected["param_groups"]:
            raise ValueError("optimizer parameter groups or hyperparameters disagree")
        values = state["state"]
        parameters = list(self.model.parameters())
        if not isinstance(values, dict) or len(values) > len(parameters):
            raise ValueError("invalid optimizer parameter state")
        for key, item in values.items():
            _integer(key, 0, len(parameters) - 1, "optimizer parameter id")
            if not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam moment fields")
            self._tensor(item["step"], (), torch.float32, "Adam step")
            if float(item["step"]) != updates:
                raise ValueError("Adam step disagrees with update count")
            self._tensor(item["exp_avg"], parameters[key].shape, parameters[key].dtype, "Adam moment")
            self._tensor(item["exp_avg_sq"], parameters[key].shape, parameters[key].dtype, "Adam second moment")
            if (item["exp_avg_sq"] < 0).any():
                raise ValueError("Adam second moment must be nonnegative")
        if updates and not values:
            raise ValueError("optimizer history is missing")
