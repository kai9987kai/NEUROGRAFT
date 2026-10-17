"""Persistent bounded online imitation in a synthetic navigation task.

The command/memory chat is deliberately local and is not a pretrained language
model. Source observations remain immutable; only circuit parameters adapt.
"""
from __future__ import annotations

from collections import deque
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

import torch
from torch import Tensor
import torch.nn.functional as F

from .dynamics import DynamicsConfig, GraphCircuit
from .environment import NavigationWorld
from .graphs import GraphBundle


MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_VERSION = 1
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

    def __post_init__(self):
        for name, low, high in (("seed", 0, 2**63 - 1), ("hidden_dim", 2, 32), ("batch_size", 1, 16),
                                ("train_every", 1, 128), ("unroll", 1, 16),
                                ("replay_capacity", 2, 512), ("threads", 1, 8)):
            _integer(getattr(self, name), low, high, name)
        _number(self.learning_rate, 1e-6, 0.1, "learning_rate")
        _number(self.teacher_probability, 0, 1, "teacher_probability")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must cover batch_size")


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
                self.model = GraphCircuit(bundle, DynamicsConfig(hidden_dim=self.config.hidden_dim, output_dim=4))
        self.model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.world = NavigationWorld(seed=self.config.seed)
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
        self.observation_history.clear()

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
        loss = F.cross_entropy(self.model(sequence).logits, labels)
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
                self.last_logits = output.logits.detach().clone()
                self.routes = output.routes[0].detach().clone()
                self.replay.append({"observations": padded, "action": expert})
                policy_action = int(output.logits.argmax(-1))
                if manual_action is not None:
                    action, source = manual_action, "manual"
                else:
                    self.policy_matches.append(policy_action == expert)
                    assisted = self.rng.random() < self.config.teacher_probability
                    action, source = (expert, "teacher") if assisted else (policy_action, "policy")
                result = self.world.step(action)
                self.steps += 1
                self.last_action, self.last_action_source = action, source
                self.reward_history.append(float(result["reward"]))
                if result["episode_done"]:
                    self._reset_inference()
                if (self._learning and self.steps % self.config.train_every == 0
                        and len(self.replay) >= self.config.batch_size):
                    self._train()
            return self.snapshot()

    def telemetry(self) -> dict[str, Any]:
        """Compute live circuit introspection, virtual lesions, LTC tau, neuromodulators, and environment telemetry."""
        with self.lock:
            probs = torch.softmax(self.last_logits[0].detach(), dim=-1).tolist()
            expert = self.world.expert_action()
            path = self.world.shortest_path()
            env_info = self.world.environment_telemetry()
            gates = [float(g.detach().sigmoid().mean()) for g in self.model.graft_gates]
            routes = self.routes.detach().clamp(min=1e-12)
            entropy = max(0.0, min(1.0, float(-(routes * routes.log()).sum(dim=-1).mean() / math.log(5.0))))
            delta_sq = sum(
                float((param.detach() - self._initial_weights[name]).pow(2).sum())
                for name, param in self.model.named_parameters()
            )
            weight_delta = math.sqrt(max(0.0, delta_sq))
            graft_impact = 0.0
            source_keys = ("worm", "fly_f", "fly_m", "human")
            source_ablation = {k: 0.0 for k in source_keys}
            if self.observation_history:
                window = list(self.observation_history)
                padded = [[[0.0] * 4 for _ in range(4)] for _ in range(self.config.unroll - len(window))] + window
                seq = torch.tensor([padded], dtype=torch.float32)
                with torch.no_grad():
                    ungrafted = self.model(seq, graft_enabled=False).logits
                    graft_impact = float((self.last_logits.detach() - ungrafted.detach()).norm())
                    for idx, skey in enumerate(source_keys):
                        ablated = self.model(seq, ablate_source=idx).logits
                        source_ablation[skey] = float((self.last_logits.detach() - ablated.detach()).norm())
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
            elif lower in ("facts", "list facts", "memory"):
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
            elif match := re.fullmatch(r"(?:world|resize|arena)\s+(\d+)(?:\s+(labyrinth|chambers|corridors|pillars))?", lower):
                new_size = max(5, min(31, int(match[1])))
                preset = match[2] or getattr(self.world, "preset", "labyrinth")
                self.resize_world(size=new_size, preset=preset)
                env_info = self.world.environment_telemetry()
                reply = (f"Resized 3D arena to {new_size}×{new_size} ({preset} topology, "
                         f"{env_info['free_cells']} free cells, {env_info['obstacle_cells']} interior obstacles, "
                         f"{env_info['corridor_cells']} corridors).")
            elif match := re.fullmatch(r"(?:preset|topology)\s+(labyrinth|chambers|corridors|pillars)", lower):
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
            elif lower in ("why", "explain", "decision", "probe"):
                info = self.telemetry()
                names = ("north", "east", "south", "west")
                dist = ", ".join(f"{n}: {p:.0%}" for n, p in zip(names, info["action_probs"]))
                teacher = names[info["expert_action"]]
                goal = self.world.snapshot()["goal"]
                reply = (f"Decision probe at ({self.world._agent[0]}, {self.world._agent[1]}) toward {goal['color']} "
                         f"({goal['x']}, {goal['z']}): policy [{dist}], shortest-path teacher recommends {teacher} "
                         f"(remaining path {max(0, len(info['expert_path']) - 1)} cells, graft Δlogits = {info['graft_impact']:.3f}).")
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
                         "Commands: pause, resume, learn on/off, evolve [1-64], world [5-31] [preset], preset [labyrinth|chambers|corridors|pillars], "
                         "neurons, lesion, neuromodulators, motifs, maze, routes, why, "
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
        return {"version": _VERSION, "graph_hash": self.bundle.digest(), "topology_hash": self.bundle.topology_digest(),
                "config": asdict(self.config), "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "world": self.world.to_dict(), "rng": self.rng.getstate(), "recurrent_state": list(self.recurrent_state),
                "replay": list(self.replay), "observation_history": list(self.observation_history),
                "steps": self.steps, "updates": self.updates, "running": self._running, "learning": self._learning,
                "last_loss": self.last_loss, "loss_history": list(self.loss_history),
                "reward_history": list(self.reward_history), "policy_matches": list(self.policy_matches),
                "last_action": self.last_action, "last_action_source": self.last_action_source,
                "last_logits": self.last_logits, "routes": self.routes, "facts": dict(self.facts),
                "chat": list(self.chat_history), "last_saved": self.last_saved}

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
            if set(payload) != set(self._payload()) or type(payload["version"]) is not int or payload["version"] != _VERSION:
                raise ValueError("unsupported checkpoint fields or version")
            _integer(payload["steps"], 0, 2**53 - 1, "steps")
            _integer(payload["updates"], 0, payload["steps"] // self.config.train_every, "updates")
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
            if payload["last_action_source"] not in (None, "manual", "policy", "teacher"):
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
            self.optimizer.load_state_dict(payload["optimizer"])
            self.world, self.rng = world, rng
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
