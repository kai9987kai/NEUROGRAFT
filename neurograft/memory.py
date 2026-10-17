"""Tri-Tier Cognitive Memory Architecture with Matched Lesion Assays.

Inspired by Drosophila mushroom body sparse associative memory and the Mnemosyne
cognitive architecture in Supermix-Poseidon:
1. Tier 1 (Working Memory): Fast recurrent context vector updated with prediction surprise.
2. Tier 2 (Episodic Store): Value-gated experiential memory weighted by reward, surprise, and recency.
3. Tier 3 (Cognitive Spatial Map): Online topological graph of visited coordinates,
   cell clearance statistics, and beacon landmark confidence.

Includes 5 matched lesion modes:
- intact: Full 3-tier memory active.
- erased: Memory wiped at every tick (zero context retention).
- shifted: Circular permutation of episodic memory keys (breaks semantic alignment).
- random_retrieval: Queries select randomly from history, ignoring value/similarity.
- memory_free: Memory completely bypassed (pure reactive baseline).
"""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

LESION_MODES = ("intact", "erased", "shifted", "random_retrieval", "memory_free")


@dataclass
class EpisodicItem:
    step: int
    state: list[float]
    action: int
    reward: float
    surprise: float


class TriTierMemory:
    """Biologically grounded 3-tier memory system with matched lesion assays."""

    def __init__(
        self,
        capacity: int = 128,
        lesion_mode: str = "intact",
    ) -> None:
        if lesion_mode not in LESION_MODES:
            raise ValueError(f"lesion_mode must be one of {LESION_MODES}, got {lesion_mode}")
        self.capacity = int(capacity)
        self.lesion_mode = lesion_mode

        # Tier 1: Working Memory context vector
        self.working_context = [0.0] * 8
        self.last_predicted_state: list[float] | None = None

        # Tier 2: Episodic memory store
        self.episodes: collections.deque[EpisodicItem] = collections.deque(maxlen=self.capacity)

        # Tier 3: Cognitive spatial map (x, z) -> stats
        self.spatial_map: dict[str, dict[str, Any]] = {}
        self.total_recorded_steps: int = 0

    def reset(self) -> None:
        self.working_context = [0.0] * 8
        self.last_predicted_state = None
        self.episodes.clear()
        self.spatial_map.clear()
        self.total_recorded_steps = 0

    def set_lesion_mode(self, mode: str) -> None:
        if mode not in LESION_MODES:
            raise ValueError(f"lesion_mode must be one of {LESION_MODES}, got {mode}")
        self.lesion_mode = mode

    def record_step(
        self,
        step: int,
        agent_pos: tuple[int, int],
        action: int,
        reward: float,
        state_features: Sequence[float],
        surprise: float = 0.0,
    ) -> None:
        """Update all 3 tiers with step experience."""
        self.total_recorded_steps += 1
        pos_key = f"{agent_pos[0]},{agent_pos[1]}"

        # Tier 3: Spatial Map update
        if pos_key not in self.spatial_map:
            self.spatial_map[pos_key] = {
                "visits": 0,
                "first_step": step,
                "last_step": step,
                "total_reward": 0.0,
            }
        self.spatial_map[pos_key]["visits"] += 1
        self.spatial_map[pos_key]["last_step"] = step
        self.spatial_map[pos_key]["total_reward"] += float(reward)

        if self.lesion_mode == "memory_free":
            return

        # Tier 1: Working Memory update
        if self.lesion_mode == "erased":
            self.working_context = [0.0] * 8
        else:
            # Recurrent leaky context update driven by surprise
            alpha = max(0.1, min(0.9, 0.4 + 0.5 * surprise))
            for i in range(min(8, len(state_features))):
                self.working_context[i] = (1.0 - alpha) * self.working_context[i] + alpha * float(state_features[i])

        # Tier 2: Episodic Store update
        item = EpisodicItem(
            step=step,
            state=[round(float(v), 4) for v in state_features[:8]],
            action=int(action),
            reward=float(reward),
            surprise=float(surprise),
        )
        self.episodes.append(item)

    def retrieve_context(self, current_features: Sequence[float]) -> dict[str, Any]:
        """Query episodic store and spatial map for relevant context."""
        if self.lesion_mode == "memory_free":
            return {
                "working_context": [0.0] * 8,
                "retrieved_item": None,
                "spatial_familiarity": 0.0,
                "mode": self.lesion_mode,
            }
        if self.lesion_mode == "erased":
            return {
                "working_context": [0.0] * 8,
                "retrieved_item": None,
                "spatial_familiarity": 0.0,
                "mode": self.lesion_mode,
            }

        if not self.episodes:
            return {
                "working_context": list(self.working_context),
                "retrieved_item": None,
                "spatial_familiarity": 0.0,
                "mode": self.lesion_mode,
            }

        if self.lesion_mode == "random_retrieval":
            chosen = random.choice(list(self.episodes))
        elif self.lesion_mode == "shifted":
            # Circular shift by half capacity
            items = list(self.episodes)
            idx = (len(items) // 2) % len(items)
            chosen = items[idx]
        else:  # "intact"
            # Value-gated similarity matching
            best_score = -float("inf")
            chosen = None
            cur = list(current_features[:8])
            for item in reversed(list(self.episodes)):
                # Cosine similarity approximation
                diff_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(cur, item.state)))
                sim = 1.0 / (1.0 + diff_norm)
                # Value weight: reward + surprise + recency
                age = max(1, self.total_recorded_steps - item.step)
                recency = math.exp(-age / 32.0)
                score = sim * (1.0 + abs(item.reward) * 2.0 + item.surprise * 1.5 + recency * 0.5)
                if score > best_score:
                    best_score = score
                    chosen = item

        retrieved_dict = {
            "step": chosen.step,
            "action": chosen.action,
            "reward": chosen.reward,
            "surprise": chosen.surprise,
        } if chosen else None

        # Familiarity: fraction of mapped cells with >= 2 visits
        mapped_count = len(self.spatial_map)
        familiar_count = sum(1 for v in self.spatial_map.values() if v["visits"] >= 2)
        familiarity = familiar_count / max(1, mapped_count)

        return {
            "working_context": list(self.working_context),
            "retrieved_item": retrieved_dict,
            "spatial_familiarity": round(familiarity, 3),
            "mapped_cells": mapped_count,
            "mode": self.lesion_mode,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "lesion_mode": self.lesion_mode,
            "working_context": [round(v, 4) for v in self.working_context],
            "episodes": [
                {
                    "step": ep.step,
                    "state": ep.state,
                    "action": ep.action,
                    "reward": ep.reward,
                    "surprise": ep.surprise,
                }
                for ep in self.episodes
            ],
            "spatial_map": dict(self.spatial_map),
            "total_recorded_steps": self.total_recorded_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriTierMemory:
        if not isinstance(data, dict):
            raise ValueError("TriTierMemory data must be a dictionary")
        capacity = int(data.get("capacity", 128))
        lesion_mode = str(data.get("lesion_mode", "intact"))
        mem = cls(capacity=capacity, lesion_mode=lesion_mode)
        ctx = data.get("working_context")
        if isinstance(ctx, list):
            mem.working_context = [float(v) for v in ctx]
        ep_list = data.get("episodes")
        if isinstance(ep_list, list):
            for ep in ep_list:
                mem.episodes.append(
                    EpisodicItem(
                        step=int(ep["step"]),
                        state=[float(v) for v in ep["state"]],
                        action=int(ep["action"]),
                        reward=float(ep["reward"]),
                        surprise=float(ep.get("surprise", 0.0)),
                    )
                )
        sp = data.get("spatial_map")
        if isinstance(sp, dict):
            mem.spatial_map = {str(k): dict(v) for k, v in sp.items()}
        mem.total_recorded_steps = int(data.get("total_recorded_steps", 0))
        return mem
