"""Active Inference & Expected Free Energy (EFE) Decision Arbiter.

Formulates decision-making under the Free Energy Principle (FEP / Friston, Parr, Pezzulo):
  G(a) = Pragmatic_Risk(a) - Epistemic_Curiosity(a) - Empowerment(a)

Where:
- Pragmatic Value: Log-likelihood under homeostatic goal prior (minimizing geodesic /
  Euclidean distance to chemoattractant goal and avoiding collision hazards).
- Epistemic Value: Information gain / entropy of LiDAR clearance over unvisited cells.
- Empowerment: Number of unblocked navigable neighbor states (escape options).
"""
from __future__ import annotations

import math
from typing import Any, Sequence


class ActiveInferenceArbiter:
    """Computes Expected Free Energy across the 4 navigation actions."""

    DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # North, East, South, West

    def __init__(
        self,
        base_epistemic_weight: float = 0.35,
        empowerment_weight: float = 0.25,
        temperature: float = 1.0,
    ) -> None:
        self.base_epistemic_weight = float(base_epistemic_weight)
        self.empowerment_weight = float(empowerment_weight)
        self.temperature = max(0.01, float(temperature))

    def evaluate_actions(
        self,
        agent_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        obstacles: set[tuple[int, int]],
        world_size: int,
        lidar_8: Sequence[float],
        visit_counts: dict[tuple[int, int], int] | None = None,
        dopamine: float = 0.5,
        octopamine: float = 0.5,
    ) -> dict[str, Any]:
        """Evaluate G(a) for actions 0=North, 1=East, 2=South, 3=West.

        Returns:
            Dict containing efe_values, efe_probs, pragmatic_scores, epistemic_scores, empowerment_scores
        """
        ax, az = agent_pos
        gx, gz = goal_pos
        visits = visit_counts or {}

        # Modulate epistemic weight by octopamine (arousal/novelty drive) vs dopamine (exploitation)
        effective_epistemic = max(0.05, self.base_epistemic_weight * (0.5 + octopamine - 0.3 * dopamine))
        effective_empowerment = self.empowerment_weight

        pragmatic_scores: list[float] = []
        epistemic_scores: list[float] = []
        empowerment_scores: list[float] = []
        efe_values: list[float] = []

        cur_dist = math.hypot(gx - ax, gz - az)

        for a_idx, (dx, dz) in enumerate(self.DELTAS):
            nx, nz = ax + dx, az + dz
            # 1. Collision check
            is_blocked = (
                nx < 0 or nx >= world_size or nz < 0 or nz >= world_size or (nx, nz) in obstacles
            )
            if is_blocked:
                # Heavy pragmatic penalty for collision
                prag = -5.0
                epist = 0.0
                emp = 0.0
            else:
                next_dist = math.hypot(gx - nx, gz - nz)
                # Pragmatic progress towards goal
                prag = (cur_dist - next_dist) * 2.0
                if next_dist < 0.5:
                    prag += 5.0  # Immediate goal achievement bonus

                # Epistemic novelty: fewer visits = higher information gain
                cell_visits = visits.get((nx, nz), 0)
                epist = 1.0 / (1.0 + cell_visits * 0.5)
                # Incorporate LiDAR open vista
                ray_clearance = lidar_8[a_idx * 2] if len(lidar_8) >= 8 else 0.5
                epist += 0.5 * ray_clearance

                # Empowerment: count open exits from next cell
                open_exits = 0
                for edx, edz in self.DELTAS:
                    enx, enz = nx + edx, nz + edz
                    if (
                        0 <= enx < world_size
                        and 0 <= enz < world_size
                        and (enx, enz) not in obstacles
                    ):
                        open_exits += 1
                emp = open_exits / 4.0  # 0 to 1

            pragmatic_scores.append(round(float(prag), 4))
            epistemic_scores.append(round(float(epist), 4))
            empowerment_scores.append(round(float(emp), 4))

            # Expected Free Energy: lower is better under Friston convention,
            # or positive value where higher is preferred policy utility:
            # Policy Utility = Pragmatic + Epistemic + Empowerment
            val = prag + effective_epistemic * epist + effective_empowerment * emp
            efe_values.append(val)

        # Softmax over EFE utility values
        max_v = max(efe_values)
        exp_vals = [math.exp((v - max_v) / self.temperature) for v in efe_values]
        sum_exp = sum(exp_vals)
        efe_probs = [v / sum_exp for v in exp_vals]

        best_action = int(efe_values.index(max(efe_values)))
        return {
            "efe_values": [round(v, 4) for v in efe_values],
            "efe_probs": [round(p, 4) for p in efe_probs],
            "pragmatic": pragmatic_scores,
            "epistemic": epistemic_scores,
            "empowerment": empowerment_scores,
            "best_action": best_action,
            "epistemic_weight": round(effective_epistemic, 3),
        }
