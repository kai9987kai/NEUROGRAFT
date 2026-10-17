"""Central Complex (CX) Ring Attractor Compass & Optomotor Stabilization.

Grounded in Drosophila neuropil models (FlyWire Whole-Brain Connectome,
Ellipsoid Body E-PG ring attractor, and Protocerebral Bridge P-EN steering).
Maintains an online neural ring attractor representing heading theta in [0, 2pi).
Computes Population Vector Average (PVA) and coherence metric R in [0, 1].
"""
from __future__ import annotations

import math
from typing import Any, Sequence
import torch


class CXRingCompass:
    """16-wedge recurrent neural ring attractor compass."""

    NUM_WEDGES: int = 16
    R_STABILIZATION_THRESHOLD: float = 0.35

    def __init__(
        self,
        num_wedges: int = 16,
        initial_heading: float = 0.0,
    ) -> None:
        if num_wedges < 4 or num_wedges > 64:
            raise ValueError(f"num_wedges must be between 4 and 64, got {num_wedges}")
        self.num_wedges = int(num_wedges)
        self.angles = [2.0 * math.pi * i / self.num_wedges for i in range(self.num_wedges)]
        
        # Precompute recurrent cosine connectivity: local excitation, global inhibition
        self.w_rec: list[list[float]] = []
        for i in range(self.num_wedges):
            row = []
            for j in range(self.num_wedges):
                diff = self.angles[i] - self.angles[j]
                row.append(-0.5 + 1.5 * math.cos(diff))
            self.w_rec.append(row)

        self.state: list[float] = [0.0] * self.num_wedges
        self.reset(initial_heading)

    def reset(self, heading: float = 0.0) -> None:
        """Initialize the ring with a Gaussian activity bump centered at heading."""
        wrapped = float(heading) % (2.0 * math.pi)
        sigma = 0.55
        self.state = []
        for angle in self.angles:
            # Shortest circular distance
            diff = math.atan2(math.sin(angle - wrapped), math.cos(angle - wrapped))
            bump = math.exp(-0.5 * (diff / sigma) ** 2)
            self.state.append(bump)
        self._normalize()

    def _normalize(self) -> None:
        """Enforce non-negative firing rates and unit sum normalization."""
        clipped = [max(0.0, v) for v in self.state]
        total = sum(clipped)
        if total > 1e-8:
            self.state = [v / total for v in clipped]
        else:
            self.state = [1.0 / self.num_wedges] * self.num_wedges

    def step(
        self,
        angular_velocity: float = 0.0,
        sensory_input: Sequence[float] | None = None,
        dt: float = 0.20,
    ) -> tuple[float, float]:
        """Update ring attractor with angular velocity integration and sensory drive.

        Returns:
            (pva_heading, pva_coherence)
        """
        # 1. Recurrent relaxation: W_rec * state
        recurrent_drive = [0.0] * self.num_wedges
        for i in range(self.num_wedges):
            s = 0.0
            row = self.w_rec[i]
            for j in range(self.num_wedges):
                s += row[j] * self.state[j]
            recurrent_drive[i] = s

        # 2. Asymmetric P-EN velocity shift: moves bump along angular velocity
        shift_amount = float(angular_velocity) * float(dt)
        shift_drive = [0.0] * self.num_wedges
        for i in range(self.num_wedges):
            s = 0.0
            for j in range(self.num_wedges):
                diff = math.atan2(
                    math.sin(self.angles[i] - self.angles[j] - shift_amount),
                    math.cos(self.angles[i] - self.angles[j] - shift_amount),
                )
                w_shift = math.exp(-0.5 * (diff / 0.6) ** 2)
                s += w_shift * self.state[j]
            shift_drive[i] = s

        # 3. Combine drives
        new_state = [0.0] * self.num_wedges
        for i in range(self.num_wedges):
            val = 0.40 * self.state[i] + 0.30 * recurrent_drive[i] + 0.30 * shift_drive[i]
            if sensory_input is not None and len(sensory_input) > 0:
                s_val = sensory_input[i % len(sensory_input)]
                val += 0.20 * float(s_val)
            new_state[i] = max(0.0, val)

        self.state = new_state
        self._normalize()
        return self.readout()

    def readout(self) -> tuple[float, float]:
        """Compute Population Vector Average (PVA) and coherence metric R."""
        vx = 0.0
        vy = 0.0
        total_activity = sum(self.state)
        for i in range(self.num_wedges):
            a = self.state[i]
            vx += a * math.cos(self.angles[i])
            vy += a * math.sin(self.angles[i])

        magnitude = math.hypot(vx, vy)
        pva_heading = math.atan2(vy, vx) % (2.0 * math.pi)
        coherence = magnitude / (total_activity + 1e-8)
        coherence = min(1.0, max(0.0, coherence))
        return pva_heading, coherence

    def is_disoriented(self) -> bool:
        """True if PVA coherence drops below the stabilization threshold."""
        _, coherence = self.readout()
        return coherence < self.R_STABILIZATION_THRESHOLD

    def optomotor_stabilization_action(
        self,
        current_action: int,
        target_heading: float,
    ) -> int:
        """If disoriented, provide stabilizing orientation bias towards target."""
        pva_heading, coherence = self.readout()
        if coherence >= self.R_STABILIZATION_THRESHOLD:
            return current_action

        # Orientation error between PVA heading and target
        err = math.atan2(
            math.sin(target_heading - pva_heading),
            math.cos(target_heading - pva_heading),
        )
        # In grid world: 0=North (+Z down or -Z), 1=East (+X), 2=South (+Z), 3=West (-X)
        # Map heading angle into discrete direction:
        target_dir = int(round((target_heading % (2 * math.pi)) / (math.pi / 2.0))) % 4
        return target_dir

    def to_dict(self) -> dict[str, Any]:
        pva, coh = self.readout()
        return {
            "num_wedges": self.num_wedges,
            "wedges": [round(float(v), 5) for v in self.state],
            "pva_heading": round(float(pva), 4),
            "pva_coherence": round(float(coh), 4),
            "disoriented": bool(self.is_disoriented()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CXRingCompass:
        if not isinstance(data, dict):
            raise ValueError("CXRingCompass data must be a dictionary")
        num_wedges = int(data.get("num_wedges", cls.NUM_WEDGES))
        compass = cls(num_wedges=num_wedges)
        wedges = data.get("wedges")
        if isinstance(wedges, list) and len(wedges) == num_wedges:
            compass.state = [float(v) for v in wedges]
            compass._normalize()
        return compass
