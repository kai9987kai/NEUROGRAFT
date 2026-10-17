"""Nonlinear Dynamical Systems Phase-Plane & Lyapunov Separatrix Stability Engine.

Implements Lyapunov dynamical stability analysis (Poincaré-Bendixson & Lyapunov
attractor basins) for autonomous connectome navigation:
1. State Space:
     x = [E, S, D, V]^T
   where:
     E = Energy efficiency / reward rate
     S = Spatial goal proximity (1.0 - dist / max_dist)
     D = Barrier clearance margin (min LiDAR clearance)
     V = Velocity / policy alignment with shortest-path direction

2. Attractor Basin & Lyapunov Energy Function:
     x* = [0.85, 0.90, 0.80, 1.00]^T  (optimal homeostatic fixed point)
     L(x) = 0.5 * sum_i w_i * (x_i - x_i*)^2

3. Orbital Drift (Lie Derivative):
     L_dot = (L(t) - L(t-1)) / dt
     - L_dot < 0: Asymptotic convergence towards homeostatic attractor basin.
     - L_dot > 0: Destabilizing divergence towards critical collision separatrix.

4. Separatrix Margin:
     Distance to critical boundary where obstacle clearance D <= 0.05.
"""
from __future__ import annotations

import math
from typing import Any, Sequence


class PhasePlaneEngine:
    """Nonlinear dynamical systems phase plane analyzer and Lyapunov stabilizer."""

    # Target homeostatic fixed point attractor
    TARGET = (0.85, 0.90, 0.80, 1.00)  # E, S, D, V
    WEIGHTS = (2.0, 3.0, 4.0, 1.5)

    def __init__(self, dt: float = 0.20) -> None:
        self.dt = float(dt)
        self.last_lyapunov: float | None = None
        self.last_state: tuple[float, float, float, float] | None = None

    def reset(self) -> None:
        self.last_lyapunov = None
        self.last_state = None

    def audit(
        self,
        energy_rate: float,
        goal_proximity: float,
        clearance_margin: float,
        heading_alignment: float,
    ) -> dict[str, Any]:
        """Compute phase plane coordinates, Lyapunov energy, drift, and separatrix margin."""
        e = max(0.0, min(1.0, float(energy_rate)))
        s = max(0.0, min(1.0, float(goal_proximity)))
        d = max(0.0, min(1.0, float(clearance_margin)))
        v = max(0.0, min(1.0, float(heading_alignment)))
        state = (e, s, d, v)

        # 1. Lyapunov Energy L(x) = 0.5 * sum w_i * (x_i - x_i*)^2
        lyapunov = 0.5 * sum(
            w * ((val - tgt) ** 2)
            for w, val, tgt in zip(self.WEIGHTS, state, self.TARGET)
        )

        # 2. Orbital drift L_dot
        if self.last_lyapunov is not None:
            drift = (lyapunov - self.last_lyapunov) / max(0.01, self.dt)
        else:
            drift = 0.0

        # Phase velocity vector field f(x) = (state - last_state) / dt
        if self.last_state is not None:
            v_field = {
                "dE_dt": round((e - self.last_state[0]) / self.dt, 4),
                "dS_dt": round((s - self.last_state[1]) / self.dt, 4),
                "dD_dt": round((d - self.last_state[2]) / self.dt, 4),
                "dV_dt": round((v - self.last_state[3]) / self.dt, 4),
            }
        else:
            v_field = {"dE_dt": 0.0, "dS_dt": 0.0, "dD_dt": 0.0, "dV_dt": 0.0}

        self.last_lyapunov = lyapunov
        self.last_state = state

        # 3. Separatrix distance: critical boundary is collision (d <= 0.0)
        separatrix_dist = max(0.0, d)
        in_critical_basin = separatrix_dist < 0.15

        # 4. Stability regime classification
        if in_critical_basin:
            regime = "extinction_hazard"
        elif drift < -0.10:
            regime = "contracting_attractor"
        elif drift > 0.10:
            regime = "divergent_saddle"
        else:
            regime = "transient_orbit"

        return {
            "state": {"energy": round(e, 4), "proximity": round(s, 4), "clearance": round(d, 4), "alignment": round(v, 4)},
            "lyapunov_energy": round(float(lyapunov), 5),
            "lyapunov_drift": round(float(drift), 5),
            "separatrix_distance": round(float(separatrix_dist), 4),
            "in_critical_basin": bool(in_critical_basin),
            "stability_regime": regime,
            "phase_velocity": v_field,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "last_lyapunov": float(self.last_lyapunov) if self.last_lyapunov is not None else None,
            "last_state": [float(v) for v in self.last_state] if self.last_state is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhasePlaneEngine:
        if not isinstance(data, dict):
            raise ValueError("PhasePlaneEngine data must be a dictionary")
        engine = cls(dt=float(data.get("dt", 0.20)))
        lyap = data.get("last_lyapunov")
        if lyap is not None:
            engine.last_lyapunov = float(lyap)
        st = data.get("last_state")
        if isinstance(st, list) and len(st) == 4:
            engine.last_state = (float(st[0]), float(st[1]), float(st[2]), float(st[3]))
        return engine
