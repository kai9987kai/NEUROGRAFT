"""Connectome Forward Self-Model & Agency Stabilization Audit.

Inspired by GhostInTheMachine, QuantumBot & Causeway:
Equips NEUROGRAFT with an introspective forward self-model that forecasts the
connectome's next pooled latent state given current activity and motor output.
By contrasting self-predictions against counterfactual null actions, the agent
measures active causal agency, detects reality drift / hallucinations, and
stabilizes internal self-representations across continuous evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SelfModelConfig:
    """Configuration parameters for connectome forward self-modeling."""
    latent_dim: int = 32
    action_dim: int = 4
    hidden_dim: int = 64
    learning_rate: float = 0.005
    drift_ema_alpha: float = 0.92
    agency_temperature: float = 4.0


class ConnectomeForwardSelfModel(nn.Module):
    """Residual forward dynamics model of the multi-species connectome."""

    def __init__(self, config: SelfModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else SelfModelConfig()
        dim = self.config.latent_dim
        h_dim = self.config.hidden_dim
        a_dim = self.config.action_dim

        # Predicts residual state shift: Delta s = f(s_t, a_t)
        self.fc_state = nn.Linear(dim, h_dim)
        self.fc_action = nn.Linear(a_dim, h_dim)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

        # Telemetry state
        self.register_buffer("_last_pred_error", torch.tensor(0.0))
        self.register_buffer("_reality_drift_ema", torch.tensor(0.0))
        self.register_buffer("_last_agency_score", torch.tensor(0.0))
        self._history_errors: list[float] = []
        self._history_agency: list[float] = []
        self.total_forecasts: int = 0

    def _adapt_latent(self, latent: Tensor) -> Tensor:
        """Dynamically pad or slice input latent vector to self.config.latent_dim."""
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        cur_dim = latent.shape[-1]
        target_dim = self.config.latent_dim
        if cur_dim == target_dim:
            return latent
        if cur_dim < target_dim:
            pad = latent.new_zeros((latent.shape[0], target_dim - cur_dim))
            return torch.cat([latent, pad], dim=-1)
        return latent[:, :target_dim]

    def _action_one_hot(self, action: int, batch_size: int, device: torch.device) -> Tensor:
        a_dim = self.config.action_dim
        one_hot = torch.zeros((batch_size, a_dim), dtype=torch.float32, device=device)
        idx = max(0, min(a_dim - 1, int(action)))
        one_hot[:, idx] = 1.0
        return one_hot

    def forecast(self, latent: Tensor, action: int) -> Tensor:
        """Forecast the next latent connectome state given action: hat_s_{t+1}."""
        x = self._adapt_latent(latent)
        a = self._action_one_hot(action, x.shape[0], x.device)
        h = self.fc_state(x) + self.fc_action(a)
        delta = self.mlp(h)
        return x + delta

    def counterfactual_null(self, latent: Tensor) -> Tensor:
        """Forecast counterfactual state assuming zero motor engagement."""
        x = self._adapt_latent(latent)
        a_null = torch.zeros((x.shape[0], self.config.action_dim), dtype=torch.float32, device=x.device)
        h = self.fc_state(x) + self.fc_action(a_null)
        delta = self.mlp(h)
        return x + delta

    def step_audit(
        self,
        current_latent: Tensor,
        action: int,
        next_latent: Tensor,
        train: bool = True,
    ) -> dict[str, Any]:
        """Evaluate forward prediction against actual transition, updating agency audit."""
        s_t = self._adapt_latent(current_latent).detach()
        s_next = self._adapt_latent(next_latent).detach()

        # Predictions
        s_pred = self.forecast(s_t, action)
        s_null = self.counterfactual_null(s_t)

        pred_err = float((s_next - s_pred).pow(2).mean().sqrt().item())
        null_err = float((s_next - s_null).pow(2).mean().sqrt().item())

        # Active causal agency metric: (Null_err - Act_err) / (Null_err + eps)
        # Clamped into [-1.0, 1.0]
        denom = max(1e-6, null_err)
        agency_score = (null_err - pred_err) / denom
        agency_score = max(-1.0, min(1.0, agency_score))

        # Reality drift EMA
        alpha = self.config.drift_ema_alpha
        drift = float(self._reality_drift_ema.item())
        if self.total_forecasts == 0:
            drift = pred_err
        else:
            drift = alpha * drift + (1.0 - alpha) * pred_err
        self._reality_drift_ema.copy_(torch.tensor(drift))
        self._last_pred_error.copy_(torch.tensor(pred_err))
        self._last_agency_score.copy_(torch.tensor(agency_score))

        # Online calibration of forward self-model
        if train and self.training:
            self.optimizer.zero_grad()
            # Forward pass with autograd
            pred_train = self.forecast(s_t, action)
            loss = F.mse_loss(pred_train, s_next)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

        self.total_forecasts += 1
        self._history_errors.append(round(pred_err, 4))
        self._history_agency.append(round(agency_score, 4))
        if len(self._history_errors) > 64:
            self._history_errors.pop(0)
            self._history_agency.pop(0)

        # Qualitative agency classification
        if agency_score >= 0.20:
            regime = "autonomous-agency"
        elif agency_score >= 0.0:
            regime = "passive-concordance"
        else:
            regime = "counterfactual-perturbation"

        alignment = max(0.0, min(1.0, 1.0 / (1.0 + pred_err)))

        return {
            "prediction_error": round(pred_err, 4),
            "counterfactual_null_error": round(null_err, 4),
            "agency_score": round(agency_score, 4),
            "reality_drift": round(drift, 4),
            "agency_regime": regime,
            "self_alignment": round(alignment, 4),
            "total_audits": self.total_forecasts,
        }

    def explain(self) -> str:
        """Introspective summary of agency and reality drift."""
        agency = float(self._last_agency_score.item())
        drift = float(self._reality_drift_ema.item())
        err = float(self._last_pred_error.item())
        return (
            f"Connectome Forward Self-Model [Audits: {self.total_forecasts}] | "
            f"Agency Gain: {agency:+.3f} | Pred RMS: {err:.4f} | Reality Drift EMA: {drift:.4f}"
        )
