"""Pure inspection diagnostics for live navigation replay and routing states."""
from __future__ import annotations

import copy
import math
from typing import Any

import torch

from .graphs import SOURCE_KEYS


def navigation_diagnostics(live) -> dict[str, Any]:
    """Capture replay and weights together, then probe outside the live lock.

    The diagnostic describes the captured instant even when online updates
    continue. Neither the live model's module modes nor nested replay values
    are changed or consulted after capture.
    """
    with live.lock:
        replay = copy.deepcopy(list(live.replay))
        world_size = live.world.size
        world_preset = getattr(live.world, "preset", "labyrinth")
        model = copy.deepcopy(live.model) if replay else None
        # Validate replay observations
        for sample in replay:
            obs = sample.get("observations")
            if not isinstance(obs, list):
                raise ValueError("invalid replay observation sequence")
            for step in obs:
                if not isinstance(step, list) or len(step) != 4:
                    raise ValueError("invalid replay observation step")
                for row in step:
                    if not isinstance(row, list) or len(row) != 4:
                        raise ValueError("invalid replay observation row")
                    for val in row:
                        if type(val) not in (int, float) or not math.isfinite(val):
                            raise ValueError("replay observation values must be finite")

    action_names = ("north", "east", "south", "west")
    if not replay:
        return {
            "replay": {
                "size": 0,
                "action_counts": {name: 0 for name in action_names},
                "goal_progress_feature_range": None,
            },
            "human_routing": None,
            "policy": None,
            "saved_world": {
                "size": world_size,
                "preset": world_preset,
            },
            "representative_windows": {
                "count": 0,
                "indices": [],
            },
        }

    action_counts = {
        name: sum(item.get("action") == index for item in replay)
        for index, name in enumerate(action_names)
    }
    progress = [item["observations"][-1][3][2] for item in replay]
    progress_range = {"min": min(progress), "max": max(progress)}

    n = len(replay)
    if n <= 32:
        rep_indices = list(range(n))
    else:
        rep_indices = [round(i * (n - 1) / 31) for i in range(32)]

    # Only the detached copy is switched to evaluation mode; probes leave
    # global RNGs and the original model's possibly mixed module modes alone.
    model.eval().requires_grad_(False)
    seq_tensor = torch.tensor([item["observations"] for item in replay], dtype=model.policy.weight.dtype)
    with torch.inference_mode():
        out = model(seq_tensor)

    # Routes received by human module: index 3
    # out.routes has shape [batch, 4, 5]
    human_routes = out.routes[:, 3, :]
    mean_mass_list = human_routes.mean(0).tolist()
    labels = (*SOURCE_KEYS, "null")
    mean_mass_dict = {label: float(val) for label, val in zip(labels, mean_mass_list)}
    goal_source_mean_mass = float(mean_mass_list[1])

    logits = out.logits
    policy_summary = {
        "logits_mean": [float(v) for v in logits.mean(0).tolist()],
        "action_distribution": [
            sum(int(a) == idx for a in logits.argmax(-1)) / max(1, n)
            for idx in range(4)
        ],
    }

    return {
        "replay": {
            "size": n,
            "action_counts": action_counts,
            "goal_progress_feature_range": progress_range,
        },
        "human_routing": {
            "mean_mass": mean_mass_dict,
            "goal_source_mean_mass": goal_source_mean_mass,
        },
        "policy": policy_summary,
        "saved_world": {
            "size": world_size,
            "preset": world_preset,
        },
        "representative_windows": {
            "count": len(rep_indices),
            "indices": rep_indices,
        },
    }
