"""Versioned, validated auxiliary state for bounded live-session continuation.

This module contains serialization only. Observer modules do not become policy
inputs merely because their state is saved alongside the trainable circuit.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import math
import re
from typing import Any

import torch

from .bioelectric import BioelectricConfig, BioelectricMorphogeneticSubstrate
from .archaeology import ArchaeologyReport
from .compass import CXRingCompass
from .memory import LESION_MODES, TriTierMemory
from .phase_plane import PhasePlaneEngine
from .stp import STPConfig, STPState, TsodyksMarkramSubstrate


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid auxiliary {name}")


def _number(value, low=-1e12, high=1e12):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError("invalid auxiliary numeric value")


def _tree(value, depth=0, budget=None):
    """Bound unknown primitive diagnostic trees before copying/installing them."""
    if budget is None:
        budget = [100_000]
    budget[0] -= 1
    if depth > 16 or budget[0] < 0:
        raise ValueError("auxiliary state exceeds structural limits")
    if value is None or type(value) is bool:
        return
    if type(value) in (int, float):
        _number(value)
    elif isinstance(value, str):
        if len(value) > 4096 or "\x00" in value:
            raise ValueError("invalid auxiliary text")
    elif isinstance(value, torch.Tensor):
        if (value.device.type != "cpu" or value.layout != torch.strided
                or value.numel() > 2_000_000 or value.dtype not in (torch.float32, torch.int64, torch.bool)
                or not torch.isfinite(value).all()
                or (value.is_floating_point() and (value.abs() > 1e9).any())):
            raise ValueError("invalid auxiliary tensor")
    elif isinstance(value, dict):
        if len(value) > 10_000:
            raise ValueError("auxiliary mapping exceeds limit")
        for key, child in value.items():
            if type(key) not in (str, int):
                raise ValueError("invalid auxiliary mapping key")
            _tree(key, depth + 1, budget)
            _tree(child, depth + 1, budget)
    elif type(value) in (tuple, list):
        if len(value) > 10_000:
            raise ValueError("auxiliary sequence exceeds limit")
        for child in value:
            _tree(child, depth + 1, budget)
    else:
        raise ValueError("unsupported auxiliary state value")


def _keys(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"invalid auxiliary {name} fields")


def _primitive_schema(value, reference):
    """Diagnostics must have their declared primitive structure, never tensors."""
    if isinstance(reference, dict):
        _keys(value, reference, "diagnostic")
        for key, child in reference.items():
            _primitive_schema(value[key], child)
    elif type(reference) in (list, tuple):
        if type(value) is not type(reference) or len(value) != len(reference):
            raise ValueError("invalid auxiliary primitive sequence")
        for child, expected in zip(value, reference):
            _primitive_schema(child, expected)
    elif reference is None:
        if value is not None:
            raise ValueError("invalid auxiliary optional value")
    elif type(reference) is bool:
        if type(value) is not bool:
            raise ValueError("invalid auxiliary diagnostic boolean")
    elif type(reference) in (float, int):
        _number(value)
        if type(reference) is int and type(value) is not int:
            raise ValueError("invalid auxiliary diagnostic integer")
    elif isinstance(reference, str):
        if not isinstance(value, str) or not value:
            raise ValueError("invalid auxiliary diagnostic label")
    else:
        raise ValueError("unknown diagnostic schema")


def _module(module, state, immutable_buffers=True):
    expected = module.state_dict()
    _keys(state, expected, "module")
    buffers = dict(module.named_buffers())
    for name, reference in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"invalid auxiliary module tensor: {name}")
        if immutable_buffers and name in buffers and not torch.equal(value, reference):
            raise ValueError(f"auxiliary immutable buffer mismatch: {name}")


def _adam(module, optimizer, state, max_steps):
    _keys(state, ("state", "param_groups"), "optimizer")
    _primitive_schema(state["param_groups"], optimizer.state_dict()["param_groups"])
    if state["param_groups"] != optimizer.state_dict()["param_groups"]:
        raise ValueError("auxiliary optimizer parameter groups disagree")
    parameters = list(module.parameters())
    if not isinstance(state["state"], dict) or len(state["state"]) > len(parameters):
        raise ValueError("invalid auxiliary optimizer history")
    for index, entry in state["state"].items():
        _integer(index, 0, len(parameters) - 1, "optimizer parameter id")
        _keys(entry, ("step", "exp_avg", "exp_avg_sq"), "Adam moment")
        step = entry["step"]
        if not isinstance(step, torch.Tensor) or step.shape != () or step.dtype != torch.float32:
            raise ValueError("invalid auxiliary Adam step")
        if not 0 <= float(step) <= max_steps or float(step) != int(float(step)):
            raise ValueError("auxiliary Adam step exceeds audit count")
        for name in ("exp_avg", "exp_avg_sq"):
            value, reference = entry[name], parameters[index]
            if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError("invalid auxiliary Adam moment shape")
        if (entry["exp_avg_sq"] < 0).any():
            raise ValueError("auxiliary Adam variance must be nonnegative")
    if state["state"] and set(state["state"]) != set(range(len(parameters))):
        raise ValueError("auxiliary Adam parameter history is incomplete")


def capture(session) -> dict[str, Any]:
    memory = session.memory.to_dict()
    # Presentation serializers round floats; continuation preserves exact values.
    memory["working_context"] = list(session.memory.working_context)
    memory["last_predicted_state"] = copy.deepcopy(session.memory.last_predicted_state)
    return {
        "version": 1,
        "anchor": session.anchor_model.state_dict(),
        "consolidation": session.consolidation.state_dict(),
        "byte_bridge": session.byte_bridge.state_dict(),
        "self_model": session.self_model.state_dict(),
        "self_model_optimizer": session.self_model.optimizer.state_dict(),
        "self_model_history": {"errors": list(session.self_model._history_errors),
                               "agency": list(session.self_model._history_agency),
                               "forecasts": session.self_model.total_forecasts},
        "training_modes": {name: getattr(session, name).training for name in
                           ("consolidation", "byte_bridge", "self_model")},
        "memory": memory,
        "compass": {"num_wedges": session.compass.num_wedges, "state": list(session.compass.state)},
        "phase_plane": session.phase_plane.to_dict(),
        "stp": {"config": asdict(session.stp.config), "u": list(session.stp_state.u), "r": list(session.stp_state.r)},
        "bioelectric": {"config": asdict(session.bioelectric.config), "voltage": session.bioelectric_state},
        "previous_latent": session._prev_latent,
        "previous_action": session._prev_action,
        "summaries": {"stp": session._stp_summary, "bioelectric": session._bioelectric_summary,
                      "self_model": session._self_model_summary, "phase_plane": session._phase_summary},
        "initial_weights": session._initial_weights,
        "anchor_kl_drift": session._anchor_kl_drift,
        "archaeology_report": asdict(session.archaeology._last_report) if session.archaeology._last_report is not None else None,
    }


def validate(session, state, steps):
    """Validate all auxiliary content before installing any base or observer state."""
    _tree(state)
    _keys(state, capture(session), "checkpoint")
    if type(state["version"]) is not int or state["version"] != 1:
        raise ValueError("unsupported auxiliary state version")
    for name, module in (("anchor", session.anchor_model), ("consolidation", session.consolidation),
                         ("byte_bridge", session.byte_bridge), ("self_model", session.self_model)):
        _module(module, state[name], immutable_buffers=name != "self_model")
    modes = state["training_modes"]
    _keys(modes, ("consolidation", "byte_bridge", "self_model"), "training modes")
    if any(type(value) is not bool for value in modes.values()):
        raise ValueError("auxiliary training modes must be booleans")
    history = state["self_model_history"]
    _keys(history, ("errors", "agency", "forecasts"), "self-model history")
    _integer(history["forecasts"], 0, steps, "forecast count")
    if (type(history["errors"]) is not list or type(history["agency"]) is not list
            or len(history["errors"]) != len(history["agency"])
            or len(history["errors"]) != min(64, history["forecasts"])):
        raise ValueError("invalid auxiliary self-model histories")
    for value in history["errors"]:
        _number(value, 0, 1e12)
    for value in history["agency"]:
        _number(value, -1, 1)
    _adam(session.self_model, session.self_model.optimizer, state["self_model_optimizer"], history["forecasts"])

    memory = state["memory"]
    _keys(memory, (*session.memory.to_dict(), "last_predicted_state"), "memory")
    _integer(memory["capacity"], 128, 128, "memory capacity")
    if memory["lesion_mode"] not in LESION_MODES:
        raise ValueError("invalid auxiliary memory configuration")
    _integer(memory["total_recorded_steps"], 0, steps, "memory step count")
    if type(memory["working_context"]) is not list or len(memory["working_context"]) != 8:
        raise ValueError("invalid auxiliary working context")
    for value in memory["working_context"]:
        _number(value)
    if memory["last_predicted_state"] is not None and (
            type(memory["last_predicted_state"]) is not list or len(memory["last_predicted_state"]) > 16):
        raise ValueError("invalid auxiliary predicted memory")
    if memory["last_predicted_state"] is not None:
        for value in memory["last_predicted_state"]:
            _number(value)
    if type(memory["episodes"]) is not list or len(memory["episodes"]) > 128:
        raise ValueError("invalid auxiliary episodic memory length")
    for episode in memory["episodes"]:
        _keys(episode, ("step", "state", "action", "reward", "surprise"), "episode")
        _integer(episode["step"], 0, steps, "episode step")
        _integer(episode["action"], 0, 3, "episode action")
        if type(episode["state"]) is not list or len(episode["state"]) != 8:
            raise ValueError("invalid auxiliary episodic feature length")
        for value in episode["state"]:
            _number(value)
        _number(episode["reward"])
        _number(episode["surprise"], 0, 1e12)
    if not isinstance(memory["spatial_map"], dict) or len(memory["spatial_map"]) > 31 * 31:
        raise ValueError("invalid auxiliary spatial map")
    for key, cell in memory["spatial_map"].items():
        if not re.fullmatch(r"\d{1,2},\d{1,2}", key) or any(int(part) > 30 for part in key.split(",")):
            raise ValueError("invalid auxiliary map coordinate")
        _keys(cell, ("visits", "first_step", "last_step", "total_reward"), "spatial map cell")
        _integer(cell["visits"], 1, max(1, steps), "map visits")
        _integer(cell["first_step"], 0, steps, "map first step")
        _integer(cell["last_step"], cell["first_step"], steps, "map last step")
        _number(cell["total_reward"])

    compass = state["compass"]
    _keys(compass, ("num_wedges", "state"), "compass")
    _integer(compass["num_wedges"], 16, 16, "compass wedges")
    if type(compass["state"]) is not list or len(compass["state"]) != 16:
        raise ValueError("invalid auxiliary compass shape")
    for value in compass["state"]:
        _number(value, 0, 1)
    if abs(sum(compass["state"]) - 1) > 1e-6:
        raise ValueError("invalid auxiliary compass normalization")
    phase = state["phase_plane"]
    _keys(phase, ("dt", "last_lyapunov", "last_state"), "phase plane")
    _number(phase["dt"], 0.20, 0.20)
    if ((phase["last_state"] is None) != (phase["last_lyapunov"] is None)):
        raise ValueError("invalid auxiliary phase plane configuration")
    if phase["last_state"] is not None:
        if type(phase["last_state"]) is not list or len(phase["last_state"]) != 4:
            raise ValueError("invalid auxiliary phase coordinates")
        for value in phase["last_state"]:
            _number(value, 0, 1)
        _number(phase["last_lyapunov"], 0, 10)

    stp = state["stp"]
    _keys(stp, ("config", "u", "r"), "STP")
    _keys(stp["config"], asdict(session.stp.config), "STP config")
    for name, value in stp["config"].items():
        if name != "enabled":
            _number(value, 0, 1e6)
    if type(stp["config"]["enabled"]) is not bool:
        raise ValueError("invalid auxiliary STP switch")
    STPConfig(**stp["config"])
    for name in ("u", "r"):
        reference = getattr(session.stp.initial_state(1), name)
        if type(stp[name]) is not list or len(stp[name]) != len(reference):
            raise ValueError("invalid auxiliary STP source count")
        for value, expected in zip(stp[name], reference):
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                    or (value < 0).any() or (value > 1).any()):
                raise ValueError("invalid auxiliary STP state")
    bio = state["bioelectric"]
    _keys(bio, ("config", "voltage"), "bioelectric")
    _keys(bio["config"], asdict(session.bioelectric.config), "bioelectric config")
    for value in bio["config"].values():
        _number(value, -1e6, 1e6)
    bio_config = BioelectricConfig(**bio["config"])
    if bio_config.c_m <= 0 or bio_config.pump_gain < 0:
        raise ValueError("invalid auxiliary membrane coefficients")
    initial_voltage = session.bioelectric.initial_voltage(1)
    _keys(bio["voltage"], initial_voltage, "voltage")
    for name, expected in initial_voltage.items():
        value = bio["voltage"][name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                or (value < bio_config.v_min).any() or (value > bio_config.v_max).any()):
            raise ValueError("invalid auxiliary membrane voltage")
    latent = state["previous_latent"]
    if latent is not None and (not isinstance(latent, torch.Tensor) or latent.shape != (1, 4 * session.config.hidden_dim)
                               or latent.dtype != torch.float32):
        raise ValueError("invalid auxiliary previous latent")
    if state["previous_action"] is not None:
        _integer(state["previous_action"], 0, 3, "previous action")
    _keys(state["summaries"], ("stp", "bioelectric", "self_model", "phase_plane"), "summaries")
    for key in ("stp", "bioelectric", "self_model"):
        if not isinstance(state["summaries"][key], dict):
            raise ValueError("invalid auxiliary diagnostic summary")
    _primitive_schema(state["summaries"]["stp"], session._stp_summary)
    _primitive_schema(state["summaries"]["self_model"], session._self_model_summary)
    _integer(state["summaries"]["self_model"]["total_audits"], history["forecasts"], history["forecasts"], "summary audit count")
    bio_summary = state["summaries"]["bioelectric"]
    expected_bio = dict(session._bioelectric_summary)
    if "species_potentials" in bio_summary:
        expected_bio["species_potentials"] = {name: {"mean_v_mv": 0.0, "var_v": 0.0, "min_v_mv": 0.0, "max_v_mv": 0.0}
                                              for name in initial_voltage}
    _primitive_schema(bio_summary, expected_bio)
    phase_summary = state["summaries"]["phase_plane"]
    if phase_summary is not None:
        _primitive_schema(phase_summary, PhasePlaneEngine().audit(0, 0, 0, 0))
    _module(session.model, {**session.model.state_dict(), **state["initial_weights"]})
    if set(state["initial_weights"]) != set(dict(session.model.named_parameters())):
        raise ValueError("invalid auxiliary initial parameter names")
    _number(state["anchor_kl_drift"], 0, 1e12)
    report = state["archaeology_report"]
    if report is not None:
        _primitive_schema(report, asdict(ArchaeologyReport(
            target_species="fly_f", resection_mode="rich_club_hubs", intact_logits_entropy=0.0,
            lesioned_fidelity=0.0, bioelectric_rescued_fidelity=0.0, mtpr=0.0, rescue_delta=0.0,
            regime="engram-preserved", species_resilience={name: 0.0 for name in session.archaeology.source_keys},
            cross_species_motif_conservation=session.archaeology._motif_conservation)))


def install(session, state):
    """Install a previously validated tree into an isolated candidate session."""
    for name, module in (("anchor", session.anchor_model), ("consolidation", session.consolidation),
                         ("byte_bridge", session.byte_bridge), ("self_model", session.self_model)):
        module.load_state_dict(state[name], strict=True)
    session.self_model.optimizer.load_state_dict(copy.deepcopy(state["self_model_optimizer"]))
    history = state["self_model_history"]
    session.self_model.total_forecasts = history["forecasts"]
    session.self_model._history_errors = list(history["errors"])
    session.self_model._history_agency = list(history["agency"])
    for name, mode in state["training_modes"].items():
        getattr(session, name).train(mode)
    session.memory = TriTierMemory.from_dict(copy.deepcopy(state["memory"]))
    session.memory.last_predicted_state = copy.deepcopy(state["memory"]["last_predicted_state"])
    session.compass = CXRingCompass(state["compass"]["num_wedges"])
    session.compass.state = list(state["compass"]["state"])
    session.phase_plane = PhasePlaneEngine.from_dict(state["phase_plane"])
    session.stp = TsodyksMarkramSubstrate(session.stp.node_counts, STPConfig(**state["stp"]["config"]))
    session.stp_state = STPState(tuple(value.clone() for value in state["stp"]["u"]),
                               tuple(value.clone() for value in state["stp"]["r"]))
    session.bioelectric = BioelectricMorphogeneticSubstrate(session.bundle, BioelectricConfig(**state["bioelectric"]["config"]))
    session.bioelectric_state = {name: value.clone() for name, value in state["bioelectric"]["voltage"].items()}
    session._prev_latent = state["previous_latent"].clone() if state["previous_latent"] is not None else None
    session._prev_action = state["previous_action"]
    session._stp_summary = copy.deepcopy(state["summaries"]["stp"])
    session._bioelectric_summary = copy.deepcopy(state["summaries"]["bioelectric"])
    session._self_model_summary = copy.deepcopy(state["summaries"]["self_model"])
    session._phase_summary = copy.deepcopy(state["summaries"]["phase_plane"])
    session._initial_weights = {name: value.clone() for name, value in state["initial_weights"].items()}
    session._anchor_kl_drift = state["anchor_kl_drift"]
    session.archaeology._last_report = (ArchaeologyReport(**copy.deepcopy(state["archaeology_report"]))
                                       if state["archaeology_report"] is not None else None)
    session._telemetry_archaeology_cache = None
