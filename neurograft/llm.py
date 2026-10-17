"""Frozen language backbone with a real differentiable graph-state pathway.

Imports are optional: graph ingestion and inspection do not load Transformers.
Generation deliberately uses a cache-free full-prefix reference implementation.
"""
from __future__ import annotations

import re
import torch
from torch import nn


class FrozenLanguageGraft(nn.Module):
    def __init__(self, backbone: nn.Module, context_dim: int):
        super().__init__()
        if not isinstance(context_dim, int) or context_dim <= 0:
            raise ValueError("context_dim must be a positive integer")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        embedding = backbone.get_input_embeddings().weight
        self.projection = nn.Linear(context_dim, embedding.shape[1], bias=False,
                                    device=embedding.device, dtype=embedding.dtype)
        # A nonzero projection lets the zero gate receive a gradient initially.
        self.gate = nn.Parameter(torch.zeros((), device=embedding.device, dtype=embedding.dtype))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, input_ids, context, attention_mask=None, enabled=True,
                injection_mask=None, **kwargs):
        if not enabled:
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if input_ids.ndim != 2 or context.ndim != 2 or input_ids.shape[0] != context.shape[0]:
            raise ValueError("input_ids [B,T] and context [B,D] must have matching batches")
        if context.shape[1] != self.projection.in_features:
            raise ValueError("graph context width does not match adapter")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache", False):
            raise ValueError("grafted reference forward requires use_cache=False and no past_key_values")
        kwargs.setdefault("use_cache", False)
        embeddings = self.backbone.get_input_embeddings()(input_ids)
        context = context.to(device=embeddings.device, dtype=embeddings.dtype)
        residual = self.projection(context)[:, None, :] * self.gate.tanh()
        for mask in (attention_mask, injection_mask):
            if mask is not None and (mask.shape != input_ids.shape or not torch.isfinite(mask).all()
                                     or not ((mask == 0) | (mask == 1)).all()):
                raise ValueError("injection/attention masks must be finite binary [B,T] matching input_ids")
        # Explicit injection positions can narrow valid tokens, never unmask padding.
        if attention_mask is not None:
            residual = residual * attention_mask.to(embeddings).unsqueeze(-1)
        if injection_mask is not None:
            residual = residual * injection_mask.to(embeddings).unsqueeze(-1)
        return self.backbone(inputs_embeds=embeddings + residual,
                             attention_mask=attention_mask, **kwargs)

    @torch.no_grad()
    def generate_greedy(self, input_ids, context, max_new_tokens=16, attention_mask=None,
                        enabled=True, eos_token_id=None):
        if not isinstance(max_new_tokens, int) or not 0 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 0 and 512")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("generation requires a nonempty [B,T] prompt")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not torch.all(attention_mask[:, -1] == 1):
            raise ValueError("generation needs matching masks and left padding (last token valid)")
        tokens = input_ids.clone()
        mask = attention_mask.clone()
        done = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        for _ in range(max_new_tokens):
            output = self(tokens, context, attention_mask=mask, enabled=enabled, use_cache=False)
            next_token = output.logits[:, -1].argmax(-1)
            if eos_token_id is not None:
                next_token = torch.where(done, int(eos_token_id), next_token)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_token[:, None])), dim=1)
            if eos_token_id is not None:
                done |= next_token == int(eos_token_id)
                if done.all():
                    break
        return tokens


class NeurograftLanguageModel(nn.Module):
    """End-to-end circuit-to-language computation with fresh graph episodes.

    ``graph_sequence`` must contain only observations available at prediction
    time. It must never be derived from the teacher-forced answer labels.
    """

    def __init__(self, circuit: nn.Module, language: FrozenLanguageGraft):
        super().__init__()
        if circuit.context_dim != language.projection.in_features:
            raise ValueError("circuit context dimension does not match language adapter")
        self.circuit = circuit
        self.language = language

    def forward(self, input_ids, graph_sequence, attention_mask=None, graft_enabled=True, **kwargs):
        context = self.circuit(graph_sequence).context if graft_enabled else None
        return self.language(input_ids, context, attention_mask=attention_mask,
                             enabled=graft_enabled, **kwargs)

    @torch.no_grad()
    def generate_greedy(self, input_ids, graph_sequence, attention_mask=None,
                        graft_enabled=True, **kwargs):
        context = self.circuit(graph_sequence).context if graft_enabled else None
        return self.language.generate_greedy(input_ids, context, attention_mask=attention_mask,
                                            enabled=graft_enabled, **kwargs)


def load_pretrained(model_id: str, revision: str, context_dim: int,
                    local_files_only: bool = True, minimum_available_gb: float = 6.0):
    """Explicit opt-in, revision-pinned FP32 loading; no silent model substitution."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("model revision must be an immutable full 40-character commit SHA")
    supported = {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"}
    if model_id not in supported:
        raise ValueError(f"model_id must be one of {sorted(supported)}")
    from .runtime import memory_info
    available = memory_info()["available_ram_gb"]
    if available is None or available < minimum_available_gb:
        raise RuntimeError(f"pretrained FP32 probe needs at least {minimum_available_gb:.1f} GiB "
                           f"available RAM; detected {available}. Free memory and rerun.")
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    loader = AutoModelForImageTextToText if model_id == "Qwen/Qwen3.5-0.8B" else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                             local_files_only=local_files_only, trust_remote_code=False)
    backbone = loader.from_pretrained(model_id, revision=revision, local_files_only=local_files_only,
                                      trust_remote_code=False, dtype=torch.float32,
                                      attn_implementation="eager")
    return FrozenLanguageGraft(backbone, context_dim), tokenizer
