"""Sparse engineered graph dynamics; these are not physiological neuron models.

Observed graphs supply immutable sparse buffers. Learned cross-source routing is
a separate latent overlay and never writes edges into a source graph. A signed
functional edge is used as an association weight, without a causal interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .graphs import SOURCE_KEYS, GraphBundle, SourceGraph


RELATIONS = ("chemical", "electrical", "macro_structural", "macro_functional")


@dataclass(frozen=True)
class DynamicsConfig:
    input_dim: int = 4
    hidden_dim: int = 12
    output_dim: int = 2
    top_k: int = 2
    readout: str = "human"

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "output_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or not 0 <= self.top_k <= 3:
            raise ValueError("top_k must be an integer from 0 through 3 cross-source senders")
        if type(self.readout) is not str or self.readout not in ("human", "context_residual"):
            raise ValueError("readout must be 'human' or 'context_residual'")


@dataclass(frozen=True)
class CircuitOutput:
    logits: Tensor
    state: tuple[Tensor, ...]
    context: Tensor
    routes: Tensor


class SparseRelation(nn.Module):
    """One relation's sparse weighted mean, normalized at each destination.

The denominator is the sum of *absolute* incoming weights, so negative
functional associations keep their sign and cancellation cannot divide by zero.
Directed computational edges are already expanded by the source encoder.
Storage and aggregation are O(N + E), with no dense adjacency allocation.
"""

    def __init__(
        self,
        num_nodes: int,
        src: Sequence[int],
        dst: Sequence[int],
        weights: Sequence[float],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if num_nodes <= 0 or len(src) != len(dst) or len(src) != len(weights):
            raise ValueError("invalid sparse relation dimensions")
        self.num_nodes = num_nodes
        src_tensor = torch.tensor(src, dtype=torch.long)
        dst_tensor = torch.tensor(dst, dtype=torch.long)
        raw = torch.tensor(weights, dtype=torch.float64)
        if src_tensor.numel() and (
            src_tensor.min() < 0 or dst_tensor.min() < 0
            or src_tensor.max() >= num_nodes or dst_tensor.max() >= num_nodes
        ):
            raise ValueError("sparse relation endpoint is outside its source graph")
        if not torch.isfinite(raw).all():
            raise ValueError("sparse relation weights must be finite")
        # A common scale avoids overflow while preserving each destination ratio.
        scaled = raw / raw.abs().max() if raw.numel() and raw.abs().max() > 0 else raw
        denominator = torch.zeros(num_nodes, dtype=torch.float64).index_add(0, dst_tensor, scaled.abs())
        safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        normalized = scaled / safe_denominator[dst_tensor]
        self.register_buffer("src", src_tensor)
        self.register_buffer("dst", dst_tensor)
        self.register_buffer("weight", normalized.to(torch.float32))
        self.register_buffer("has_incoming", denominator > 0)
        self.transform = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, states: Tensor) -> Tensor:
        result = states.new_zeros(states.shape[0], self.num_nodes, states.shape[-1])
        if not self.src.numel():
            return result
        messages = self.transform(states.index_select(1, self.src)) * self.weight[None, :, None]
        return result.index_add(1, self.dst, messages)


class _SourceEncoder(nn.Module):
    def __init__(self, graph: SourceGraph, config: DynamicsConfig) -> None:
        super().__init__()
        self.num_nodes = len(graph.nodes)
        if not self.num_nodes:
            raise ValueError("each source graph must contain at least one node")
        positions = {node.id: index for index, node in enumerate(graph.nodes)}
        edges: dict[str, tuple[list[int], list[int], list[float]]] = {
            relation: ([], [], []) for relation in RELATIONS
        }
        for edge in graph.edges:
            if edge.relation not in edges:
                raise ValueError(f"unsupported observed relation: {edge.relation}")
            src, dst, weights = edges[edge.relation]
            source, destination = positions[edge.src], positions[edge.dst]
            src.append(source)
            dst.append(destination)
            weights.append(edge.weight)
            # An undirected observation is expanded for computation only. A loop
            # remains one computational edge, and raw biological counts are intact.
            if not edge.directed and source != destination:
                src.append(destination)
                dst.append(source)
                weights.append(edge.weight)
        self.relations = nn.ModuleDict({
            relation: SparseRelation(self.num_nodes, *edges[relation], config.hidden_dim)
            for relation in RELATIONS
        })
        incoming = torch.zeros(self.num_nodes)
        outgoing = torch.zeros(self.num_nodes)
        signed_incoming = torch.zeros(self.num_nodes)
        active_relations = torch.zeros(self.num_nodes)
        for relation in self.relations.values():
            one = torch.ones_like(relation.weight)
            incoming.index_add_(0, relation.dst, one)
            outgoing.index_add_(0, relation.src, one)
            signed_incoming.index_add_(0, relation.dst, relation.weight.sign())
            active_relations += relation.has_incoming.to(torch.float32)
        scale = math.log1p(max(1, self.num_nodes))
        # Structural features break identical broadcast-node states without
        # adding node-count-dependent trainable embeddings or graph-size shortcuts.
        features = torch.stack((
            incoming.log1p() / scale,
            outgoing.log1p() / scale,
            signed_incoming / incoming.clamp_min(1),
            (incoming + outgoing == 0).to(torch.float32),
        ), dim=-1)
        self.register_buffer("node_features", features)
        self.register_buffer("relation_normalizer", active_relations.clamp_min(1)[None, :, None])
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.feature_projection = nn.Linear(4, config.hidden_dim, bias=False)
        self.gru = nn.GRUCell(2 * config.hidden_dim, config.hidden_dim)

    def forward(self, observation: Tensor, previous: Tensor) -> Tensor:
        messages = torch.zeros_like(previous)
        for relation in self.relations.values():
            messages = messages + relation(previous)
        messages = messages / self.relation_normalizer
        drive = self.input_projection(observation)[:, None, :] + self.feature_projection(self.node_features)[None, :, :]
        drive = torch.tanh(drive).expand(-1, self.num_nodes, -1)
        inputs = torch.cat((drive, messages), dim=-1)
        batch_size, num_nodes, hidden_dim = previous.shape
        updated = self.gru(inputs.reshape(batch_size * num_nodes, -1), previous.reshape(batch_size * num_nodes, hidden_dim))
        return updated.reshape(batch_size, num_nodes, hidden_dim)


class GraphCircuit(nn.Module):
    """Four separate recurrent modules with an explicit synthetic routing overlay.

The legacy ``policy`` reads the human module. Other sources can influence that
head through the graft. The optional zero-initialized ``context_residual`` head
adds a direct readout from all four pooled modules. ``context`` also exposes
these pooled modules to adapters. Inference flags never remove parameters.
    """

    def __init__(self, bundle: GraphBundle, config: DynamicsConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else DynamicsConfig()
        graph_map = {graph.manifest.source_id: graph for graph in bundle.graphs}
        if set(graph_map) != set(SOURCE_KEYS) or len(bundle.graphs) != len(SOURCE_KEYS):
            raise ValueError(f"a circuit requires exactly these source graphs: {SOURCE_KEYS}")
        self.bundle_digest = bundle.digest()
        hidden_dim = self.config.hidden_dim
        self.context_dim = 4 * hidden_dim
        self.encoders = nn.ModuleList([_SourceEncoder(graph_map[source], self.config) for source in SOURCE_KEYS])
        self.queries = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in SOURCE_KEYS])
        self.keys = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in SOURCE_KEYS])
        self.values = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in SOURCE_KEYS])
        self.bridge_projections = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in SOURCE_KEYS])
        self.route_bias = nn.Parameter(torch.zeros(4, 4))
        self.null_logits = nn.Parameter(torch.zeros(4))
        self.graft_gates = nn.Parameter(torch.zeros(4, hidden_dim))
        self.policy = nn.Linear(hidden_dim, self.config.output_dim)
        self.register_buffer("self_route_mask", torch.eye(4, dtype=torch.bool))
        self.context_policy = None
        if self.config.readout == "context_residual":
            # Appending a zero residual must preserve legacy initialization and
            # the caller's RNG stream, including subsequent observer modules.
            with torch.random.fork_rng(devices=[]):
                self.context_policy = nn.Linear(self.context_dim, self.config.output_dim)
            nn.init.zeros_(self.context_policy.weight)
            nn.init.zeros_(self.context_policy.bias)

    def initial_state(self, batch_size: int) -> tuple[Tensor, ...]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        parameter = self.policy.weight
        return tuple(parameter.new_zeros(batch_size, encoder.num_nodes, self.config.hidden_dim) for encoder in self.encoders)

    def _validate_observations(self, observations: Tensor) -> None:
        if not isinstance(observations, Tensor) or observations.ndim != 3:
            raise ValueError("observations must have shape [batch,4,input_dim]")
        if observations.shape[0] <= 0 or observations.shape[1:] != (4, self.config.input_dim):
            raise ValueError("observations must have shape [positive batch,4,input_dim]")
        if observations.dtype != self.policy.weight.dtype or observations.device != self.policy.weight.device:
            raise ValueError("observations must use the circuit parameter dtype and device")
        if not torch.isfinite(observations).all():
            raise ValueError("observations must be finite")

    def _validate_state(self, state: Sequence[Tensor], batch_size: int) -> None:
        if not isinstance(state, (tuple, list)) or len(state) != 4:
            raise ValueError("state must contain one tensor per source")
        for tensor, encoder in zip(state, self.encoders):
            if not isinstance(tensor, Tensor) or tensor.shape != (batch_size, encoder.num_nodes, self.config.hidden_dim):
                raise ValueError("state dimensions must match batch, source node counts and hidden_dim")
            if tensor.dtype != self.policy.weight.dtype or tensor.device != self.policy.weight.device:
                raise ValueError("state must use the circuit parameter dtype and device")
            if not torch.isfinite(tensor).all():
                raise ValueError("state must be finite")

    def _route(self, pooled: Tensor, ablate_source: int | None = None) -> tuple[Tensor, Tensor]:
        queries = torch.stack([projection(pooled[:, index]) for index, projection in enumerate(self.queries)], dim=1)
        keys = torch.stack([projection(pooled[:, index]) for index, projection in enumerate(self.keys)], dim=1)
        values = torch.stack([projection(pooled[:, index]) for index, projection in enumerate(self.values)], dim=1)
        if ablate_source is not None and 0 <= ablate_source < 4:
            mask = torch.ones_like(values)
            mask[:, ablate_source, :] = 0.0
            values = values * mask
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(self.config.hidden_dim) + self.route_bias
        scores = scores.masked_fill(self.self_route_mask, float("-inf"))
        selected = torch.zeros_like(scores, dtype=torch.bool)
        if self.config.top_k:
            indices = scores.topk(self.config.top_k, dim=-1).indices
            selected.scatter_(-1, indices, True)
        scores = scores.masked_fill(~selected, float("-inf"))
        null = self.null_logits[None, :, None].expand(pooled.shape[0], -1, -1)
        routes = torch.softmax(torch.cat((scores, null), dim=-1), dim=-1)
        bridge = torch.matmul(routes[..., :4], values)
        return bridge, routes

    def step(
        self,
        observations: Tensor,
        state: Sequence[Tensor] | None = None,
        graft_enabled: bool = True,
        recurrence: bool = True,
        liquid_alpha: Sequence[float] | None = None,
        ablate_source: int | None = None,
        context_enabled: bool = True,
    ) -> CircuitOutput:
        self._validate_observations(observations)
        if state is not None:
            self._validate_state(state, observations.shape[0])
        previous = self.initial_state(observations.shape[0]) if state is None or not recurrence else state
        local = tuple(encoder(observations[:, index], previous[index]) for index, encoder in enumerate(self.encoders))
        if liquid_alpha is not None:
            if len(liquid_alpha) != 4:
                raise ValueError("liquid_alpha must provide 4 source retention coefficients in (0, 1]")
            local = tuple(
                (1.0 - float(alpha)) * prev + float(alpha) * cur
                for alpha, prev, cur in zip(liquid_alpha, previous, local)
            )
        if self.context_policy is not None and ablate_source is not None and 0 <= ablate_source < 4:
            # A direct readout requires a complete source-input lesion. Zero
            # local states before routing so keys and queries cannot leak it.
            local = tuple(torch.zeros_like(states) if index == ablate_source else states
                          for index, states in enumerate(local))
        elif ablate_source == 3:
            # Keep legacy intervention behavior exact for existing archives.
            local = (*local[:3], torch.zeros_like(local[3]))
        pooled = torch.stack([states.mean(dim=1) for states in local], dim=1)
        if graft_enabled:
            bridge, routes = self._route(pooled, ablate_source=ablate_source)
            state_out = tuple(
                states + self.graft_gates[index].sigmoid()[None, None, :]
                * torch.tanh(self.bridge_projections[index](bridge[:, index]))[:, None, :]
                for index, states in enumerate(local)
            )
        else:
            state_out = local
            routes = pooled.new_zeros(observations.shape[0], 4, 5)
            routes[..., 4] = 1
        final_pooled = torch.stack([states.mean(dim=1) for states in state_out], dim=1)
        if self.context_policy is not None and ablate_source is not None and 0 <= ablate_source < 4:
            # Incoming grafts can reconstruct a lesioned module from other
            # sources; exclude that module from both readout paths as well.
            source_mask = torch.ones_like(final_pooled)
            source_mask[:, ablate_source] = 0
            final_pooled = final_pooled * source_mask
        context = final_pooled.flatten(start_dim=1)
        logits = self.policy(final_pooled[:, 3])
        if self.context_policy is not None and context_enabled:
            logits = logits + self.context_policy(context)
        return CircuitOutput(
            logits=logits,
            state=state_out,
            context=context,
            routes=routes,
        )

    def forward(
        self,
        sequence: Tensor,
        graft_enabled: bool = True,
        recurrence: bool = True,
        liquid_alpha: Sequence[float] | None = None,
        ablate_source: int | None = None,
        context_enabled: bool = True,
    ) -> CircuitOutput:
        if not isinstance(sequence, Tensor) or sequence.ndim != 4 or sequence.shape[1] <= 0:
            raise ValueError("sequence must have shape [batch,positive time,4,input_dim]")
        state = None
        for time in range(sequence.shape[1]):
            output = self.step(
                sequence[:, time],
                state,
                graft_enabled=graft_enabled,
                recurrence=recurrence,
                liquid_alpha=liquid_alpha,
                ablate_source=ablate_source,
                context_enabled=context_enabled,
            )
            state = output.state
        return output
