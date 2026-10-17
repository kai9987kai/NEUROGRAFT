"""Dale's law polarity audit, input-fraction scaling, and contractive spectral bounds.

In biological connectomics (e.g. Janelia FlyEM, C. elegans connectome), chemical
synapses adhere to Dale's principle: each presynaptic neuron releases a characteristic
neurotransmitter, giving all of its chemical output terminals a uniform polarity
(excitatory or inhibitory). Furthermore, synaptic weights scaled by measured input
fraction produce contractive Perron roots bounded by Collatz-Wielandt quotients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor

from .graphs import Edge, GraphBundle, Node, SourceGraph, SourceManifest


# Canonical neurotransmitter polarities for well-characterized C. elegans and insect types.
# Excitatory (+1): Acetylcholine (ACh), Octopamine/Monoamines (+1 modulatory)
# Inhibitory (-1): GABA, Glutamate (in insect/nematode inhibitory channels like GluCl), Histamine
KNOWN_INHIBITORY_PREFIXES = ("VD", "DD", "AVL", "DVB", "RME", "RIS", "GABA", "GluCl", "HIS")
KNOWN_EXCITATORY_PREFIXES = ("DA", "DB", "VA", "VB", "AS", "AVA", "AVB", "PVC", "AVD", "ACh")


def infer_node_polarity(original_id: str) -> int:
    """Infer Dale's neurotransmitter polarity from biological neuron classification.

    Returns:
        +1 for excitatory (e.g. cholinergic motor/command neurons)
        -1 for inhibitory (e.g. GABAergic motor/cross-inhibitory neurons)
    """
    clean = original_id.upper().strip()
    for prefix in KNOWN_INHIBITORY_PREFIXES:
        if clean.startswith(prefix.upper()):
            return -1
    return 1


@dataclass(frozen=True)
class DalesLawReport:
    """Rigorous audit of Dale's law compliance and spectral contraction bounds."""
    source_id: str
    total_nodes: int
    multi_target_nodes: int
    compliant_nodes: int
    compliance_ratio: float
    excitatory_types: int
    inhibitory_types: int
    mixed_violations: int
    spectral_radius_bound: float
    perron_root_estimate: float
    input_fraction_coverage: float
    is_contractive: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shifted_power_iteration(
    adj_abs: Tensor,
    iters: int = 500,
    tol: float = 1e-8,
) -> tuple[float, float, int]:
    """Perron root and Collatz-Wielandt upper bound via shifted power iteration.

    For a nonnegative matrix A, iterating (A + c*I) removes periodicity, guaranteeing
    convergence to the dominant eigenvector x > 0. The Collatz-Wielandt quotient
    max_i ((A x)_i / x_i) provides a rigorous upper bound on the spectral radius rho(A).
    """
    n = adj_abs.shape[0]
    if n == 0:
        return 0.0, 0.0, 0
    row_sum = adj_abs.sum(dim=1)
    shift = float(row_sum.mean().item()) if row_sum.numel() else 0.0
    x = torch.full((n,), 1.0 / math.sqrt(max(1, n)), dtype=torch.float64)
    lam = prev = 0.0
    iteration = 0
    for iteration in range(1, iters + 1):
        y = torch.mv(adj_abs, x) + shift * x
        norm = float(torch.linalg.norm(y).item())
        if norm <= 1e-12:
            return 0.0, 0.0, iteration
        lam = norm - shift
        x = y / norm
        if iteration > 10 and abs(lam - prev) <= tol * max(1.0, abs(lam)):
            break
        prev = lam
    ax = torch.mv(adj_abs, x)
    safe_x = torch.where(x > 1e-12, x, torch.ones_like(x))
    quotient = torch.where(x > 1e-12, ax / safe_x, torch.zeros_like(x))
    upper_bound = float(quotient.max().item())
    estimate = float((x @ ax).item())
    return estimate, upper_bound, iteration


def audit_dales_law(graph: SourceGraph, radius: float = 0.9) -> DalesLawReport:
    """Audit Dale's law compliance and contractive spectral bounds on a SourceGraph.

    Args:
        graph: Input SourceGraph to audit.
        radius: Target contraction spectral scaling radius (default 0.9).

    Returns:
        DalesLawReport containing compliance metrics, sign counts, and Collatz-Wielandt bounds.
    """
    n = len(graph.nodes)
    if n == 0:
        return DalesLawReport(
            source_id=graph.manifest.source_id,
            total_nodes=0,
            multi_target_nodes=0,
            compliant_nodes=0,
            compliance_ratio=1.0,
            excitatory_types=0,
            inhibitory_types=0,
            mixed_violations=0,
            spectral_radius_bound=0.0,
            perron_root_estimate=0.0,
            input_fraction_coverage=0.0,
            is_contractive=True,
        )

    node_indices = {node.id: i for i, node in enumerate(graph.nodes)}
    node_polarities = {i: infer_node_polarity(node.original_id) for i, node in enumerate(graph.nodes)}
    outgoing_edges: dict[int, list[float]] = {i: [] for i in range(n)}
    incoming_weights: dict[int, list[float]] = {i: [] for i in range(n)}
    chem_edges: list[tuple[int, int, float]] = []

    for edge in graph.edges:
        if edge.relation in ("chemical", "macro_functional"):
            src = node_indices.get(edge.src)
            dst = node_indices.get(edge.dst)
            if src is not None and dst is not None:
                outgoing_edges[src].append(edge.weight)
                incoming_weights[dst].append(abs(edge.weight))
                chem_edges.append((src, dst, edge.weight))

    multi_target = 0
    compliant = 0
    excitatory = 0
    inhibitory = 0
    mixed = 0

    for src_idx, weights in outgoing_edges.items():
        pol = node_polarities[src_idx]
        if pol > 0:
            excitatory += 1
        else:
            inhibitory += 1

        if len(weights) >= 2:
            multi_target += 1
            # If the source graph has signed edges (like macro_functional), check sign consistency
            has_pos = any(w > 0 for w in weights)
            has_neg = any(w < 0 for w in weights)
            if has_pos and has_neg:
                mixed += 1
            else:
                compliant += 1

    ratio = compliant / max(1, multi_target) if multi_target > 0 else 1.0

    # Build input-fraction weighted adjacency matrix
    adj = torch.zeros((n, n), dtype=torch.float64)
    total_in = torch.zeros(n, dtype=torch.float64)
    for src, dst, w in chem_edges:
        total_in[dst] += abs(w)

    for src, dst, w in chem_edges:
        denom = total_in[dst].item()
        frac = abs(w) / denom if denom > 0 else 0.0
        sign = float(node_polarities[src])
        adj[dst, src] += frac * sign * radius

    coverage = float((total_in > 0).float().mean().item())
    est, ub, _ = _shifted_power_iteration(adj.abs())

    return DalesLawReport(
        source_id=graph.manifest.source_id,
        total_nodes=n,
        multi_target_nodes=multi_target,
        compliant_nodes=compliant,
        compliance_ratio=round(ratio, 4),
        excitatory_types=excitatory,
        inhibitory_types=inhibitory,
        mixed_violations=mixed,
        spectral_radius_bound=round(ub, 4),
        perron_root_estimate=round(est, 4),
        input_fraction_coverage=round(coverage, 4),
        is_contractive=ub < 1.0,
    )


def audit_bundle_dales_law(bundle: GraphBundle, radius: float = 0.9) -> dict[str, dict[str, Any]]:
    """Audit all source graphs within a bundle for Dale's law compliance."""
    reports = {}
    for graph in bundle.graphs:
        report = audit_dales_law(graph, radius=radius)
        reports[graph.manifest.source_id] = report.to_dict()
    return reports


def project_dales_law(graph: SourceGraph, radius: float = 0.9) -> SourceGraph:
    """Project a connectome graph into input-fraction scaled contractive bounds.

    Scales each incoming edge by input fraction times radius, guaranteeing that
    each receiving node's incoming weight sum is at most radius <= 0.9.
    """
    node_indices = {node.id: i for i, node in enumerate(graph.nodes)}
    n = len(graph.nodes)
    total_in: dict[int, float] = {i: 0.0 for i in range(n)}

    for edge in graph.edges:
        if edge.relation in ("chemical", "macro_functional"):
            d = node_indices.get(edge.dst)
            if d is not None:
                total_in[d] += abs(edge.weight)

    new_edges: list[Edge] = []
    for edge in graph.edges:
        if edge.relation in ("chemical", "macro_functional"):
            d = node_indices.get(edge.dst)
            if d is not None and total_in[d] > 0:
                frac = abs(edge.weight) / total_in[d]
                new_w = float(frac * radius)
                # Keep original sign for macro_functional if negative
                if edge.weight < 0 and edge.relation == "macro_functional":
                    new_w = -new_w
                new_edges.append(
                    Edge(
                        src=edge.src,
                        dst=edge.dst,
                        relation=edge.relation,
                        weight=float(new_w),
                        source_id=edge.source_id,
                        directed=edge.directed,
                    )
                )
            else:
                new_edges.append(edge)
        else:
            new_edges.append(edge)

    return SourceGraph(
        manifest=graph.manifest,
        nodes=graph.nodes,
        edges=tuple(new_edges),
    )
