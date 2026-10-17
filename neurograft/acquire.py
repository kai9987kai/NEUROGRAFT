"""Bounded acquisition of a revision-pinned public worm connectivity source.

The source connectivity licence is deliberately unresolved here. Local acquisition
does not grant redistribution rights to the data or its derivatives.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone
import urllib.request
import xml.etree.ElementTree as ET


PINNED_WORM_REVISION = "1ca08b14f4e795a973ce447565e0c97b897ee4dc"
REPOSITORY = "https://github.com/openworm/CElegansNeuroML"


def neuron_ids_from_network(payload: bytes) -> set[str]:
    """Use CellsOnly's declared neuron populations, excluding other cell files."""
    root = ET.fromstring(payload)
    populations = [entry for entry in root.iter() if entry.tag.rsplit('}', 1)[-1] == 'population']
    result = {entry.attrib['id'] for entry in populations}
    if any(entry.attrib.get('size') != '1' for entry in populations) or len(result) != len(populations):
        raise ValueError("neuron network must declare unique one-cell populations")
    if not result:
        raise ValueError("no declared neuron populations in reference network")
    return result


def align_worm_neurons(payload: bytes, model_ids: set[str]) -> tuple[set[str], dict[str, str]]:
    """Preserve edge-table motor IDs, recording explicit zero-padding aliases."""
    rows = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    observed = {row[key].strip() for row in rows for key in ('Source', 'Target')}
    allowed, aliases = set(), {}
    for name in sorted(model_ids):
        selected = name
        match = re.fullmatch(r'(AS|DA|DB|DD|VA|VB|VC|VD)([1-9])', name)
        if name not in observed and match:
            candidate = f'{match[1]}{int(match[2]):02d}'
            if candidate in observed:
                selected = candidate
                aliases[selected] = name
        if selected in allowed:
            raise ValueError("ambiguous neuron identifier alignment")
        allowed.add(selected)
    return allowed, aliases


def normalize_worm_csv(payload: bytes, neuron_ids: set[str],
                       electrical_conflicts: str = 'error') -> tuple[list[dict], dict]:
    """Filter to explicit neuron IDs, preserving synapse counts and gap contacts."""
    if electrical_conflicts not in {'error', 'exclude'}:
        raise ValueError('electrical_conflicts must be error or exclude')
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames != ["Source", "Target", "Weight", "Type"]:
        raise ValueError("worm CSV requires Source,Target,Weight,Type headers")
    rows = []
    seen: dict[tuple, float] = {}
    electrical_rows: dict[tuple, list[float]] = {}
    conflicts: set[tuple] = set()
    receipt = {"raw_rows": 0, "excluded_non_neuron_rows": 0,
               "deduplicated_electrical_rows": 0}
    for line, item in enumerate(reader, 2):
        receipt["raw_rows"] += 1
        src, dst = item["Source"].strip(), item["Target"].strip()
        relation = item["Type"].strip().lower()
        if relation not in {"chemical", "electrical"}:
            raise ValueError(f"unknown worm relation at line {line}: {relation}")
        try:
            weight = float(item["Weight"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid weight at line {line}") from exc
        if not math.isfinite(weight) or weight <= 0 or not weight.is_integer():
            raise ValueError(f"invalid synapse weight at line {line}")
        if src not in neuron_ids or dst not in neuron_ids:
            receipt["excluded_non_neuron_rows"] += 1
            continue
        directed = relation == "chemical"
        if not directed:
            src, dst = sorted((src, dst))
        key = (relation, src, dst)
        if not directed:
            electrical_rows.setdefault(key, []).append(weight)
        if key in seen:
            if not directed and seen[key] == weight:
                receipt["deduplicated_electrical_rows"] += 1
                continue
            if not directed and electrical_conflicts == 'exclude':
                conflicts.add(key)
                continue
            raise ValueError(f"conflicting or ambiguous duplicate worm edge at line {line}")
        seen[key] = weight
        rows.append(dict(src=src, dst=dst, relation=relation, weight=weight, directed=directed))
    rows = [row for row in rows if (row['relation'], row['src'], row['dst']) not in conflicts]
    receipt['conflicting_electrical_pairs'] = [
        {'src': key[1], 'dst': key[2], 'weights': electrical_rows[key]} for key in sorted(conflicts)]
    receipt['excluded_conflicting_electrical_rows'] = sum(len(electrical_rows[key]) for key in conflicts)
    receipt['electrical_conflicts_policy'] = electrical_conflicts
    if not rows:
        raise ValueError("no neuron-to-neuron edges survived filtering")
    receipt["retained_edges"] = len(rows)
    receipt["declared_neurons"] = len(neuron_ids)
    return rows, receipt


def _download(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NEUROGRAFT-research/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"download exceeds {max_bytes} byte cap")
    return payload


def fetch_worm(output_dir: str | Path, revision: str = PINNED_WORM_REVISION,
               max_bytes: int = 4_000_000, electrical_conflicts: str = 'error') -> dict:
    """Fetch two small immutable public source files and save a validated bundle."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("worm revision must be a full immutable 40-character commit SHA")
    if not 1 <= max_bytes <= 20_000_000:
        raise ValueError("download cap must be between 1 and 20,000,000 bytes")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("acquisition output directory must be empty; existing data are preserved")
    csv_url = f"https://raw.githubusercontent.com/openworm/CElegansNeuroML/{revision}/herm_full_edgelist.csv"
    network_url = f"https://raw.githubusercontent.com/openworm/CElegansNeuroML/{revision}/CElegans/generatedNeuroML2/CellsOnly.net.nml"
    payload = _download(csv_url, max_bytes)
    network_bytes = _download(network_url, max_bytes)
    neuron_ids, aliases = align_worm_neurons(payload, neuron_ids_from_network(network_bytes))
    records, receipt = normalize_worm_csv(payload, neuron_ids, electrical_conflicts)
    receipt['neuron_model_aliases'] = aliases
    raw_sha = hashlib.sha256(payload).hexdigest()
    network_sha = hashlib.sha256(network_bytes).hexdigest()
    from .graphs import SourceManifest, Node, Edge, SourceGraph, GraphBundle
    manifest = SourceManifest(
        source_id="worm", species="Caenorhabditis elegans", version=revision,
        url=csv_url, citation=REPOSITORY,
        license="UNRESOLVED_CONNECTIVITY_TERMS; local research; do not redistribute",
        scale="single_neuron", status="imported", sha256=raw_sha,
        retrieved_at=datetime.now(timezone.utc).isoformat(), authorized=False,
        processing=(
            "neuron-only graph using CellsOnly.net.nml populations with explicit source-ID aliases",
            f"neuron_network_url={network_url}", f"neuron_network_sha256={network_sha}",
            "mirrored identical electrical contacts canonicalized once; chemical direction preserved",
            json.dumps(receipt, sort_keys=True),
        ),
    )
    nodes = tuple(Node(f"worm:{node}", "worm", "neuron", node) for node in sorted(neuron_ids))
    edges = tuple(Edge(f"worm:{r['src']}", f"worm:{r['dst']}", r["relation"],
                       r["weight"], "worm", directed=r["directed"]) for r in records)
    bundle = GraphBundle((SourceGraph(manifest, nodes, edges),))
    receipt.update(revision=revision, source_url=csv_url, raw_sha256=raw_sha,
                   network_sha256=network_sha, bundle_sha256=bundle.digest(),
                   source_data_license_resolved=False, stats=bundle.stats())
    # Validate everything before writing; do not overwrite existing acquisition files.
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "herm_full_edgelist.csv").write_bytes(payload)
    (destination / "CellsOnly.net.nml").write_bytes(network_bytes)
    bundle.save(destination / "worm.bundle.json")
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
