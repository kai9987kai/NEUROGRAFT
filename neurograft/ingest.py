"""Bounded local CSV ingestion with raw-byte checksums and explicit semantics."""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path

from .graphs import Edge, Node, SourceGraph, SourceManifest


MAX_INPUT_BYTES = 64 * 1024 * 1024


def _read(path: str | Path) -> tuple[bytes, str]:
    path = Path(path)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("local input size exceeds the 64 MiB first-build limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("local input size exceeds the 64 MiB first-build limit")
    try:
        return raw, raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("CSV must use UTF-8 encoding") from exc


def _rows(text: str, headers: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != set(headers):
            raise ValueError(f"CSV requires exactly these headers: {','.join(headers)}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row has a different field count from the header")
            rows.append({key: value.strip() for key, value in row.items()})
        return rows
    except csv.Error as exc:
        raise ValueError("malformed CSV") from exc


def _float(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _bool(value: str) -> bool:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise ValueError("directed must be true, false, 1, or 0")


def import_edge_csv(nodes_path: str | Path, edges_path: str | Path, manifest: SourceManifest) -> SourceGraph:
    """Import typed local IDs. Header fields are mandatory; confidence may be blank.

    The manifest's claimed raw hash is replaced by a hash of a canonical map of
    the two actual file hashes; both individual hashes are preserved in receipts.
    Fixture status is preserved, so importing a synthetic file cannot make it
    authentic. Rights authorization remains separately declared by the caller.
    """
    nodes_raw, nodes_text = _read(nodes_path)
    edges_raw, edges_text = _read(edges_path)
    node_rows = _rows(nodes_text, ("id", "kind"))
    edge_rows = _rows(edges_text, ("src", "dst", "relation", "weight", "directed", "confidence"))
    source = manifest.source_id
    nodes = tuple(Node(f"{source}:{row['id']}", source, row["kind"], row["id"]) for row in node_rows)
    edges = tuple(Edge(f"{source}:{row['src']}", f"{source}:{row['dst']}", row["relation"],
                       _float(row["weight"], "weight"), source, _bool(row["directed"]),
                       _float(row["confidence"], "confidence") if row["confidence"] else None) for row in edge_rows)
    hashes = {"nodes_sha256": hashlib.sha256(nodes_raw).hexdigest(), "edges_sha256": hashlib.sha256(edges_raw).hexdigest()}
    checksum = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    receipt = (f"import_edge_csv: nodes_sha256={hashes['nodes_sha256']}; edges_sha256={hashes['edges_sha256']}; "
               f"node_rows={len(nodes)}; edge_rows={len(edges)}; namespace={source}:; "
               "weight_semantics=relation-defined; undirected=one-observation")
    return SourceGraph(replace(manifest, sha256=checksum, processing=manifest.processing + (receipt,)), nodes, edges)


def import_human_matrix(path: str | Path, manifest: SourceManifest,
                        relation: str = "macro_functional", threshold: float = 0.0) -> SourceGraph:
    """Import a labelled symmetric region matrix, retaining its upper triangle.

    Thresholding uses absolute value and keeps nonzero values with abs(weight)
    >= threshold. The diagonal is validated then excluded. Functional values,
    including diagonal entries, are correlations in [-1, 1], never synapses.
    """
    if manifest.source_id != "human" or manifest.scale != "brain_region":
        raise ValueError("human matrix requires a human brain_region manifest")
    if relation not in ("macro_structural", "macro_functional"):
        raise ValueError("human matrix relation must be macro_structural or macro_functional")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")
    raw, text = _read(path)
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError("malformed matrix CSV") from exc
    if not rows or len(rows[0]) < 2:
        raise ValueError("matrix requires a header and at least one labelled region")
    labels = [label.strip() for label in rows[0][1:]]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("matrix column labels must be nonempty and unique")
    size = len(labels)
    if len(rows) != size + 1 or any(len(row) != size + 1 for row in rows[1:]):
        raise ValueError("matrix must be square with matching row lengths")
    if [row[0].strip() for row in rows[1:]] != labels:
        raise ValueError("matrix row labels must match column labels in order")
    matrix = [[_float(value.strip(), "matrix weight") for value in row[1:]] for row in rows[1:]]
    for i in range(size):
        for j in range(size):
            value = matrix[i][j]
            if relation == "macro_functional" and not -1 <= value <= 1:
                raise ValueError("functional matrix correlation must be in [-1, 1]")
            if relation == "macro_structural" and value < 0:
                raise ValueError("structural matrix weights must be nonnegative")
            if not math.isclose(value, matrix[j][i], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("human matrix must be symmetric")
    nodes = tuple(Node(f"human:{label}", "human", "region", label) for label in labels)
    edges = tuple(Edge(nodes[i].id, nodes[j].id, relation, matrix[i][j], "human", False)
                  for i in range(size) for j in range(i + 1, size)
                  if matrix[i][j] != 0 and abs(matrix[i][j]) >= threshold)
    checksum = hashlib.sha256(raw).hexdigest()
    receipt = (f"import_human_matrix: input_sha256={checksum}; relation={relation}; threshold_abs={threshold}; "
               f"diagonal=validated-excluded; symmetry_rtol=1e-9; symmetry_atol=1e-12; "
               f"regions={size}; retained_upper_triangle={len(edges)}; zero_entries=not-retained; "
               "functional_semantics=signed-association-not-causal-transmission")
    return SourceGraph(replace(manifest, sha256=checksum, processing=manifest.processing + (receipt,)), nodes, edges)
