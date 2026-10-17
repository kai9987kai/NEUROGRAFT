# Worm acquisition and normalization

The local source is pinned to OpenWorm CElegansNeuroML commit `1ca08b14f4e795a973ce447565e0c97b897ee4dc`.

Raw files:

- `data/raw/openworm-pinned/herm_full_edgelist.csv`
- `data/raw/openworm-pinned/CellsOnly.net.nml`

The full byte hashes, URLs and normalization decisions are recorded in `data/raw/openworm-pinned/receipt.json` and the graph manifest. The raw CSV SHA-256 is `142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca`.

The cells-only NeuroML network declares 302 neurons. This is preferable to enumerating every generated `.cell.nml` file: that directory also includes a muscle example. Explicit zero-padding aliases map reference motor-neuron names to the original edge-table IDs; the manifest preserves the mapping.

| Normalization receipt | Count |
|---|---:|
| Raw source edge rows | 7,379 |
| Rows with a non-neuron endpoint excluded | 1,573 |
| Equal mirrored electrical rows deduplicated | 1,063 |
| Inconsistent electrical source rows excluded by explicit policy | 24 |
| Retained chemical observations | 3,638 |
| Retained undirected electrical observations | 1,081 |
| Retained observations total | 4,719 |
| Declared neuron nodes, including isolates | 302 |
| Expanded computational arcs | 5,787 |
| Observed self-loops | 47 |
| Weak components | 3 |
| Isolated nodes | 2 |

The 24 inconsistent rows form 12 mirrored electrical pairs. Both source weights and affected endpoints are retained in the receipt. Acquisition rejects these inconsistencies by default. The local build explicitly used `--electrical-conflicts exclude`, removing the ambiguous pairs from the working graph without choosing an invented weight or altering the raw data.

These counts describe this filtered source derivative, not a universal modern worm connectome or a calibrated simulation. The acquired connectivity table's redistribution licence remains unresolved. Technical authenticity is true; rights readiness is false. The other three source slots in the mixed graph are labelled synthetic fixtures.
