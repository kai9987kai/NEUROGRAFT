"""Standalone inspectable HTML reports; values are escaped and remain local."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path


def write_html_report(report: dict, output_path: str | Path) -> Path:
    def text(value):
        return escape(str(value))

    runs = report.get("runs", [])
    summary = report.get("summary", {})
    bars = []
    colors = ("#70e0c3", "#88afff", "#ffc478", "#cf9cff")
    for index, (name, values) in enumerate(summary.items()):
        if not isinstance(values, dict):
            continue
        accuracy = values.get("mean_test_accuracy", values.get("mean_accuracy", 0))
        accuracy = float(accuracy)
        bars.append(f'<div class="bar-label"><span>{text(name)}</span><strong>{accuracy:.1%}</strong></div>'
                    f'<div class="track"><div style="width:{max(0,min(100,accuracy*100)):.2f}%;'
                    f'background:{colors[index % len(colors)]}"></div></div>')
    rows = []
    for run in runs:
        accuracy = run.get("test_accuracy", 0)
        rows.append(f'<tr><td>{text(run.get("variant"))}</td><td>{text(run.get("seed"))}</td>'
                    f'<td>{float(accuracy):.1%}</td><td>{text(run.get("parameter_count", run.get("trainable_parameters", "—")))}</td>'
                    f'<td>{text(run.get("duration_seconds", run.get("elapsed_seconds", "—")))}</td></tr>')
    readiness = report.get("source_readiness", report.get("graph_stats", {}))
    sources = readiness.get("sources", {}) if isinstance(readiness, dict) else {}
    cards = []
    for name in ("worm", "fly_f", "fly_m", "human"):
        source = sources.get(name, {})
        cards.append(f'<article><span class="eyebrow">{text(name)}</span><h3>{text(source.get("status", "unavailable"))}</h3>'
                     f'<p>{text(source.get("node_count", "—"))} nodes · {text(source.get("edge_count", "—"))} observations</p></article>')
    scope = report.get("evidence_scope", "synthetic task; inspect full receipts below")
    scope_html = '<br>'.join(text(item) for item in scope) if isinstance(scope, list) else text(scope)
    content = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEUROGRAFT · Experiment report</title><style>
:root{font:16px/1.55 system-ui,sans-serif;color:#dfedf2;background:#0b121a;color-scheme:dark}
body{max-width:1080px;margin:auto;padding:42px 24px}h1{font-size:clamp(2rem,6vw,3.9rem);letter-spacing:-.05em;margin:.15em 0}
h2{font-size:1.4rem;margin-top:2.5rem}p{color:#a8becb}a{color:#70e0c3}.eyebrow{color:#70e0c3;font-size:.76rem;letter-spacing:.14em;text-transform:uppercase}
.notice{border-left:3px solid #ffc478;padding:12px 18px;background:#1d211f;margin:26px 0;color:#f3dcba}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}article,.panel{border:1px solid #293b49;background:#101d28;border-radius:12px;padding:20px}article h3{margin:7px 0}article p{font-size:.8rem}
.bar-label{display:flex;justify-content:space-between;margin-top:18px}.track{height:13px;border-radius:8px;background:#253544;margin:7px 0 18px;overflow:hidden}.track>div{height:100%}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;border-bottom:1px solid #293b49;padding:12px}th{color:#a8becb}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.78rem;color:#adc7d6}summary{cursor:pointer;padding:12px 0}.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.flow span{padding:9px 15px;border:1px solid #3a5965;border-radius:9px}.flow b{color:#70e0c3}footer{color:#7996a8;margin:34px 0}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}body{padding:24px 14px}.scroll{overflow:auto}}
</style><header><div class="eyebrow">Experimental neural circuit laboratory · v0.1</div><h1>NEUROGRAFT</h1><p>Sparse source graphs. Explicit synthetic bridges. Measured outcomes.</p></header>'''
    content += f'<div class="notice"><strong>Evidence scope:</strong> {scope_html}<br>These synthetic-task results do not establish a biological-connectome advantage or pretrained language-model improvement.</div>'
    content += '<h2>Graphs used in this experiment</h2><div class="grid">' + ''.join(cards) + '</div>'
    if report.get('build_verification'):
        content += '<details><summary>Build verification and other acquired sources</summary><pre>' + text(json.dumps(report['build_verification'], indent=2)) + '</pre></details>'
    content += '<h2>Computational pathway</h2><div class="panel flow"><span>Separate source encoders</span><b>→</b><span>Artificial sparse graft</span><b>→</b><span>Recurrent state</span><b>→</b><span>Human-module task head</span></div>'
    content += '<p>The cross-species routes are engineered computations. Qwen integration has a separate readiness gate.</p>'
    content += '<h2>Held-out synthetic task</h2><div class="panel">' + (''.join(bars) or '<p>See run receipts below.</p>') + '</div>'
    content += '<h2>Every seed and control</h2><div class="scroll"><table><thead><tr><th>Variant</th><th>Seed</th><th>Test accuracy</th><th>Parameters</th><th>Seconds</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    comparisons = report.get('paired_comparisons', [])
    if comparisons:
        content += '<h2>Paired seed comparisons</h2><div class="panel">'
        for pair in comparisons:
            interval = pair.get('bootstrap_95_ci')
            interval_text = 'not estimated' if interval is None else f"[{interval[0]:+.1%}, {interval[1]:+.1%}]"
            content += f'<p><strong>{text(pair["comparison"])}</strong>: {float(pair["mean_accuracy_difference"]):+.1%} · descriptive 95% interval {text(interval_text)}</p>'
        content += '<p>Five seeds give exploratory intervals. All runs share one fixed dataset split; topology rewiring also varies with control seed.</p></div>'
    content += '<details><summary>Full report, configurations and reproducibility receipts</summary><pre>' + text(json.dumps(report, indent=2, allow_nan=False)) + '</pre></details>'
    content += '<footer>Generated locally by NEUROGRAFT. Source and synthetic-edge provenance remain separate.</footer></html>'
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
