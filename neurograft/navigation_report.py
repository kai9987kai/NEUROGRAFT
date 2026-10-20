"""Offline candidate navigation reports with explicit selection and promotion scope."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from urllib.parse import quote


def write_navigation_report(result: dict, output_path: str | Path) -> Path:
    """Write one new standalone HTML artifact; never replace existing evidence."""
    path = Path(output_path)
    if path.exists():
        raise ValueError(f'output already exists: {path}; choose a new path')
    raw = json.dumps(result, indent=2, allow_nan=False)

    def text(value):
        return escape(str(value))

    def number(value, digits=3):
        return f'{value:.{digits}f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else '—'

    def percentage(value):
        return f'{value:.1%}' if isinstance(value, (int, float)) and not isinstance(value, bool) else '—'

    def summary(report, variant):
        return report.get('variants', {}).get(variant, {}).get('summary', {})

    baseline, candidate = result.get('baseline', {}), result.get('candidate', {})
    selected_round = result.get('selected_round', 'unavailable')
    selected_hash = result.get('selected_model_hash', candidate.get('model_hash', 'unavailable'))
    promotion = ('No promotion performed' if result.get('promotion_performed') is False
                 else f'Promotion flag: {result.get("promotion_performed", "unavailable")}')
    scope = result.get('evidence_scope', result.get('scope', 'Synthetic navigation experiment; inspect raw receipts.'))
    if isinstance(scope, list):
        scope = ' '.join(str(item) for item in scope)
    protocols_match = bool(baseline.get('protocol_id')) and baseline.get('protocol_id') == candidate.get('protocol_id')
    protocol_notice = ('Baseline and candidate use the same frozen test protocol.' if protocols_match
                       else 'Protocols differ or are missing; the comparison is not certified as matched.')
    labels = {'policy': 'Full policy', 'no_graft': 'Without grafts', 'no_recurrence': 'Without recurrence',
              'random': 'Random actions', 'shortest_path': 'Shortest-path oracle',
              'no_context': 'Without context residual'}
    rows = []
    for variant, label in labels.items():
        before, after = summary(baseline, variant), summary(candidate, variant)
        if not before and not after:
            continue
        cells = [text(label)]
        for values in (before, after):
            cells.extend([text(values.get('episodes', '—')), percentage(values.get('success_rate')),
                          number(values.get('spl')), percentage(values.get('collision_rate')),
                          number(values.get('mean_steps'), 1)])
        rows.append('<tr><th scope="row">' + cells[0] + '</th>' + ''.join(f'<td>{value}</td>' for value in cells[1:]) + '</tr>')

    progress_rows = []
    for entry in result.get('progress', []):
        values = entry.get('validation', {})
        if 'variants' in values:
            values = summary(values, 'policy')
        elif 'summary' in values:
            values = values['summary']
        selected = entry.get('round') == selected_round
        progress_rows.append(f'<tr{" class=selected" if selected else ""}><th scope="row">Round {text(entry.get("round", "—"))}'
                             f'{" · selected" if selected else ""}</th><td>{percentage(values.get("success_rate"))}</td>'
                             f'<td>{number(values.get("spl"))}</td><td>{percentage(values.get("collision_rate"))}</td>'
                             f'<td>{text(entry.get("updates", entry.get("training_updates", "—")))}</td></tr>')

    hashes = [('Initial candidate', result.get('initial_model_hash', 'unavailable')),
              ('Selected candidate', selected_hash),
              ('Captured baseline', result.get('baseline_model_hash', baseline.get('model_hash', 'unavailable'))),
              ('Source graph', candidate.get('graph_hash', baseline.get('graph_hash', 'unavailable'))),
              ('Baseline test protocol', baseline.get('protocol_id', 'unavailable')),
              ('Candidate test protocol', candidate.get('protocol_id', 'unavailable'))]
    hash_rows = ''.join(f'<dt>{text(label)}</dt><dd>{text(value)}</dd>' for label, value in hashes)
    config = text(json.dumps(result.get('config', {}), indent=2, allow_nan=False))
    download = escape('data:application/json;charset=utf-8,' + quote(raw, safe=''), quote=True)
    content = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEUROGRAFT · Candidate navigation experiment</title><style>
:root{font:16px/1.6 system-ui,sans-serif;color:#dceaf2;background:#0b1420;color-scheme:dark}*{box-sizing:border-box}body{max-width:1200px;margin:auto;padding:40px 24px}h1{font-size:clamp(1.8rem,5vw,3rem);line-height:1.2;margin:10px 0}h2{font-size:1.3rem;margin:30px 0 12px}.eyebrow{color:#67dfc3;font-size:.75rem;letter-spacing:.12em;text-transform:uppercase}p{color:#a9bfce}.notice{border:1px solid #76623e;background:#302a1d;border-radius:10px;padding:18px;color:#ead4a8}.notice strong{display:block;color:#f6dfb0}.panel{background:#111f2e;border:1px solid #2a4054;border-radius:10px;padding:20px}.scroll{overflow-x:auto;border:1px solid #2a4054;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:.86rem;white-space:nowrap}th,td{padding:11px 13px;text-align:right;border-bottom:1px solid #2a4054}th:first-child{text-align:left}thead th{color:#9fbdcf;background:#152839}thead tr:first-child th{color:#76e6cc}.selected{background:#153b37}.selected th{color:#78ead4}a{display:inline-block;color:#0b3028;background:#70e0c3;padding:9px 16px;text-decoration:none;border-radius:7px;font-weight:600}pre,dd{overflow-wrap:anywhere;white-space:pre-wrap;font:.8rem/1.7 Consolas,monospace}dl{display:grid;grid-template-columns:180px minmax(0,1fr);gap:8px 18px}dt{color:#a6bdcb;font-size:.85rem}dd{margin:0;color:#c3dedb}summary{cursor:pointer;color:#8ee7d1}footer{margin-top:32px;color:#86a1b6;font-size:.8rem}.scope{line-height:1.8}@media(max-width:650px){body{padding:24px 14px}.panel{padding:14px}dl{grid-template-columns:1fr;gap:2px}dd{margin-bottom:12px}}
</style></head><body><header><span class="eyebrow">NEUROGRAFT · Isolated candidate experiment</span><h1>Navigation training &amp; frozen trials</h1><p>A captured baseline, a separately trained candidate, and a recorded selection rule.</p></header>'''
    content += f'<section class="notice"><strong>{text(promotion)}</strong>Selected round: {text(selected_round)}. Candidate artifacts are separate from the active live session.</section>'
    content += f'<p class="scope">{text(scope)}</p><p>{text(protocol_notice)}</p>'
    content += '<h2>Frozen test comparison</h2><p>Teacher assistance and learning are disabled during these trials. Inference controls share weights within each checkpoint; they are not separately trained baselines. SPL scores failed trials as zero; collision rate counts blocked action attempts.</p>'
    content += '<div class="scroll"><table><caption class="eyebrow">Unchanged baseline versus selected candidate</caption><thead><tr><th scope="col" rowspan="2">Variant</th><th scope="colgroup" colspan="5">Captured baseline</th><th scope="colgroup" colspan="5">Selected candidate</th></tr><tr>'
    content += '<th scope="col">Trials</th><th scope="col">Success</th><th scope="col">SPL</th><th scope="col">Collisions</th><th scope="col">Mean steps</th>' * 2
    content += '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    content += '<h2>Validation progression</h2><p>Checkpoint selection uses validation results. Test results above describe the selected checkpoint; they must not be reused to choose a round.</p><div class="scroll"><table><thead><tr><th scope="col">Training round</th><th scope="col">Success</th><th scope="col">SPL</th><th scope="col">Collisions</th><th scope="col">Updates</th></tr></thead><tbody>'
    content += ''.join(progress_rows) + '</tbody></table></div>'
    training_rows = []
    covered = set()
    for entry in result.get('progress', []):
        collection = entry.get('collection', {})
        replay = entry.get('replay', {})
        training = entry.get('training', {})
        if not collection:
            continue
        for condition in collection.get('condition_counts', []):
            covered.add((condition['size'], condition['preset']))
        cells = [text(entry.get('round')), text(collection.get('transitions', '—')),
                 text(replay.get('retained_total', '—')), text(replay.get('seen_total', '—')),
                 number(training.get('mean_loss')), number(training.get('final_loss'))]
        training_rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    if training_rows:
        content += '<h2>Training coverage &amp; replay</h2><p>Finite action-stratified reservoirs retain expert-labelled visits across worlds and rounds. Loss is imitation cross-entropy; it is not a navigation success metric.</p>'
        content += f'<p>{len(covered)} size/topology combinations collected. Retained and lifetime samples describe training memory, not independent trials.</p>'
        content += '<div class="scroll"><table><thead><tr><th>Round</th><th>Collected</th><th>Retained</th><th>Lifetime samples</th><th>Mean loss</th><th>Final loss</th></tr></thead><tbody>' + ''.join(training_rows) + '</tbody></table></div>'
    content += '<h2>Protocol &amp; artifact identities</h2><div class="panel"><dl>' + hash_rows + '</dl><details><summary>Curriculum and split configuration</summary><pre>' + config + '</pre></details></div>'
    content += f'<h2>Complete evidence receipt</h2><p><a href="{download}" download="neurograft-navigation-experiment.json">Download raw JSON</a></p><details class="panel"><summary>Inspect full report</summary><pre>{text(raw)}</pre></details>'
    content += '<footer>Generated locally. Synthetic navigation outcomes do not establish biological-connectome advantage, independent replication, or pretrained language ability.</footer></body></html>'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(content)
    return path


def write_navigation_evaluation_report(report: dict, output_path: str | Path) -> Path:
    """Standalone fixed-weight controller comparison, separate from training reports."""
    path = Path(output_path)
    if path.exists():
        raise ValueError(f'output already exists: {path}')
    raw = json.dumps(report, indent=2, allow_nan=False)
    text = lambda value: escape(str(value))
    def number(value, percent=False):
        if type(value) not in (int, float):
            return '—'
        return f'{value:.1%}' if percent else f'{value:.3f}'
    labels = {'policy': 'Raw policy', 'masked_policy': 'Policy + legal mask',
              'observed_map': 'Observed map + policy', 'map_without_policy': 'Observed map only',
              'no_graft': 'Without grafts', 'no_recurrence': 'Without recurrence',
              'random': 'Random actions', 'shortest_path': 'Full-map shortest-path oracle'}
    rows = []
    for name, label in labels.items():
        values = report.get('variants', {}).get(name, {}).get('summary')
        if values is None:
            continue
        cells = [text(label), text(values.get('episodes', '—')), number(values.get('success_rate'), True),
                 number(values.get('spl')), number(values.get('collision_rate'), True),
                 number(values.get('mean_steps')), number(values.get('proposed_wall_rate'), True),
                 number(values.get('policy_override_rate'), True)]
        rows.append('<tr><th scope="row">' + cells[0] + '</th>' + ''.join('<td>' + c + '</td>' for c in cells[1:]) + '</tr>')
    identities = ''.join('<dt>' + text(label) + '</dt><dd>' + text(value) + '</dd>' for label, value in (
        ('Model SHA-256', report.get('model_hash', 'unavailable')),
        ('Graph SHA-256', report.get('graph_hash', 'unavailable')),
        ('Study / protocol identity', report.get('study_id', report.get('protocol_id', 'unavailable')))))
    download = escape('data:application/json;charset=utf-8,' + quote(raw, safe=''), quote=True)
    content = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NEUROGRAFT · Observation-only navigation</title><style>
:root{font:16px/1.6 system-ui,sans-serif;color:#dceaf2;background:#0b1420;color-scheme:dark}*{box-sizing:border-box}body{max-width:1200px;margin:auto;padding:36px 24px}h1{font-size:clamp(2rem,5vw,3.3rem);line-height:1.15}h2{margin-top:32px;font-size:1.3rem}.eyebrow{color:#70e0c3;letter-spacing:.12em;font-size:.75rem}p{color:#afc4d2}.notice{padding:18px;border:1px solid #7c6944;border-radius:10px;background:#30291e;color:#f1d9ab}.scroll{overflow:auto;border:1px solid #294658;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:.85rem;white-space:nowrap}th,td{padding:12px;border-bottom:1px solid #294658;text-align:right}th:first-child{text-align:left}thead{background:#152a3a;color:#8de5d1}dl{display:grid;grid-template-columns:190px minmax(0,1fr);gap:10px}dd{margin:0;overflow-wrap:anywhere;color:#a9d9cb}dt{color:#9db5c8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem}details{margin:16px 0;padding:18px;border:1px solid #294658;border-radius:10px}summary{cursor:pointer;color:#8de5d1}a{display:inline-block;padding:10px 16px;background:#70e0c3;color:#0b3028;text-decoration:none;border-radius:7px;font-weight:600}@media(max-width:650px){body{padding:24px 14px}dl{grid-template-columns:minmax(0,1fr)}dd{margin-bottom:12px}}
</style></head><body><span class="eyebrow">NEUROGRAFT · FIXED WEIGHTS · MATCHED WORLDS</span><h1>What makes navigation work?</h1><p>Separate raw policy behavior, collision filtering and episodic observed-map planning.</p>'''
    content += '<div class="notice">Masking and mapping are engineered assistance. They do not establish stronger raw neural navigation. No model training or promotion occurs in these trials.</div>'
    content += '<h2>Matched controller comparison</h2><p>All variants share worlds, goals and step budgets. Map controllers receive only public grid size, current position/goal and adjacent-wall sensors. The map-only control removes learned route tie-breaking. The oracle has privileged full-map access.</p>'
    content += '<div class="scroll"><table><thead><tr><th>Variant</th><th>Worlds</th><th>Success</th><th>SPL</th><th>Collisions</th><th>Mean steps</th><th>Proposed walls</th><th>Overrides</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    content += '<p>Proposed walls counts blocked raw-policy suggestions; overrides counts decisions changed by a controller. Zero executed collisions follow legal filtering. Failed episodes score zero SPL. These bounded synthetic trials are exploratory.</p>'
    content += '<h2>Evidence scope</h2><p>' + text(report.get('evidence_scope', 'Synthetic grid task; inspect complete receipts.')) + '</p>'
    content += '<h2>Captured identities</h2><dl>' + identities + '</dl><details><summary>Declared scenarios</summary><pre>' + text(json.dumps(report.get('config', {}), indent=2)) + '</pre></details>'
    content += f'<p><a href="{download}" download="neurograft-controller-comparison.json">Download complete JSON</a></p><details><summary>Inspect episodes, actions and controller traces</summary><pre>{text(raw)}</pre></details></body></html>'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(content)
    return path
