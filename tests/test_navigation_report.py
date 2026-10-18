import json
from html.parser import HTMLParser

import pytest


def experiment():
    def report(success, spl):
        return {'protocol_id': 'p' * 64, 'model_hash': 'a' * 64,
                'graph_hash': 'b' * 64, 'config': {'seeds': [30000, 30001], 'size': 5, 'preset': 'labyrinth', 'max_steps': 16},
                'variants': {'policy': {'summary': {'episodes': 2, 'success_rate': success,
                    'spl': spl, 'collision_rate': .125, 'mean_steps': 12, 'teacher_agreement': .5}},
                    'random': {'summary': {'episodes': 2, 'success_rate': 0, 'spl': 0,
                        'collision_rate': .25, 'mean_steps': 16, 'teacher_agreement': .25}}}}
    return {'config': {'seed': 17, 'rounds': 2, 'validation_seeds': [20000], 'test_seeds': [30000, 30001]},
            'initial_model_hash': 'i' * 64, 'selected_model_hash': 's' * 64, 'baseline_model_hash': 'a' * 64,
            'selected_round': 1, 'progress': [{'round': 1, 'validation': {'success_rate': .5, 'spl': .4, 'collision_rate': .2}},
                {'round': 2, 'validation': {'success_rate': .25, 'spl': .2, 'collision_rate': .3}}],
            'baseline': report(0, 0), 'candidate': report(.5, .4), 'promotion_performed': False,
            'evidence_scope': 'Synthetic candidate training; no biological advantage established.'}


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_data(self, data):
        self.values.append(data)


def test_navigation_report_compares_baseline_and_candidate_and_shows_selection_boundary(tmp_path):
    from neurograft.navigation_report import write_navigation_report
    result = experiment()
    destination = tmp_path / 'report.html'
    assert write_navigation_report(result, destination) == destination
    content = destination.read_text(encoding='utf-8')
    text = VisibleText()
    text.feed(content)
    visible = ' '.join(text.values)
    assert '50.0%' in visible and '0.400' in visible
    assert 'Round 1' in visible and 'Round 2' in visible
    assert 'Selected round: 1' in visible
    assert 'No promotion performed' in visible
    assert 'Inference controls share weights' in visible
    assert 's' * 64 in visible and 'a' * 64 in visible and 'p' * 64 in visible
    assert result['evidence_scope'] in visible


def test_navigation_report_escapes_untrusted_content_and_contains_no_remote_resources(tmp_path):
    from neurograft.navigation_report import write_navigation_report
    result = experiment()
    result['evidence_scope'] = '</script><img src=x onerror=alert(1)>'
    result['config']['label'] = '<script>alert(2)</script>'
    destination = write_navigation_report(result, tmp_path / 'report.html')
    content = destination.read_text(encoding='utf-8')
    assert '<img src=x' not in content and '<script>alert(2)' not in content
    assert '&lt;img src=x' in content
    assert 'src="https://' not in content and 'href="https://' not in content
    assert 'onerror=' not in content.replace('&lt;img src=x onerror=alert(1)&gt;', '')


def test_navigation_report_does_not_replace_existing_output_or_accept_nonfinite_numbers(tmp_path):
    from neurograft.navigation_report import write_navigation_report
    destination = tmp_path / 'report.html'
    destination.write_text('retained evidence', encoding='utf-8')
    with pytest.raises(ValueError, match='exists'):
        write_navigation_report(experiment(), destination)
    assert destination.read_text(encoding='utf-8') == 'retained evidence'
    result = experiment()
    result['candidate']['variants']['policy']['summary']['spl'] = float('nan')
    with pytest.raises(ValueError):
        write_navigation_report(result, tmp_path / 'invalid.html')
    assert not (tmp_path / 'invalid.html').exists()


def test_train_navigation_cli_exposes_isolated_candidate_inputs_and_config():
    from neurograft.cli import parser
    args = parser().parse_args(['train-navigation', '--bundle', 'graph.json', '--checkpoint', 'capture.pt',
        '--out', 'candidate-run', '--config', 'curriculum.json', '--rounds', '2', '--worlds', '4',
        '--updates', '8', '--seed', '23', '--lr', '.001', '--init', 'checkpoint'])
    assert args.command == 'train-navigation' and args.out == 'candidate-run'
    assert args.rounds == 2 and args.worlds_per_round == 4 and args.updates_per_round == 8
    assert args.seed == 23 and args.learning_rate == .001 and args.init == 'checkpoint'
