"""CLI exposes the causal study and its strict verifier without live state."""
import json

from neurograft.cli import main, parser


def test_adaptation_commands_create_and_replay_fresh_archive(tmp_path, capsys):
    assert parser().parse_args(['study-adaptation', '--out', 'new']).command == 'study-adaptation'
    out = tmp_path / 'adaptation'
    assert main(['study-adaptation', '--out', str(out), '--seeds', '15000', '--budget', '16']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['promotion_performed'] is False
    assert result['summary']
    assert main(['verify-adaptation-study', str(out), '--replay']) == 0
    assert json.loads(capsys.readouterr().out)['deterministic_replay'] is True
    assert not list(tmp_path.rglob('*.pt'))
