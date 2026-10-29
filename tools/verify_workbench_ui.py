"""Exercise the actual workbench UI against an isolated live fixture."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-modules', required=True)
    parser.add_argument('--chromium', required=True)
    parser.add_argument('--out', default='output/workbench-ui')
    args = parser.parse_args()
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig, _checkpoint_digest
    from neurograft.server import create_server
    from neurograft.service import LiveApp
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='workbench-browser-', dir=out) as temp:
        app = LiveApp(make_fixture_bundle(nodes_per_source=3), Path(temp) / 'session.pt',
                      background=False, config=LiveConfig(hidden_dim=4, unroll=2))
        app.session.running = False
        app.session.tick(steps=8)
        before = _checkpoint_digest(app.session._payload())
        server = create_server(app, port=0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            env = dict(os.environ, NODE_PATH=str(Path(args.node_modules).resolve()))
            result = subprocess.run(['node', str(Path(__file__).with_suffix('.cjs')),
                                     f'http://127.0.0.1:{server.server_port}', args.chromium, str(out)],
                                    env=env, text=True, capture_output=True, timeout=120)
            print(result.stdout, end='')
            if result.returncode:
                raise RuntimeError(result.stderr or 'workbench browser checks failed')
            assert before == _checkpoint_digest(app.session._payload()), 'workbench changed the live session'
            receipt = json.loads((out / 'workbench-browser.json').read_text())
            receipt['live_session_unchanged'] = True
            (out / 'workbench-browser.json').write_text(json.dumps(receipt, indent=2) + '\n')
        finally:
            server.shutdown()
            server.server_close()
            worker.join(3)
            app.close(save=False)


if __name__ == '__main__':
    main()
