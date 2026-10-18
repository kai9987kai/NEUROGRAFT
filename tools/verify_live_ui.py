"""Run browser regression checks against an isolated, temporary fixture session.

No existing server or session file is opened. Supply an installed Playwright
module directory and Chromium executable; this tool never downloads a browser.
"""
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
    parser.add_argument('--out', default='output/playwright')
    args = parser.parse_args()
    from neurograft.fixtures import make_fixture_bundle
    from neurograft.live import LiveConfig
    from neurograft.server import create_server
    from neurograft.service import LiveApp
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='isolated-live-', dir=out) as directory:
        app = LiveApp(make_fixture_bundle(nodes_per_source=4), Path(directory) / 'session.pt',
                      background=False, config=LiveConfig(hidden_dim=4, batch_size=2))
        server = None
        try:
            app.control({'action': 'pause'})
            app.control({'action': 'step', 'steps': 8})
            server = create_server(app, port=0)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            env = dict(os.environ, NODE_PATH=str(Path(args.node_modules).resolve()))
            print(f'Isolated browser server: http://127.0.0.1:{server.server_port}', flush=True)
            try:
                result = subprocess.run(['node', str(Path(__file__).with_name('verify_live_ui.cjs')),
                                         f'http://127.0.0.1:{server.server_port}', args.chromium, str(out)],
                                        env=env, text=True, capture_output=True, timeout=120)
            except subprocess.TimeoutExpired as exc:
                for output in (exc.stdout, exc.stderr):
                    if output:
                        print(output.decode(errors='replace') if isinstance(output, bytes) else output, flush=True)
                raise RuntimeError(f'browser verification exceeded 120s; inspect {out / "browser-stages.log"}') from exc
            if result.stdout:
                print(result.stdout, end='')
            if result.returncode:
                raise RuntimeError(result.stderr or 'browser verification failed')
            report = json.loads((out / 'browser-verification.json').read_text(encoding='utf-8'))
            if not report['ok']:
                raise RuntimeError('browser verification reported errors')
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            app.close()


if __name__ == '__main__':
    main()
