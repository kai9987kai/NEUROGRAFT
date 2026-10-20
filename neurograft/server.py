"""Loopback HTTP API and offline assets for the NEUROGRAFT live workspace."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from urllib.parse import unquote, urlsplit

from . import __version__
from .service import ConflictError, LiveApp


ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/index.html': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8'),
          '/vendor/three.module.js': ('vendor/three.module.js', 'text/javascript; charset=utf-8'),
          '/vendor/three.core.js': ('vendor/three.core.js', 'text/javascript; charset=utf-8'),
          '/vendor/three.LICENSE.txt': ('vendor/three.LICENSE.txt', 'text/plain; charset=utf-8')}


class LiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(app: LiveApp, host='127.0.0.1', port=8768, *, ready=None):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('live service binds only to loopback')

    class Handler(BaseHTTPRequestHandler):
        server_version = f'NEUROGRAFT/{__version__}'

        def log_message(self, format, *args):
            return  # Keep continuous polling quiet; errors are returned visibly.

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def send_data(self, code, payload, kind='application/json; charset=utf-8'):
            raw = json.dumps(payload, allow_nan=False).encode('utf-8') if isinstance(payload, (dict, list)) else payload
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return  # Reloads and closed browser tabs can cancel in-flight polls.

        def origin_ok(self):
            port = self.server.server_address[1]
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            if self.headers.get('Host') not in hosts:
                self.send_data(403, {'error': 'loopback Host required'})
                return False
            origin = self.headers.get('Origin')
            if origin is not None and origin not in {f'http://{host}' for host in hosts}:
                self.send_data(403, {'error': 'same-origin requests required'})
                return False
            return True

        def do_GET(self):
            if not self.origin_ok():
                return
            path = unquote(urlsplit(self.path).path)
            if path == '/api/state':
                self.send_data(200, app.state())
            elif path == '/api/checkpoints':
                self.send_data(200, app.checkpoints())
            elif path == '/api/health':
                health = app.health()
                self.send_data(200 if health['ok'] else 503, health)
            elif path == '/api/evaluation':
                report = app.evaluation()
                self.send_data(200, b'null' if report is None else report)
            elif path == '/favicon.ico':
                self.send_data(204, b'')
            elif path in ('/report.html', '/v0.1-report.html'):
                from pathlib import Path
                report_file = Path('reports/index.html')
                if report_file.is_file():
                    self.send_data(200, report_file.read_bytes(), 'text/html; charset=utf-8')
                else:
                    self.send_data(404, {'error': 'report file not found'})
            elif path in ASSETS:
                relative, kind = ASSETS[path]
                try:
                    self.send_data(200, files('neurograft').joinpath('web', relative).read_bytes(), kind)
                except FileNotFoundError:
                    self.send_data(404, {'error': 'UI asset is unavailable'})
            else:
                self.send_data(404, {'error': 'not found'})

        def do_POST(self):
            if not self.origin_ok():
                return
            if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                self.send_data(415, {'error': 'application/json required'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 65536:
                    self.send_data(413, {'error': 'JSON body must contain 1-65536 bytes'})
                    return
                def reject_constant(value):
                    raise ValueError('nonfinite JSON numbers are invalid')
                payload = json.loads(self.rfile.read(length).decode('utf-8'), parse_constant=reject_constant)
                if not isinstance(payload, dict):
                    raise ValueError('JSON body must be an object')
                # Consume the bounded request before closing on a startup
                # rejection. Unread body bytes can cause a TCP reset that hides
                # the 503 response from the client (RFC 9112 section 9.6).
                if ready is not None and not ready.is_set():
                    self.send_data(503, {'error': 'live session startup is not complete'})
                    return
                path = unquote(urlsplit(self.path).path)
                if path == '/api/control':
                    result = app.control(payload)
                elif path == '/api/chat':
                    result = app.chat(payload.get('message'))
                elif path == '/api/save':
                    result = app.save(payload.get('name', 'default'))
                elif path == '/api/load':
                    result = app.load(payload.get('name', 'default'))
                elif path == '/api/evaluate':
                    result = app.evaluate(payload)
                else:
                    self.send_data(404, {'error': 'not found'})
                    return
                self.send_data(200, result)
            except ConflictError as exc:
                self.send_data(409, {'error': str(exc)})
            except FileNotFoundError as exc:
                self.send_data(404, {'error': str(exc)})
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                self.send_data(400, {'error': str(exc)})
            except (OSError, RuntimeError) as exc:
                self.send_data(500, {'error': str(exc)})

    return LiveHTTPServer((host, port), Handler)


def serve_live(bundle, state_path, port=8768, interval=.2, autosave=5):
    import threading
    app = None
    server = None
    mirror = None
    mirror_thread = None
    started = False
    ready = threading.Event()
    try:
        # Binding must succeed before a worker can advance or persist the session.
        app = LiveApp(bundle, state_path, interval=interval, autosave=autosave, background=False)
        server = create_server(app, port=port, ready=ready)
        if port != 8767:
            try:
                mirror = create_server(app, port=8767, ready=ready)
            except OSError:
                mirror = None
        # Restored running/learning flags are authoritative. New sessions already
        # default to both enabled; only new sessions need an initial checkpoint.
        if not app.state_path.exists():
            app.save('default')
        if mirror is not None:
            mirror_thread = threading.Thread(target=mirror.serve_forever,
                                             kwargs={'poll_interval': .2}, daemon=True)
            mirror_thread.start()
        # No fallible setup may follow worker startup before the transaction
        # commits: even a delayed mirror failure must leave saved state intact.
        app.start_background()
        started = True
        ready.set()
        if mirror is not None:
            print('NEUROGRAFT Live (Web Preview mirror): http://127.0.0.1:8767/index.html', flush=True)
        print(f'NEUROGRAFT Live: http://127.0.0.1:{server.server_address[1]}/', flush=True)
        print(f'Session: {app.state_path} | autosave every {autosave:g}s | Ctrl+C saves and stops.', flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        print('\nSaving full live state and stopping.' if started
              else '\nStartup interrupted; the session was not advanced.', flush=True)
    finally:
        ready.clear()
        try:
            if mirror is not None:
                try:
                    if mirror_thread is not None and mirror_thread.is_alive():
                        mirror.shutdown()
                        mirror_thread.join(3)
                finally:
                    mirror.server_close()
        finally:
            try:
                if server is not None:
                    server.server_close()
            finally:
                if app is not None:
                    app.close(save=started)
