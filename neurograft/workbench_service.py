"""Independent workbench ownership and a fixture-only launch mode."""
from __future__ import annotations

import threading

from .memory import TopologicalMemoryGraph
from .workbench import ReplayWorkbench


class WorkbenchStore:
    def __init__(self, seed=0, capture=None):
        self.lock = threading.RLock()
        self.current = ReplayWorkbench(seed=seed)
        self.capture = capture

    def snapshot(self):
        with self.lock:
            return self.current.snapshot()

    def receipt(self):
        with self.lock:
            return self.current.export_receipt()

    @staticmethod
    def _fields(payload, allowed):
        if type(payload) is not dict or set(payload) - set(allowed):
            raise ValueError('invalid workbench command fields')

    def control(self, payload):
        if type(payload) is not dict:
            raise ValueError('workbench command must be an object')
        action = payload.get('action')
        if action == 'fixture':
            self._fields(payload, ('action', 'seed', 'window'))
            candidate = ReplayWorkbench(seed=payload.get('seed', 0), window=payload.get('window', 8))
        elif action == 'capture':
            self._fields(payload, ('action', 'window'))
            if self.capture is None:
                raise ValueError('Live capture requires the main live server; this tool uses an isolated fixture.')
            # Copy live data first; release the session lock before model/oracle work.
            graph, goal, provenance = self.capture()
            candidate = ReplayWorkbench(TopologicalMemoryGraph.from_state(graph), tuple(goal),
                                        provenance=provenance, window=payload.get('window', 8))
        elif action == 'import':
            self._fields(payload, ('action', 'receipt'))
            candidate = ReplayWorkbench.from_receipt(payload.get('receipt'))
        else:
            with self.lock:
                return self.current.apply(payload)
        # Failed candidate construction/verification never replaces the current lab.
        with self.lock:
            self.current = candidate
            return self.current.snapshot()


class FixtureWorkbenchApp:
    """HTTP adapter with no neural model, live worker, session or checkpoint path."""
    workbench_only = True

    def __init__(self, seed=0):
        self.store = WorkbenchStore(seed=seed)

    def workbench_state(self):
        return self.store.snapshot()

    def workbench_control(self, payload):
        return self.store.control(payload)

    def workbench_receipt(self):
        return self.store.receipt()

    def health(self):
        return {'ok': True, 'mode': 'fixture_workbench', 'live_session_opened': False}


def serve_workbench(port=8770, seed=0):
    from .server import create_server
    app = FixtureWorkbenchApp(seed=seed)
    server = create_server(app, port=port)
    print(f'NEUROGRAFT Replay Workbench: http://127.0.0.1:{server.server_port}/workbench', flush=True)
    print('Isolated fixture; no live checkpoint opened. Ctrl+C stops the tool.', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
