"""Single-writer live-session ownership, periodic learning and checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import re
import sys
import threading
import time


class ConflictError(ValueError):
    """An action conflicts with the current running state."""


class SessionLease:
    """OS-released exclusive lock; a stale lock file is not a stale lease."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open('a+b')
        if self.handle.seek(0, 2) == 0:
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError('another writer already has this live session locked') from exc

    def close(self):
        if not self.handle.closed:
            self.handle.close()  # Closing releases the OS lock on all platforms.


def _bounded_int(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{label} must be an integer between {minimum} and {maximum}')
    return value


class LiveApp:
    def __init__(self, bundle, state_path, interval=.25, autosave=15, background=True, config=None):
        if not isinstance(interval, (float, int)) or not math.isfinite(interval) or not .05 <= interval <= 10:
            raise ValueError('interval must be from 0.05 to 10 seconds')
        if not isinstance(autosave, (float, int)) or not math.isfinite(autosave) or not 1 <= autosave <= 3600:
            raise ValueError('autosave must be from 1 to 3600 seconds')
        if sum(len(g.nodes) for g in bundle.graphs) > 2000 or sum(len(g.edges) for g in bundle.graphs) > 50000:
            raise ValueError('live CPU session cap is 2,000 nodes and 50,000 observations')
        self.bundle = bundle
        self.state_path = Path(state_path).resolve()
        self.slots = self.state_path.parent / 'checkpoints'
        self.interval, self.autosave = float(interval), float(autosave)
        self.guard = threading.RLock()
        self.evaluation_guard = threading.Lock()
        self.workbench_guard = threading.Lock()
        self._workbench_store = None
        self.latest_evaluation = None
        self.stop_event = threading.Event()
        self.last_error = None
        self.worker = None
        self.closed = False
        self.lease = SessionLease(self.state_path.with_suffix(self.state_path.suffix + '.lock'))
        try:
            from .live import LiveSession
            self.session = (LiveSession.load(self.state_path, bundle) if self.state_path.exists()
                            else LiveSession(bundle, config))
            self.last_autosave = time.monotonic()
            if background:
                self.start_background()
        except BaseException:
            self.lease.close()
            raise

    def start_background(self):
        """Start the single worker after callers have acquired their other resources."""
        with self.guard:
            if self.closed:
                raise RuntimeError('cannot start a closed live session')
            if self.worker is not None:
                if self.worker.is_alive():
                    return
                raise RuntimeError('live background worker is no longer running')
            self.worker = threading.Thread(target=self._loop, name='neurograft-learning', daemon=True)
            try:
                self.worker.start()
            except BaseException:
                self.worker = None
                raise

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                with self.guard:
                    if self.session.running:
                        self.session.tick()
                    if time.monotonic() - self.last_autosave >= self.autosave:
                        self.save('default')
            except Exception as exc:
                with self.guard, self.session.lock:
                    self.last_error = str(exc)
                    self.session.running = False

    def _slot(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name):
            raise ValueError('checkpoint name must be 1-40 letters, numbers, underscores or hyphens')
        if name == 'default':
            return self.state_path
        candidate = self.slots / f'{name}.pt'
        if not candidate.resolve().is_relative_to(self.slots.resolve()):
            raise ValueError('checkpoint path leaves its slot directory')
        return candidate

    def checkpoints(self):
        result = []
        paths = [('default', self.state_path)]
        if self.slots.exists():
            paths += [(p.stem, p) for p in sorted(self.slots.glob('*.pt'))
                      if re.fullmatch(r'[A-Za-z0-9_-]{1,40}', p.stem)]
        for name, path in paths:
            if path.exists() and (name == 'default' or path.resolve().is_relative_to(self.slots.resolve())):
                info = path.stat()
                result.append({'name': name, 'bytes': info.st_size,
                               'modified': datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat()})
        return result

    def state(self):
        with self.guard:
            return {**self.session.snapshot(), 'telemetry': self.session.telemetry(),
                    'checkpoints': self.checkpoints(),
                    'runtime': {'autosave_seconds': self.autosave, 'interval_seconds': self.interval,
                                'last_error': self.last_error,
                                'checkpoint_compatibility': self.session.checkpoint_compatibility}}

    def health(self):
        with self.guard:
            worker_alive = bool(self.worker and self.worker.is_alive())
            return {'ok': not self.closed and self.last_error is None and (self.worker is None or worker_alive),
                    'running': self.session.running, 'learning': self.session.learning,
                    'worker': {'enabled': self.worker is not None, 'alive': worker_alive},
                    'last_error': self.last_error, 'closed': self.closed,
                    'evaluation_running': self.evaluation_guard.locked()}

    def _capture_workbench(self):
        from .live import _checkpoint_digest
        with self.guard, self.session.lock:
            return (self.session.memory.topological_graph.to_state(), tuple(self.session.world._goal),
                    {'kind': 'live_capture', 'source_steps': self.session.steps,
                     'source_model_sha256': _checkpoint_digest(self.session.model.state_dict()),
                     'source_bundle_sha256': self.session.bundle.digest()})

    def _workbench(self):
        from .workbench_service import WorkbenchStore
        with self.workbench_guard:
            if self._workbench_store is None:
                self._workbench_store = WorkbenchStore(capture=self._capture_workbench)
            return self._workbench_store

    def workbench_state(self):
        return self._workbench().snapshot()

    def workbench_control(self, payload):
        return self._workbench().control(payload)

    def workbench_receipt(self):
        return self._workbench().receipt()

    def evaluation(self):
        with self.guard:
            return self.latest_evaluation

    def evaluate(self, values):
        from .navigation_evaluation import NavigationEvalConfig, frozen_navigation_evaluation
        if not isinstance(values, dict) or set(values) - {'seeds', 'size', 'preset', 'max_steps', 'include_controllers', 'include_steering_controls'}:
            raise ValueError('evaluation accepts only seeds, size, preset, max_steps, include_controllers and include_steering_controls')
        seeds = values.get('seeds', [10000, 10001, 10002])
        if not isinstance(seeds, list):
            raise ValueError('evaluation seeds must be a JSON array of integers')
        config = NavigationEvalConfig(seeds=tuple(seeds), size=values.get('size', 11),
                                      preset=values.get('preset', 'labyrinth'),
                                      max_steps=values.get('max_steps', 48),
                                      include_controllers=values.get('include_controllers', False),
                                      include_steering_controls=values.get('include_steering_controls', False))
        if not self.evaluation_guard.acquire(blocking=False):
            raise ConflictError('a frozen navigation evaluation is already running')
        try:
            with self.guard:
                candidate = self.session
            report = frozen_navigation_evaluation(candidate, config)
            with self.guard:
                self.latest_evaluation = report
            return report
        finally:
            self.evaluation_guard.release()

    def save(self, name='default'):
        path = self._slot(name)
        with self.guard:
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt = self.session.save(path)
            if name == 'default':
                self.last_autosave = time.monotonic()
            return {'saved': {'name': name, **(receipt if isinstance(receipt, dict) else {})},
                    'state': self.state()}

    def load(self, name='default'):
        from .live import LiveSession
        path = self._slot(name)
        with self.guard:
            if not path.is_file():
                raise FileNotFoundError(f'checkpoint does not exist: {name}')
            candidate = LiveSession.load(path, self.bundle)
            self.session = candidate
            self.last_error = None
            return self.state()

    def control(self, values):
        if not isinstance(values, dict):
            raise ValueError('control payload must be an object')
        action = values.get('action')
        # Validate inputs before mutating the session.
        if action in ('step', 'move'):
            count = _bounded_int(values.get('steps', 1), 'steps', 1, 32)
            if action == 'move':
                direction = _bounded_int(values.get('direction'), 'direction', 0, 3)
        elif action == 'boost':
            count = _bounded_int(values.get('steps', 16), 'steps', 1, 64)
        elif action == 'replay':
            sweeps = _bounded_int(values.get('sweeps', 4), 'sweeps', 1, 16)
        elif action == 'world':
            from .environment import NavigationWorld
            wsize = _bounded_int(values.get('size', 15), 'size', 5, 31)
            wpreset = values.get('preset', 'labyrinth')
            if not isinstance(wpreset, str) or wpreset not in NavigationWorld.PRESETS:
                raise ValueError('invalid world preset')
        elif action == 'memory_lesion':
            mmode = values.get('mode', 'intact')
            if mmode not in ('intact', 'erased', 'shifted', 'random_retrieval', 'memory_free'):
                raise ValueError('invalid memory lesion mode')
        elif action == 'learning':
            if not isinstance(values.get('enabled'), bool):
                raise ValueError('enabled must be boolean')
        elif action == 'navigation':
            from .navigation_controller import ALL_MODES, MODES
            mode = values.get('mode')
            if not isinstance(mode, str) or mode not in ALL_MODES:
                raise ValueError('invalid navigation mode')
        elif action not in ('pause', 'resume', 'goal', 'replay'):
            raise ValueError('unknown control action')
        with self.guard, self.session.lock:
            if action == 'pause':
                self.session.running = False
            elif action == 'resume':
                self.last_error = None
                self.session.running = True
            elif action == 'learning':
                self.session.learning = values['enabled']
            elif action == 'navigation':
                self.session.set_navigation_mode(mode)
            elif action == 'boost':
                self.session.tick(steps=count)
            elif action == 'world':
                self.session.resize_world(size=wsize, preset=wpreset)
            elif action == 'memory_lesion':
                self.session.memory.set_lesion_mode(mmode)
            elif action in ('step', 'move'):
                if self.session.running:
                    raise ConflictError('pause the session before stepping or moving manually')
                self.session.tick(steps=count, manual_action=direction if action == 'move' else None)
            elif action == 'goal':
                self.session.set_goal(color=values.get('color'), x=values.get('x'), z=values.get('z'))
            elif action == 'replay':
                replay_summary = (self.session.replay_memory(budget=sweeps * 8)
                                  if self.session.navigation_mode == 'replay_steer'
                                  else self.session.memory.trigger_swr_replay(max_sweeps=sweeps))
                return {**self.state(), 'replay': replay_summary}
            return self.state()

    def chat(self, message):
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= 2000:
            raise ValueError('chat message must contain 1-2000 characters')
        with self.guard:
            result = self.session.chat(message)
            if result.get('action') == 'save':
                self.save('default')
            return {**result, 'state': self.state()}

    def close(self, *, save=True):
        """Stop and release ownership; failed startup can roll back without saving."""
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.worker and self.worker is not threading.current_thread():
            self.worker.join(10)
        try:
            with self.guard:
                if save:
                    self.save('default')
        finally:
            self.lease.close()
