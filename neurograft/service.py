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
        self.stop_event = threading.Event()
        self.last_error = None
        self.worker = None
        self.closed = False
        self.lease = SessionLease(self.state_path.with_suffix(self.state_path.suffix + '.lock'))
        try:
            from .live import LiveSession
            self.session = (LiveSession.load(self.state_path, bundle) if self.state_path.exists()
                            else LiveSession(bundle, config))
        except BaseException:
            self.lease.close()
            raise
        self.last_autosave = time.monotonic()
        if background:
            self.worker = threading.Thread(target=self._loop, name='neurograft-learning', daemon=True)
            self.worker.start()

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
                                'last_error': self.last_error}}

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
        elif action == 'learning':
            if not isinstance(values.get('enabled'), bool):
                raise ValueError('enabled must be boolean')
        elif action not in ('pause', 'resume', 'goal'):
            raise ValueError('unknown control action')
        with self.guard, self.session.lock:
            if action == 'pause':
                self.session.running = False
            elif action == 'resume':
                self.last_error = None
                self.session.running = True
            elif action == 'learning':
                self.session.learning = values['enabled']
            elif action == 'boost':
                self.session.tick(steps=count)
            elif action in ('step', 'move'):
                if self.session.running:
                    raise ConflictError('pause the session before stepping or moving manually')
                self.session.tick(steps=count, manual_action=direction if action == 'move' else None)
            elif action == 'goal':
                self.session.set_goal(color=values.get('color'), x=values.get('x'), z=values.get('z'))
            return self.state()

    def chat(self, message):
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= 2000:
            raise ValueError('chat message must contain 1-2000 characters')
        with self.guard:
            result = self.session.chat(message)
            if result.get('action') == 'save':
                self.save('default')
            return {**result, 'state': self.state()}

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.worker and self.worker is not threading.current_thread():
            self.worker.join(10)
        try:
            with self.guard:
                self.save('default')
        finally:
            self.lease.close()
