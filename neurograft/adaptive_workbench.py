"""Version-two workbench; the historical v1 engine and receipts stay intact."""
from __future__ import annotations

from .adaptive_memory import AdaptiveTransitionMemory
from .workbench import ReplayWorkbench


class AdaptiveReplayWorkbench(ReplayWorkbench):
    """Three evidence policies crossed with three independent replay schedules."""

    VERSION = 2
    MODELS = ('cumulative', 'recent', 'adaptive')
    EVIDENCE_SCOPE = (
        ReplayWorkbench.EVIDENCE_SCOPE + ' Adaptive evidence selects between cumulative and recent '
        'observed counts using discounted pre-update predictive log scores. The score is a heuristic, '
        'not a posterior, confidence level, or certified change detector. Binary forecasts use '
        'Beta(0.5,0.5) smoothing; planning uses unsmoothed actual counts. Identical observed '
        'prefixes cannot distinguish a real closure from an unlucky stationary run.'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evidence = AdaptiveTransitionMemory(self._models['cumulative'], window=self.window)

    def _execute(self, command):
        # The parent validates command-wide capacity and updates both v1 models.
        # This object is already a detached candidate under inherited apply().
        super()._execute(command)
        if command['action'] == 'observe':
            for _ in range(command['count']):
                self._evidence.observe(command['source'], command['action_index'], command['destination'])
            self._models['adaptive'] = self._evidence.graph

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot['adaptive_evidence'] = self._evidence.diagnostics()
        return snapshot
