"""Causal predictive-score selection between cumulative and recent evidence.

This engineering baseline is neither a posterior nor a calibrated change test.
Forecast smoothing never enters the planner's observed integer-count graph.
Captured aggregate counts have no reconstructed order. Only new events update
the per-action score, and every forecast is scored before incorporating its event.
"""
from __future__ import annotations

from collections import Counter
import copy
import math

from .memory import TopologicalMemoryGraph
from .replay import ReplayPlanner


class AdaptiveTransitionMemory:
    MAX_EVENTS = 2048
    MAX_NODES = 256
    DECAY = .95
    THRESHOLD = math.log(4.)
    MINIMUM_FRESH = 4
    SCORING_PSEUDOCOUNT = .5
    FORECAST_MODELS = ('cumulative', 'recent', 'guarded_recent', 'adaptive')

    def __init__(self, graph: TopologicalMemoryGraph, window: int = 8):
        if type(window) is not int or window not in (4, 8, 16):
            raise ValueError('adaptive window must be 4, 8, or 16')
        ReplayPlanner._model(graph)
        if len(graph.nodes) > self.MAX_NODES:
            raise ValueError('adaptive node limit exceeded')
        state = graph.to_state()
        self._cumulative = TopologicalMemoryGraph.from_state(state)
        self._recent = TopologicalMemoryGraph.from_state(state)
        self._graph = TopologicalMemoryGraph.from_state(state)
        self.window = window
        self.events = 0
        self._windows = {}
        self._pairs = {}
        self._losses = {model: {'brier': 0., 'log_loss': 0.} for model in self.FORECAST_MODELS}

    @property
    def graph(self):
        """A detached graph containing only selected, actually observed counts."""
        return TopologicalMemoryGraph.from_state(self._graph.to_state())

    @property
    def cumulative(self):
        return TopologicalMemoryGraph.from_state(self._cumulative.to_state())

    @property
    def recent(self):
        return TopologicalMemoryGraph.from_state(self._recent.to_state())

    @staticmethod
    def _key(source, action):
        source = TopologicalMemoryGraph._position(source)
        if type(action) is not int or not 0 <= action < 4:
            raise ValueError('invalid adaptive action')
        return source, action

    @classmethod
    def _move_probability(cls, graph, key):
        counts = graph.transitions.get(key, {})
        moved = sum(count for destination, count in counts.items() if destination != key[0])
        return (moved + cls.SCORING_PSEUDOCOUNT) / (sum(counts.values()) + 2 * cls.SCORING_PSEUDOCOUNT)

    def predict(self, source, action):
        """Return pre-update binary move probabilities without mutating evidence."""
        key = self._key(source, action)
        cumulative = self._move_probability(self._cumulative, key)
        recent = self._move_probability(self._recent, key)
        pair = self._pairs.get(key, {})
        return {'cumulative': cumulative, 'recent': recent,
                'guarded_recent': recent if pair.get('fresh_count', 0) >= self.MINIMUM_FRESH else cumulative,
                'adaptive': recent if pair.get('selected') == 'recent' else cumulative}

    @staticmethod
    def _replace_counts(graph, key, counts):
        previous = graph.transitions.get(key, {})
        graph.total_transitions += sum(counts.values()) - sum(previous.values())
        graph.transitions[key] = dict(counts)
        source = key[0]
        graph.adjacency[source] = {destination for action in range(4)
                                   for destination in graph.transitions.get((source, action), {})}

    def observe(self, source, action, destination):
        """Score once, then update this pair. No hidden truth or phase is accepted."""
        source, destination = TopologicalMemoryGraph._transition(source, action, destination)
        key = source, action
        if self.events >= self.MAX_EVENTS:
            raise ValueError('adaptive event limit exceeded')
        if len(self._cumulative.nodes | {source, destination}) > self.MAX_NODES:
            raise ValueError('adaptive node limit exceeded')
        if self._cumulative.total_transitions >= TopologicalMemoryGraph.MAX_TRANSITIONS:
            raise ValueError('adaptive transition count exceeds limit')

        predictions = self.predict(source, action)
        moved = int(destination != source)
        brier = {model: (probability - moved) ** 2 for model, probability in predictions.items()}
        # Compute the event probabilities directly from integer counts. At the
        # largest admissible counts, 1-p_move can round to zero despite smoothing.
        event_probabilities = {}
        for model, graph in (('cumulative', self._cumulative), ('recent', self._recent)):
            counts = graph.transitions.get(key, {})
            matching = sum(count for outcome, count in counts.items() if (outcome != source) == bool(moved))
            event_probabilities[model] = ((matching + self.SCORING_PSEUDOCOUNT)
                                          / (sum(counts.values()) + 2 * self.SCORING_PSEUDOCOUNT))
        previous = self._pairs.get(key, {'fresh_count': 0, 'evidence': 0.,
                                          'selected': 'cumulative', 'switches': 0})
        event_probabilities['guarded_recent'] = event_probabilities[
            'recent' if previous['fresh_count'] >= self.MINIMUM_FRESH else 'cumulative']
        event_probabilities['adaptive'] = event_probabilities[previous['selected']]
        log_loss = {model: -math.log(probability) for model, probability in event_probabilities.items()}
        evidence = (self.DECAY * previous['evidence']
                    + math.log(event_probabilities['recent']) - math.log(event_probabilities['cumulative']))
        fresh_count = previous['fresh_count'] + 1
        selected = ('recent' if fresh_count >= self.MINIMUM_FRESH and evidence > self.THRESHOLD
                    else 'cumulative')
        switched = selected != previous['selected']

        # Everything that can reject an input was checked before any mutation.
        self._cumulative.record_transition(source, action, destination)
        outcomes = (self._windows.get(key, []) + [destination])[-self.window:]
        self._windows[key] = outcomes
        self._recent.add_node(source)
        self._recent.add_node(destination)
        self._replace_counts(self._recent, key, Counter(outcomes))
        self._graph.add_node(source)
        self._graph.add_node(destination)
        selected_graph = self._recent if selected == 'recent' else self._cumulative
        self._replace_counts(self._graph, key, selected_graph.transitions[key])
        self._pairs[key] = {'fresh_count': fresh_count, 'evidence': evidence, 'selected': selected,
                            'switches': previous['switches'] + int(switched)}
        for model in self.FORECAST_MODELS:
            self._losses[model]['brier'] += brier[model]
            self._losses[model]['log_loss'] += log_loss[model]
        self.events += 1
        return {'predictions': predictions, 'brier': brier, 'log_loss': log_loss,
                'selected_before': previous['selected'], 'selected_after': selected,
                'evidence': evidence, 'fresh_count': fresh_count, 'switch': switched}

    def diagnostics(self):
        """Detached deterministic diagnostics; scores are not confidence levels."""
        return {'parameters': {'decay': self.DECAY, 'threshold': self.THRESHOLD,
                               'minimum_fresh': self.MINIMUM_FRESH,
                               'scoring_pseudocount': self.SCORING_PSEUDOCOUNT,
                               'window': self.window},
                'events': self.events,
                'pairs': [{'source': list(source), 'action': action, **copy.deepcopy(pair),
                           'cumulative_count': sum(self._cumulative.transitions[(source, action)].values()),
                           'recent_count': sum(self._recent.transitions[(source, action)].values()),
                           'retained_count': sum(self._graph.transitions[(source, action)].values())}
                          for (source, action), pair in sorted(self._pairs.items())],
                'losses': copy.deepcopy(self._losses)}
