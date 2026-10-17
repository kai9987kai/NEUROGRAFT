# Dynamics implementation evidence

17 October 2026. This report covers `neurograft/dynamics.py` and its focused
behavioral tests. All inputs in these tests are explicitly synthetic fixtures.

## Computation delivered

`GraphCircuit` requires the four canonical source modules in the shared plan.
Each has its own GRU, input projection, structural-feature projection and four
relation transforms. Trainable parameter allocation does not depend on the
graft/recurrence flags or observed relation counts. An episode state is an
external tuple of `[batch,nodes,hidden]` tensors; `step` does not modify the
supplied tensors. `forward` starts from zero state on each call.

Observed edges become registered sparse source/destination/weight buffers. No
biological-node adjacency matrix is allocated. For each relation and destination,
messages use `weight / sum(abs(incoming weight))`, followed by a relation-specific
linear map and `index_add`. The incoming absolute sum preserves negative
functional-association signs and prevents positive/negative cancellation in the
denominator. Independent incoming relation channels are averaged per destination
before the source GRU. Empty channels and zero-weight edges produce zero messages.
Undirected observations expand to two computational arcs except that a self-loop
remains one arc; the original graph is never edited.

Four deterministic node features describe normalized incoming/outgoing degree,
incoming sign balance and isolation. They break the identical-node-state symmetry
that would otherwise arise from broadcasting a module observation to every node.
There are no learned node embeddings or biological-size dense matrices. Prior
node states travel through actual observed-edge transforms and GRUs, then node
states are mean-pooled into source ports.

The artificial overlay computes a query and key for every source, masks self
routes before selection, chooses at most `top_k` other senders independently for
each receiver and example, and normalizes those scores together with a learned
null-route score. The null route carries a zero message. Gated projected messages
are added to each receiving source's node states. `routes` has shape `[B,4,5]` and
`context` has shape `[B,4H]`. The task policy reads only the pooled human state.

`graft_enabled=False` returns the local source states and null-only routes.
`top_k=0` has the same numerical output as that bypass. `recurrence=False`
discards the previous state at every step, so earlier observations cannot affect
the final output. These controls retain all allocated trainable parameters, while
their active computation and parameter gradients differ.

## Tests and observations

The initial test run failed because `neurograft.dynamics` did not exist. The first
implemented run passed 17 tests and exposed an invalid test construction that
negated structural counts as well as functional correlations. The graph validator
correctly rejected it. The test was corrected to negate only functional edges;
no graph-validator relaxation was made.

Final focused command:

```text
python -m pytest tests/test_dynamics.py -q
19 passed in 12.97s
exit code 0
```

The tests exercise actual tensor computation, including:

- deterministic episode reset and separate per-node state shapes;
- caller-state and source-digest preservation;
- self-route masking even with a deliberately dominant self score;
- top-k selection independently for each example and receiver, and null mass;
- exact null-only/bypass equivalence;
- human-policy independence from other sources when grafting is disabled;
- final-output independence from earlier cues without recurrence;
- finite nonzero gradients to previous node states, early nonhuman observations,
  source GRUs, query/key/value transforms and actual observed-relation transforms;
- a real Adam parameter update and lower loss on a bounded synthetic example;
- hand-derived signed weighted messages and their gradients;
- invariant normalized messages under a common edge-weight rescaling;
- undirected expansion without doubling an observed self-loop;
- changed logits when human node state or functional edge signs change, with
  identical trainable parameters between the graph variants;
- rejected malformed shapes, nonfinite observations and invalid dimensions.

PyTorch tests used one Python process and `torch.set_num_threads(2)`. The installed
environment prints an ignored `pyreadline3.console.Console.__del__` Windows handle
error after pytest output; the final process exit status was zero. The warning is
not silently counted as a test failure or described as a clean environment.

The root integrator owns the full-project test run and evaluation integration;
this focused report does not claim those checks before they occur.

## Boundaries

These are compact engineered recurrent graph computations. Signed human
correlations remain associations; using them in this software is not a claim of
causal transmission or physiological calibration. Structural degree features and
learned routing are engineering choices, and confidence metadata are preserved in
source records rather than interpreted as a calibrated propagation probability.

Tests establish trainability and contracts on small synthetic graphs, not a
generalization advantage, authentic four-source readiness, brain emulation,
consciousness, or biological evidence. Sparse top-k selection has a discrete
selection boundary; gradients flow through selected scores and messages, while
unselected routes receive no score gradient on that example. Identical parameter
allocation does not imply identical active compute across ablations. A connected
module routing diagram does not establish connectivity of every observed node.
