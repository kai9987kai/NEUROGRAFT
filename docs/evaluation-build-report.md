# Evaluation build report

17 October 2026

## Delivered interfaces

`neurograft.evaluation.BenchmarkConfig`, `generate_task(size, delay, seed)`,
`run_benchmark(bundle, config, output_dir, variants=...)`, and
`load_checkpoint(path, bundle)` implement the plan's shared contract. The
checkpoint loader accepts the original input bundle, reconstructs separately
saved degree-rewired controls, and returns the model plus metadata.

The synthetic task has `delay + 2` timesteps and four features per source.
At time zero, worm and female fly each receive one signed XOR bit and a cue
indicator. Only these two sources receive small label-independent nuisance
inputs. Male fly receives zeros. The human module receives a common query
indicator at the last step and no cue or label-dependent input. Labels are
balanced for sizes divisible by four. The four underlying XOR cue patterns
necessarily recur across independently seeded splits.

## Experimental controls and artifacts

- Fixed train/validation/test generator seeds are `3 * data_seed + {0,1,2}`;
  defaults are 5187, 5188, and 5189. Model seeds are separate.
- Every model seed has identical initial named parameters, paired examples,
  epoch permutations, optimizer type, learning rate and number of optimizer
  steps for all requested variants. Graph buffers are deliberately preserved
  when transferring initialized parameters to the rewired control.
- Controls are `grafted`, `no_graft`, `no_recurrence`, and `rewired`. They all
  allocate the full model. Active computation and active gradients can differ.
- Final-step cross-entropy trains each variant for every configured epoch.
  Highest validation accuracy selects the checkpoint; validation loss breaks
  ties. The test split is evaluated once after selection.
- Each epoch writes JSONL training and validation metrics. Checkpoints contain
  selected weights, complete saved control graph, graph and topology hashes,
  benchmark and model configs, selected epoch, seed, variant, split/data hashes,
  parameter hashes, and a deterministic test-logit replay hash.
- Original graph hashes and topology hashes are distinguished from control
  provenance. Each source's real attempted/successful-rewire receipt is saved;
  `topology_changed` reports whether edge records actually changed.
- Outputs are written atomically. Nonempty destinations are refused and an
  exclusive lock prevents competing writers from silently sharing a run.
  Interrupted output is inspectable and is not claimed as resumable training.
- CPU thread count defaults to two and cannot exceed two. Input allocation and
  estimated training/held-out tensor sizes have bounded resource checks. These
  are allocation estimates, not guarantees about total process memory.
- Reports record every seed's accuracy, parameter count, optimizer steps,
  selection epoch, elapsed wall time and process RSS sampled every 50 ms.
  RSS includes PyTorch, allocator retention and earlier sequential runs; it is
  not tensor-only memory or an exact peak. Timing includes initialization,
  validation, optimizer setup and artifact writes, so it is not a compute-matched
  speed comparison.
- Reports embed the original source-readiness statistics and a per-variant
  mean accuracy/sample standard deviation summary. Standard deviation has
  `ddof=1` and is null for fewer than two seeds; fixtures never satisfy the
  authentic four-source readiness gate.
- Paired bootstrap intervals use 2,000 deterministic resamples over seed-level
  graft-minus-control differences. One seed gives no interval. Very small seed
  samples give unstable descriptive intervals, and the fixed common task splits
  mean these intervals reflect run-seed variation in initialization/training
  and, for rewired controls, topology rewiring. They do not measure data-split
  uncertainty.

## Verification evidence

The initial test-first check failed because `neurograft.evaluation` did not yet
exist: one expected failing test. The implementation and tests then passed
Python compilation.

Focused command:

```powershell
python -m pytest tests/test_evaluation.py -q
```

Result: **28 passed in 19.29 seconds, exit code 0**. An ignored environment
`pyreadline3.Console.__del__` Windows error appeared after pytest completed;
it was also observed by the dynamics test worker and is not a test failure.

The integration fixture trained all four variants for model seeds 7 and 11,
two epochs, 16 training examples, eight validation examples and 12 test
examples, batch size eight, hidden width four, delay one, and six synthetic
nodes per source. Each variant took exactly four optimizer steps. Tests checked
identical parameter counts, initial parameter hashes, data hashes and batch
order hashes; held-out test was evaluated once per variant. Every selected
checkpoint reproduced its exact recorded test-logit bytes, including rewired
controls. A second independent benchmark reproduced selected parameter hashes,
epochs, accuracies and test-logit hashes. Tests also covered invalid configs,
task leakage boundaries, refusal to clobber existing output, unknown variants,
wrong input graph, corrupted graph hash, invalid seed/variant/selection epoch,
and nonfinite saved weights.

The fixture's temporary benchmark artifacts are sandbox-scoped. This report
does not invent per-seed smoke scores or claim those temporary files are a
durable experiment. The root integration's persistent benchmark `report.json`
is the intended source for final seed-level scores, timings and sampled RSS.
The whole-project suite and the separately declared five-seed experiment are
root-owned integration steps and were not run concurrently by this worker.

Integration follow-up: source-readiness and summary metadata were added with
a focused regression test while the root-owned experiment was running. This
worker did not launch another Torch process. The regression test and final
whole-project verification are delegated to the root integration run.

## Evidence boundaries

This benchmark tests engineered communication and memory on synthetic delayed
XOR. Its information placement is designed to require source communication and
recurrence. A graft advantage therefore does not establish a biological benefit,
a benefit from authentic connectome topology, model capability, or a powered
scientific result. Rewired comparisons can help investigate topology dependence
but small seed counts and this toy task limit inference. These checkpoints
support replay and inference; training resume is not implemented.
