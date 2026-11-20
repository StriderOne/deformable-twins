# Graph Residual Flow pilot

Implementation for [[E-0014]], [[P-0009]], [[BL-0011]]. This is an exploratory
posterior-diversity experiment, not evidence of multiple modes or sim-to-real
improvement. Authoritative settings are in `experiments/E-0014.yaml`.

## Model and limitations

The flow operates on every spring's unconstrained logit coordinate, followed by
the existing bounded stiffness transform. Physical adjacency uses normalized
spring–vertex incidence; spatial adjacency uses symmetric normalized neighbors
of initial spring centers. Types are mixed with an identity path by averaging.
Both operators have spectral norm at most one. Shared linear maps use the
explicit bound `sqrt(||W||_1 ||W||_infinity)`, and tanh is 1-Lipschitz. Thus each
residual branch obeys the configured contraction bound. This graph and network
are an initial architectural choice, not an experimentally preferred design.
Spatial neighbor count is capped before symmetrization; final degree may exceed
that cap. Physical incidence includes self messages. Geometry is fixed and
includes controller endpoints; no new forces are added to the simulator.

The log density uses a Hutchinson/Russian-roulette series. A detached Neumann
series supplies an unbiased gradient estimate through one differentiable VJP,
so memory does not grow with the number of series terms. The gradient is an
estimator, not the exact derivative of the particular sampled log-density.
The continuation probability exceeds the squared contraction bound to avoid
an unbounded-variance geometric-tail bound. This changes the initial plan's
engineering default; it does not change the scientific objective.

Training retains framewise updates and truncated physics gradients. A fixed
base draw is reused within a pass, while parameters change after each frame.
This is an approximate inference procedure, not an exact full-trajectory ELBO
gradient. Diagnostics use a separate reproducible random stream. The base mean
and variance are not physical posterior moments; physical widths are sampled.
TensorBoard and online Comet logging are enabled. Scalar metrics are logged
after every complete pass; spatial posterior images follow the YAML cadence.
The container reads the host Comet config through a read-only mount; credentials
are never stored in Git. JSON history and plots remain available locally. New NF samples have no analytic marginal-median candidate.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_residual_flow.py -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_vi_*.py' -v
```

These tests exercise the graph norm, inverse, exact small Jacobian comparison,
stochastic gradient, support, RNG isolation, framewise resume and pair metrics.
They do not run a physical experiment or establish full-scale GPU feasibility.

Opt-in synthetic CUDA capacity test (no real observations or physics):

```bash
PYTHONPATH=src python tests/smoke_residual_flow.py
```

Validation on 2026-11-20: all five NF tests passed locally and in the server's
training container; eighteen existing VI tests passed locally. A synthetic
114479-spring graph, one forward/backward/Adam step on the RTX 5080 took about
0.62 seconds and peaked at 1883 MiB of PyTorch allocations. This excludes the
simulator, graph construction and diagnostics and is not an end-to-end training
estimate. The server image was
`sha256:d2ff62b2e7d095be53b30070cdd69911b15af2a21c06513635b5368fa8c419f7`;
Torch 2.9.1+cu128, SciPy 1.17.1, CUDA available, `pip check` passed.

## Server execution

Use the existing `environments/container.sh train` container on
`roman@100.64.0.9`. Its data mount preserves the paths in the original dataset
manifest. Verify available GPU memory and input hashes before execution.

Before **each** scientific launch, inspect open researcher questions and run:

```bash
git status --porcelain=v1 --untracked-files=all --ignore-submodules=none
git rev-parse HEAD
```

The first command must succeed with no output. Do not commit, stash or clean
automatically to satisfy it. The CLI independently enforces this for E-0014.
Use unused R-NNNN IDs; no real run IDs are reserved by this documentation.
Store active manifests outside the worktree to keep it clean for evaluation:

```bash
environments/container.sh train python -m lab.calibration.phystwin_vi \
  --config experiments/E-0014.yaml \
  --run-dir /home/homa/.local/share/deformable-twins/run-records/R-NNNN --mode vi
```

The time budget applies to training after model/graph construction. An active
pass finishes beyond the deadline; actual elapsed time is reported. Checkpoints
are saved at complete-pass boundaries. A crash retains the previous complete
checkpoint and writes partial-state diagnostics when possible.

For extension, prepare an explicitly approved YAML variant with a new additional
time budget; preserve model, optimizer and data settings. Use a new run ID and
`--resume-checkpoint` with the previous artifact's `checkpoint.pt`. Graph and
configuration compatibility are checked. Prior run records are immutable.

Evaluation uses a new run ID, `--mode evaluate --vi-checkpoint PATH`. The plan
also accepts an existing BL-0010 checkpoint for descriptive control evaluation;
its actual baseline ID is recorded. Evaluate both with the same E-0014 config.
A separate evaluation budget applies. Joint stiffness vectors, completed
trajectory arrays, masked observation errors and pair tables/plots are saved.
Prediction distances use all simulated particles; observation RMSE retains the
original masks. Partial evaluation records its completed/requested counts.
No clustering threshold or mode count is assigned automatically.

After execution, copy completed manifests into `runs/` without rewriting them,
write a run review including failures and artifact links, and ask for the
researcher's interpretation before advancing the research.
