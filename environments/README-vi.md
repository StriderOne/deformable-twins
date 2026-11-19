# PhysTwin VI pilot

Offline training only. The pinned PhysTwin source is read without importing its
renderer; no upstream files are modified. Executable settings live in
[`E-0001`](../experiments/E-0001.yaml), evaluation rules in
[`P-0001`](../registry/protocols/P-0001.yaml).

## Environment used

`~/.venvs/phystwin-vi` was created with Python 3.11 from the existing
`~/anaconda3/envs/bridge_sbi` interpreter and `--system-site-packages` to reuse
Torch. Warp and Open3D were installed into the new venv; existing environments
were not modified. It is a separate venv, not a hermetic environment.
Validation artifacts and their hashes are listed in
[`runs/vi-validation.json`](../runs/vi-validation.json).

Direct dependencies are pinned in `requirements-phystwin-vi.txt`; the complete
installed package list is retained in the validation artifacts. CUDA device
support must be checked on a new machine. Raw processed data remain outside Git.

## Checks

From the repository root:

```bash
PYTHONPATH=src ~/.venvs/phystwin-vi/bin/python -m unittest discover -s tests -p test_vi_posterior.py -v
PYTHONPATH=src ~/.venvs/phystwin-vi/bin/python tests/smoke_vi_gradients.py --sloth
~/.venvs/phystwin-vi/bin/python -m pip check
```

The second command explicitly uses the GPU and local sloth data. It is not an
ordinary CI test. It checks gradients on a tiny system before a short sloth
rollout. Warp warns that the upstream `set_control_points` kernel has backward
disabled: controller trajectories are fixed inputs, so they need no derivatives
with respect to stiffness. This warning is retained in validation logs.

## Training and evaluation

Choose **unused** run IDs; the CLI refuses existing directories. `--mode` is
`map`, `vi` (default), or `evaluate`. Run one process at a time on this GPU.

```bash
PYTHONPATH=src ~/.venvs/phystwin-vi/bin/python -m lab.calibration.phystwin_vi \
  --config experiments/E-0001.yaml --run-dir runs/R-NNNN --mode vi
```

Evaluation accepts `--map-checkpoint PATH` and `--vi-checkpoint PATH` and always
includes the CMA initialization. Use a separate unused run ID. Checkpoints and
plots go to the artifact root in the plan; Git stores manifests and hashes.

Each iteration resets to frame zero, samples one fixed log-stiffness vector,
accumulates per-frame gradients, and performs one optimizer update. The
simulation is continuous across frames but gradients are truncated. `VI_mu`
means `exp(mu)` (the marginal median stiffness), not the lognormal mean.

The objective logs omit the fixed Gaussian normalization constant. The reported
RMSE is per valid coordinate, not mean Euclidean point distance. Training VI
losses use a different sampled model each iteration, whereas MAP is deterministic
up to GPU floating-point effects. The posterior is not a calibrated confidence
interval for physical stiffness.

A timed-out partial rollout does not update parameters or enter iteration
metrics. Checkpoints include the last completed optimizer step and RNG state;
a resume CLI is intentionally not included. Non-finite trajectories stop the run
and preserve the failed sample and traceback. Final manifests are not rewritten.
