# Bug Report: Explicit-Euler Optimization Demo Failure

Date: 2026-04-28

## Summary

The `ntrain=10` optimization demo does **not** currently complete under the requested configuration

- `time_integrator = "explicit_euler"`
- `max_dt = 0.01`
- `optimizer = "lbfgs"`

because the very first forward evaluation already fails for the `OpInf`-initialized latent dynamics.

## What was fixed immediately

While reproducing the failure, a real exception-handling bug was found and fixed:

- `ForwardRolloutFailure` was implemented as a frozen dataclass subclassing `RuntimeError`
- Python attempted to assign `__traceback__` during exception propagation
- that triggered `FrozenInstanceError`

This has been fixed by removing the frozen dataclass restriction from `ForwardRolloutFailure`.

## Current reproducible failure

Command:

```bash
python /storage/yuhang/Myresearch/GOATTM/module_test/reduced_qoi_best_response/run_small_optimization_demo.py
```

Observed run directory:

- `/storage/yuhang/Myresearch/GOATTM/module_test/output_plots/reduced_qoi_optimization_demo/runs/small_opt_demo_opinf_20260428_042333_7876b519`

Failure artifact:

- `failures/training_exception_20260428_042333_04f66087.json`
- `failures/training_exception_20260428_042333_04f66087.npz`

Reported failure:

- sample id: `sample-000`
- sample index: `0`
- latent sample path:
  `/storage/yuhang/Myresearch/GOATTM/module_test/output_plots/reduced_qoi_optimization_demo/workflow_dataset/20260428_042332/opinf_init/latent_dataset/train/000000_sample-000.npz`
- failure time: `t = 0.11999999999999998`

## Parameter magnitudes at failure start

The run fails during the **initial** gradient evaluation, before any optimization step is accepted.

Initial dynamics norms from `initial_parameters.npz`:

- `||A||_F = 775.2194310175025`
- `||A||_2 = 540.5272806952482`
- `||H||_F = 345.065768978669`
- `||B||_F = 420.44157757097946`
- `||c||_2 = 1159.4545628086335`
- full parameter norm: `1380.3636388145794`

These norms are very large for an explicit Euler rollout with `dt = 0.01`.

## Interpretation

The failure is **not** currently evidence of a discrete-adjoint bug.

What is happening is:

1. raw data is preprocessed
2. `OpInf` initialization produces a latent model
3. that latent model has very large coefficients
4. explicit Euler with `dt=0.01` is not stable enough for that initialization
5. the first training-set forward rollout already leaves the numerically safe region and fails

So this is primarily an **initialization / stability / step-size compatibility** issue.

## Why this matters

The explicit-Euler first-order and second-order derivative chains have already passed:

- solver-level tangent / incremental-adjoint checks
- first-order reduced-objective Taylor test
- second-order reduced-objective Taylor test
- full `unit_test` regression sweep

Therefore, the blocking issue for this demo is not “the explicit Euler adjoint is wrong”.
It is “the initialized dynamics are too aggressive for the requested explicit rollout”.

## Recommended next actions

1. add a pre-training stability sanity check on initialized dynamics
2. rescale or regularize the `OpInf` initialization more aggressively before training
3. reduce `max_dt` further for the explicit-Euler demo
4. optionally reject or damp initializations whose explicit-Euler rollout already fails on the training set

## Status

- library-level explicit-Euler implementation: done
- explicit-Euler unit and Taylor tests: passing
- explicit-Euler optimization demo with current `OpInf` initialization: failing at the initial forward rollout
