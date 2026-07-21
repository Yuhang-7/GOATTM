# Update Log

## 2026-07-04 RK4 substeps between observation times

Added an explicit substep RK4 path for cases where QoI is observed every
coarse interval but the latent dynamics should be advanced with a smaller
time step.

- `ContinuousBatch.substep_midpoint_inputs(substeps)` interpolates input values
  at substep midpoints inside each observation interval.
- `SubstepRungeKutta4Stepper(substeps)` advances the latent dynamics with
  multiple RK4 steps per observation interval, but stores states only at the
  original QoI observation times.
- `runge_kutta4_substep_rollout_adjoint` implements the corresponding exact
  discrete adjoint by recomputing per-interval internal RK4 states during the
  backward sweep.
- `ReducedObjective` and `DistributedReducedObjective` route
  `gradient_mode="rk4_adjoint"` through the substep adjoint when the stepper
  has `substeps > 1`.
- Added `tools/benchmark_cascadia_steppers.py` for Cascadia closure timing with
  dissipative low-rank `A` and energy-Tucker low-rank `H`.

First smoke comparison on `nid001021`, using 16 Cascadia samples,
`r=20`, `A` rank 6, `H` reduced rank 6:

| stepper | physical latent dt | closure seconds | peak memory |
| --- | ---: | ---: | ---: |
| lagged midpoint, `K=2` | 5.0 s | 5.97 | 66.9 MiB |
| RK4, `substeps=10` | 0.5 s | 72.14 | 211.4 MiB |

The substep RK4 path is correct and useful as a reference, but the current
Python-level implementation is not a fast Adam-training path: every observation
interval expands into 10 RK4 steps and 40 stage VJP assemblies in the adjoint.
Before using this for thousands of Adam steps, the substep RK4 loop needs
fusion, checkpointing, or a coarser substep count.

## 2026-07-04 RK4 explicit forward option

Added `RungeKutta4Stepper` as a pure explicit forward solver:

- The stepper evaluates the quadratic ODE with classical fourth-order
  Runge--Kutta and performs no implicit linear solve.
- The current input API supplies one midpoint input per step, so RK4 freezes
  that midpoint input across all four stages.
- `rollout` and `rollout_with_lags` are implemented, making RK4 usable in
  `ReducedObjective.evaluate` and in the autograd reduced-gradient path.
- `rollout_with_picard_history` intentionally raises `NotImplementedError`
  because RK4 has no Picard history.  The exact RK4 discrete adjoint has not
  yet been implemented.
- `tests/run_dataset_training.py`, `tests/run_speed_benchmark.py`, and
  `tests/run_training_smoke.py` accept `--stepper rk4`.
- Dataset training automatically uses `gradient_mode="autograd"` for RK4 when
  the default lagged adjoint was requested; non-autograd RK4 gradients are
  rejected explicitly.
- Backend metadata reports `solver_backend="explicit_runge_kutta4"` for RK4.

Smoke checks on Perlmutter:

```text
pytest -q tests/test_core_exact.py::test_rk4_rollout_matches_constant_source_solution
2 passed in 3.69s

python tests/run_speed_benchmark.py --latent-dim 8 --h-rank 4 --dp 4 \
  --output-dim 4 --batch-size 16 --steps 8 --step-size 0.01 \
  --stepper rk4 --modes evaluate,autograd --warmup 1 --repeat 1
custom evaluate mean=0.0093s peak=8.3 MiB
custom autograd mean=0.0419s peak=17.5 MiB
```

## 2026-07-04 RK4 exact discrete adjoint

Added `gradient_mode="rk4_adjoint"` for the explicit RK4 solver.  For one
step, with midpoint input fixed over the step,

```math
\begin{aligned}
k_1 &= f(u_n,p_n;\theta),\\
k_2 &= f(u_n+\tfrac{h}{2}k_1,p_n;\theta),\\
k_3 &= f(u_n+\tfrac{h}{2}k_2,p_n;\theta),\\
k_4 &= f(u_n+h k_3,p_n;\theta),\\
u_{n+1} &= u_n+\tfrac{h}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
```

Given an incoming cotangent `lambda_{n+1}`, initialize

```math
\bar u_n \mathrel{+}= \lambda_{n+1},\quad
\bar k_1 \mathrel{+}= \tfrac{h}{6}\lambda_{n+1},\quad
\bar k_2 \mathrel{+}= \tfrac{h}{3}\lambda_{n+1},\quad
\bar k_3 \mathrel{+}= \tfrac{h}{3}\lambda_{n+1},\quad
\bar k_4 \mathrel{+}= \tfrac{h}{6}\lambda_{n+1}.
```

Then reverse the stages:

```math
\begin{aligned}
(\bar x_4,\bar p,\bar\theta) &\mathrel{+}= Df(x_4,p_n;\theta)^T\bar k_4,
&x_4 &= u_n+h k_3,\\
\bar u_n &\mathrel{+}= \bar x_4,
&\bar k_3 &\mathrel{+}= h\bar x_4,\\
(\bar x_3,\bar p,\bar\theta) &\mathrel{+}= Df(x_3,p_n;\theta)^T\bar k_3,
&x_3 &= u_n+\tfrac{h}{2}k_2,\\
\bar u_n &\mathrel{+}= \bar x_3,
&\bar k_2 &\mathrel{+}= \tfrac{h}{2}\bar x_3,\\
(\bar x_2,\bar p,\bar\theta) &\mathrel{+}= Df(x_2,p_n;\theta)^T\bar k_2,
&x_2 &= u_n+\tfrac{h}{2}k_1,\\
\bar u_n &\mathrel{+}= \bar x_2,
&\bar k_1 &\mathrel{+}= \tfrac{h}{2}\bar x_2,\\
(\bar x_1,\bar p,\bar\theta) &\mathrel{+}= Df(u_n,p_n;\theta)^T\bar k_1,
&\bar u_n &\mathrel{+}= \bar x_1.
\end{aligned}
```

The rollout adjoint applies this one-step reverse sweep backward in time:

```math
\lambda_n =
\frac{\partial \Phi}{\partial u_n}
\left(\frac{\partial u_{n+1}}{\partial u_n}\right)^T\lambda_{n+1}.
```

Implementation notes:

- `runge_kutta4_step_adjoint` implements the stage reverse sweep above.
- `runge_kutta4_rollout_adjoint` stores or reuses primal RK4 states and builds
  only one local RHS VJP graph at a time; it does not retain a full rollout
  autograd graph.
- `ReducedObjective.value_and_grad` now supports `gradient_mode="rk4_adjoint"`.
- `DistributedReducedObjective` supports both `lagged_adjoint` and
  `rk4_adjoint` for data-parallel training.
- Dataset training with `--stepper rk4` maps the old default
  `--gradient-mode lagged_adjoint` to `rk4_adjoint`; explicit
  `--gradient-mode autograd` remains available as a correctness/speed baseline.

Correctness tests compare one-step and rollout RK4 adjoints against full
PyTorch autograd.

## 2026-07-04 RK4 RHS VJP without autograd

Replaced the RK4 adjoint's local RHS VJP with a fully assembled manual VJP.
The production RK4 adjoint path now computes

```math
Df(x,p;\theta)^T\bar f
= A^T\bar f
  + D_x H(x,x)^T\bar f
  + B^T\bar f,
```

and accumulates parameter gradients block by block:

- `linear.*` through `_linear_vjp`;
- `quadratic.*` through `_quadratic_frozen_vjp` with both quadratic arguments
  set to `x`;
- `source.B` and `source.c` through `_source_vjp`.

This removes `torch.autograd.grad` from the RK4 discrete-adjoint production
path.  Autograd remains only as a test/baseline path and for unsupported
fallback routines.

## 2026-07-03

Initialized a fresh GOATTM-style GPU rewrite in `/storage/yuhang/quadrode_gpu_goattm`.

Initial scope:

- Added `DenseLinearA` as the reference general dense `A`, matching GOATTM's
  skewCP dynamics model where only the quadratic term is skewCP-constrained.
- Added `DissipativeSkewA` as an optional structured `A`.
- Added `SkewCPQuadratic` with exact energy-preserving skewCP action and frozen
  linearization.
- Added `DenseLaggedMidpointStepper` as the correctness-first GPU baseline.
- Reserved `ExactSMWLaggedMidpointStepper` and `SkewCPDefectLaggedMidpointStepper`
  APIs for future exact SMW and adaptive defect reduced solvers.
- Added exact dense frozen-step adjoint for general dense `A` and skewCP `H`.
- Added quadratic decoder readout and ridge normal-equation variable projection.
- Added an initial smoke test checking dense-A frozen-step adjoint against
  PyTorch autograd on GPU.

ccgo3 smoke result:

```text
torch 2.5.1 cuda True
smoke ok torch.Size([8, 4, 5]) cuda:0
```

## 2026-07-03 Step 1--3 pass

Implemented the first three rewrite priorities:

- Step 1: added `data.py` with `ContinuousSample`, `ContinuousBatch`,
  `SampleManifest`, NPZ sample/manifest IO, strict time-grid validation, and
  bounded piecewise-linear input interpolation.
- Step 2: strengthened the dense GPU baseline with `rollout_with_lags`, keeping
  `DenseLinearA + SkewCPQuadratic + DenseLaggedMidpointStepper` as the
  correctness path.
- Step 3: upgraded decoder variable projection to streaming normal-equation
  assembly with residual reporting.

BFGS/LBFGS and GOATTM-style initialization are explicitly part of the next
training layer, after this IO/numerics/normal-solve base is stable.

## 2026-07-03 Reduced objective base

Added the first reduced-objective callable:

- `ReducedObjective.evaluate(batch)` performs dense rollout, decoder normal
  solve, prediction, and weighted QoI loss evaluation.
- `ReducedObjective.value_and_grad(batch)` supports the exact PyTorch-VJP
  gradient for the finite Picard dense rollout and the frozen-lag discrete
  adjoint smoke path; both write gradients into dynamics parameters.
- Added trapezoidal observation weights and a weighted trajectory loss helper.
- Included the decoder ridge term in the reduced objective after the normal
  solve. This is required by the envelope theorem: the finite-difference
  objective and the analytic reduced gradient must differentiate the same
  regularized decoder best-response objective.

This is the core callable that BFGS/LBFGS should wrap next.

## 2026-07-03 Taylor sweep design

Added `tests/run_taylor_sweep.py` for reduced-objective finite-difference
checks.  The sweep uses random synthetic continuous batches because the goal is
gradient correctness, not generalization:

- The decoder is eliminated by variable projection at every perturbed dynamics
  parameter.
- The checked parameter block is the dynamics block `(A,H,B,c)`.
- For a random unit direction `p`, the script reports
  `|J(theta+eps p)-J(theta)|`, which should be first order in `eps`, and
  `|J(theta+eps p)-J(theta)-eps <grad J,p>|`, which should be second order.
- The default sweep covers eight cases over multiple random seeds, latent
  ranks, batch sizes, step counts, time steps, and Picard iteration counts.
- Outputs are written as JSON plus a dependency-free SVG log-log plot so the
  ccgo3 `quadrode_torch` environment does not need matplotlib.

## 2026-07-03 Lagged midpoint adjoint reference

Added dense correctness adjoints for the actual finite-Picard lagged midpoint
map:

- `lagged_midpoint_step_adjoint` reverses every Picard substep, including the
  dependence of each lag `ell = (u_n + u_{n+1}^{(k)})/2` on the previous Picard
  iterate.
- `lagged_midpoint_rollout_adjoint` applies the step adjoint in a reverse-time
  sweep for an entire rollout with arbitrary state cotangents.
- Added tests comparing both the one-step and rollout adjoints against
  differentiating `DenseLaggedMidpointStepper` directly with PyTorch autograd.
- Added `gradient_mode="lagged_adjoint"` in `ReducedObjective`, using the
  full finite-Picard rollout adjoint as the reduced-gradient path.

This is distinct from the earlier frozen-lag adjoint: the new reference
matches the finite Picard solver itself, while the frozen-lag adjoint treats
the final lag as a constant.

## 2026-07-03 IO, training logs, and speed baselines

Added first-pass library wrappers beyond correctness kernels:

- `ContinuousDataset` and `split_manifest` provide manifest-backed NPZ loading,
  train/test splitting, deterministic shuffling, and batch construction.
- `LBFGSReducedTrainer` wraps `ReducedObjective.value_and_grad` so the decoder
  is still eliminated by variable projection at every closure call.
- `JsonlOptimizationLogger` writes closure-level optimization logs containing
  loss, data loss, decoder regularization, normal-equation residual, gradient
  norm, wall-clock time, and gradient mode.
- `tests/run_speed_benchmark.py` records repeatable timing and peak allocated
  CUDA memory for `evaluate`, `autograd`, `lagged_adjoint`, and
  `frozen_adjoint` paths.

This is not yet application-specific full HDF5 IO or a production trainer, but
it establishes the missing IO/training/logging/speed measurement spine.

## 2026-07-03 r=60, dt=1e-3, dp=150 benchmark

Extended `tests/run_speed_benchmark.py` with custom case arguments and ran a
larger continuous-time pressure test:

- latent dimension `r=60`
- input dimension `d_p=150`
- output dimension `d_q=150`
- time step `dt=0.001`
- final time `T=1`, hence `1000` steps
- Picard iterations `K=2`

Dense-reference timings on ccgo3, float64/CUDA:

| case | mode | mean seconds | peak memory |
| --- | --- | ---: | ---: |
| batch 64, Hrank 32 | evaluate | 2.7225 | 2139.7 MiB |
| batch 64, Hrank 32 | autograd | 7.5989 | 17254.5 MiB |
| batch 64, Hrank 32 | lagged_adjoint | 22.6381 | 2326.8 MiB |
| batch 128, Hrank 32 | evaluate | 2.6696 | 4236.9 MiB |
| batch 128, Hrank 32 | lagged_adjoint | 22.6366 | 4598.9 MiB |
| batch 256, Hrank 32 | evaluate | 2.7814 | 8430.9 MiB |
| batch 256, Hrank 32 | lagged_adjoint | 25.2230 | 9145.5 MiB |
| batch 128, Hrank 60 | evaluate | 2.6373 | 4236.9 MiB |
| batch 128, Hrank 60 | lagged_adjoint | 24.6424 | 4598.9 MiB |

Interpretation:

- Increasing batch from 64 to 256 mostly increases memory, not wall-clock time,
  so this case is dominated by the 1000 serial time steps and per-step dense
  solves rather than by insufficient batch parallelism.
- Autograd is faster at batch 64 but uses far more memory because it stores the
  full rollout graph.
- The current full lagged adjoint is memory efficient but slow because it is a
  dense correctness reference implemented as a reverse sweep of local VJPs.
- Hrank has little effect in this path because the dense reference materializes
  and solves dense `60 x 60` systems. Low-rank SMW/defect solvers are required
  before Hrank can translate into speed.

Additional batch scaling on the same 40GB A100:

| case | mode | mean seconds | peak memory |
| --- | --- | ---: | ---: |
| batch 512, Hrank 32 | evaluate | 3.3759 | 16816.5 MiB |
| batch 512, Hrank 32 | lagged_adjoint | 25.7660 | 18231.8 MiB |
| batch 768, Hrank 32 | evaluate | 3.9810 | 25204.6 MiB |
| batch 768, Hrank 32 | lagged_adjoint | 27.8084 | 27325.4 MiB |
| batch 1024, Hrank 32 | evaluate | 5.1993 | 33591.8 MiB |

The full `lagged_adjoint` path fails at batch 1024 on the current single A100.
The failure occurs while differentiating the decoder loss with respect to all
stored states, before the reverse-time dynamics sweep.  This identifies the
quadratic decoder state-gradient construction as the next memory bottleneck for
large batches.

## 2026-07-03 decoder-gradient and interpolation memory optimization

The first large-batch profile found two accidental memory issues:

- `decoder_loss_and_state_grad` was intended to be a manual state-gradient
  routine, but it used decoder parameters with `requires_grad=True`.  PyTorch
  therefore built a chunk-by-chunk autograd graph and retained many quadratic
  feature intermediates.  On `r=60, dp=150, batch=512`, component profiling
  showed allocated memory jumping from about `2.9 GiB` after normal solve to
  about `26 GiB` after decoder state-gradient construction.
- `ReducedObjective` recomputed midpoint inputs in the adjoint path, creating a
  second large `p_mid` tensor.
- `linear_interpolate` formed extra large temporaries via
  `(1-w) * left + w * right`.

Fixes:

- Detached decoder weights/bias and state inputs in the manual decoder
  state-gradient path.
- Used a smaller chunk size for adjoint-mode decoder state gradients and stopped
  returning full predictions from `value_and_grad` adjoint modes.
- Reused one `p_mid` tensor through rollout and adjoint sweep.
- Rewrote linear interpolation as in-place weighted accumulation on the gathered
  left/right buffers.

After these fixes, the same component profile for
`r=60, dp=150, batch=512, steps=1000` became:

| component | seconds | allocated after component |
| --- | ---: | ---: |
| midpoint inputs | 0.1231 | 2347.6 MiB |
| rollout with lags | 2.7601 | 2824.7 MiB |
| normal solve | 0.4202 | 2856.3 MiB |
| decoder state grad | 0.4069 | 3090.9 MiB |
| adjoint sweep | 20.2825 | 3685.3 MiB |

Large-batch `lagged_adjoint` now scales to much larger batches on one 40GB A100:

| batch | seconds | peak memory |
| ---: | ---: | ---: |
| 512 | 24.3405 | 3776.3 MiB |
| 1024 | 25.7611 | 10828.5 MiB |
| 2048 | 29.4766 | 21613.4 MiB |
| 3072 | 34.9961 | 32399.1 MiB |
| 3584 | 38.1791 | 32827.1 MiB |
| 4096 | 42.3581 | 29860.5 MiB |

The non-monotone peak memory numbers are due to allocator reuse and different
temporary lifetimes, but the practical result is clear: batch 4096 is now
feasible for this test on a single A100.

## 2026-07-03 dynamics regularization and optimization accounting

Completed the reduced-objective loss decomposition used by the optimizer:

\[
J_{\mathrm{red}}(\theta)
=
\frac12\|D(u_\theta)-q\|_{W_t}^2
+\frac{\lambda_D}{2}\|D^\star_\theta\|_F^2
+\frac{\lambda_\theta}{2}\|\theta\|_2^2,
\]

where \(D^\star_\theta\) is the variable-projection decoder best response and
\(\theta=(A,H,B,c)\) denotes the dynamics parameters.  The default dynamics
ridge is now \(10^{-7}\), matching the intended weak regularization of the
outer dynamics parameters.

Implementation notes:

- `ReducedObjective.evaluate` and all gradient modes now include
  `dynamics_regularization_loss`.
- The autograd gradient path differentiates this term directly.
- The manual frozen and lagged adjoint paths add the missing
  \(\lambda_\theta \theta\) contribution explicitly after the dynamics adjoint
  sweep.
- `JsonlOptimizationLogger` records data loss, decoder regularization, dynamics
  regularization, normal-equation residual, gradient norm, closure time, and
  elapsed time for every LBFGS closure.

Added `tests/run_training_smoke.py` as a reusable optimization smoke runner.
The runner executes the intended optimization loop:

1. roll out the latent dynamics,
2. solve the decoder normal equation by variable projection,
3. evaluate the reduced objective,
4. compute dynamics gradients by the selected gradient mode,
5. let LBFGS update only the dynamics parameters.

The first synthetic optimization run on ccgo3 used
`r=20`, `Hrank=12`, `d_p=d_q=32`, `batch=256`, `steps=80`, `K=2`,
`decoder_ridge=1e-5`, and `dynamics_ridge=1e-7`.  It made 21 closure calls in
8.891 seconds.  The objective decreased from `3.251436e+02` to
`3.233446e+02`; the final split was
`data=3.232682e+02`, `decoder_reg=7.458250e-02`,
`dynamics_reg=1.781747e-03`.

## 2026-07-03 large-batch streaming evaluation fixes

The first `batch=4096` benchmark revealed two unnecessary large tensors:

- `ReducedObjective.evaluate` materialized the full quadratic decoder feature
  and prediction tensors, which is not needed for loss-only optimization or
  timing.
- `lagged_midpoint_rollout_adjoint` always allocated
  \(\lambda_{p,\mathrm{mid}}\in\mathbb{R}^{N_t\times N_b\times d_p}\), even
  though the reduced dynamics optimizer does not update the input trajectory.

Fixes:

- `decoder_loss_and_state_grad` now has independent switches for returning
  prediction and state gradients.
- `ReducedObjective.evaluate(batch)` defaults to a streaming loss-only
  evaluation.  Prediction can still be requested explicitly with
  `return_prediction=True`.
- `lagged_midpoint_rollout_adjoint` and `lagged_midpoint_step_adjoint` now
  support `return_input_adjoint=False`; `ReducedObjective` uses this mode.

Current single-A100 float64 benchmark for the large synthetic case
`r=60`, `Hrank=60`, `d_p=d_q=150`, `batch=4096`, `steps=1000`, `dt=1e-3`,
`K=2`:

| mode | seconds | peak memory | normal residual |
| --- | ---: | ---: | ---: |
| evaluate | 14.1709 | 26391.5 MiB | 1.3e-14 |
| lagged_adjoint | 24.1708 | 31279.3 MiB | 1.3e-14 |

This is still a dense correctness path.  The main remaining runtime cost is the
serial time sweep of dense frozen systems in the forward rollout and adjoint
sweep.  For train sets above 10k samples, the next production-level wrapper
should accumulate decoder normal matrices and dynamics gradients over
micro-batches rather than requiring one monolithic batch.

## 2026-07-03 full-objective micro-batch accumulation

Added `MicroBatchReducedObjective` for train sets that cannot fit as one GPU
batch.  This is a full reduced objective wrapper, not mini-batch SGD:

1. scan all micro-batches once and accumulate the global decoder normal system,
2. solve one global variable-projection decoder,
3. scan all micro-batches again with that fixed decoder,
4. accumulate data loss and dynamics gradients,
5. add the dynamics ridge gradient once.

This preserves the intended objective

\[
J_{\mathrm{red}}(\theta)
=
\min_D
\frac12\|D(u_\theta)-q\|_{W_t}^2
+\frac{\lambda_D}{2}\|D\|_F^2
+\frac{\lambda_\theta}{2}\|\theta\|_2^2,
\]

rather than solving a different decoder on each micro-batch.

Implementation details:

- `assemble_decoder_normal_terms` and `solve_decoder_normal_terms` split the
  variable-projection normal equation into accumulation and solve phases.
- `slice_batch` splits a `ContinuousBatch` along the sample dimension.
- `MicroBatchReducedObjective` accepts either a full `ContinuousBatch`, a
  reusable iterable of `ContinuousBatch` objects, or a callable batch factory.
  The callable form is required for streaming data because the full objective
  needs two passes.
- The wrapper supports the manual `lagged_adjoint` and `frozen_adjoint` modes;
  `autograd` is intentionally rejected for the streamed full objective.
- `tests/run_training_smoke.py` now supports `--micro-batch-size` and
  `--stream-synthetic`; the streaming mode generates deterministic synthetic
  micro-batches without materializing the full train set on GPU.

Correctness check:

- A new reduced-objective test verifies that micro-batch lagged adjoint matches
  full-batch lagged adjoint in loss, decoder regularization, normal observation
  count, and every dynamics parameter gradient.

Synthetic optimization smoke:

| case | setting | closures | seconds | result |
| --- | --- | ---: | ---: | --- |
| `micro_opt_r20_b512` | `batch=512`, `micro=128`, `steps=80` | 10 | 20.986 | loss `6.513283e+02 -> 6.503763e+02` |
| `stream_micro_r20_b512` | streaming, `batch=512`, `micro=128`, `steps=80` | 2 | 4.499 | loss `6.535980e+02 -> 6.535919e+02` |
| `micro_r60_b8192_steps100` | `batch=8192`, `micro=2048`, `steps=100` | 1 | 8.362 | full closure completed |
| `stream_micro_r60_b8192_steps1000` | streaming, `batch=8192`, `micro=2048`, `steps=1000` | 1 | 80.328 | full closure completed |

The failed non-streaming `batch=8192`, `steps=1000` run OOMed before the solver
started because the synthetic `input_values` tensor alone required about
18.3 GiB on GPU.  This confirms the important production rule: large train sets
must be streamed as micro-batches instead of assembled as one GPU tensor.

## 2026-07-03 cached micro-batch full objective

Added `cache_mode` to `MicroBatchReducedObjective` and to
`tests/run_training_smoke.py`:

- `cache_mode="none"` keeps the previous memory-minimal two-pass behavior:
  assemble the global decoder normal system, then recompute rollouts for the
  loss/adjoint pass.
- `cache_mode="full"` stores the first-pass rollout data and reuses it in the
  adjoint pass.  For `lagged_adjoint`, this stores states and Picard histories;
  for `frozen_adjoint`, this stores states and lags.

This removes redundant forward rollout work without changing the full reduced
objective.  A correctness test now checks both `cache_mode="none"` and
`cache_mode="full"` against the full-batch lagged adjoint in loss and every
dynamics parameter gradient.

Measured tradeoff on ccgo3, float64/CUDA, streaming synthetic data,
`r=60`, `Hrank=60`, `d_p=d_q=150`, `batch=4096`, `micro=1024`, `steps=1000`,
`dt=1e-3`, `K=2`:

| cache mode | closure seconds | peak memory |
| --- | ---: | ---: |
| `none` | 62.286 | 13411.5 MiB |
| `full` | 40.651 | 21411.9 MiB |

The cached path is about `1.53x` faster in this test, at the cost of about
8 GiB additional peak GPU memory.  The remaining runtime is dominated by the
serial adjoint sweep and dense per-step solves, not by repeated forward rollout.

## 2026-07-03 exact skewCP SMW solve path

The previous skewCP implementation used low-rank contractions for
`H(u,u)` and the manual VJP, but the implicit midpoint solve still materialized
the frozen matrix

\[
F(\ell)=U\operatorname{diag}(W^T\ell)V^T
-V\operatorname{diag}(W^T\ell)U^T
\]

as a dense batched \(r\times r\) matrix.  This meant the most expensive
operation did not actually use the low-rank structure.

Added an exact Woodbury path for frozen skewCP systems.  With

\[
Z=[U,V],\qquad
C(\ell)=
\begin{bmatrix}
0 & \operatorname{diag}(W^T\ell)\\
-\operatorname{diag}(W^T\ell) & 0
\end{bmatrix},
\]

the frozen matrix is \(F(\ell)=ZC(\ell)Z^T\), and the forward system is

\[
\left(M_A-\tau ZC(\ell)Z^T\right)x=b,\qquad M_A=I-\tau A.
\]

The implementation solves this by

\[
x
=
y
-K
\left(I-\tau C(\ell)G\right)^{-1}
\left(-\tau C(\ell)\right)Z^Ty,
\]

where

\[
y=M_A^{-1}b,\qquad K=M_A^{-1}Z,\qquad G=Z^TK.
\]

`K` and `G` are precomputed once per rollout because they depend only on
\((A,U,V,\tau)\), not on the trajectory state.  The transpose systems in the
lagged adjoint use the corresponding cached transpose precompute once per
adjoint sweep.

Implementation changes:

- `exact_frozen_smw_solve` solves frozen skewCP systems without building
  `frozen_matrix`.
- `ExactSMWLaggedMidpointStepper` now uses the exact SMW solve in
  `step`, `rollout_with_lags`, and `rollout_with_picard_history`.
- The manual lagged adjoint uses SMW transpose solves and low-rank frozen
  transpose actions instead of dense `l_mat` materialization.
- `tests/run_speed_benchmark.py` and `tests/run_training_smoke.py` accept
  `--stepper dense|smw`.
- Core tests verify that SMW frozen solves and SMW rollouts match the dense
  reference, and that lagged adjoints still match autograd.

Representative ccgo3 float64/CUDA benchmark,
`r=128`, `Hrank=8`, `d_p=d_q=64`, `batch=512`, `steps=100`, `K=2`:

| stepper | evaluate | lagged adjoint |
| --- | ---: | ---: |
| dense | 2.6121 s | 2.8928 s |
| exact SMW | 1.2875 s | 1.3810 s |

This is the first benchmark where the low-rank skewCP structure is used in the
implicit solve itself.  The benefit depends on \(Hrank\): if \(2\,Hrank\) is
comparable to or larger than \(r\), the SMW small system may not beat the dense
solve.

## 2026-07-03 larger Hrank SMW scaling

Ran a larger single-GPU scaling check after introducing the exact SMW stepper.
The setup was

\[
r=128,\qquad d_p=150,\qquad d_q=50,\qquad N_t=1000,\qquad \Delta t=10^{-3},
\qquad K=2.
\]

For `batch=2048`, exact SMW timings were:

| Hrank | evaluate | lagged adjoint | peak memory | normal residual |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 26.3737 s | 31.8712 s | 21049.2 MiB | 5.4e-13 |
| 32 | 29.0242 s | 37.2410 s | 21048.9 MiB | 6.0e-13 |
| 64 | 38.1690 s | 55.0722 s | 22007.4 MiB | 4.8e-13 |

Dense baseline at `Hrank=64`, same setup:

| stepper | evaluate | lagged adjoint | peak memory | normal residual |
| --- | ---: | ---: | ---: | ---: |
| dense | 40.6589 s | 57.8260 s | 22007.4 MiB | 5.7e-13 |
| exact SMW | 38.1690 s | 55.0722 s | 22007.4 MiB | 4.8e-13 |

The speed benefit decreases as \(Hrank\) grows because the SMW system has size
\(2\,Hrank\).  At `Hrank=64` and `r=128`, the reduced system is already
`128 x 128`, so the SMW and dense methods are close.

Batch-size probe at `Hrank=64`:

| batch | evaluate | lagged adjoint | peak memory / status |
| ---: | ---: | ---: | --- |
| 3072 | 55.5932 s | 79.6055 s | 32734.0 MiB |
| 4096 | 72.4203 s | OOM | evaluate peak 31661.0 MiB; adjoint OOM when allocating Picard histories |

The `batch=4096` failure occurred when allocating the Picard history tensor
with shape roughly `(1000, 3, 4096, 128)` in float64, requiring about
11.72 GiB on top of the tensors already resident on GPU.  For this full-batch
cached lagged adjoint path on the current 40GB A100, `batch=3072` is the stable
upper point for this configuration.  Larger training sets should use the
streamed micro-batch full-objective wrapper.

Nsight Compute:

- `ncu` is available on ccgo3 after `module load cuda/12.4`.
- A representative `r=60, dp=150, batch=512, steps=100` lagged-adjoint run was
  profiled with `ncu --set basic --launch-skip 50 --launch-count 80`.
- The captured kernels were dominated by small batched dense linear algebra:
  `dgetf2_fused_batched_kernel`, batched `trsm`, small batched GEMM, and
  vectorized elementwise kernels.
- Nsight reported many kernels with low waves per SM and launch grids smaller
  than the 108 SMs on the A100; for example a CUTLASS dgemm kernel had grid
  size 16 and waves per SM about 0.04, and ncu flagged launch configuration as
  the optimization issue.

Interpretation: after memory fixes, the main time bottleneck is not batch size.
It is the reverse-time sweep over 1000 time steps, where each step launches many
small dense batched-solve and VJP kernels.  The next performance step is a
hand-written/fused low-rank adjoint algebra path, not simply increasing batch.

## 2026-07-03 Picard-history precompute for adjoint

Added a precompute/cache path for the finite-Picard lagged midpoint adjoint:

- `DenseLaggedMidpointStepper.rollout_with_picard_history` stores the predictor
  and all Picard iterates for each time step during the forward rollout.
- `lagged_midpoint_rollout_adjoint` can consume these cached histories, so the
  reverse sweep no longer reruns the forward Picard iteration inside every
  step-adjoint call.
- The history tensor is preallocated as one contiguous block to avoid the large
  peak memory caused by collecting 1000 small tensors and then calling
  `torch.stack`.

This is a real precompute/reuse win for dense correctness adjoints.  It trades
memory for time:

| case | before cache | after cache | peak memory after cache |
| --- | ---: | ---: | ---: |
| `r=60, dp=150, batch=512, steps=1000` | 24.3405 s | 19.6601 s | 4715.5 MiB |
| `r=60, dp=150, batch=4096, steps=1000` | 42.3581 s | 31.8425 s | 35472.3 MiB |

Component timing for `batch=512` after history caching:

| component | seconds | allocated after component |
| --- | ---: | ---: |
| midpoint inputs | 0.1211 | 2347.5 MiB |
| rollout with history | 2.9523 | 3528.6 MiB |
| normal solve | 0.4224 | 3560.2 MiB |
| decoder state grad | 0.4160 | 3794.8 MiB |
| adjoint sweep | 15.9081 | 4389.3 MiB |

The cache removes several seconds of redundant no-grad Picard recomputation.
The remaining large cost is still the local-VJP reverse sweep itself.

## 2026-07-03 manual lagged adjoint sensitivities

Replaced the dense correctness local-VJP path for the main benchmark model
`DenseLinearA + SkewCPQuadratic + LinearSource` with a hand-coded adjoint
equation:

- Each Picard substep solves the transpose linear system
  `(I - tau L(ell))^T mu = lambda`.
- The state adjoints are propagated analytically through
  `(I + tau L(ell))u + h(Bp+c)` and through
  `ell = (u + y_prev)/2`.
- Sensitivities for dense `A`, skewCP factors `(U,V,W)`, source matrix `B`, and
  source bias `c` are accumulated by explicit batched tensor contractions.
- The predictor substep uses the same explicit skewCP VJP, with both frozen
  arguments identified with `u`.
- Unsupported linear/source modules still have the older autograd-VJP fallback.

This is the intended adjoint-equation path: no `torch.autograd.grad` is used for
the main dense/skewCP/source dynamics adjoint sweep.

Correctness:

- Existing one-step lagged midpoint adjoint vs PyTorch autograd passed.
- Existing multi-step rollout adjoint vs PyTorch autograd passed.
- Existing reduced-objective `lagged_adjoint` gradient vs autograd passed.

Performance for `r=60, dp=150, dq=150, steps=1000, K=2`:

| batch | previous cached adjoint | manual adjoint | peak memory |
| ---: | ---: | ---: | ---: |
| 512 | 19.6601 s | 8.6065 s | 4506.0 MiB |
| 4096 | 31.8425 s | 25.8108 s | 35749.5 MiB |

Component profile at batch 512:

| component | seconds |
| --- | ---: |
| midpoint inputs | 0.1245 |
| rollout with history | 2.7648 |
| normal solve | 0.4224 |
| decoder state grad | 0.3962 |
| adjoint sweep | 4.8870 |

The adjoint sweep is now much closer to forward-solve complexity.  The remaining
gap is mostly from explicit sensitivity accumulation for all dynamics
parameters and from many small dense batched solves/kernels across 1000 time
steps.

## 2026-07-03 manual-adjoint Taylor test and precompute notes

Added a `--gradient-mode` option to `tests/run_taylor_sweep.py` so Taylor tests
can check the hand-coded `lagged_adjoint` path directly, rather than only the
PyTorch-autograd reference path.

Full 8-case Taylor sweep for `gradient_mode=lagged_adjoint`:

| case | zeroth slope | first-order slope |
| --- | ---: | ---: |
| seed12_p2_h004 | 1.035 | 1.998 |
| seed13_p2_h002 | 0.999 | 2.000 |
| seed21_p1_h003 | 1.007 | 1.999 |
| seed22_p3_h003 | 0.999 | 2.006 |
| seed31_rank4 | 1.005 | 2.000 |
| seed32_batch8 | 0.990 | 2.001 |
| seed41_steps8 | 0.907 | 2.000 |
| seed42_rank5_p3 | 0.968 | 2.000 |

The first-order Taylor remainder is consistently second order, so the manual
reduced-gradient path passes finite-difference checks.

Precompute assessment:

- Already used: Picard histories are precomputed during forward rollout and
  reused by the adjoint sweep.
- Useful next: for fixed dynamics parameters in one closure, precompute a base
  factorization of `M_A = I - tau A` and use SMW for the state-dependent
  low-rank skewCP update `H_ell`.
- Not useful as a global cache: the full matrix
  `M(ell)=I-tau(A+H_ell)` changes with time step, batch member, and Picard
  iterate.  Its inverse/SVD cannot be reused globally unless the same `ell`
  repeats.
- If a matrix is truly repeated, LU/Cholesky is the preferred solve precompute.
  SVD is more appropriate for rank truncation or ill-conditioned least-squares
  problems, not for routine repeated square solves.

Tried and rejected:

- Rewriting dense solve construction from `(I +/- tau L)` to diagonal in-place
  shifts plus `u + tau L u` was slightly slower on the current PyTorch/CUDA
  path, so it was not kept as an optimization.

## 2026-07-03 dense epsilon Taylor curve

Reran the manual `lagged_adjoint` Taylor test with 33 log-spaced epsilon values
from `1e-2` down to `1e-9`.  The plotted quantity is the directional derivative
error requested for Taylor testing:

`|J(theta + eps d) - J(theta) - eps <grad J, d>| / eps`.

Interpretation:

- On the asymptotic range `1e-6 <= eps <= 1e-3`, all eight cases have slope
  essentially one.
- Over the full range `1e-9 <= eps <= 1e-2`, the fitted slope is smaller because
  the curve enters the expected double-precision roundoff/cancellation region
  near the smallest epsilon values.

Representative slopes:

| case | full range | `1e-7..1e-3` | `1e-6..1e-3` |
| --- | ---: | ---: | ---: |
| seed12_p2_h004 | 0.724 | 1.004 | 1.000 |
| seed13_p2_h002 | 0.648 | 1.029 | 1.002 |
| seed21_p1_h003 | 0.574 | 0.974 | 1.000 |
| seed22_p3_h003 | 0.679 | 1.013 | 1.001 |
| seed31_rank4 | 0.556 | 0.972 | 0.999 |
| seed32_batch8 | 0.724 | 0.999 | 1.000 |
| seed41_steps8 | 0.528 | 0.975 | 0.999 |
| seed42_rank5_p3 | 0.659 | 0.994 | 0.999 |
