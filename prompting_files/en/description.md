# GOATTM Library Description

This document gives a code-faithful description of the current `GOATTM` library.
It is intended to help future prompting, design discussions, and onboarding stay aligned
with what the repository actually implements today.

## 1. What the library is

`GOATTM` is a reduced-order modeling and QoI-training library centered on:

- latent quadratic dynamics,
- quadratic decoders from latent states to QoIs,
- exact discrete derivatives for selected time integrators,
- decoder best-response reduction, and
- optimization workflows that can run in serial or sample-parallel MPI mode.

The package is not just a collection of experiments. It already exposes a real public API
through `src/goattm/__init__.py`, with reusable components for data loading, preprocessing,
rollout, loss evaluation, reduced optimization, and training logs/checkpoints.

## 2. Main package structure

### `core/`

Pure algebra and parameterization helpers live here. This includes:

- quadratic feature construction,
- compressed quadratic tensor representations,
- `mu_h` parameterization utilities,
- stabilized linear parameterization of the form `A = -S S^T + W`,
- pullback maps from explicit matrix gradients to structured parameters.

This layer is intentionally low level and mostly NumPy-only.

### `models/`

This layer defines the state-space and decoder objects:

- `QuadraticDynamics`,
- `StabilizedQuadraticDynamics`,
- `QuadraticDecoder`.

`StabilizedQuadraticDynamics` stores the linear block in structured stabilized coordinates
but still exposes explicit state-space operators such as `rhs` and Jacobian-related helpers.

### `solvers/`

This layer contains rollout and tangent/discrete-adjoint support for the available time integrators.

The current public dispatcher supports three integrators:

- `implicit_midpoint`,
- `explicit_euler`,
- `rk4`.

This is an important correction relative to older notes: the library is no longer only a
midpoint-plus-explicit-Euler codebase. `rk4` is wired into the public time-integration layer,
the loss layer, and the training configuration, and `ReducedQoiTrainerConfig` currently defaults
to `rk4`.

The implicit-midpoint nonlinear solve now has a homotopy continuation fallback.  The solver first
tries the direct Newton solve for the full midpoint residual.  If that fails, it tracks the root of

`F_lambda(u_next) = u_next - u_prev - lambda * dt * f((u_prev + u_next) / 2)`

from `lambda=0` to `lambda=1`, using a default lambda step of `0.1` and halving the step locally
when a subproblem fails.  This is meant to keep Newton on the physical root branch rather than
jumping to a far-field spurious root of the quadratic implicit equation.

### `losses/`

This layer evaluates trajectory-level QoI losses and assembles first-order derivatives.

Important responsibilities include:

- trapezoidal observation weighting,
- decoder partial assembly,
- state loss derivatives,
- exact discrete adjoint recursion,
- exact first-order dynamics gradient assembly for supported integrators.

At the first-order level, the implementation supports midpoint, explicit Euler, and RK4 branches.

### `problems/`

This is the reduced-objective workflow layer. It contains:

- decoder normal-equation assembly/solve,
- dataset-level loss-and-gradient evaluation,
- decoder best-response reduction,
- reduced objective preparation,
- reduced gradient and Hessian-action evaluation.

This layer is the bridge between local derivative formulas and optimization-ready objectives.

One subtle but important caveat:

- the higher-level reduced-objective workflow accepts all public time integrators;
- some lower-level helper routines are still specialized.

For example, `decoder_normal_equation.py` currently rolls trajectories with
`rollout_implicit_midpoint_to_observation_times`, so that helper is still midpoint-specific even
though the broader training stack is more general.

### `data/`

This layer defines the `.npz` dataset contract and manifest helpers.

Current data support includes:

- `NpzQoiSample`,
- `NpzSampleManifest`,
- reproducible train/test splits,
- explicit train/test splits by sample id,
- cubic-spline input interpolation,
- piecewise-linear input interpolation,
- save/load helpers for samples and manifests.

The expected sample payload is latent-state initial data plus QoI observations, with optional
time-varying inputs and free-form metadata.

### `preprocess/`

This layer prepares datasets and initial models before training.

Implemented capabilities include:

- train-set normalization for QoIs and optional inputs,
- materialization of normalized train/test datasets,
- constrained least-squares helpers for energy-preserving quadratic structure,
- `OpInf`-based initialization of a reduced stabilized model,
- optional latent embedding construction before regression.

Normalization is currently centered max-absolute scaling rather than z-score scaling.  For each
QoI/input dimension, the train-set mean is subtracted and the centered values are divided by
`max_train_abs / target_max_abs`, with `target_max_abs=0.9` by default.  This keeps the normalized
training set within about `[-0.9, 0.9]` per dimension while still recording the scale needed for
denormalization.

`OpInf` preprocessing is more than a single regression call: it can also normalize data,
materialize latent datasets, validate forward rollouts, and record initialization provenance.

### `train/`

This layer exposes the optimization and logging interface.

Key components include:

- `ReducedQoiTrainer`,
- `ReducedQoiTrainerConfig`,
- updater configs for Adam, gradient descent, L-BFGS, and Newton-action steps,
- metrics logging,
- timing summaries,
- checkpoints,
- run-directory provenance records.

The trainer is designed around the reduced objective in which decoder parameters are treated
as an inner best response to the current dynamics parameters.

## 3. Data and preprocessing workflow

The intended workflow is roughly:

1. Build or load a `.npz` manifest.
2. Split the dataset into train/test sets, either explicitly or from a seed.
3. Optionally compute normalization statistics and materialize normalized artifacts.
4. Optionally run `OpInf` initialization to produce a latent dataset, an initialized dynamics model,
   and an initialized decoder.
5. Train using the reduced QoI objective on the resulting manifests.

Two clarifications matter here:

- normalization is now a library-owned preprocessing step, not something that must happen outside;
- the preprocessing stage writes artifacts and provenance so later runs can be reconstructed.

## 4. Optimization model

The central training object is the reduced best-response objective

`J(mu_g) = J(mu_f*(mu_g), mu_g)`,

where the decoder parameters are solved as an inner problem for each current choice of dynamics
parameters.

Operationally, the workflow is:

1. choose dynamics parameters,
2. roll out trajectories over the dataset,
3. solve the decoder normal equation,
4. evaluate the QoI loss and regularization terms,
5. assemble the reduced gradient,
6. optionally assemble a reduced Hessian action or explicit Hessian.

This is one of the main design differences from the older GOAM-style code: the library is built
around a reusable reduced-objective workflow instead of only one-off experiment scripts.

## 5. Derivatives and second-order support

### First order

For supported branches, the implementation uses exact discrete derivatives of the chosen
time discretization. It does not merely discretize a continuous-time adjoint afterward.

### Second order

The library also supports reduced Hessian-action evaluation, and the trainer exposes Newton-style
updates that can use either Hessian actions or an explicitly assembled reduced Hessian.

This second-order machinery is present for midpoint, explicit Euler, and RK4 paths in the main
reduced-objective evaluator.

## 6. Distributed execution

MPI support is sample-parallel.

Each rank owns a subset of manifest entries, loads only its local samples, computes local
contributions, and participates in collective reductions.

Manifest partitioning is round-robin by sample index (`idx % world_size == rank`).  It is not
locked to one ODE per rank: a rank may own many samples, one sample, or no samples, depending on
`ntrain` and MPI world size.

Typical distributed patterns include:

- dataset loss accumulation,
- decoder normal-equation assembly,
- reduced gradient accumulation,
- Hessian-action accumulation.

L-BFGS is root-led.  Only rank 0 owns the SciPy `minimize(..., method="L-BFGS-B")` object.  Worker
ranks sit in a command loop and respond to root broadcasts such as `evaluate`, `snapshot`, `stop`,
and `abort`.  This avoids the previous confusing situation where every MPI rank appeared to own an
independent optimizer state.

Forward rollout failures are collective-safe.  Each rank catches failures from any local sample,
records sample id/index/path/reason, and then all ranks participate in failure collection.  During
L-BFGS objective evaluation, a failed rollout returns a large penalty objective/gradient to steer
the optimizer away from the candidate point.  Failures during mandatory snapshots or initialization
are treated as abort conditions because those points are being accepted or recorded.

Logging and checkpoint writing are primarily root-rank responsibilities.

## 7. Provenance and run artifacts

Training runs create output directories with records such as:

- `config.json`,
- `split.json`,
- `preprocess.json`,
- `metrics.jsonl`,
- `summary.txt`,
- `timing_summary.txt`,
- `timing_summary.json`,
- checkpoints,
- failure records,
- stdout/stderr logs.

This is a meaningful strength of the current codebase: it aims to make experiments replayable
instead of relying on ad hoc notebook state.

## 8. What is strong today

The current library is strongest in:

- structured latent quadratic parameterization,
- reduced best-response optimization organization,
- discrete derivative support across multiple integrators,
- root-led L-BFGS with MPI sample-parallel dataset evaluation,
- preprocessing and initialization provenance,
- centered max-absolute normalization that keeps normalized training data bounded,
- implicit-midpoint homotopy fallback for difficult nonlinear solves,
- test-oriented development around rollout/derivative correctness.

## 9. What should be described carefully

The library description should avoid overstating a few things:

- not every low-level helper is fully integrator-agnostic yet;
- `OpInf` initialization can still produce models whose forward rollout is numerically fragile;
- explicit schemes can be limited by stability regions even when adjoint formulas are correct;
- implicit midpoint is more robust with homotopy fallback, but this does not remove all nonlinear
  solve or bad-parameter failure modes;
- L-BFGS penalty handling treats failed candidate rollouts as infeasible points; if a data sample is
  intrinsically bad, that should be diagnosed separately rather than hidden as an optimizer issue;
- some modules still reflect active research iteration rather than a fully frozen product API.

So the right framing is:

- the repository already contains a substantial reusable library,
- the public workflow is broader than the old midpoint-only or explicit-Euler-only notes suggest,
- but a few internals still encode narrower assumptions and should be documented honestly.

## 10. Recommended prompt framing for future work

When prompting against this repository, it is more accurate to describe `GOATTM` as:

`A reduced-order modeling and reduced-QoI training library for quadratic latent dynamics, with
structured parameterizations, multiple time integrators, exact discrete derivatives, decoder
best-response optimization, bounded preprocessing normalization, root-led MPI L-BFGS training, and
homotopy-guarded implicit nonlinear solves.`

That description is closer to the current code than framing it as only an explicit-Euler branch,
only a midpoint implementation, or only an `OpInf` experiment sandbox.
