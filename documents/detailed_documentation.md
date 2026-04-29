# GOATTM Detailed Documentation

## 1. Scope

This document is a detailed guide to the current `GOATTM` codebase. It focuses on the parts that
already exist in `src/goattm`, how they fit together, and what assumptions or limitations are
important for users and future developers.

## 2. Conceptual model

The library works with reduced latent states `u(t)` and trains two coupled pieces:

- a latent dynamics model that advances `u(t)` in time,
- a decoder that maps latent states to observed quantities of interest.

The training objective is not organized as a naive joint optimization over all dynamics and decoder
parameters at once. Instead, the library is built around a reduced best-response viewpoint:

- the dynamics parameters are the outer variables,
- the decoder parameters are treated as an inner least-squares best response.

This makes the library especially suited for GOAM-style reduced training pipelines where QoI fit,
structured dynamics, and efficient reduced derivatives all matter.

## 3. Public package layout

### `goattm.core`

This module family contains low-level algebraic utilities used by the rest of the package.

Responsibilities include:

- generating quadratic feature vectors,
- converting between explicit and compressed quadratic representations,
- handling the `mu_h` parameterization,
- constructing stabilized linear operators from `S` and `W` parameters,
- pulling explicit gradients back to structured coordinates.

This is the mathematical infrastructure layer. It does not define training workflows by itself.

### `goattm.models`

The model layer exposes the main mathematical objects:

- `QuadraticDynamics`
- `StabilizedQuadraticDynamics`
- `QuadraticDecoder`

`QuadraticDynamics` represents the latent state equation with linear, quadratic, input, and bias
terms. `StabilizedQuadraticDynamics` stores the linear block through stabilized coordinates while
still exposing explicit operators. `QuadraticDecoder` maps latent states to QoIs using linear,
quadratic, and affine decoder terms.

### `goattm.data`

This layer defines the library's `.npz` data contract.

Important objects:

- `NpzQoiSample`
- `NpzSampleManifest`
- `NpzTrainTestSplit`

The current sample structure includes:

- `sample_id`
- `observation_times`
- latent initial condition `u0`
- `qoi_observations`
- optional `input_times`
- optional `input_values`
- optional metadata

The module also supports:

- manifest persistence,
- deterministic seeded train/test splits,
- explicit train/test splits by sample ids,
- cubic-spline input interpolation,
- piecewise-linear input interpolation.

This is one of the areas where the library is already quite clean and reusable.

### `goattm.solvers`

This layer handles rollout logic, tangent propagation, and integrator dispatch.

The public dispatcher currently supports:

- `implicit_midpoint`
- `explicit_euler`
- `rk4`

This matters because some older project notes still read as though `GOATTM` were mainly a
midpoint codebase with a newer explicit-Euler branch attached. That is no longer the best summary.
`rk4` is part of the public integration interface and is also the current default in
`ReducedQoiTrainerConfig`.

Integrator support is not only forward rollout. The package also includes derivative-related
helpers such as:

- discrete adjoints,
- incremental discrete adjoints,
- parameter-gradient accumulation,
- Hessian-action term accumulation.

### `goattm.losses`

The losses layer evaluates observation-aligned QoI objectives and first-order derivatives.

It currently includes:

- trapezoidal quadrature weights,
- decoder partial assembly,
- state-space residual derivatives,
- exact discrete adjoints for supported integrators,
- exact first-order parameter gradient assembly.

The important implementation fact is that these are discrete derivatives of the chosen scheme,
not post hoc continuous-adjoint approximations.

### `goattm.problems`

This is the workflow layer for building optimization-ready reduced objectives.

Key capabilities include:

- dataset-level QoI loss/gradient evaluation,
- decoder normal-equation assembly and solve,
- reduced objective preparation,
- reduced gradient evaluation,
- reduced Hessian-action evaluation,
- explicit reduced Hessian assembly by repeated actions.

This layer is where the library starts to feel like a coherent optimization framework instead of
just a set of numerical kernels.

There is one nuance worth calling out very explicitly:

- the main reduced-objective evaluator accepts the public time-integrator choices,
- but not every lower-level helper is equally general.

In particular, `decoder_normal_equation.py` currently assembles decoder normal equations from
midpoint rollouts specifically. That does not invalidate the broader multi-integrator workflow,
but it does mean that "full integrator-agnosticism everywhere" would be an overstatement.

### `goattm.preprocess`

This layer is responsible for preparing data and initial models before training.

Current capabilities:

- QoI normalization using training statistics,
- optional input normalization,
- materialization of normalized train/test artifacts,
- constrained least-squares utilities for energy-preserving quadratic fitting,
- `OpInf`-based initialization of a stabilized latent model,
- latent embedding construction before regression,
- validation of initialized models through forward rollouts.

The `OpInf` initialization path is substantial. It does more than estimate coefficients:

- it can normalize the data,
- generate latent datasets,
- fit a structured dynamics model,
- validate rollouts,
- record artifacts and diagnostics.

### `goattm.train`

This layer exposes the training interface and run management.

Important pieces:

- `ReducedQoiTrainer`
- `ReducedQoiTrainerConfig`
- optimizer/update configs for Adam, gradient descent, L-BFGS, and Newton-action steps
- metrics and summary logging
- checkpointing
- timing instrumentation
- output-directory provenance

The trainer operates on the reduced best-response objective and records the full run context to
make later reconstruction possible.

## 4. End-to-end workflow

The intended workflow today is:

1. Prepare `.npz` samples and a manifest.
2. Create a train/test split, either by seed or explicit ids.
3. Optionally normalize the training and test data.
4. Optionally run `OpInf` initialization to produce a latent dataset and initial model.
5. Build the reduced training workflow.
6. Optimize the outer dynamics parameters while solving the decoder inner problem repeatedly.
7. Inspect run logs, checkpoints, metrics, and timing summaries.

This is a cleaner and more self-contained workflow than the older pattern where preprocessing,
initialization, and optimization logic were scattered across external scripts.

## 5. Reduced best-response formulation

The core reduced training idea is:

`J(mu_g) = J(mu_f*(mu_g), mu_g)`

where:

- `mu_g` denotes the outer dynamics parameters,
- `mu_f*(mu_g)` is the decoder best response for those dynamics.

Operationally, one evaluation of the reduced objective usually requires:

1. distributed forward rollout across the training samples,
2. decoder normal-equation assembly,
3. decoder solve,
4. QoI loss evaluation,
5. reduced gradient assembly,
6. optional Hessian-action assembly.

This structure is central to the identity of the library and should show up in any serious
description of what `GOATTM` does.

## 6. Time-integrator support

At the public solver level, the library supports:

- `implicit_midpoint`
- `explicit_euler`
- `rk4`

At the workflow level:

- rollout dispatch supports all three,
- first-order loss gradients support all three,
- reduced Hessian actions support all three,
- trainer configuration accepts all three.

At the helper level:

- some routines remain specialized,
- midpoint still appears in certain lower-level assembly paths.

So the accurate wording is: the public training stack is multi-integrator, while some internals are
still partially specialized.

## 7. Derivative support

### First-order derivatives

The code includes exact discrete-adjoint-based first-order derivatives for supported time
integrators. This is a major strength because it keeps optimization logic aligned with the actual
discretized dynamics used in forward rollout.

### Second-order derivatives

The reduced-objective workflow also supports Hessian-action computations and can explicitly build a
reduced Hessian by repeated action evaluations. This enables Newton-style or quasi-Newton-like
experiments at the reduced level.

The trainer already reflects this design by exposing a `NewtonActionUpdater` with configurable
action-based or explicit-Hessian modes.

## 8. MPI execution model

Distributed execution is sample-parallel.

Each rank:

- owns a subset of manifest entries,
- loads only local samples,
- computes local rollout/loss/gradient contributions,
- participates in global reductions.

This pattern is used in:

- dataset loss evaluation,
- decoder normal-equation assembly,
- reduced gradient assembly,
- Hessian-action assembly.

Root-rank responsibilities typically include:

- solving certain global systems,
- writing consolidated logs,
- writing checkpoints and summaries.

## 9. Output artifacts and provenance

The training and preprocessing code aim to preserve reproducibility by writing structured outputs,
for example:

- `config.json`
- `split.json`
- `preprocess.json`
- `metrics.jsonl`
- `summary.txt`
- `timing_summary.txt`
- `timing_summary.json`
- checkpoint files
- stdout/stderr logs
- failure records

This is an underrated strength of the library because it reduces ambiguity around how a result was
actually produced.

## 10. Strengths

The codebase is currently strongest in the following areas:

- structured parameterizations for quadratic latent dynamics,
- reduced best-response training organization,
- exact discrete derivatives across multiple solver branches,
- MPI-aware dataset evaluation,
- preprocessing and initialization artifact management,
- test-driven verification habits around rollout and derivative logic.

## 11. Risks and limitations

The main limitations are not best summarized as "the math is broken." A more accurate list is:

- `OpInf` initialization can still generate numerically aggressive models,
- explicit schemes can become unstable for poor initializations or large step sizes,
- some low-level helpers still encode midpoint assumptions,
- parts of the package are still evolving research infrastructure rather than a frozen product API.

That means downstream documentation should be ambitious but careful:

- strong enough to reflect the real library value,
- honest enough not to claim universal generality where the code is still specialized.

## 12. Recommended canonical description

If we want one library description that is both concise and accurate, a good default is:

`GOATTM is a reduced-order modeling and reduced-QoI training library for quadratic latent dynamics,
with structured parameterizations, multiple time integrators, exact discrete derivatives, decoder
best-response optimization, preprocessing, and MPI-aware training workflows.`
