# GOATTM Implementation Description

This document gives a more detailed implementation-level description of the current `GOATTM` library.
It is intentionally more operational than the scope notes: the goal is to make it easy to understand
what the code is doing, how the pieces fit together, and where the remaining risks are.

## 1. High-level architecture

The library is organized into a few main layers.

### `core/`

This layer contains pure algebraic utilities:

- quadratic feature construction
- compressed quadratic representations
- `mu_H` parameterization
- stabilized linear parameterization
  \[
  A = -S S^T + W
  \]
- pullback maps from explicit matrix gradients back to structured parameters

The key design choice here is that the structured parameterization is the public one, but most
solver and derivative assembly logic still works with explicit matrix blocks internally because that
keeps the calculus simple.

### `models/`

This layer defines the reduced dynamics and decoder objects.

- `QuadraticDynamics`
- `StabilizedQuadraticDynamics`
- `QuadraticDecoder`

`StabilizedQuadraticDynamics` is a wrapper over an explicit `QuadraticDynamics` instance.
It exposes the same `rhs`, `rhs_jacobian`, and quadratic helper interfaces, but stores the
trainable linear block in stabilized coordinates.

### `solvers/`

This layer contains time discretization logic.

Originally only `implicit_midpoint` existed.
The current code now also supports `explicit_euler`.

Important files:

- `implicit_midpoint.py`
- `explicit_euler.py`
- `time_integration.py`

`time_integration.py` provides the dispatch layer. This is the piece that makes the time integrator
a configurable option rather than something hard-coded into the workflow.

### `losses/`

This layer contains trajectory-level QoI loss evaluation and the exact discrete adjoint machinery
for the currently supported time integrators.

Important file:

- `qoi_loss.py`

It is responsible for:

- observation-time trapezoidal QoI loss
- decoder gradients
- state-space loss gradients
- discrete adjoint recursion
- exact first-order dynamics gradient assembly

### `problems/`

This is the workflow layer that builds optimization-ready problems from data manifests.

Important files:

- `decoder_normal_equation.py`
- `qoi_dataset_problem.py`
- `reduced_qoi_best_response.py`

This layer handles:

- distributed forward rollout over the dataset
- decoder normal-equation best response
- reduced objective evaluation
- gradient assembly with `mu_f = mu_f^*(mu_g)`
- Hessian-action assembly
- caching

### `data/`

This layer defines the `.npz` dataset schema and dataset splitting utilities.

Important pieces:

- `NpzQoiSample`
- `NpzSampleManifest`
- explicit or seed-based train/test split
- cubic spline input interpolation

### `preprocess/`

This layer takes raw data and prepares it for reduced training.

It currently contains:

- training-statistics normalization
- materialized normalized train/test artifacts
- constrained least squares for energy-preserving quadratic fitting
- `OpInf` initialization

### `train/`

This layer builds the actual optimization interface.

Important pieces:

- `ReducedQoiTrainer`
- `ReducedQoiTrainerConfig`
- optimizer adapters
- run logging
- timing summaries
- checkpoints

## 2. Data workflow

The intended user-facing data workflow is:

1. Provide raw `.npz` samples.
2. Build a manifest.
3. Optionally split train/test explicitly, or let the library do it from a seed.
4. Apply training-statistics normalization through `preprocess/`.
5. Optionally run `OpInf` initialization.
6. Train on the resulting latent or normalized manifests.

The key point is that normalization is no longer expected to happen outside the library.
The preprocess stage is intended to own that responsibility, and the run directory now records
whether preprocessing was applied.

## 3. Reduced optimization workflow

The central optimization object is the reduced objective

\[
J(\mu_g) = J(\mu_f^*(\mu_g), \mu_g),
\]

where the decoder parameters are treated as an inner best response.

The current workflow is:

1. Fix `mu_g`.
2. Run forward rollout on the training set.
3. Assemble and solve the decoder normal equation.
4. Evaluate QoI data loss and decoder regularization.
5. Compute the reduced first derivative with the GOAM-style outer gradient path.
6. Optionally compute an exact Hessian-action.

The most important implementation decision here is that the training path no longer uses the old
slow basis-sweep reduced gradient as its default.
That older path is still useful for verification and some research experiments, but the default
optimization route is now the faster reduced objective gradient chain.

## 4. Time-integrator dispatch

The current code preserves the old midpoint path and adds an explicit-Euler path as a selectable option.

Supported values:

- `implicit_midpoint`
- `explicit_euler`

This choice now enters through the evaluator and trainer configuration, rather than being hidden in
the solver calls.

That means:

- old workflows remain available
- new workflows can switch integrators without rewriting upper layers
- testing and benchmarking can compare integrators under the same optimization interface

## 5. First-order and second-order derivatives

### First order

For each supported time integrator, the code computes the **exact discrete adjoint** for the chosen
discretization and uses that to assemble the first-order dynamics gradient.

This is important: the implemented gradient is not based on discretizing a continuous-time adjoint
after the fact.

### Second order

The library also supports exact Hessian-action evaluation for the reduced objective.

The main ingredients are:

- tangent forward rollout
- incremental discrete adjoint
- differentiation of the parameter pullback terms

The midpoint and explicit-Euler branches use different discrete formulas, but both are assembled in
the same `ReducedObjectiveWorkflow` / evaluator interface.

## 6. MPI and distributed execution

Distributed execution is currently sample-parallel.

Each rank:

- owns a static subset of the manifest
- loads only its own local samples
- computes local rollout and gradient contributions
- participates in global reductions

The decoder normal equation is built in this way:

1. local normal matrix / RHS assembly
2. `allreduce`
3. root solve
4. broadcast decoder parameters back to all ranks

The optimizer control flow is replicated, but the expensive data evaluations are distributed.
Logging and checkpoint writing are root-only.

## 7. Logging and provenance

Each training run creates a run directory.

Typical files are:

- `config.json`
- `split.json`
- `preprocess.json`
- `stdout.log`
- `stderr.log`
- `metrics.jsonl`
- `summary.txt`
- `timing_summary.txt`
- `timing_summary.json`
- `initial_parameters.npz`
- `checkpoints/`
- `failures/`

The goal is that a failed or successful run can later be reconstructed without guessing:

- what data split was used
- what preprocess was applied
- what optimizer settings were used
- what dynamics/decoder parameters were active

## 8. Current strengths

At this point the library is strongest in these areas:

- structured reduced-model parameterization
- distributed decoder best-response solve
- reduced objective gradient / Hessian-action workflow
- experiment logging and checkpointing
- exact discrete-derivative testing culture

The testing story is substantially better than in the old library because Taylor tests, unit tests,
and workflow-level smoke tests are now part of normal development.

## 9. Current weaknesses / risks

The main remaining risks are not hidden calculus bugs in the parts already Taylor-tested.
They are more often:

- bad initialization quality
- unstable latent dynamics from `OpInf`
- integrator stability limits for certain initialized models
- optimizer trial points leaving the numerically safe region

This is especially visible when a coarse or poor `OpInf` initialization is combined with a
non-A-stable explicit scheme.

## 10. Recent explicit-Euler branch status

The explicit-Euler branch is now fully wired through the first-order and second-order derivative
pipelines and validated by unit tests and Taylor tests.

However, the current `OpInf`-initialized optimization demo can still fail at the very first forward
evaluation because the initialized latent dynamics are too large in norm for the requested explicit
Euler step size.

That is a modeling / initialization stability problem, not an adjoint-formula problem.

## 11. Suggested near-term next steps

The most valuable next improvements are:

1. make `OpInf` initialization quality checks stricter
2. add safer scaling or projection on initialized latent dynamics
3. add automatic fallback logic when an explicit integrator is requested for a clearly unstable initialization
4. benchmark explicit Euler against midpoint under matched datasets and step sizes

These are the changes most likely to convert the current implementation from “correct and testable”
to “robust for the more difficult demos”.
