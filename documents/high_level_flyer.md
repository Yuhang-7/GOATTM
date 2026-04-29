# GOATTM High-Level Flyer

## What GOATTM is

`GOATTM` is a reduced-order modeling library for training quadratic latent dynamics against
quantity-of-interest (QoI) observations.

It combines:

- structured latent dynamics models,
- quadratic decoders from latent states to observables,
- exact discrete derivative machinery,
- decoder best-response reduction,
- MPI-aware training and evaluation workflows.

## What it is good at

`GOATTM` is designed for workflows where we want to:

- fit latent reduced models from time-dependent data,
- keep the dynamics parameterization structured and interpretable,
- train against QoI trajectories rather than only full-state reconstruction,
- use exact discrete gradients and Hessian actions,
- scale dataset evaluation across samples with MPI.

## Main ingredients

The library is organized into a few practical layers:

- `core`: algebra and parameterization utilities
- `models`: latent dynamics and decoder objects
- `solvers`: rollout support for `implicit_midpoint`, `explicit_euler`, and `rk4`
- `losses`: QoI losses and discrete adjoints
- `problems`: decoder best-response and reduced-objective assembly
- `preprocess`: normalization and `OpInf` initialization
- `train`: optimizers, checkpoints, metrics, and run logging

## Practical workflow

A typical workflow looks like:

1. load `.npz` samples and build a manifest,
2. split train/test data,
3. optionally normalize the dataset,
4. optionally run `OpInf` initialization,
5. train the reduced model with decoder best-response updates,
6. inspect metrics, checkpoints, and timing summaries.

## Current status

The codebase already behaves like a reusable library, not only a research scratchpad.

Its strongest areas today are:

- structured quadratic reduced dynamics,
- reduced best-response training organization,
- exact discrete derivative support across multiple integrators,
- provenance-aware preprocessing and training outputs,
- MPI-aware dataset evaluation.

## Important caveat

The high-level workflow is broader than some older notes suggest, but a few lower-level helpers
still carry narrower assumptions. In particular, some internals remain midpoint-specific even
though the public training stack now supports multiple integrators.

## One-sentence description

`GOATTM is a reduced-order modeling and reduced-QoI training library for quadratic latent dynamics,
with structured parameterizations, multiple time integrators, exact discrete derivatives, decoder
best-response optimization, preprocessing, and MPI-aware training workflows.`
