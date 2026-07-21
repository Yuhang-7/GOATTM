# Design Notes

## Exactness modes

The library must expose correctness-first modes before accelerated modes.

- `DenseLaggedMidpointStepper`: dense GPU solve of every frozen lagged midpoint
  linear system. This is the baseline and supports a general dense `A`.
- `ExactSMWLaggedMidpointStepper`: reserved API for exact low-rank/SMW reduced
  solves. It must match the dense stepper to roundoff.
- `SkewCPDefectLaggedMidpointStepper`: adaptive defect iteration. It must report
  the iteration count and residual once the reduced solver is implemented, and
  it must be able to fall back to exact solve.

## A operator policy

GOATTM's skewCP dynamics allow a general dense linear term `A`. Therefore
`DenseLinearA` is the reference operator. Dissipative/skew low-rank `A` is an
optional structured model, not the only model class.

## Adjoint policy

The first target is the exact frozen-lag discrete adjoint: the lag state is
treated as fixed, while the frozen linear system is differentiated exactly with
respect to `u`, `A`, skewCP `H`, and source parameters. Full differentiation
through lag predictors/Picard lag updates is a separate mode and should not be
mixed with the frozen-lag semantics.

For the dense P0 correctness path, the reduced objective also exposes an
`autograd` gradient mode. This differentiates the actual finite-Picard dense
rollout graph while treating the decoder best response by the envelope theorem.
This mode is the Taylor-test reference for BFGS/LBFGS. The `frozen_adjoint`
mode is the first hand-coded adjoint path and is checked separately against its
own frozen-lag semantics.

## Variable projection

The decoder readout is eliminated by a ridge normal equation for fixed latent
states. The reduced objective layer should call the normal solve before
evaluating dynamics gradients.

## Training layer commitments

BFGS/LBFGS is a required training capability, not an optional extra.  Once the
P0 IO, dense rollout, normal solve, and exact adjoint path are stable, the next
training layer should expose a reduced objective callable suitable for BFGS and
LBFGS line-search optimizers.

GOATTM-style initialization should be migrated by following the existing GOATTM
logic as closely as possible: preprocessing/normalization, OpInf or exported
initial dynamics, decoder template initialization, rollout validation, and
artifact logging.  The GPU rewrite should change the tensor backend and batching
strategy, not the mathematical initialization story.
