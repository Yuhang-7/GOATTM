# Skew-CP Parameterization For The Quadratic Term

This document is an implementation guide for replacing the dense quadratic
tensor with a `skewCP` parameterization. The design goal is simple:

- keep the energy-preserving structure exactly,
- avoid explicitly storing the dense tensor `H`,
- expose a small set of trainable matrices that can be optimized like the other
  model parameters.

## Object Being Parameterized

The latent dynamics contain a quadratic term:

```text
dx/dt = A x + H (x tensor x) + B p + c
```

The energy-preserving requirement is:

```text
x^T H (x tensor x) = 0, for every x
```

Rather than parameterizing `H` directly, parameterize an auxiliary tensor `N`
with skew-symmetry in the first two indices:

```text
N_{k i j} = -N_{i k j}
```

Then define:

```text
H_{k i j} = N_{k i j} + N_{k j i}
```

This construction guarantees the energy constraint by design.

## Skew-CP Definition

Skew-CP is a low-rank matrix factorization of the natural unfolding:

```text
N in Lambda^2(R^d) tensor R^d
```

The left factor is restricted to decomposable bivectors. A rank-one component is:

```text
(u^(a) wedge v^(a)) tensor z^(a)
```

The coordinate formula is:

```text
N_{k i j}
= sum_{a=1}^R
  (u_k^(a) v_i^(a) - v_k^(a) u_i^(a))
  z_j^(a)
```

This is the main parameterization. It automatically enforces
`N_{k i j} = -N_{i k j}`.

Implementation parameters:

```text
U[:, a] = u^(a)
V[:, a] = v^(a)
Z[:, a] = z^(a)
```

with:

```text
U, V, Z: shape (d, R)
```

Do not add separate component weights in the first implementation. They can be
absorbed into `Z`.

## Forward Formula

Never form `N` or `H` during training.

For a single state `x`, compute:

```text
alpha = U.T @ x
beta  = V.T @ x
gamma = Z.T @ x
```

Then:

```text
H (x tensor x)
= 2 sum_{a=1}^R gamma_a (beta_a u^(a) - alpha_a v^(a))
```

Vectorized single-state implementation:

```text
quad = 2.0 * (U @ (gamma * beta) - V @ (gamma * alpha))
```

Batched implementation for `X` with shape `(n, d)`:

```text
alpha = X @ U
beta  = X @ V
gamma = X @ Z
quad = 2.0 * ((gamma * beta) @ U.T - (gamma * alpha) @ V.T)
```

This should be the only forward path used by the dynamics code.

## Complexity

For latent dimension `d` and skew-CP rank `R`:

```text
parameter count = 3 d R
forward cost    = O(d R) per state
memory          = O(d R)
```

This replaces dense storage of `H`, which would scale like `O(d^3)`.

## Implementation Checklist

- Add a quadratic form option, for example `quadratic_form = "skew_cp"`.
- Add trainable matrices `U`, `V`, and `Z` with shape `(d, R)`.
- Route the quadratic dynamics through the forward formula above.
- Keep the existing dense/general quadratic code path untouched.
- Do not materialize `H` except in optional debug utilities.
- If a dense `H` export is needed, build it from `N` only for inspection.

## Minimum Tests

- Single-state output has shape `(d,)`.
- Batched output has shape `(n, d)`.
- Batched output matches a Python loop over single states.
- Energy identity holds numerically:

```text
x dot quad(x) = 0
```

- Gradients flow through `U`, `V`, and `Z`.
- Parameter packing and unpacking preserve all shapes.

## Notes

The decomposition is not identifiable: scaling can move between `U`, `V`, and
`Z`, and different component choices can represent the same `N`. This is fine
for the first implementation. Add normalization or regularization only if
optimization becomes unstable.
