## Nonlinear Quadratic Solver Comparison Test

### Purpose

This module test studies the nonlinear per-step solve that appears when we replace the legacy Crank-Nicolson step with the **implicit midpoint** rule for quadratic latent dynamics, and compares the two discretizations directly.

The target dynamics are

`du/dt = A u + H(u, u) + c`

with:

- `A + A^T` negative definite,
- `H` symmetric in its two state arguments,
- and `u^T H(u, u) = 0` for all `u`, i.e. `H` is energy-preserving.

This test is motivated by the observation that the legacy solver path used a CN discretization together with an ad hoc nonlinear step solver. That old approach did not guarantee that a valid `u_{n+1}` satisfying the discrete equation would be found for a given step size.


### New Testing Goal

We want to test the following design direction:

1. Use **implicit midpoint** instead of CN for forward evolution.
2. Do not assume we know a usable bound on the operator norm of `H`.
3. If a chosen `dt` fails, retry with `dt <- 0.8 * dt`.
4. Use **Newton-Raphson iteration** for the nonlinear midpoint equation.
5. Exploit the fact that for quadratic dynamics all residuals and Jacobians are easy to compute explicitly.
6. Compare the behavior of the new midpoint rule against the legacy CN rule under the same random test regime.


### Discrete Equations

Given a current state `u_n` and a step size `dt`, the implicit midpoint update solves for `u_{n+1}` such that

`u_{n+1} - u_n - dt * F((u_n + u_{n+1}) / 2) = 0`

where

`F(u) = A u + H(u, u) + c`.

Defining the midpoint

`m = (u_n + u_{n+1}) / 2`,

the residual is

`R(u_{n+1}) = u_{n+1} - u_n - dt * (A m + H(m, m) + c)`.

Because `H` is quadratic and symmetric, the Jacobian is available in closed form:

`J = I - 0.5 * dt * A - dt * H(m, ·)`.

This makes Newton-Raphson a natural per-step nonlinear solver.

For comparison, the CN step solves

`u_{n+1} - u_n - 0.5 * dt * (F(u_n) + F(u_{n+1})) = 0`.

For quadratic dynamics, its Jacobian is also explicit:

`J_CN = I - 0.5 * dt * A - dt * H(u_{n+1}, ·)`.


### Practical Step-Retry Rule

In theory, existence and uniqueness can be derived under structural assumptions.
In practice, we still need a robust numerical path that does not depend on knowing `||H||`.

The test therefore uses the following practical rule:

- try solving the midpoint equation with a chosen `dt`,
- if Newton fails,
- reduce the step by `dt <- 0.8 * dt`,
- and retry until success or until a minimum step threshold is reached.

This is meant to reflect the intended design direction for the rewrite.


### What Is Tested

For each `r in {10, 15, 20}`, the script:

1. Generates random matrices/tensors:
   - `A` with `A + A^T` strictly negative definite,
   - `H` energy-preserving,
   - `c` random with a deliberately larger scale than before.
2. Tests a **single nonlinear solve** for both implicit midpoint and CN.
3. Tests a **full forward rollout** over a fixed time horizon for both methods.
4. Records:
   - success/failure counts,
   - number of Newton failures,
   - number of step-size reductions,
   - accepted step counts for rollouts.


### Files

- `run_midpoint_quadratic_tests.py`: executable module test script.


### How To Run

From the repository root:

```bash
python module_test/nonlinear_quadratic_midpoint/run_midpoint_quadratic_tests.py
```


### Deterministic Test Configuration

The script uses a fixed RNG seed and currently tests:

- `r = 10, 15, 20`
- methods: `midpoint`, `cn`
- 20 single-step solve trials for each `r`
- 10 rollout trials for each `r`
- `c_scale = 1.0`
- `t_final = 1.0`
- `n_time_steps = 100`
- `dt_initial = t_final / n_time_steps = 0.01`

This keeps the test reproducible and lightweight enough for iterative module work.


### Observed Results

The current observed results from the module test script are:

| method | r | step solve success | step solve failure | step dt reductions | step Newton failures | rollout success | rollout failure | rollout dt reductions | rollout Newton failures | rollout accepted steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| midpoint | 10 | 20 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 1000 |
| midpoint | 15 | 20 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 1000 |
| midpoint | 20 | 20 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 1000 |
| cn | 10 | 20 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 1000 |
| cn | 15 | 20 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 1000 |
| cn | 20 | 20 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 1000 |


### Interpretation

For the current random test regime:

- the statistics below should now be interpreted for 100 nominal time steps per rollout,
- midpoint and CN are still being judged on the same random test regime,
- and any extra `dt *= 0.8` triggers should be treated as meaningful stress-test information.

Even if both methods succeed frequently, the midpoint rule remains the preferred rewrite target because it matches the intended discrete structure more cleanly for the energy-preserving quadratic setting.
