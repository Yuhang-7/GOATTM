# Explicit Euler Time Discretization and Exact Discrete Adjoint

This note records the explicit-Euler branch that is currently implemented in `GOATTM`.
It supersedes the earlier midpoint-only path for the new optional
`time_integrator="explicit_euler"` workflow.

## 1. Forward scheme

We consider the reduced dynamics

\[
\dot u = f(u,t;\mu_g)
= A u + H(u,u) + B p(t) + c.
\]

With explicit Euler and step size \(\Delta t_n\),

\[
u_{n+1} = u_n + \Delta t_n f(u_n,t_n;\mu_g).
\]

If the rollout is aligned to observation times, the final step on each interval is shortened so that the
trajectory lands exactly on the target observation time.

## 2. Exact discrete adjoint

Let the discrete objective be

\[
J = \sum_{n=0}^{N} \ell_n(u_n),
\]

where \(\ell_n\) is the observation-space loss contribution pulled back to state space.
Define

\[
g_n := \frac{\partial \ell_n}{\partial u_n}.
\]

For the explicit Euler step

\[
F_n(u_n) = u_n + \Delta t_n f(u_n,t_n),
\]

the exact discrete adjoint recursion is

\[
\lambda_N = g_N,
\]

\[
\lambda_n = g_n + \left(I + \Delta t_n \frac{\partial f}{\partial u}(u_n,t_n)\right)^T \lambda_{n+1},
\qquad n=N-1,\dots,0.
\]

This is the exact reverse-mode differentiation of the explicit Euler map, not a continuous-time adjoint
that is later discretized.

## 3. Exact first-order parameter gradient

For one step,

\[
u_{n+1} = u_n + \Delta t_n f(u_n,t_n;\mu_g),
\]

the parameter contribution is

\[
\frac{\partial J}{\partial \mu_g}
= \sum_{n=0}^{N-1}
\left(\frac{\partial f}{\partial \mu_g}(u_n,t_n)\right)^T
\Delta t_n \lambda_{n+1}.
\]

For the explicit matrix blocks \((A,H,B,c)\),

\[
\delta f
= \delta A\,u_n + \delta H\,\zeta(u_n) + \delta B\,p(t_n) + \delta c,
\]

so the exact stepwise pullbacks are

\[
\nabla_A J \;{+}{=}\; \Delta t_n\, \lambda_{n+1} u_n^T,
\]

\[
\nabla_H J \;{+}{=}\; \Delta t_n\, \lambda_{n+1} \zeta(u_n)^T,
\]

\[
\nabla_B J \;{+}{=}\; \Delta t_n\, \lambda_{n+1} p(t_n)^T,
\]

\[
\nabla_c J \;{+}{=}\; \Delta t_n\, \lambda_{n+1}.
\]

After this explicit-matrix gradient is assembled, `GOATTM` pulls it back to the structured
parameterization \((S,W,\mu_H,B,c)\) when stabilized dynamics are used.

## 4. Incremental forward / tangent scheme

Given a direction \(\delta \mu_g\), define \(\tilde u_n := D u_n[\delta\mu_g]\).
Then the exact tangent recursion is

\[
\tilde u_0 = 0,
\]

\[
\tilde u_{n+1}
= \tilde u_n
+ \Delta t_n
\left(
\frac{\partial f}{\partial u}(u_n,t_n)\tilde u_n
+ \frac{\partial f}{\partial \mu_g}(u_n,t_n)[\delta\mu_g]
\right).
\]

This is exactly what the implemented
`rollout_explicit_euler_tangent_from_base_rollout(...)` computes.

## 5. Exact incremental adjoint

Let

\[
\tilde g_n := D g_n[\delta\mu_g].
\]

Differentiating the discrete adjoint recursion gives the exact incremental adjoint

\[
\tilde\lambda_N = \tilde g_N,
\]

\[
\tilde\lambda_n
= \tilde g_n
+ \left(I + \Delta t_n \frac{\partial f}{\partial u}(u_n,t_n)\right)^T \tilde\lambda_{n+1}
+ \Delta t_n
\left(
D\!\left[\frac{\partial f}{\partial u}(u_n,t_n)\right][\tilde u_n,\delta\mu_g]
\right)^T
\lambda_{n+1}.
\]

For quadratic dynamics,

\[
\frac{\partial f}{\partial u}(u_n,t_n)
= A + J_H(u_n),
\]

and its directional derivative is

\[
\delta J_n
= \delta A + J_H(\tilde u_n) + J_{\delta H}(u_n).
\]

This is the quantity provided to the incremental adjoint solver through the callback
`jacobian_direction(...)`.

## 6. Exact Hessian-action assembly

Once \(\tilde u_n\) and \(\tilde\lambda_n\) are available, the explicit matrix-block Hessian action follows
from differentiating the first-order pullbacks:

\[
\delta(\nabla_A J)\;{+}{=}\;
\Delta t_n\left(\tilde\lambda_{n+1} u_n^T + \lambda_{n+1}\tilde u_n^T\right),
\]

\[
\delta(\nabla_H J)\;{+}{=}\;
\Delta t_n\left(\tilde\lambda_{n+1}\zeta(u_n)^T + \lambda_{n+1}\delta\zeta(u_n)^T\right),
\]

\[
\delta(\nabla_B J)\;{+}{=}\;
\Delta t_n\,\tilde\lambda_{n+1} p(t_n)^T,
\]

\[
\delta(\nabla_c J)\;{+}{=}\;
\Delta t_n\,\tilde\lambda_{n+1}.
\]

These explicit matrix-block actions are then pulled back to the structured stabilized parameterization.

## 7. Implementation map

The current implementation lives in:

- `src/goattm/solvers/explicit_euler.py`
- `src/goattm/solvers/time_integration.py`
- `src/goattm/losses/qoi_loss.py`
- `src/goattm/problems/reduced_qoi_best_response.py`

The public choice is:

- `time_integrator="implicit_midpoint"`
- `time_integrator="explicit_euler"`

and the old midpoint path is still preserved.

## 8. Current validation status

The explicit-Euler branch is numerically validated by:

- solver-level tangent / incremental-adjoint finite-difference checks
- first-order reduced-objective Taylor test
- second-order reduced-objective Taylor test
- full `unit_test` regression sweep

At the time this note was written, the full unit suite passed after the explicit-Euler integration changes.
