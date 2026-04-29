# Why RK4 and Implicit Midpoint Blow Up on the Stabilized Quadratic ODE

## Current Implementation Status

This note was originally written as a diagnosis and design memo.  The current code now implements
the first major implicit-midpoint fix proposed here: `solve_implicit_midpoint_step_with_retry`
keeps the direct Newton solve as the fast path, then falls back to a homotopy continuation solve
when direct Newton fails.

The implemented homotopy is

$$
F_\lambda(u_{n+1})
= u_{n+1} - u_n
- \lambda \Delta t\, f\!\left(\frac{u_n + u_{n+1}}{2}\right),
\qquad \lambda \in [0,1].
$$

At $\lambda=0$, the root is exactly $u_n$.  The solver tracks that root branch to $\lambda=1$
with a default lambda step of `0.1`; if a lambda subproblem fails, the lambda step is halved down
to `1e-4`.  This is intentionally finer than the earlier sketch of `0.25, 0.5, 0.75, 1.0`,
because the whole point is to avoid jumping branches in a multi-root quadratic implicit equation.

This does not eliminate all possible bad-parameter failures.  It only makes the nonlinear solve
more likely to follow the physical root branch.  The diagnostic and structural recommendations
below are still relevant, especially monitoring $\|H\|$, $\|J(u)\|$, $S$ singular values, and
rollout state norms during training.

## Abstract

We consider the latent quadratic ODE

$$
\dot u = A u + H(u, u) + c, \qquad u \in \mathbb{R}^r, \tag{0.1}
$$

with the *stabilized* parametrization

$$
A = -SS^\top + W, \quad W^\top = -W, \tag{0.2}
$$

and an energy-preserving quadratic term satisfying $u^\top H(u, u) = 0$ for all $u$. Despite these structural guarantees — $A + A^\top = -2SS^\top \preceq 0$ and the radial energy bound — both `rollout_rk4` and `rollout_implicit_midpoint` produce non-finite states in finite time. This note explains *why*, separates analytic from numerical stability, identifies two distinct failure mechanisms (one for each integrator), and proposes a prioritized list of fixes.

The TL;DR:

1. **Energy preservation is a radial constraint.** It bounds $\frac{d}{dt}\tfrac12\|u\|^2$ but does **not** bound $\|H(u,u)\|$, $\|H\|_\mathrm{op}$, or the spectral radius of the Jacobian along the trajectory.
2. **RK4 fails by CFL violation.** The instantaneous Jacobian $J(u) = A + 2 M(u)$ has spectral radius $\sim \|H\|_\mathrm{op} \cdot \|u\|$. Once $\|u(t)\|$ peaks, $|\lambda(J)| \cdot \Delta t$ leaves the RK4 stability disk; error is amplified and the loop runs away.
3. **Implicit midpoint fails by Newton convergence to a spurious root.** The per-step residual is *quadratic* in $u_{n+1}$ and admits multiple solutions; the line-search criterion in [`solve_implicit_midpoint_step`](../src/goattm/solvers/implicit_midpoint.py#L83-L154) accepts whichever root has smaller residual norm, including the unphysical far-field one.
4. **Fixes** range from cheap diagnostic guards (Lyapunov monitor, contraction-radius line search) to structural redesigns (IMEX/ETD-RK4 splitting, AVF integrators, an operator-norm penalty on $H$ during training).

---

## 1. What the Stabilized Parametrization Actually Buys Us

### 1.1 Radial energy identity

Take the inner product of (0.1) with $u$:

$$
\frac{d}{dt}\tfrac12\|u\|^2 \;=\; u^\top A u + u^\top H(u,u) + u^\top c \;=\; -\|S^\top u\|^2 + u^\top c. \tag{1.1}
$$

Two regimes:

- **Unforced case ($c = 0$).** $\frac{d}{dt}\tfrac12\|u\|^2 = -\|S^\top u\|^2 \le 0$. The energy is non-increasing. If $S$ is full row-rank, $\|S^\top u\|^2 \ge \sigma_{\min}(S)^2 \|u\|^2$ and we get exponential decay.
- **Forced case ($c \ne 0$).** Cauchy–Schwarz gives
  $$\frac{d}{dt}\tfrac12\|u\|^2 \le -\sigma_{\min}(S)^2 \|u\|^2 + \|c\|\,\|u\|,$$
  yielding an absorbing ball $\|u\|_\infty \le \|c\| / \sigma_{\min}(S)^2$ provided $S$ has full row rank.

So **continuous-time trajectories are bounded** under the stabilized parametrization, *conditional on* $\sigma_{\min}(S)$ staying away from zero. Any blow-up observed in numerical integration is therefore a property of the *discretization*, not of the analytic flow.

### 1.2 What energy preservation does *not* give us

The structural identity $u^\top H(u,u) = 0$ says only that $H(u,u)$ is **orthogonal to $u$**. It does not bound the magnitude of $H(u,u)$. Concretely, if $\|H\|_\mathrm{op}$ denotes the operator norm of the symmetric bilinear map $H$, then

$$
\|H(u,u)\| \;\le\; \|H\|_\mathrm{op} \cdot \|u\|^2, \tag{1.2}
$$

and (1.2) is the only generic bound. In particular:

- Training has **no incentive** to keep $\|H\|_\mathrm{op}$ small. The energy-preserving manifold contains operators of arbitrary norm.
- The Jacobian along the trajectory is $J(u) = A + 2 M(u)$ where $M(u) v = H(u,v)$, and $\|M(u)\|_\mathrm{op} \le \|H\|_\mathrm{op} \cdot \|u\|$. So $\|J(u)\|$ scales **linearly with $\|u\|$**.
- Even when $\|u\|$ is bounded analytically, $\|J(u)\|$ can be large during the transient peak.

This is the deepest point: *the stabilized parametrization protects the radial mode of $u$, but not the spectral norm of the dynamics linearization*. The CFL constraint is set by the latter.

### 1.3 A subtlety: rank-deficient $S$

If during training the singular values of $S$ collapse, $\sigma_{\min}(S) \to 0$ and the absorbing ball in §1.1 expands to infinity. In that regime the rotation $W$ and the quadratic $H$ can pump energy into the (approximate) null space of $S^\top$ unboundedly. Worth checking `np.linalg.svd(S)` periodically during training.

---

## 2. Why RK4 Blows Up

### 2.1 The CFL bound depends on $\|u(t)\|$

RK4's linear stability region is bounded: along the negative real axis $|\lambda \Delta t| \lesssim 2.78$, along the imaginary axis $|\lambda \Delta t| \lesssim 2.83$. For our problem,

$$
\rho(J(u)) \;\le\; \sigma_{\max}(S)^2 + \|W\|_\mathrm{op} + 2 \|H\|_\mathrm{op}\,\|u\|, \tag{2.1}
$$

so a sufficient step-size condition is

$$
\Delta t \;\lesssim\; \frac{2.83}{\sigma_{\max}(S)^2 + \|W\|_\mathrm{op} + 2\|H\|_\mathrm{op}\,\|u(t)\|_\infty}. \tag{2.2}
$$

The denominator is dominated by the **time-varying** term $2 \|H\|_\mathrm{op} \|u(t)\|$. Two consequences:

- **Transient tightening.** Even if the asymptotic absorbing ball is small, $\|u(t)\|$ may overshoot it during the initial transient (especially for OpInf-style initial conditions far from the slow manifold). The CFL is set by the *peak*, not the steady-state.
- **Skew-symmetric / rotational eigenvalues.** $W$ contributes purely imaginary eigenvalues to $J$, and $M(u)$ frequently contributes near-imaginary eigenvalues for energy-preserving $H$ (rotation of $u$ around the energy ball). RK4's imaginary-axis margin is the binding one.

### 2.2 The runaway feedback

If a single step violates (2.2), error is *amplified* rather than damped. The amplified error increases $\|u(t)\|$, which in turn tightens the CFL bound, which causes the next step to amplify error more. This is a *finite-time numerical blow-up of an analytically bounded ODE*. The check `np.all(np.isfinite(current_state))` at [`rk4.py:115`](../src/goattm/solvers/rk4.py#L115) is the symptom; the cause is several steps earlier.

### 2.3 Diagnostic

For a short trajectory of length $T$ at fixed $\Delta t$, log

```
m = max_t |Im eig(rhs_jacobian(u(t)))|
```

and compare against $2.83 / \Delta t$. If $m \cdot \Delta t > 2.83$ at any time, the integration is in the unstable regime regardless of how small the residual *appears* per step.

---

## 3. Why Implicit Midpoint Also Blows Up

This is the more subtle failure. Implicit midpoint is A-stable, symplectic, and exactly preserves any quadratic invariant of the form $u^\top Q u$ when the ODE is linear. For our quadratic right-hand side it has none of these guarantees in their strict form; in particular it admits *multiple roots per step*.

### 3.1 The per-step residual is quadratic

The implicit-midpoint update solves $F(u_{n+1}) = 0$ where ([`implicit_midpoint.py:55-67`](../src/goattm/solvers/implicit_midpoint.py#L55-L67))

$$
F(u_{n+1}) \;=\; u_{n+1} - u_n - \Delta t\, f\!\left(\tfrac{u_n + u_{n+1}}{2}\right), \tag{3.1}
$$

with $f(u) = A u + H(u,u) + c$. Because $f$ is quadratic, $F$ is a *quadratic* vector-valued map of $u_{n+1}$:

$$
F(u_{n+1}) \;=\; u_{n+1} \;-\; u_n \;-\; \Delta t\, A \tfrac{u_n + u_{n+1}}{2} \;-\; \tfrac{\Delta t}{4} H(u_n + u_{n+1},\, u_n + u_{n+1}) \;-\; \Delta t\, c. \tag{3.2}
$$

A quadratic equation in $\mathbb{R}^r$ generically admits multiple roots. One root is close to $u_n + \Delta t f(u_n)$ for small $\Delta t$ — the *physical* root. Other roots are far from $u_n$ — *spurious* roots that do not correspond to any continuous trajectory. As $\Delta t$ grows, the spurious roots move closer to the physical one and become harder to discriminate.

### 3.2 Newton can converge to the spurious root

Inspect [`solve_implicit_midpoint_step`](../src/goattm/solvers/implicit_midpoint.py#L83-L154). The acceptance criterion in the line search (line 141) is

```python
if np.isfinite(trial_norm) and trial_norm < residual_norm:
    accepted = True
```

— pure residual decrease. There is no check that the candidate $u_{n+1}$ lies in any contraction neighborhood of $u_n$. If Newton's update happens to point toward the spurious root (which can happen when $\Delta t$ is moderate, $\|u_n\|$ is large, or the explicit-Euler initial guess `explicit_euler_guess` is itself outside the basin of the physical root), the iteration converges to it. The residual is satisfied to tolerance, the step is *reported as successful*, and $\|u_{n+1}\|$ jumps to a large value.

The next step now starts from a state with much larger $\|u\|$, where the spurious root is even farther, and the Newton solve finds a *yet-more-spurious* root. This is **a different failure mode from the RK4 mode**: the residual is satisfied at every step, no `NaN` is produced, but the trajectory diverges from the physical solution exponentially.

### 3.3 What about A-stability?

A-stability is a property of the integrator applied to the *linear* test equation $\dot u = \lambda u$. For our nonlinear $f$, it tells us that the *frozen-Jacobian* linearization is unconditionally stable, which is necessary but far from sufficient. The non-uniqueness of the implicit equation is a purely nonlinear phenomenon; A-stability cannot see it. Hairer–Lubich–Wanner (Geometric Numerical Integration, §IV.8) and Hairer–Wanner Vol. II discuss this explicitly.

### 3.4 Aside: implicit midpoint is *not* exactly energy-preserving here

Even when Newton finds the correct root, implicit midpoint exactly preserves only invariants of the form $\Phi(u) = u^\top Q u$ when $f(u) \in \ker(Q + Q^\top)$ everywhere. For our problem there is no such global invariant — the analytic energy *decreases*, it doesn't conserve. So implicit midpoint provides no discrete energy bound that we could lean on for stability.

---

## 4. Fixes, in Priority Order

### 4.1 Cheap diagnostic guards (do these first)

**(a) Lyapunov monitor.** During the rollout, track $E_n = \tfrac12 \|u_n\|^2$ and compare against the analytic bound from §1.1:
- If $c = 0$, require $E_n$ non-increasing.
- If $c \ne 0$, require $E_n \le \max(E_0, \|c\|^2 / (2 \sigma_{\min}(S S^\top)))$.

Any step that violates this bound is *prima facie* numerical. Reject the step, shrink $\Delta t$ by a factor of $\tfrac12$, retry. This single guard would catch both failure modes in §2 and §3 well before they cascade.

**(b) Contraction-radius line search for implicit midpoint.** Replace the bare `trial_norm < residual_norm` check with a compound criterion:

```
trial_norm < residual_norm
AND ||u_next - u_prev|| <= kappa * dt * ||f(u_prev)||      # kappa in [2, 5]
```

This rejects the spurious root in §3.2 because the spurious root violates the second inequality dramatically. The justification: the contraction-mapping fixed-point existence theorem for implicit midpoint requires $\Delta t < c / L$ where $L$ is the local Lipschitz constant of $f$; a candidate solution outside the contraction ball is *guaranteed* to be the wrong root.

**(c) Aggressive `dt_shrink`.** The current `dt_shrink = 0.8` ([`implicit_midpoint.py:163`](../src/goattm/solvers/implicit_midpoint.py#L163)) recovers slowly from a bad regime. Use $0.5$ on Newton failure, and require Newton to converge in $\le 5$ iterations rather than $25$ before declaring success.

### 4.2 IMEX / ETD splitting (the principled fix)

Split the right-hand side into a stiff linear part and a non-stiff quadratic part:

$$
\dot u \;=\; \underbrace{A u + c}_{\text{stiff, linear}} \;+\; \underbrace{H(u, u)}_{\text{quadratic}}. \tag{4.1}
$$

The linear flow $\Phi^L_t$ has a closed form: $\Phi^L_t(u) = e^{At}(u + A^{-1} c) - A^{-1} c$. Use Exponential Time Differencing (ETD) to absorb the stiff part exactly:

$$
u_{n+1} \;=\; e^{A \Delta t} u_n \;+\; \int_0^{\Delta t} e^{A(\Delta t - s)} \bigl[ H(u(s), u(s)) + c \bigr] \, ds. \tag{4.2}
$$

Approximate the integral with explicit RK-style stages (Cox–Matthews ETD-RK4 or Krogstad's variant). Properties:

- **CFL is set by $H$ alone**, not by $A$. The blow-up mechanism in §2 is mitigated because the stiff $A$-modes are integrated exactly.
- **No nonlinear root-finding**, so the spurious-root mechanism in §3 vanishes.
- For our problem, $A$ is small (rank $r$, typically $r \le 50$), so $e^{A \Delta t}$ via `scipy.linalg.expm` is cheap and computed once per step.
- 4th-order accurate, comparable to RK4.

This is the textbook gold-standard for "linear stiff + non-stiff polynomial nonlinearity" and is the recommended path forward for production training.

### 4.3 AVF (Average Vector Field) method

For systems with a first integral, the AVF method

$$
u_{n+1} \;=\; u_n \;+\; \Delta t \int_0^1 f\bigl((1-s) u_n + s\, u_{n+1}\bigr)\, ds \tag{4.3}
$$

exactly preserves any first integral of the form $V(u)$ for which $\nabla V \cdot f = 0$. For our quadratic $f$, the integral evaluates to a closed-form polynomial in $(u_n, u_{n+1})$ and the per-step equation is again quadratic, so the spurious-root issue from §3 returns. *AVF is appropriate only if $H$ corresponds to a Hamiltonian gradient structure with a known invariant*; for the generic energy-preserving $H$ in this codebase it gives no improvement over implicit midpoint and is not recommended.

Reference: Celledoni–McLachlan–Owren–Quispel, *Energy-preserving integrators and the structure of B-series*, 2010.

### 4.4 Structural: bound $\|H\|_\mathrm{op}$ during training

The deepest fix is to constrain $\|H\|_\mathrm{op}$ in the parametrization itself, so that the CFL in (2.2) is bounded *a priori*. Two routes:

- **Penalty.** Add $\lambda \|H\|_F^2$ or $\lambda \|H\|_\mathrm{op}^2$ to the training loss. Cheap to implement, but the choice of $\lambda$ trades expressivity for stability.
- **Reparametrization.** Decompose $H = R \, \tilde H \, R^{-1}$ where $\tilde H$ is bounded by construction (e.g. parametrized on a sphere) and $R$ is a learned scaling. Analogous to how `a_from_stabilized_params` decomposes $A$.

The current stabilized parametrization gives $A$ a proper manifold structure. The same treatment should be extended to $H$ — otherwise stabilization is just *displaced* from $A$ to $H$.

---

## 5. The Question to Ask Yourself

> What is the maximum value of $\|u(t)\| \cdot \|H\|_\mathrm{op}$ over my trajectory, and how does $\Delta t$ compare to $1 / (\|A\| + 2 \|H\|_\mathrm{op} \|u\|_{\max})$?

If the latter is smaller than the former, **your blow-up is not a bug — it is a CFL violation**, and structural integrators cannot save you. Structure-preserving methods preserve invariants; they do not relax CFL. Long-time numerical stability is a separate property that requires either (i) an a priori bound on the Jacobian spectral radius (achievable only via constraints on $\|H\|$), or (ii) a stiff integrator that absorbs the linear dissipation exactly (ETD/IMEX).

---

## 6. References

1. Hairer, Lubich, Wanner. *Geometric Numerical Integration*, 2nd ed., Springer 2006. §IV.8 on multiple roots of implicit one-step schemes.
2. Hairer, Wanner. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed., Springer 1996.
3. Cox, Matthews. *Exponential time differencing for stiff systems*, J. Comp. Phys. 176 (2002).
4. Krogstad. *Generalized integrating factor methods for stiff PDEs*, J. Comp. Phys. 203 (2005).
5. Celledoni, McLachlan, Owren, Quispel. *Energy-preserving integrators and the structure of B-series*, Foundations of Computational Mathematics 10 (2010).
6. Schlegel, Knoth, Arnold, Wolke. *Multirate Runge–Kutta schemes for advection equations*, J. Comp. App. Math. 226 (2009) — for IMEX context.
