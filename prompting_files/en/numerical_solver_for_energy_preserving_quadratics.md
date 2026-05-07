# Notes on the Implicit Midpoint Scheme for Quadratic Latent Dynamics

## Abstract

We consider the time integration of the latent quadratic ODE

$$
\dot u = A u + H(u, u) + c,
$$

arising in latent reduced-order modeling of convection-dominated PDEs. We use the *implicit midpoint* scheme as the forward integrator because it exactly preserves quadratic invariants, making it the natural choice when the quadratic operator $H$ is constructed to be *energy-preserving*, i.e. $u^\top H(u, u) = 0$. This note documents three things:

1. The forward scheme.
2. The corresponding discrete adjoint equation and parameter gradient formulae used during training under a QoI loss.
3. A rigorous proof of existence and uniqueness of the per-step solution under the assumption $A + A^\top \preceq 0$.

---

## 1. Forward Scheme: Implicit Midpoint

### 1.1 Setup

Let $u(t) \in \mathbb{R}^k$ be the latent state and let $\theta$ collect all trainable parameters of the right-hand side. The continuous-time dynamics are

$$
\dot u = F(u, \theta), \qquad F(u, \theta) = A(\theta)\, u + H(\theta)(u, u) + c(\theta), \tag{1.1}
$$

where

- $A(\theta) \in \mathbb{R}^{k \times k}$ is a linear operator,
- $H(\theta) : \mathbb{R}^k \times \mathbb{R}^k \to \mathbb{R}^k$ is a symmetric bilinear map, i.e. $H(u, v) = H(v, u)$,
- $c(\theta) \in \mathbb{R}^k$ is a forcing term (possibly time-dependent; we drop the $t$-dependence for notational simplicity).

We impose the structural constraint

$$
u^\top H(u, u) = 0 \quad \text{for all } u \in \mathbb{R}^k, \tag{1.2}
$$

which we refer to as the *energy-preserving* property of $H$. In practice this constraint should be enforced *by construction* at the parameterization level rather than by a posteriori projection. The `GOAM_clean` reference implementation does this through a reduced parameter vector $\mu_H$ and a sparse expansion map $\mu_H \mapsto H$; other implementations may realize the same idea through an equivalent structured tensor parameterization.

#### GOAM_clean reference parametrization of the energy-preserving quadratic term

The `GOAM_clean` reference implementation makes this structural constraint explicit through a reduced parameter vector `muH` together with a deterministic expansion map

$$
\mu_H \in \mathbb{R}^{d_H},
\qquad
d_H = \frac{r(r-1)(r+1)}{3},
\qquad
H = \mathcal{E}_r(\mu_H) \in \mathbb{R}^{r \times s},
\qquad
s = \frac{r(r+1)}{2}.
$$

Here the quadratic operator is not stored as a full three-tensor. Instead, `GOAM_clean` stores the compressed matrix

$$
H = \big[ H_{a;bc} \big]_{a=0,\dots,r-1}^{0 \le c \le b \le r-1},
$$

whose columns correspond to the lower-triangular quadratic monomials

$$
\zeta(u) = \big(u_0^2,\; u_1 u_0,\; u_1^2,\; u_2 u_0,\; u_2 u_1,\; u_2^2,\; \dots,\; u_{r-1}^2 \big)^\top \in \mathbb{R}^s,
$$

so that

$$
H(u,u) = H\,\zeta(u). \tag{1.2a}
$$

The key point is that `GOAM_clean` does **not** optimize over all entries of $H$. It uses the class `dynamic_func_stablized` and stores the dynamic parameters in the order

$$
\theta_{\mathrm{dyn}} = (A,\mu_H,B,c),
$$

whereas the unconstrained mode `dynamic_func_general` stores $(A,H,B,c)$ directly. The stabilized mode expands $\mu_H$ into $H$ using the routine `muH_to_fullH(muH, r)`.

To describe the expansion, write $H_{a;bc}$ for the coefficient multiplying the monomial $u_b u_c$ in the $a$-th output component, with the convention $b \ge c$. Then `muH_to_fullH` fills the entries of $H$ by local cancellation rules:

1. For each triple of distinct indices $0 \le i < j < k \le r-1$, introduce two free parameters $\alpha_{ijk}$ and $\beta_{ijk}$ and set

$$
H_{i;kj} = \alpha_{ijk},
\qquad
H_{j;ki} = \beta_{ijk},
\qquad
H_{k;ji} = -\alpha_{ijk} - \beta_{ijk}. \tag{1.2b}
$$

2. For each pair $0 \le j < k \le r-1$, introduce one free parameter $\gamma_{jk}$ and set

$$
H_{j;kj} = \gamma_{jk},
\qquad
H_{k;jj} = -\gamma_{jk}. \tag{1.2c}
$$

3. For each pair $0 \le i < k \le r-1$, introduce one free parameter $\delta_{ik}$ and set

$$
H_{i;kk} = \delta_{ik},
\qquad
H_{k;ki} = -\delta_{ik}. \tag{1.2d}
$$

4. All remaining entries are zero; in particular

$$
H_{k;kk} = 0 \qquad \text{for every } k. \tag{1.2e}
$$

This is exactly what the code in `GOAM_clean/GOAM/baselibrary/staticfunction.py` constructs. The counting also matches:

$$
2\binom{r}{3} + 2\binom{r}{2}
= \frac{r(r-1)(r+1)}{3}
= d_H.
$$

Why does this guarantee energy preservation? Because every cubic monomial cancels *locally* in $u^\top H(u,u)$:

- For $u_i u_j u_k$ with $i<j<k$, the total coefficient is $\alpha_{ijk} + \beta_{ijk} - (\alpha_{ijk} + \beta_{ijk}) = 0$.
- For $u_j^2 u_k$, the total coefficient is $\gamma_{jk} - \gamma_{jk} = 0$.
- For $u_i u_k^2$, the total coefficient is $\delta_{ik} - \delta_{ik} = 0$.
- For $u_k^3$, the coefficient is zero because $H_{k;kk}=0$.

Hence

$$
u^\top H(u,u) = 0 \qquad \text{for all } u, \tag{1.2f}
$$

without any additional projection or post-processing step.

Equivalently, the expansion is a sparse linear map

$$
\mathrm{vec}(H) = L_r \mu_H, \tag{1.2g}
$$

where `GOAM_clean` builds $L_r$ explicitly through `muH_to_fullH_matrix(r)`. The reverse-mode pullback from an unconstrained $H$-gradient to a $\mu_H$-gradient is the transpose action

$$
\nabla_{\mu_H} \mathcal{L} = L_r^\top \, \mathrm{vec}\!\left(\nabla_H \mathcal{L}\right), \tag{1.2h}
$$

which in the code appears as `Hdiff_to_muH` (or, equivalently, `L_r.T @ H.flatten()` in the initialization routines). This is the form that is most useful when implementing module tests for nonlinear solvers and discrete adjoints.

#### Rewrite implementation note: use the compressed matrix directly

For the rewrite, the most practical internal representation is usually **not** a full three-tensor.
Instead, one should store

$$
H_{\mathrm{comp}} = \mathcal{E}_r(\mu_H) \in \mathbb{R}^{r \times s},
\qquad
s = \frac{r(r+1)}{2},
$$

and evaluate the quadratic term through the compressed monomial vector

$$
\zeta(u) = (u_0^2,\; u_1u_0,\; u_1^2,\; \dots,\; u_{r-1}^2)^\top.
$$

Then

$$
H(u,u) = H_{\mathrm{comp}} \, \zeta(u). \tag{1.2i}
$$

If the $a$-th row of $H_{\mathrm{comp}}$ is interpreted as a lower-triangular matrix $L_a$, then the symmetric bilinear map associated with the quadratic term is

$$
H(u,v)_a = \frac{1}{2}\left(u^\top L_a v + v^\top L_a u\right). \tag{1.2j}
$$

This formula is what should be used in the rewrite for testing identities such as $H(s,w)$ and for building the midpoint Jacobian. In particular,

$$
\frac{\partial}{\partial u} H(u,u)_a = (L_a + L_a^\top)u
= 2\,H(u,\cdot)_a, \tag{1.2k}
$$

so the quadratic contribution to the state Jacobian is obtained directly from the compressed representation, without ever expanding to a dense three-tensor.

### 1.2 The implicit midpoint discretization

Given a step size $\Delta t > 0$ and the current state $u_n$, the implicit midpoint scheme defines $u_{n+1}$ as the solution of

$$
\boxed{\;\frac{u_{n+1} - u_n}{\Delta t} = F\!\left( \frac{u_n + u_{n+1}}{2},\ \theta\right).\;} \tag{1.3}
$$

We denote the midpoint by

$$
\bar u_n := \frac{u_n + u_{n+1}}{2}.
$$

The implicit midpoint scheme is second-order accurate (local truncation error $\mathcal{O}(\Delta t^3)$, global error $\mathcal{O}(\Delta t^2)$), $A$-stable, symmetric (self-adjoint), and—most importantly for our purposes—it *exactly preserves all quadratic first integrals* of the underlying ODE. In particular, when $F$ has no forcing ($c = 0$) and $A + A^\top = 0$, the scheme exactly preserves $\|u\|^2$ at every step, regardless of $\Delta t$.

### 1.3 Why this scheme

A few standard alternatives and their failure modes for our problem:

- **Forward Euler**: explicit, but $\|u_{n+1}\|^2 - \|u_n\|^2$ contains an uncontrolled $\mathcal{O}(\Delta t^2)$ term that drifts upward, so even continuously dissipative systems can blow up at the discrete level.
- **RK4**: fourth-order accurate, but does not preserve quadratic invariants; a slow but persistent energy drift accumulates over long horizons.
- **Crank–Nicolson**: takes $\frac{1}{2}[F(u_n) + F(u_{n+1})]$ rather than $F(\bar u_n)$. Coincides with implicit midpoint for *linear* $F$ but *not* for our quadratic $F$, and does *not* preserve quadratic invariants in general.

---

## 2. Adjoint Equation and Gradient

We now derive the discrete adjoint corresponding to the forward scheme (1.3), under a generic state-tracking loss

$$
\mathcal{L}(\theta) = \sum_{n=0}^{N} \ell_n(u_n), \tag{2.1}
$$

where $\ell_n : \mathbb{R}^k \to \mathbb{R}$ is differentiable. In our application $\ell_n(u_n) = \|q_n - D(u_n)\|^2$ where $D$ is the decoder.

### 2.1 Lagrangian and adjoint variables

Introduce Lagrange multipliers $\lambda_{n+1} \in \mathbb{R}^k$ for $n = 0, \dots, N-1$ and form

$$
\mathcal{J}(\theta) = \sum_{n=0}^{N} \ell_n(u_n) + \sum_{n=0}^{N-1} \lambda_{n+1}^\top \big[ u_{n+1} - u_n - \Delta t\, F(\bar u_n, \theta) \big].
$$

On any forward trajectory, the bracket is identically zero, so $\mathcal{J} = \mathcal{L}$ and consequently $\nabla_\theta \mathcal{J} = \nabla_\theta \mathcal{L}$ for any choice of $\lambda$. We choose $\lambda$ to make the gradient with respect to the state variables vanish.

### 2.2 State Jacobians

Differentiating $F$ at the midpoint:

$$
J_n \;:=\; \frac{\partial F}{\partial u}(\bar u_n, \theta) = A + 2 H(\bar u_n, \cdot), \tag{2.2}
$$

where $H(\bar u_n, \cdot) : v \mapsto H(\bar u_n, v)$ is the linear map obtained by freezing the first argument. The factor of $2$ comes from $\partial_u\, H(u, u) = 2 H(u, \cdot)$, which uses the symmetry of $H$.

### 2.3 Adjoint recursion

The state $u_n$ for $1 \le n \le N-1$ appears in three places in $\mathcal{J}$: in $\ell_n(u_n)$, in step $n \to n+1$ (through both $-u_n$ and $\bar u_n$), and in step $n-1 \to n$ (through both $u_n$ and $\bar u_{n-1}$). Setting $\partial \mathcal{J} / \partial u_n = 0$ yields, after rearranging,

$$
\boxed{\;\left(I - \frac{\Delta t}{2} J_{n-1}^\top \right) \lambda_n = \left(I + \frac{\Delta t}{2} J_n^\top \right) \lambda_{n+1} - \nabla \ell_n(u_n).\;} \tag{2.3}
$$

This is the backward-in-time adjoint recursion, valid for $n = N-1, N-2, \dots, 1$.

### 2.4 Terminal condition

The state $u_N$ appears only in $\ell_N(u_N)$ and in step $N-1 \to N$. Setting $\partial \mathcal{J} / \partial u_N = 0$ gives

$$
\left( I - \frac{\Delta t}{2} J_{N-1}^\top \right) \lambda_N = -\nabla \ell_N(u_N). \tag{2.4}
$$

This is the starting condition for the backward sweep.

### 2.5 Initial state gradient

If $u_0$ is itself trainable (e.g. a learned initial condition),

$$
\frac{d \mathcal{L}}{d u_0} = \nabla \ell_0(u_0) - \left( I + \frac{\Delta t}{2} J_0^\top \right) \lambda_1.
$$

### 2.6 Parameter gradient

The parameter gradient is the sum over all forward steps:

$$
\boxed{\;\nabla_\theta \mathcal{L} = -\sum_{n=0}^{N-1} \Delta t\, \left[ \frac{\partial F}{\partial \theta}(\bar u_n, \theta) \right]^\top \lambda_{n+1}.\;} \tag{2.5}
$$

Each term $\partial F / \partial \theta$ is a Jacobian of the right-hand side with respect to a parameter block, and is local to a single time step. In a deep-learning implementation, this contraction is performed automatically by reverse-mode automatic differentiation when each forward step is implemented in a differentiable framework.

### 2.6a Practical gradient formula for the compressed $H$ representation

When the rewrite stores the quadratic operator through the compressed matrix $H_{\mathrm{comp}}$, the quadratic part of the forward model is

$$
q(u) := H_{\mathrm{comp}} \, \zeta(u). \tag{2.5a}
$$

If a scalar objective $\mathcal{L}$ depends on $q(u)$ and the upstream adjoint with respect to $q$ is

$$
g := \frac{\partial \mathcal{L}}{\partial q} \in \mathbb{R}^r,
$$

then the gradient with respect to the compressed matrix is simply

$$
\frac{\partial \mathcal{L}}{\partial H_{\mathrm{comp}}}
= g \, \zeta(u)^\top. \tag{2.5b}
$$

This is the form that should be accumulated inside the rewritten adjoint code, because it is local, cheap, and directly matches the stored representation.

### 2.6b Pullback from compressed $H$ to $\mu_H$

Since the parametrization map $\mu_H \mapsto H_{\mathrm{comp}}$ is linear, the reverse-mode pullback is also linear. If we denote the compressed-matrix gradient by

$$
G_H := \frac{\partial \mathcal{L}}{\partial H_{\mathrm{comp}}} \in \mathbb{R}^{r \times s},
$$

then

$$
\nabla_{\mu_H}\mathcal{L} = \mathcal{E}_r^\ast(G_H), \tag{2.5c}
$$

where $\mathcal{E}_r^\ast$ is exactly the transpose action implemented in the old code by `Hdiff_to_muH`.

In coordinates, the pullback follows the same local cancellation pattern as the forward map:

- for a triple parameter contributing to `H[i;kj]`, `H[j;ki]`, and `H[k;ji]`, the corresponding $\mu_H$ gradient is the sum of the first two compressed gradients minus the third;
- for a pair parameter contributing to `H[j;kj]` and `H[k;jj]`, the corresponding $\mu_H$ gradient is the first compressed gradient minus the second;
- for a pair parameter contributing to `H[i;kk]` and `H[k;ki]`, the corresponding $\mu_H$ gradient is the first compressed gradient minus the second.

This is the key implementation point: in the rewrite, the optimizer should see **unconstrained variables $\mu_H$**, while every structural property is enforced by the deterministic expansion and its pullback.

### 2.7 Exact per-step adjoint evolution

Assume now that each nonlinear solve in (1.3) is carried out to exact convergence, so that locally the implicit midpoint rule defines a differentiable one-step map

$$
u_{n+1} = \Psi_{\Delta t}(u_n, \theta).
$$

Define the step residual

$$
R_n(u_n, u_{n+1}, \theta) := u_{n+1} - u_n - \Delta t\, F\!\left(\frac{u_n + u_{n+1}}{2}, \theta\right).
$$

At an exact forward solution $R_n = 0$, the partial derivatives of the residual are

$$
\frac{\partial R_n}{\partial u_n} = -I - \frac{\Delta t}{2} J_n,
\qquad
\frac{\partial R_n}{\partial u_{n+1}} = I - \frac{\Delta t}{2} J_n. \tag{2.6a}
$$

Therefore the linearized forward step satisfies

$$
\boxed{\;\left(I - \frac{\Delta t}{2} J_n\right)\delta u_{n+1}
= \left(I + \frac{\Delta t}{2} J_n\right)\delta u_n
+ \Delta t\, \left[\frac{\partial F}{\partial \theta}(\bar u_n, \theta)\right]\delta\theta.\;} \tag{2.6b}
$$

This immediately gives the exact discrete adjoint evolution. Define the cost-to-go cotangent

$$
p_n := \nabla_{u_n}\!\left( \sum_{m=n}^{N} \ell_m(u_m) \right),
\qquad
p_N = \nabla \ell_N(u_N).
$$

Then, for $n = N-1, N-2, \dots, 0$, one backward step consists of:

$$
\boxed{\;\left(I - \frac{\Delta t}{2} J_n^\top\right) z_n = p_{n+1},\;} \tag{2.6c}
$$

followed by

$$
\boxed{\;p_n = \nabla \ell_n(u_n) + \left(I + \frac{\Delta t}{2} J_n^\top\right) z_n.\;} \tag{2.6d}
$$

Equivalently, eliminating $z_n$ gives the closed-form cotangent update

$$
\boxed{\;p_n
= \nabla \ell_n(u_n)
+ \left[\left(I - \frac{\Delta t}{2} J_n\right)^{-1}
\left(I + \frac{\Delta t}{2} J_n\right)\right]^\top p_{n+1}.\;} \tag{2.6e}
$$

The corresponding parameter contribution from step $n$ is

$$
\boxed{\;g_n
= \Delta t\, \left[\frac{\partial F}{\partial \theta}(\bar u_n, \theta)\right]^\top z_n,
\qquad
\nabla_\theta \mathcal{L} = \sum_{n=0}^{N-1} g_n.\;} \tag{2.6f}
$$

Equations (2.6c)–(2.6f) are the same discrete adjoint written as an explicit stepwise pullback. They are equivalent to the Lagrange-multiplier form (2.3)–(2.5): if we define $z_n := -\lambda_{n+1}$ and $p_n := \nabla \ell_n(u_n) + \left(I + \frac{\Delta t}{2} J_n^\top\right) z_n$, then the two formulations coincide.

### 2.8 Summary of the algorithm

1. **Forward sweep.** For $n = 0, 1, \dots, N-1$: solve the nonlinear system (1.3) for $u_{n+1}$ by Newton iteration; cache $u_n$, $\bar u_n$, and the Jacobian $J_n$ (or the operations needed to apply $J_n^\top$).
2. **Loss.** Accumulate $\mathcal{L} = \sum_n \ell_n(u_n)$.
3. **Backward sweep.** Either solve the terminal condition (2.4) and propagate (2.3) backward to obtain $\lambda_{N-1}, \dots, \lambda_1$, or equivalently use the stepwise pullback (2.6c)–(2.6d) for $p_n$.
4. **Gradient.** Assemble $\nabla_\theta \mathcal{L}$ via (2.5), or equivalently by accumulating the local contributions (2.6f).

### 2.9 Implementation remarks

1. *Same linear operator structure on both passes.* The forward Newton iteration and the adjoint recursion both involve linear systems whose operators differ only by a transpose. A matrix-free Krylov solver (e.g. GMRES) using Jacobian-vector products ($Jv$ for forward, $J^\top w$ for adjoint) reuses the same code path. For tensor-train compressed $H$, both Jacobian–vector products cost $\mathcal{O}(k R^2)$ per call.

2. *Memory.* The discrete adjoint requires storing $\{u_n, \bar u_n\}_{n=0}^{N}$ during the forward pass. For our typical scales ($k \sim 10^4$, $N \sim 10^2$), this is on the order of $10^7$ floats $\sim 10^2$ MB, fitting comfortably in GPU memory without checkpointing.

3. *Discrete vs. continuous adjoint.* We use the *discrete* adjoint derived above. Because the forward scheme is structure-preserving, an arbitrary discretization of the continuous adjoint ODE $-\dot \lambda = J^\top \lambda - \nabla\ell(u)$ would be inconsistent with the forward scheme and produce an approximate gradient rather than the exact gradient of $\mathcal{L}$. The discrete adjoint guarantees that $\nabla_\theta \mathcal{L}$ matches finite-difference checks up to round-off.

4. *Gradient verification.* Always verify the implementation with finite-difference checks: pick a random direction $e$, compute $g_{\text{FD}} = [\mathcal{L}(\theta + \epsilon e) - \mathcal{L}(\theta - \epsilon e)] / (2 \epsilon)$ and confirm $\langle \nabla_\theta \mathcal{L}, e \rangle \approx g_{\text{FD}}$ to relative accuracy $\sim 10^{-4}$ for $\epsilon \sim 10^{-5}$.

---

## 3. Well-Posedness of the Forward Step

We now prove that under the structural assumptions on $A$ and $H$, and a mild step size restriction, the implicit midpoint equation (1.3) has a unique solution $u_{n+1}$.

### 3.1 Assumptions and notation

**Assumption 3.1 (Standing assumptions).** We assume:

1. **Dissipative symmetric part**: $A + A^\top \preceq -2 \mu I$ for some $\mu > 0$. Equivalently, $\lambda_{\max}\!\left( \tfrac{A + A^\top}{2} \right) \le -\mu$.
2. **Symmetric bilinear $H$**: $H(u, v) = H(v, u)$ for all $u, v \in \mathbb{R}^k$.
3. **Energy preservation**: $u^\top H(u, u) = 0$ for all $u \in \mathbb{R}^k$.

We use the operator norm

$$
\|H\|_{\mathrm{op}} \;:=\; \sup_{\|x\| = \|y\| = \|z\| = 1} \big| x^\top H(y, z) \big|,
$$

which is finite since $H$ is bilinear on a finite-dimensional space.

### 3.2 Reformulation in midpoint variables

Write $\bar u := (u_n + u_{n+1})/2$. Then $u_{n+1} = 2 \bar u - u_n$, and (1.3) becomes

$$
\frac{2(\bar u - u_n)}{\Delta t} = A \bar u + H(\bar u, \bar u) + c.
$$

Rearranging,

$$
\Phi(\bar u) := \left( \frac{2}{\Delta t} I - A \right) \bar u - H(\bar u, \bar u) = b, \qquad b := \frac{2}{\Delta t} u_n + c. \tag{3.1}
$$

Showing existence and uniqueness of $u_{n+1}$ is equivalent to showing that (3.1) has exactly one solution.

We will establish:

1. An a priori bound on any solution of (3.1), using the energy-preserving structure of $H$.
2. Existence via Brouwer's fixed-point theorem applied to a continuous fixed-point map on a closed ball.
3. Uniqueness within the ball, via a derived identity from energy preservation.

### 3.3 An a priori energy estimate

**Lemma 3.2 (A priori bound).** Let $\eta := \frac{2}{\Delta t} + \mu > 0$. Then any solution $\bar u$ of (3.1) satisfies

$$
\|\bar u\| \;\le\; \frac{\|b\|}{\eta} \;=\; \frac{ \left\| \frac{2}{\Delta t} u_n + c \right\| }{ \frac{2}{\Delta t} + \mu }.
$$

*Proof.* Take the inner product of (3.1) with $\bar u$:

$$
\bar u^\top \!\left( \tfrac{2}{\Delta t} I - A \right) \bar u - \bar u^\top H(\bar u, \bar u) = \bar u^\top b.
$$

The second term on the left vanishes by Assumption 3.1(iii). For the first term, write

$$
\bar u^\top \!\left( \tfrac{2}{\Delta t} I - A \right) \bar u = \tfrac{2}{\Delta t} \|\bar u\|^2 - \bar u^\top A \bar u = \tfrac{2}{\Delta t} \|\bar u\|^2 - \tfrac{1}{2} \bar u^\top (A + A^\top) \bar u \;\ge\; \left( \tfrac{2}{\Delta t} + \mu \right) \|\bar u\|^2,
$$

using Assumption 3.1(i). Therefore $\eta \|\bar u\|^2 \le \bar u^\top b \le \|\bar u\| \|b\|$, which gives the claim. $\square$

We will denote the bound by $R^\star := \|b\|/\eta$.

### 3.4 A polarization identity

**Lemma 3.3 (Polarization of energy preservation).** For all $s, w \in \mathbb{R}^k$,

$$
2\, w^\top H(s, w) + s^\top H(w, w) \;=\; 0. \tag{3.2}
$$

*Proof.* Fix $s, w \in \mathbb{R}^k$ and define the scalar function

$$
P(t) \;:=\; (s + tw)^\top H(s + tw,\, s + tw), \qquad t \in \mathbb{R}.
$$

By Assumption 3.1(iii), $P(t) = 0$ for all $t$. Using bilinearity and symmetry of $H$,

$$
H(s + tw, s + tw) = H(s, s) + 2 t H(s, w) + t^2 H(w, w),
$$

hence

$$
P(t) = s^\top H(s, s) + t \big[ 2 s^\top H(s, w) + w^\top H(s, s) \big] + t^2 \big[ s^\top H(w, w) + 2 w^\top H(s, w) \big] + t^3\, w^\top H(w, w).
$$

Since $P$ is a polynomial of degree at most $3$ and vanishes for all $t$, every coefficient is zero. The coefficient of $t^2$ gives (3.2). (The coefficients of $t^0$ and $t^3$ recover Assumption 3.1(iii) for $s$ and $w$ separately; the coefficient of $t$ gives the related identity $2 s^\top H(s, w) + w^\top H(s, s) = 0$.) $\square$

### 3.5 Existence

**Proposition 3.4 (Existence).** Suppose

$$
\eta^2 \;\ge\; 4 \|H\|_{\mathrm{op}} \|b\|. \tag{3.3}
$$

Then (3.1) has a solution $\bar u \in \mathbb{R}^k$ with $\|\bar u\| \le R^\star$.

*Proof.* Under Assumption 3.1(i), the symmetric part of $\frac{2}{\Delta t} I - A$ is at least $\eta I$, so $\frac{2}{\Delta t} I - A$ is invertible with

$$
\left\| \big( \tfrac{2}{\Delta t} I - A \big)^{-1} \right\|_2 \;\le\; \frac{1}{\eta}. \tag{3.4}
$$

(For any $M$ with $M + M^\top \succeq 2 \nu I$, $\|Mv\|\|v\| \ge v^\top M v \ge \nu \|v\|^2$, so $\|Mv\| \ge \nu \|v\|$, giving $\|M^{-1}\|_2 \le 1/\nu$.)

Define the fixed-point map

$$
T(\bar u) \;:=\; \left( \tfrac{2}{\Delta t} I - A \right)^{-1} \big[\, H(\bar u, \bar u) + b \,\big].
$$

Solutions of (3.1) are exactly fixed points of $T$.

If $\|H\|_{\mathrm{op}} = 0$, then $T$ is the constant map $T(\bar u) = \left( \tfrac{2}{\Delta t} I - A \right)^{-1} b$, which is already a fixed point of $T$. Thus assume $\|H\|_{\mathrm{op}} > 0$. We show $T$ maps a closed ball $B_R := \{ u \in \mathbb{R}^k : \|u\| \le R \}$ into itself for some $R \in [0, \eta / (2 \|H\|_{\mathrm{op}})]$.

For $\bar u \in B_R$,

$$
\|T(\bar u)\| \;\le\; \tfrac{1}{\eta} \big[\, \|H\|_{\mathrm{op}} R^2 + \|b\| \,\big].
$$

The condition $\|T(\bar u)\| \le R$ becomes

$$
\|H\|_{\mathrm{op}} R^2 - \eta R + \|b\| \;\le\; 0,
$$

which has a real solution iff the discriminant is non-negative: $\eta^2 \ge 4 \|H\|_{\mathrm{op}} \|b\|$, i.e. (3.3). Choose

$$
R = R_- \;:=\; \frac{\eta - \sqrt{\eta^2 - 4 \|H\|_{\mathrm{op}} \|b\|}}{2 \|H\|_{\mathrm{op}}}.
$$

Then $T(B_{R_-}) \subset B_{R_-}$. Since $T$ is continuous and $B_{R_-}$ is convex and compact, Brouwer's fixed-point theorem yields a fixed point $\bar u \in B_{R_-}$. Note that this fixed point is a solution of (3.1), so Lemma 3.2 implies the sharper bound $\|\bar u\| \le R^\star$. $\square$

### 3.6 Uniqueness

**Proposition 3.5 (Uniqueness).** Under condition (3.3), the fixed point of $T$ in $B_{R^\star}$ is unique.

*Proof.* Let $\bar u_1, \bar u_2$ be two solutions of (3.1). By Lemma 3.2, $\|\bar u_i\| \le R^\star$ for $i = 1, 2$. Set $w := \bar u_1 - \bar u_2$ and $s := \bar u_1 + \bar u_2$, so $\|s\| \le 2 R^\star$.

Subtracting the two copies of (3.1),

$$
\left( \tfrac{2}{\Delta t} I - A \right) w = H(\bar u_1, \bar u_1) - H(\bar u_2, \bar u_2).
$$

Using bilinearity and symmetry,

$$
H(\bar u_1, \bar u_1) - H(\bar u_2, \bar u_2) = H(\bar u_1, \bar u_1 - \bar u_2) + H(\bar u_1 - \bar u_2, \bar u_2) = H(s, w),
$$

hence

$$
\left( \tfrac{2}{\Delta t} I - A \right) w = H(s, w). \tag{3.5}
$$

Take the inner product with $w$:

$$
\eta \|w\|^2 \;\le\; w^\top \!\left( \tfrac{2}{\Delta t} I - A \right) w = w^\top H(s, w).
$$

By Lemma 3.3, $w^\top H(s, w) = -\tfrac{1}{2}\, s^\top H(w, w)$, so

$$
\eta \|w\|^2 \;\le\; -\tfrac{1}{2}\, s^\top H(w, w) \;\le\; \tfrac{1}{2} \|s\| \cdot \|H\|_{\mathrm{op}} \|w\|^2 \;\le\; \|H\|_{\mathrm{op}} R^\star \|w\|^2.
$$

Therefore

$$
\big( \eta - \|H\|_{\mathrm{op}} R^\star \big) \|w\|^2 \;\le\; 0.
$$

Now, $R^\star = \|b\| / \eta$, so $\eta - \|H\|_{\mathrm{op}} R^\star = \eta - \|H\|_{\mathrm{op}} \|b\| / \eta = (\eta^2 - \|H\|_{\mathrm{op}} \|b\|) / \eta$. Condition (3.3) gives $\|H\|_{\mathrm{op}} \|b\| \le \eta^2 / 4$, hence

$$
\eta - \|H\|_{\mathrm{op}} R^\star \;\ge\; \frac{3\eta}{4} \;>\; 0.
$$

We conclude $w = 0$, i.e. $\bar u_1 = \bar u_2$. $\square$

### 3.7 Main result

**Theorem 3.6 (Well-posedness of implicit midpoint).** Under Assumption 3.1, suppose the step size $\Delta t$ satisfies

$$
\boxed{\;\left( \frac{2}{\Delta t} + \mu \right)^2 \;\ge\; 4 \|H\|_{\mathrm{op}}\, \left\| \frac{2}{\Delta t} u_n + c \right\|.\;} \tag{3.6}
$$

Then the implicit midpoint equation (1.3) has a unique solution $u_{n+1} \in \mathbb{R}^k$.

*Proof.* Combine Propositions 3.4 and 3.5. $\square$

### 3.8 Discussion

**Remark 3.7 (Asymptotic step size restriction).** For small $\Delta t$, condition (3.6) reads, to leading order,

$$
\frac{4}{\Delta t^2} \;\gtrsim\; \frac{8 \|H\|_{\mathrm{op}} \|u_n\|}{\Delta t}, \qquad \text{i.e. } \quad \Delta t \;\lesssim\; \frac{1}{2\, \|H\|_{\mathrm{op}} \|u_n\|}.
$$

The restriction is therefore mild: $\Delta t$ is bounded only by the inverse of the local quadratic "speed" $\|H\|_{\mathrm{op}} \|u_n\|$, and is independent of any CFL-type condition on linear stiffness.

**Remark 3.8 (Uniform step size over a simulation).** By the energy estimate

$$
\tfrac{1}{2} \tfrac{d}{dt} \|u\|^2 = u^\top A u + u^\top c \;\le\; -\mu \|u\|^2 + \|u\|\|c\|
$$

combined with Grönwall, $\|u(t)\|$ is uniformly bounded on $[0, T]$ for bounded forcing $c$. Consequently a single $\Delta t$ satisfying (3.6) for the maximum attained $\|u_n\|$ guarantees well-posedness of every forward step.

**Remark 3.9 (The role of energy preservation).** The proof critically uses Lemma 3.3, which is a polarization of the identity $u^\top H(u, u) = 0$. Without energy preservation, the term $\bar u^\top H(\bar u, \bar u)$ in the proof of Lemma 3.2 does not vanish, and the a priori bound is lost. The architecture's by-construction energy-preserving parameterization of $H$ is therefore not merely a desirable qualitative feature; it is what makes the implicit midpoint scheme *unconditionally well-posed in $\|u_n\|$* (modulo the explicit step size restriction (3.6)).

**Remark 3.10 (Numerical solution of the per-step system).** In practice, (1.3) is solved by Newton iteration. The Newton system at iterate $u_{n+1}^{(k)}$ is

$$
\left( I - \tfrac{\Delta t}{2} J_n^{(k)} \right) \delta = -\!\left[ u_{n+1}^{(k)} - u_n - \Delta t\, F\!\left( \tfrac{u_n + u_{n+1}^{(k)}}{2} \right) \right],
$$

where $J_n^{(k)} = A + 2 H(\bar u_n^{(k)}, \cdot)$. Under the conditions above the operator $I - \tfrac{\Delta t}{2} J_n^{(k)}$ has positive definite symmetric part, so GMRES applied with Jacobian–vector products (no Jacobian assembly) converges in a small number of iterations. The same operator, transposed, drives the adjoint linear solves in Section 2, allowing the forward and backward solvers to share infrastructure.
