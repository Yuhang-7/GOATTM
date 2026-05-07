# Optimizer Logic in `GOAM_clean`

This note summarizes how optimization is actually organized in `GOAM_clean`, with emphasis on the code paths in

- `GOAM/solver/optimizer.py`
- `GOAM/solver/multiple.py`
- `GOAM/solver/cross_validation_problem.py`
- `Example/ADR/main_ADR.py`
- `Example/ADR_using_general/main_ADR.py`

The goal is not to document every helper function, but to explain the optimization architecture clearly enough to reimplement it in a new library.

---

## 1. High-Level Picture

`GOAM_clean` does **not** optimize the entire ROM in one flat black-box loop by default.

Instead, the optimization logic is built around the following decomposition:

1. Choose the dynamic parameters `g`.
2. For those dynamic parameters, solve the reduced dynamics forward.
3. Assemble and solve a linear normal equation for the decoder parameters `f`.
4. Evaluate the loss.
5. Compute the gradient with respect to the dynamic parameters using a discrete adjoint.
6. Run an outer optimizer on the dynamic parameters only.

So the main training variable in the outer nonlinear optimization is the **dynamic parameter block**, while the **decoder block** is re-identified inside each loss/gradient evaluation.

This is the most important design choice to preserve if you want behavior similar to `GOAM_clean`.

---

## 2. Parameter Blocks

The ROM is split into two learnable components:

### 2.1 Decoder parameters `f`

The decoder maps reduced states to QoIs. In the default quadratic decoder form,

$$
q \approx D(u) = V_1 u + V_2(u,u) + v.
$$

In code, this block is stored in `self.f`.

Typical parameter vector:

$$
\mu_f = (V_1, V_2, v).
$$

### 2.2 Dynamic parameters `g`

The reduced dynamics are of the form

$$
\dot u = A u + H(u,u) + Bp + c.
$$

In code, this block is stored in `self.g`.

There are two modes:

- `dynamic_type='stablized'`: optimize `(A, muH, B, c)`, where `muH` is the compressed energy-preserving parameterization.
- `dynamic_type='general'`: optimize `(A, H, B, c)` directly.

So the dynamic parameter vector is

$$
\mu_g =
\begin{cases}
(A,\mu_H,B,c), & \text{stablized},\\
(A,H,B,c), & \text{general}.
\end{cases}
$$

---

## 3. What the Outer Optimizer Actually Optimizes

The outer objective used in `cross_validation_problem.optimize_solver()` is effectively

$$
\min_{\mu_g} \; \mathcal{L}(\mu_g),
$$

where for each trial value of `\mu_g` the code does:

1. Update the dynamic model.
2. Solve the reduced forward problem.
3. Assemble the decoder normal equation.
4. Solve for the decoder parameters `\mu_f^\star(\mu_g)`.
5. Evaluate the total loss using this decoder.

So the optimized objective is really the **reduced objective**

$$
\mathcal{L}(\mu_g) := \mathcal{J}(\mu_f^\star(\mu_g), \mu_g).
$$

This logic appears in:

- `multiple.py::lossfunc`
- `multiple.py::lossfunc_jacobian`
- `multiple.py::find_decode_param`

### 3.1 Consensus for the `fix mu_g` case

One detail that should be made explicit for any rewrite is the meaning of "optimize `mu_f` after `mu_g` is fixed".

In `GOAM_clean`, once `mu_g` is fixed, the decoder block is **not** normally advanced by the outer nonlinear optimizer. Instead, `mu_f` is treated as a linear subproblem and is recovered by

$$
\mu_f^\star(\mu_g)=\arg\min_{\mu_f}\mathcal{J}(\mu_f,\mu_g),
$$

through least squares / normal equations.

So the practical consensus is:

- fixing `mu_g` implies recomputing the best-response `mu_f^\star(\mu_g)` by `assemble_normal_system -> normal_solve`,
- this decoder recovery is part of every loss/gradient/Hessian evaluation on the main training path,
- the final exported decoder should also be the decoder obtained from that fixed final `mu_g`,
- the `lossfunc_full` / `lossfunc_jacobian_full` branch is a full-space auxiliary path, useful for diagnostics or experiments, but not the default production training logic.

For the rewrite, one additional implementation constraint should be made explicit here: when MPI is used, this decoder-recovery step should be treated as a **distributed normal-equation assembly problem**. Each rank may own only a subset of training samples, so the local contributions to the decoder least-squares system must be accumulated globally, typically by MPI reductions of the normal matrix and right-hand side before solving for `\mu_f`.

The full joint objective also exists:

- `lossfunc_full`
- `lossfunc_jacobian_full`
- `lossfunc_hessian_action_full`

but the main training scripts usually optimize only the dynamic block and recover the decoder block by normal solve.

---

## 4. Loss Composition

The total training loss stored in `self.train_problem.totalloss` is

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{QoI}}
+ \beta \mathcal{L}_{\mathrm{QoI\_dt}}
+ \mathcal{R}_f
+ \mathcal{R}_g.
$$

### 4.1 QoI misfit

The main data misfit is the QoI mismatch over all training trajectories:

$$
\mathcal{L}_{\mathrm{QoI}} = \sum_{\text{samples}} \sum_n \ell(q_n, D(u_n)).
$$

In code this is accumulated through `compute_QoImisfit`.

### 4.2 Optional time-derivative QoI misfit

If `use_timederi=True`, an extra derivative misfit term is added:

$$
\beta \mathcal{L}_{\mathrm{QoI\_dt}}.
$$

### 4.3 Decoder regularization

For `regf_type='Tikhonov'`,

$$
\mathcal{R}_f
= c_{V_1}\|V_1\|_F^2
+ c_{V_2}\|V_2\|_F^2
+ c_v\|v\|^2.
$$

### 4.4 Dynamic regularization

There are two modes.

#### `general` mode

Standard Tikhonov regularization:

$$
\mathcal{R}_g
= c_A\|A\|_F^2
+ c_H\|H\|_F^2
+ c_B\|B\|_F^2
+ c_c\|c\|^2.
$$

#### `stablized` mode

This mode adds a stability-oriented penalty to the symmetric part of `A`.

Let

$$
S(A) := \frac{A + A^\top}{2}.
$$

The code constructs a `Stable_Matrix` object and penalizes a scalar quantity called `sm_spec_abs`, which is a smooth spectral-abscissa-like measure of instability. The penalty is

$$
\mathcal{R}_{A,\mathrm{stable}}
= c_{A,\mathrm{stable}} \, \varphi(\mathrm{sm\_spec\_abs}(S(A))),
$$

with default smooth barrier-like function

$$
\varphi(x) = \log(1 + e^{x-\alpha})^2.
$$

The total dynamic regularization in stabilized mode is

$$
\mathcal{R}_g
= c_{A,\mathrm{stable}} \varphi(\mathrm{sm\_spec\_abs}(S(A)))
+ c_A\|A\|_F^2
+ c_H\|H\|_F^2
+ c_B\|B\|_F^2
+ c_c\|c\|^2.
$$

This is the main mechanism by which `GOAM_clean` encourages stable reduced dynamics.

---

## 5. Initialization Strategy

`GOAM_clean` does not rely purely on random initialization.

### 5.1 OpInf initialization

The preferred initialization path is an operator-inference-style least-squares solve:

- project full-order trajectories to a reduced basis,
- build regression features from midpoint-like reduced data,
- solve for dynamic parameters,
- for stabilized dynamics, solve in the compressed `muH` coordinates.

Relevant functions:

- `multiple.py::optimal_initialize_opinf_parameter`
- `multiple.py::initialize_opinf`

For the `AHBc` form, `muH_to_fullH_matrix(r)` is used to convert between compressed and expanded quadratic operators.

### 5.2 Decoder initialization and fixed-`mu_g` recovery

Once dynamic parameters are initialized, the decoder is identified by solving a linear normal equation:

- forward solve with the current dynamics,
- assemble the normal system,
- solve for `mu_f`.

This logic is in `find_decode_param`.

Conceptually, this is not only an initialization trick. It is the same best-response rule used throughout the main optimization path: whenever `mu_g` is fixed, `mu_f` is recovered by this linear solve rather than by another outer Adam/BFGS loop.

### 5.3 Random initialization

There is also `initialize_random(scale)`, but the main example scripts usually start from OpInf-seeded dynamic parameters loaded from disk.

---

## 6. Available Optimizers

There are two layers of optimizer code.

### 6.1 Low-level optimizers in `optimizer.py`

This file contains reusable solvers:

- `gradient_descend` with Armijo backtracking
- `gradient_descent_with_line_search` using SciPy line search
- `bfgs_with_line_search`
- `adam_solver`
- `LRSFN_solver`
- `exact_LRSFN_solver`
- partial `inexact_newton_cg`

In practice, the most important custom solver here is `adam_solver`.

Its behavior:

1. standard Adam moments `(m, v)`,
2. cosine-annealed learning rate,
3. trial step rejection if the loss becomes `NaN` or too large,
4. repeated halving of the step until the loss is finite and acceptable,
5. stop on `maxiter` or gradient norm tolerance.

### 6.2 Main training optimizer dispatch in `cross_validation_problem.py`

The real training entry point is

`cross_validation_opinf_problem.optimize_solver(...)`.

This function dispatches among:

- `Newton_CG`
- `GD`
- `Adam+Newton_CG`
- `Adam+BFGS`
- `Adam+L_BFGS`
- `BFGS`
- `L_BFGS`
- placeholders for `Gauss_Newton` and `LM`

The production scripts usually use:

- `Adam+BFGS`

as the default method.

---

## 7. The Core Training Pattern: Adam Warm Start, Then Second-Order Refinement

This is the central optimization logic used in the example scripts.

### 7.1 Stage A: Adam

Run a small number of Adam iterations first:

$$
\mu_g^{(0)} \to \mu_g^{(\mathrm{Adam})}.
$$

Purpose:

- reduce the objective quickly,
- move away from poor initial scaling,
- avoid giving Newton/BFGS a terrible starting point.

In the example scripts, this is often just `maxiter[0] = 1`, but the machinery supports longer Adam warm starts.

### 7.2 Stage B: deterministic refinement

After Adam, switch to one of:

- `Newton-CG`
- `BFGS`
- `L-BFGS-B`

This is exactly why the method names are composite, such as:

- `Adam+Newton_CG`
- `Adam+BFGS`
- `Adam+L_BFGS`

So the actual workflow is not “one optimizer only”, but rather:

$$
\text{coarse first-order phase} \;\to\; \text{refinement phase}.
$$

This staged structure is worth preserving in a reimplementation.

---

## 8. Adaptive Inner-Solver Refinement During Optimization

One subtle but very important part of `GOAM_clean` is that the ODE/discretization accuracy is also refined during optimization.

### 8.1 For `RK4`

The code progressively tightens the ODE tolerance:

$$
\text{tol}_{\mathrm{ODE}}:
\text{initialtol} \to \text{finaltol}.
$$

### 8.2 For `CN`

The code progressively refines the number of compute steps:

$$
N_t^{\mathrm{compute}} = N_{t,0} \cdot 2^{\text{subiter}}.
$$

This means the training objective is solved first on a cheaper time grid, then on finer grids.

That is, the outer optimization is staged not only in optimizer type, but also in forward/adjoint solve fidelity.

This is one of the most important architectural ideas in the whole codebase.

---

## 9. Gradient Computation Path

For a given trial dynamic parameter:

1. update dynamic parameters,
2. update decoder parameters if needed,
3. forward solve,
4. solve the decoder normal system,
5. adjoint solve,
6. assemble the gradient.

This is implemented in

- `lossfunc_jacobian`
- `lossfunc_jacobian_full`

For stabilized dynamics, gradients with respect to the expanded `H` are pushed back to compressed coordinates via

$$
\nabla_{\mu_H}\mathcal{L} = L^\top \nabla_H \mathcal{L},
$$

implemented through `Hdiff_to_muH`.

---

## 10. Hessian and Hessian-Vector Products

When `Newton-CG` is used, `GOAM_clean` does not form the full Hessian by default.

Instead it provides a Hessian-vector product interface:

- `lossfunc_hessp(x, dx)`

which calls

- `self.lossfunc_hessp(x, dx, use_GN)`

and optionally adds Levenberg-Marquardt damping:

$$
H p \mapsto H p + \gamma p.
$$

There are also full-space Hessian-action routines for:

- exact Hessian
- Gauss-Newton Hessian
- dynamic-only Hessian
- full `(decoder, dynamics)` Hessian

but these are primarily for diagnostics, checking, or plotting, not the main fast path.

---

## 11. MPI Parallel Structure

The optimization is rank-0-driven.

### Rank 0 responsibilities

- own the outer optimizer,
- call SciPy `minimize`,
- choose step directions and trial iterates,
- log progress,
- save checkpoints,
- broadcast commands to workers.

### Worker responsibilities

Workers stay inside a command loop and respond to tasks such as:

- `update_dynamic_param`
- `update_decode_param`
- `forward_solve`
- `assemble_normal_system`
- `normal_solve`
- `adjoint_solve`
- `compute_gradient`
- `compute_totalloss`

This means the optimizer is architecturally centralized, but the expensive trajectory computations are distributed.

If you rebuild this in a new library, it is helpful to separate:

1. the optimizer state machine,
2. the distributed forward/adjoint service layer.

---

## 12. Callback and Logging Logic

Every outer iteration logs much more than just the loss.

The callback records:

- iteration number,
- total loss,
- QoI misfit,
- QoI-derivative misfit if enabled,
- relative training error,
- regularization contribution,
- norms of decoder and dynamic parameter blocks,
- eigenvalues of `A`,
- eigenvalues of the symmetric part of `A`,
- norms of `B`, `H`, `c`,
- gradient norms,
- Hessian evaluation count.

It also saves:

- current decoder parameters,
- current dynamic parameters,
- optional normal matrix,
- optional current iterate `x`.

This is why `GOAM_clean` leaves a rich optimization trace on disk.

---

## 13. What the Example Scripts Actually Do

The example drivers such as `Example/ADR/main_ADR.py` and `Example/ADR_using_general/main_ADR.py` follow this pattern:

1. build `cross_validation_opinf_problem`,
2. set regularization coefficients,
3. load OpInf-initialized dynamic parameters from file,
4. compute decoder parameters by `find_decode_param()`,
5. run gradient checks,
6. call `optimize_solver(...)`,
7. validate the final parameters.

For the default non-GD path they typically use:

- `method = "Adam+BFGS"`
- `maxiter = [1, 5000]`
- `lr = 1e-3`
- `maxsubiter = 1`
- `solver_type = "CN"`

So the practical default in the repository is:

$$
\text{OpInf init} \;\to\; \text{decoder normal solve} \;\to\; \text{gradient check} \;\to\; \text{Adam} \;\to\; \text{BFGS}.
$$

---

## 14. Design Principles Worth Carrying Into a New Library

If you want to reproduce the optimizer logic of `GOAM_clean`, the important ideas are:

1. **Optimize dynamic parameters externally; recover decoder parameters internally.**
   This is the biggest structural choice.

2. **Use a staged optimizer.**
   Adam first, then a stronger local method such as BFGS or Newton-CG.

3. **Refine the fidelity of the forward/adjoint solve during optimization.**
   Cheap early iterations, more accurate later iterations.

4. **Keep stabilized and general dynamics separate.**
   In particular, stabilized dynamics use compressed `muH` coordinates and stability-aware regularization.

5. **Expose Hessian-vector products even if you do not always use them.**
   This keeps Newton-type methods possible later.

6. **Treat logging and checkpointing as part of the optimizer design.**
   `GOAM_clean` relies heavily on saved traces for debugging and long runs.

7. **Separate the optimization controller from the expensive distributed solves.**
   This is essential if you want MPI or future accelerator support.

---

## 15. A Good Minimal Reimplementation

If you do not want to replicate everything, the closest minimal version of the `GOAM_clean` optimizer would be:

1. Parameter blocks:
   `mu_g` optimized externally, `mu_f` solved by normal equation.

2. Objective:
   QoI misfit + optional derivative misfit + Tikhonov/stability regularization.

3. Initialization:
   OpInf-style least squares for dynamics, then decoder normal solve.

4. Optimizer:
   `Adam -> BFGS` pipeline.

5. Gradient:
   discrete adjoint gradient with `muH` pullback.

6. Time discretization refinement:
   coarse `computeNt` early, finer `computeNt` later.

That subset would already capture most of the practical behavior of the original codebase.

---

## 16. Short Summary

`GOAM_clean` is best viewed as a **variable-projection-style ROM training framework**:

- the decoder is solved in closed form for each dynamic iterate,
- the outer nonlinear optimizer acts mainly on the dynamics,
- initialization comes from OpInf,
- optimization is staged (`Adam` then `BFGS`/`Newton-CG`),
- stabilized dynamics use a structure-preserving `muH` parameterization plus spectral stability regularization,
- MPI workers perform the expensive forward/adjoint tasks while rank 0 drives the optimizer.

That is the logic you should reproduce if you want a new library to behave “like `GOAM_clean`”.
