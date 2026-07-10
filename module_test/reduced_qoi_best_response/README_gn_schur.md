# VarPro GN / Schur Curvature Notes

This note documents the second-order object used for the Hessian landscape
research branch. It is deliberately about the Gauss-Newton / linearized
least-squares curvature, not the exact reduced Hessian.

## Four Formulations

The first landscape comparison should use the two independent modeling choices:

1. Quadratic dynamics parameterization:
   - general \(H\): optimize the compressed matrix
     \[
     H\in\mathbb R^{r\times r(r+1)/2};
     \]
   - energy-preserving \(H\): optimize \(\mu_H\), with
     \[
     H_{ijk}+H_{jki}+H_{kij}=0.
     \]
2. Decoder treatment:
   - joint decoder: optimize \((\theta,\psi)\) together;
   - VarPro decoder: eliminate \(\psi\) by the decoder normal equation.

Here
\[
\theta=(A,H,B,c),\qquad \psi=(V_1,V_2,v_0).
\]

The current code supports the first-order general \(H\) path through
`GeneralQuadraticDynamics`, and the second-order GN objects described below.

## Joint GN Action

For the joint problem
\[
L(\theta,\psi)=\frac12\|r(\theta,\psi)\|_W^2+R_D(\psi)+R_G(\theta),
\]
the joint Gauss-Newton matrix is
\[
G_{\rm joint}
=
\begin{bmatrix}
J_\theta & J_D
\end{bmatrix}^T
W
\begin{bmatrix}
J_\theta & J_D
\end{bmatrix}
+ \nabla^2 R_D+\nabla^2 R_G.
\]

The implementation is
`ObservationAlignedBestResponseEvaluator.evaluate_joint_gauss_newton_hessian_action`.
It is matrix-free and MPI-compatible: each rank accumulates local contributions,
then uses existing array reductions.

## VarPro GN / Schur Action

For VarPro, the decoder is at the regularized best response:
\[
\psi^*(\theta)
=
\arg\min_\psi
\frac12\|r(\theta,\psi)\|_W^2+R_D(\psi).
\]

The linearized reduced problem for a dynamics perturbation \(\delta\theta\) is
\[
\min_{\delta\psi}
\frac12
\|J_\theta\delta\theta+J_D\delta\psi\|_W^2
+\frac12\,\delta\psi^T R_D\delta\psi.
\]

The decoder correction solves
\[
(J_D^T W J_D+R_D)\delta\psi
=
-J_D^T W J_\theta\delta\theta.
\]

The Schur/GN quadratic form is
\[
\delta\theta^T G_{\rm VP}\delta\theta
=
\|J_\theta\delta\theta+J_D\delta\psi\|_W^2
+\delta\psi^T R_D\delta\psi
+\delta\theta^T\nabla^2 R_G\delta\theta.
\]

The implementation is
`ReducedObjectiveWorkflow.evaluate_gauss_newton_hessian_action`.

Important distinction:

- `compute_decoder_best_response_action` differentiates the full decoder
  best-response normal equation. It includes residual-weighted terms and is for
  exact reduced-Hessian actions.
- `compute_decoder_gauss_newton_schur_action` solves only the linearized
  Gauss-Newton Schur correction above. It should be used for GN landscape
  diagnostics.

## Lagged-Midpoint Tangents

Matern52 uses the lagged-midpoint rollout. Its step is not simply an implicit or
explicit step applied to \(f(u,\theta)\); it freezes a linear operator at a
predictor state:
\[
z_n = \operatorname{RK4}_{\Delta t/2}(u_n),
\qquad
M_n = A + H(z_n,\cdot),
\]
\[
(I-\tfrac{\Delta t}{2}M_n)u_{n+1}
=
(I+\tfrac{\Delta t}{2}M_n)u_n
+\Delta t\,f_{\rm force}(t_{n+1/2}).
\]

Therefore a parameter tangent needs more than the RHS parameter action
\(\delta f(u,t)\). It also needs the parameter action on the frozen linear
operator:
\[
\delta M_n
=
\delta A+\delta H(z_n,\cdot)+H(\delta z_n,\cdot).
\]

The solver implements this analytically through
`rollout_lagged_midpoint_tangent_from_base_rollout` with two optional callbacks:

- `parameter_linear_operator_action(z,t)` returns
  \(\delta A+\delta H(z,\cdot)\);
- `forcing_parameter_action(t)` returns \(\delta Bp(t)+\delta c\).

The reduced/GN code supplies those callbacks automatically for dynamics
parameter directions. This avoids finite differencing the whole time step.

## General \(H\) vs Energy \(H\)

When a general \(H\) is initialized from an energy-preserving \(H(\mu_H)\), the
forward model is identical:
\[
Au+H(\mu_H)(u,u)+Bp+c.
\]

The energy case is a restriction of the general \(H\) parameter space. If
\[
\delta H = D_\mu H\,\delta\mu_H,
\]
then the energy-space GN action should satisfy
\[
G_\mu\delta\mu_H
=
D_\mu H^T\,G_H\,D_\mu H\,\delta\mu_H.
\]

The unit tests check this pullback relation for the joint GN action.

## Current Tests

`unit_test/test_general_quadratic_dynamics.py` checks:

1. `QuadraticDynamics(mu_h)` and `GeneralQuadraticDynamics(h_matrix)` produce
   identical lagged-midpoint trajectories and QoI when `h_matrix = H(mu_h)`.
2. General \(H\) works with the MPI-aware decoder normal equation and passes a
   first-order VarPro Taylor test.
3. The VarPro Schur/GN action is symmetric and its Rayleigh quotient matches the
   projected-tangent quadratic form.
4. The energy \(H\) joint GN action agrees with the pullback of the general
   \(H\) joint GN action when the perturbation lies in the energy subspace.
5. The analytic lagged-midpoint tangent matches finite differences for a general
   \(H\) dynamics direction.

Run:

```bash
python -m unittest unit_test.test_general_quadratic_dynamics -v
python -m unittest discover -s unit_test -p "test_*.py"
```
