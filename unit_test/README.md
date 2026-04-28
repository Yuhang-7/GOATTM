## Unit Test Inventory

### Purpose

The `unit_test` directory is used to verify that the code in `src` is correct at the component level.
Unlike `module_test`, which is used for algorithm prototyping and broader workflow experiments, the tests here are meant to validate concrete implementation details in a focused and reproducible way.


### Current Scope

At the moment, the unit tests cover six layers of the new quadratic-dynamics and QoI-loss stack:

1. the `mu_H` energy-preserving parametrization,
2. the `A = -S S^T + W` stabilized parametrization,
3. compressed quadratic kernels,
4. the quadratic dynamics model,
5. the implicit midpoint nonlinear solver,
6. QoI loss evaluation and gradient computation.


### Test Files

- `test_mu_h_parametrization.py`
- `test_a_parametrization.py`
- `test_quadratic_core.py`
- `test_quadratic_dynamics.py`
- `test_implicit_midpoint_solver.py`
- `test_qoi_loss.py`


### Detailed Test List

#### `test_mu_h_parametrization.py`

1. `test_mu_h_dimension_matches_formula`
   - Purpose:
     Verify that the parametrization dimension is
     `d_H = r(r-1)(r+1)/3`.
   - Method:
     Check the implementation against the closed-form formula for several `r`.

2. `test_mu_h_to_compressed_h_has_expected_shape`
   - Purpose:
     Verify that `mu_H` expands to the expected compressed matrix shape `(r, r(r+1)/2)`.
   - Method:
     Expand a random `mu_H` and check the output shape.

3. `test_compressed_h_to_mu_h_inverts_parametrization`
   - Purpose:
     Verify that the inverse map recovers `mu_H` from the compressed matrix representation.
   - Method:
     Expand `mu_H` to `H`, then pull back to `mu_H` and compare.

4. `test_mu_h_parametrization_is_energy_preserving`
   - Purpose:
     Verify the base structural identity
     `u^T H(u, u) = 0`.
   - Method:
     Sample random `u` and check the scalar defect numerically.

5. `test_mu_h_parametrization_satisfies_polarization_identity`
   - Purpose:
     Verify Lemma 3.3 from the midpoint note:
     `2 w^T H(s, w) + s^T H(w, w) = 0`.
   - Method:
     Sample random `s, w` and check the identity numerically.

6. `test_compressed_h_gradient_to_mu_h_matches_finite_difference`
   - Purpose:
     Verify the reverse-mode pullback from compressed `H` gradients to `mu_H`.
   - Method:
     Compare the analytic pullback against centered finite differences for a scalar objective.


#### `test_a_parametrization.py`

7. `test_param_dimensions_match_closed_form`
   - Purpose:
     Verify the free-parameter dimensions for the stabilized linear block.
   - Method:
     Check that `dim(S) = r(r+1)/2` and `dim(W) = r(r-1)/2`.

8. `test_s_and_w_matrices_have_expected_structure`
   - Purpose:
     Verify that the reconstruction maps produce an upper-triangular `S` and a skew-symmetric `W`.
   - Method:
     Build both matrices from parameter vectors and inspect their structural identities.

9. `test_a_parametrization_has_negative_semidefinite_symmetric_part`
   - Purpose:
     Verify that `A = -S S^T + W` has the intended stabilized symmetric part.
   - Method:
     Form `A`, compute `(A + A^T)/2`, and check that its eigenvalues are non-positive.

10. `test_a_gradient_pullback_matches_finite_difference`
   - Purpose:
     Verify the reverse-mode pullback from `grad_A` to `(grad_S, grad_W)`.
   - Method:
     Compare the analytic pullback against centered finite differences for a scalar objective.


#### `test_quadratic_core.py`

11. `test_quadratic_eval_matches_manual_contraction`
   - Purpose:
     Verify that `quadratic_eval(H, u, v)` matches the bilinear action matrix representation.
   - Method:
     Compare direct evaluation against `M(u) @ v`.

12. `test_quadratic_eval_matches_compressed_quadratic_features`
   - Purpose:
     Verify that `quadratic_eval(H, u)` matches the compressed monomial form
     `H zeta(u)`.
   - Method:
     Compare the implementation against explicit quadratic feature assembly.

13. `test_bilinear_action_matrix_matches_bilinear_eval`
   - Purpose:
     Verify that the bilinear action matrix satisfies
     `M(u) @ v = H(u, v)`.
   - Method:
     Compare the matrix action against direct bilinear evaluation.

14. `test_quadratic_jacobian_matches_finite_difference`
   - Purpose:
     Verify that the Jacobian of `H(u, u)` is implemented correctly.
   - Method:
     Compare Jacobian-vector products against centered finite differences.

15. `test_energy_preserving_parametrized_h_satisfies_u_h_u_u_zero`
   - Purpose:
     Verify that a parametrized `H` satisfies
     `u^T H(u, u) = 0`.
   - Method:
     Sample random `u` and check the scalar defect.

16. `test_energy_preserving_parametrized_h_satisfies_polarization_identity`
   - Purpose:
     Verify the polarization identity
     `2 w^T H(s, w) + s^T H(w, w) = 0`.
   - Method:
     Sample random `s, w` and check the identity numerically.

17. `test_energy_preserving_parametrized_h_satisfies_companion_identity`
   - Purpose:
     Verify the companion identity
     `2 s^T H(s, w) + w^T H(s, s) = 0`.
   - Method:
     Sample random `s, w` and check the identity numerically.


#### `test_quadratic_dynamics.py`

18. `test_rhs_matches_manual_formula`
   - Purpose:
     Verify that the full right-hand side
     `f(u) = A u + H(u, u) + c`
     is assembled correctly.
   - Method:
     Compare the implementation against the compressed quadratic-feature formula.

19. `test_rhs_jacobian_matches_finite_difference`
   - Purpose:
     Verify that the Jacobian of the full RHS is correct.
   - Method:
     Compare Jacobian-vector products against centered finite differences.

20. `test_invalid_shapes_raise`
   - Purpose:
     Verify that invalid input shapes are rejected early.
   - Method:
     Construct invalid `A`, `mu_H`, or `c` shapes and confirm that `ValueError` is raised.


#### `test_implicit_midpoint_solver.py`

21. `test_single_linear_step_matches_closed_form`
   - Purpose:
     Verify that the implicit midpoint step matches the known closed-form linear solve when the quadratic term is zero.
   - Method:
     Use diagonal `A`, zero `mu_H`, and compare the solver output to the exact midpoint linear algebra formula.

22. `test_quadratic_step_satisfies_midpoint_residual`
   - Purpose:
     Verify that the computed nonlinear step actually satisfies the implicit midpoint discrete equation.
   - Method:
     Solve one random quadratic step and evaluate the residual norm directly.

23. `test_rollout_reaches_final_time_with_expected_step_count`
   - Purpose:
     Verify that rollout bookkeeping is correct.
   - Method:
     Check that:
     - the solve reaches `t_final`,
     - the number of accepted steps is correct,
     - `states`, `times`, and `dt_history` have the expected shapes,
     - and `np.diff(times) == dt_history`.

24. `test_retry_path_reports_failure_when_newton_is_disabled`
   - Purpose:
     Verify that the retry/failure path is recorded correctly.
   - Method:
     Force the Newton solve to fail by setting `max_iter=0`, then check that failure, retry counts, and shrink logic are reported.


#### `test_qoi_loss.py`

25. `test_qoi_trajectory_loss_matches_manual_trapezoid_formula`
   - Purpose:
     Verify that discrete QoI loss evaluation matches the trapezoidal-rule formula.
   - Method:
     Compare the implementation against a manual sum over trajectory points.

26. `test_decoder_loss_partials_match_finite_difference`
   - Purpose:
     Verify decoder-side loss derivatives with respect to `V1`, `V2`, and `v0`.
   - Method:
     Compare analytic gradients against centered finite differences.

27. `test_rollout_loss_gradients_match_finite_difference`
   - Purpose:
     Verify rollout-based partial derivatives with respect to both decoder and dynamics parameters.
   - Method:
     For a small stabilized latent system with input forcing, compare analytic gradients with respect to
     `S`, `W`, `mu_H`, `B`, `c`, `V1`, `V2`, and `v0` against centered finite differences.


### How To Run

From the repository root:

```bash
python -m unittest discover -s unit_test -p 'test_*.py'
```


### Design Notes

- These tests are intended to validate correctness of the implementation in `src`.
- They are deliberately smaller and more local than the experiments in `module_test`.
- The `mu_H` tests are important because the whole point of the rewrite is to replace constrained optimization over `H` with unconstrained optimization over `mu_H`.
- The loss-gradient tests intentionally stop short of the variable-projection derivative through `mu_f(mu_g)`.
- As the adjoint, optimizer, and data layers evolve further, this document should continue to be extended so the unit-test intent stays easy to audit.
