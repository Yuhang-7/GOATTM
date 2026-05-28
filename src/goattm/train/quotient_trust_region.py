from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from goattm.core.parametrization import (
    compressed_h_to_mu_h,
    lower_triangular_index,
    mu_h_to_compressed_h,
    quadratic_features,
    s_params_to_matrix,
    w_params_to_matrix,
)
from goattm.core.quadratic import quadratic_eval
from goattm.models.linear_dynamics import LinearDynamics
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems.decoder_normal_equation import decoder_feature_vector, decoder_parameter_matrix
from goattm.problems.reduced_qoi_best_response import (
    DynamicsLike,
    ReducedObjectivePreparedState,
    ReducedObjectiveWorkflow,
    decoder_feature_directional_derivative,
    dynamics_from_parameter_vector,
    dynamics_parameter_dimension,
    dynamics_parameter_vector,
    rhs_parameter_action,
    unpack_dynamics_parameter_vector,
)
from goattm.runtime import timed
from goattm.solvers import rollout_tangent_from_base_rollout


@dataclass(frozen=True)
class QuotientTrustRegionConfig:
    subproblem_solver: str = "matrix_free"
    initial_radius: float = 1.0
    max_radius: float = 100.0
    acceptance_threshold: float = 1e-4
    shrink_factor: float = 0.25
    expand_factor: float = 2.0
    metric_ridge: float = 1e-8
    hessian_damping: float = 1e-8
    max_dense_dimension: int = 120
    cg_tolerance: float = 1e-6
    cg_max_iterations: int | None = None
    max_backtracks: int = 8

    def __post_init__(self) -> None:
        if self.subproblem_solver not in {"matrix_free", "dense"}:
            raise ValueError("subproblem_solver must be either 'matrix_free' or 'dense'.")
        if self.initial_radius <= 0.0:
            raise ValueError("initial_radius must be positive.")
        if self.max_radius < self.initial_radius:
            raise ValueError("max_radius must be at least initial_radius.")
        if self.metric_ridge <= 0.0:
            raise ValueError("metric_ridge must be positive.")
        if self.hessian_damping < 0.0:
            raise ValueError("hessian_damping must be nonnegative.")
        if self.cg_tolerance <= 0.0:
            raise ValueError("cg_tolerance must be positive.")
        if self.cg_max_iterations is not None and self.cg_max_iterations <= 0:
            raise ValueError("cg_max_iterations must be positive when provided.")


@dataclass(frozen=True)
class VpMetricCheckResult:
    predicted_quadratic: float
    finite_difference_quadratic: float
    relative_error: float


@dataclass(frozen=True)
class TrustRegionSubproblemResult:
    step: np.ndarray
    predicted_decrease: float
    boundary_hit: bool
    iterations: int


class QuotientTrustRegionUpdater:
    def __init__(self, config: QuotientTrustRegionConfig) -> None:
        self.config = config
        self.radius = float(config.initial_radius)

    @timed("goattm.train.QuotientTrustRegionUpdater.step")
    def step(
        self,
        current_dynamics: DynamicsLike,
        prepared_state: ReducedObjectivePreparedState,
        workflow: ReducedObjectiveWorkflow,
    ) -> tuple[np.ndarray, float]:
        if not isinstance(current_dynamics, (LinearDynamics, QuadraticDynamics, StabilizedQuadraticDynamics)):
            raise NotImplementedError("quotient_trust_region currently supports LinearDynamics, QuadraticDynamics, and StabilizedQuadraticDynamics.")
        dimension = dynamics_parameter_dimension(current_dynamics)
        parameter_vector = dynamics_parameter_vector(current_dynamics)
        vertical = quotient_vertical_basis(current_dynamics)
        projector = EuclideanHorizontalProjector(dimension, vertical)
        if projector.horizontal_dimension == 0:
            return parameter_vector.copy(), 0.0

        if self.config.subproblem_solver == "dense":
            if dimension > self.config.max_dense_dimension:
                raise ValueError(
                    f"quotient_trust_region dense dimension {dimension} exceeds "
                    f"max_dense_dimension={self.config.max_dense_dimension}; use subproblem_solver='matrix_free'."
                )
            subproblem = self._solve_dense_subproblem(prepared_state, workflow, projector)
        else:
            subproblem = self._solve_matrix_free_subproblem(prepared_state, workflow, projector)

        step = subproblem.step
        predicted_decrease = subproblem.predicted_decrease

        base_objective = float(prepared_state.objective_value)
        best_vector = parameter_vector.copy()
        best_objective = base_objective
        accepted_step = np.zeros_like(parameter_vector)
        step_scale = 1.0
        rho = -np.inf
        for _ in range(self.config.max_backtracks + 1):
            trial_step = step_scale * step
            trial_vector = parameter_vector + trial_step
            trial_dynamics = dynamics_from_parameter_vector(current_dynamics, trial_vector)
            trial_objective = float(workflow.evaluate_objective(trial_dynamics))
            if np.isfinite(trial_objective):
                actual_decrease = base_objective - trial_objective
                scaled_prediction = max(step_scale * predicted_decrease, 1e-300)
                rho = actual_decrease / scaled_prediction
                if actual_decrease > 0.0 and rho >= self.config.acceptance_threshold:
                    best_vector = trial_vector
                    best_objective = trial_objective
                    accepted_step = trial_step
                    break
            step_scale *= 0.5

        if best_objective < base_objective and rho > 0.75:
            self.radius = min(self.config.max_radius, self.config.expand_factor * self.radius)
        elif best_objective >= base_objective or rho < 0.25:
            self.radius = max(1e-14, self.config.shrink_factor * self.radius)
        return best_vector, float(np.linalg.norm(accepted_step))

    def _solve_dense_subproblem(
        self,
        prepared_state: ReducedObjectivePreparedState,
        workflow: ReducedObjectiveWorkflow,
        projector: "EuclideanHorizontalProjector",
    ) -> TrustRegionSubproblemResult:
        dimension = dynamics_parameter_dimension(prepared_state.dynamics)
        gradient = projector.project(np.asarray(prepared_state.gradient, dtype=np.float64))
        hessian = workflow.evaluate_explicit_hessian_from_prepared_state(prepared_state).hessian
        hessian = 0.5 * (np.asarray(hessian, dtype=np.float64) + np.asarray(hessian, dtype=np.float64).T)
        if self.config.hessian_damping > 0.0:
            hessian = hessian + self.config.hessian_damping * np.eye(dimension, dtype=np.float64)
        hessian = projector.project_matrix(hessian)
        vp_metric = compute_dense_vp_metric_matrix(prepared_state, workflow)
        trust_metric = projector.project_matrix(vp_metric + self.config.metric_ridge * np.eye(dimension, dtype=np.float64))

        horizontal = projector.horizontal_basis
        if horizontal.shape[1] == 0:
            return TrustRegionSubproblemResult(np.zeros(dimension, dtype=np.float64), 0.0, False, 0)
        q_grad = horizontal.T @ gradient
        q_hessian = horizontal.T @ hessian @ horizontal
        q_metric = horizontal.T @ trust_metric @ horizontal
        q_step = solve_generalized_trust_region(q_hessian, q_grad, q_metric, self.radius)
        step = projector.project(horizontal @ q_step)
        predicted_decrease = -float(gradient @ step + 0.5 * step @ hessian @ step)
        if predicted_decrease <= 0.0 or not np.isfinite(predicted_decrease):
            step = scaled_metric_steepest_descent_step(
                gradient=gradient,
                metric_quadratic=lambda vector: trust_metric_quadratic(
                    prepared_state,
                    workflow,
                    projector.project(vector),
                    self.config.metric_ridge,
                ),
                radius=self.radius,
                projector=projector,
            )
            predicted_decrease = model_predicted_decrease(gradient, lambda vector: hessian @ projector.project(vector), step)
        return TrustRegionSubproblemResult(step, predicted_decrease, False, 0)

    def _solve_matrix_free_subproblem(
        self,
        prepared_state: ReducedObjectivePreparedState,
        workflow: ReducedObjectiveWorkflow,
        projector: "EuclideanHorizontalProjector",
    ) -> TrustRegionSubproblemResult:
        gradient = projector.project(np.asarray(prepared_state.gradient, dtype=np.float64))
        dimension = int(gradient.shape[0])
        max_iterations = self.config.cg_max_iterations
        if max_iterations is None:
            max_iterations = max(2 * projector.horizontal_dimension, 25)

        def hessian_action(vector: np.ndarray) -> np.ndarray:
            horizontal_vector = projector.project(vector)
            if float(np.linalg.norm(horizontal_vector)) <= 0.0:
                return np.zeros(dimension, dtype=np.float64)
            action = workflow.evaluate_hessian_action_from_prepared_state(
                prepared_state=prepared_state,
                direction=horizontal_vector,
            ).action
            action = np.asarray(action, dtype=np.float64)
            if self.config.hessian_damping > 0.0:
                action = action + self.config.hessian_damping * horizontal_vector
            return projector.project(action)

        def metric_quadratic(vector: np.ndarray) -> float:
            return trust_metric_quadratic(
                prepared_state=prepared_state,
                workflow=workflow,
                direction=projector.project(vector),
                metric_ridge=self.config.metric_ridge,
            )

        subproblem = solve_metric_steihaug_trust_region(
            gradient=gradient,
            hessian_action=hessian_action,
            metric_quadratic=metric_quadratic,
            radius=self.radius,
            tolerance=self.config.cg_tolerance,
            max_iterations=max_iterations,
            projector=projector,
        )
        if subproblem.predicted_decrease > 0.0 and np.isfinite(subproblem.predicted_decrease):
            return subproblem
        step = scaled_metric_steepest_descent_step(
            gradient=gradient,
            metric_quadratic=metric_quadratic,
            radius=self.radius,
            projector=projector,
        )
        predicted = model_predicted_decrease(gradient, hessian_action, step)
        return TrustRegionSubproblemResult(step, predicted, True, subproblem.iterations)


class EuclideanHorizontalProjector:
    def __init__(self, dimension: int, vertical: np.ndarray, tolerance: float = 1e-10) -> None:
        self.dimension = int(dimension)
        vertical = np.asarray(vertical, dtype=np.float64)
        if vertical.size == 0 or vertical.shape[1] == 0:
            self.vertical_basis = np.zeros((self.dimension, 0), dtype=np.float64)
            self.horizontal_basis = np.eye(self.dimension, dtype=np.float64)
            return
        q, r_matrix = np.linalg.qr(vertical, mode="reduced")
        diagonal = np.abs(np.diag(r_matrix))
        if diagonal.size == 0:
            rank = 0
        else:
            rank = int(np.sum(diagonal > tolerance * max(vertical.shape) * max(1.0, float(diagonal[0]))))
        self.vertical_basis = q[:, :rank].copy()
        self.horizontal_basis = euclidean_horizontal_basis(self.dimension, self.vertical_basis, tolerance=tolerance)

    @property
    def horizontal_dimension(self) -> int:
        return int(self.horizontal_basis.shape[1])

    def project(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64).reshape(-1)
        if value.shape[0] != self.dimension:
            raise ValueError(f"vector must have length {self.dimension}, got {value.shape[0]}")
        if self.vertical_basis.shape[1] == 0:
            return value.copy()
        return value - self.vertical_basis @ (self.vertical_basis.T @ value)

    def project_matrix(self, matrix: np.ndarray) -> np.ndarray:
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape != (self.dimension, self.dimension):
            raise ValueError(f"matrix must have shape {(self.dimension, self.dimension)}, got {mat.shape}")
        if self.vertical_basis.shape[1] == 0:
            return mat.copy()
        projector_matrix = np.eye(self.dimension, dtype=np.float64) - self.vertical_basis @ self.vertical_basis.T
        return projector_matrix @ mat @ projector_matrix


def trust_metric_quadratic(
    prepared_state: ReducedObjectivePreparedState,
    workflow: ReducedObjectiveWorkflow,
    direction: np.ndarray,
    metric_ridge: float,
) -> float:
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    vp_value = compute_vp_metric_quadratic(prepared_state, workflow, vector)
    return float(vp_value + float(metric_ridge) * float(vector @ vector))


def metric_bilinear_from_quadratic(
    metric_quadratic,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left_vector = np.asarray(left, dtype=np.float64).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float64).reshape(-1)
    return 0.5 * float(
        metric_quadratic(left_vector + right_vector)
        - metric_quadratic(left_vector)
        - metric_quadratic(right_vector)
    )


def boundary_tau_for_metric(
    current: np.ndarray,
    direction: np.ndarray,
    metric_quadratic,
    radius: float,
) -> float:
    current_vector = np.asarray(current, dtype=np.float64).reshape(-1)
    direction_vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    q_current = float(metric_quadratic(current_vector))
    q_direction = float(metric_quadratic(direction_vector))
    if q_direction <= 0.0 or not np.isfinite(q_direction):
        direction_norm = float(np.linalg.norm(direction_vector))
        if direction_norm <= 0.0:
            return 0.0
        return float(radius / direction_norm)
    cross = metric_bilinear_from_quadratic(metric_quadratic, current_vector, direction_vector)
    a = q_direction
    b = 2.0 * cross
    c = q_current - float(radius) ** 2
    discriminant = max(0.0, b * b - 4.0 * a * c)
    roots = [(-b + np.sqrt(discriminant)) / (2.0 * a), (-b - np.sqrt(discriminant)) / (2.0 * a)]
    positive_roots = [float(root) for root in roots if np.isfinite(root) and root >= 0.0]
    if not positive_roots:
        return 0.0
    return min(positive_roots)


def model_predicted_decrease(gradient: np.ndarray, hessian_action, step: np.ndarray) -> float:
    step_vector = np.asarray(step, dtype=np.float64).reshape(-1)
    h_step = np.asarray(hessian_action(step_vector), dtype=np.float64).reshape(-1)
    return -float(np.asarray(gradient, dtype=np.float64).reshape(-1) @ step_vector + 0.5 * step_vector @ h_step)


def scaled_metric_steepest_descent_step(
    gradient: np.ndarray,
    metric_quadratic,
    radius: float,
    projector: EuclideanHorizontalProjector,
) -> np.ndarray:
    step = -projector.project(gradient)
    metric_norm_sq = float(metric_quadratic(step))
    if metric_norm_sq <= 0.0 or not np.isfinite(metric_norm_sq):
        norm = float(np.linalg.norm(step))
        if norm <= 0.0:
            return np.zeros_like(step)
        return step * (float(radius) / norm)
    metric_norm = float(np.sqrt(metric_norm_sq))
    if metric_norm > float(radius):
        step = step * (float(radius) / metric_norm)
    return projector.project(step)


@timed("goattm.train.solve_metric_steihaug_trust_region")
def solve_metric_steihaug_trust_region(
    gradient: np.ndarray,
    hessian_action,
    metric_quadratic,
    radius: float,
    tolerance: float,
    max_iterations: int,
    projector: EuclideanHorizontalProjector,
) -> TrustRegionSubproblemResult:
    gradient = projector.project(gradient)
    step = np.zeros_like(gradient, dtype=np.float64)
    residual = gradient.copy()
    direction = -residual
    residual_norm_initial = float(np.linalg.norm(residual))
    if residual_norm_initial <= float(tolerance):
        return TrustRegionSubproblemResult(step, 0.0, False, 0)
    residual_norm_sq = float(residual @ residual)

    for iteration in range(1, int(max_iterations) + 1):
        direction = projector.project(direction)
        h_direction = projector.project(hessian_action(direction))
        curvature = float(direction @ h_direction)
        if (not np.isfinite(curvature)) or curvature <= 1e-14 * max(1.0, float(np.linalg.norm(direction)) ** 2):
            tau = boundary_tau_for_metric(step, direction, metric_quadratic, radius)
            boundary_step = projector.project(step + tau * direction)
            predicted = model_predicted_decrease(gradient, hessian_action, boundary_step)
            return TrustRegionSubproblemResult(boundary_step, predicted, True, iteration)

        alpha = residual_norm_sq / curvature
        trial_step = projector.project(step + alpha * direction)
        if float(metric_quadratic(trial_step)) >= float(radius) ** 2:
            tau = boundary_tau_for_metric(step, direction, metric_quadratic, radius)
            boundary_step = projector.project(step + tau * direction)
            predicted = model_predicted_decrease(gradient, hessian_action, boundary_step)
            return TrustRegionSubproblemResult(boundary_step, predicted, True, iteration)

        step = trial_step
        residual = projector.project(residual + alpha * h_direction)
        new_residual_norm_sq = float(residual @ residual)
        if np.sqrt(new_residual_norm_sq) <= float(tolerance) * max(1.0, residual_norm_initial):
            predicted = model_predicted_decrease(gradient, hessian_action, step)
            return TrustRegionSubproblemResult(step, predicted, False, iteration)
        beta = new_residual_norm_sq / max(residual_norm_sq, 1e-300)
        direction = projector.project(-residual + beta * direction)
        residual_norm_sq = new_residual_norm_sq

    predicted = model_predicted_decrease(gradient, hessian_action, step)
    return TrustRegionSubproblemResult(step, predicted, False, int(max_iterations))


@timed("goattm.train.compute_vp_metric_quadratic")
def compute_vp_metric_quadratic(
    prepared_state: ReducedObjectivePreparedState,
    workflow: ReducedObjectiveWorkflow,
    direction: np.ndarray,
) -> float:
    direction_vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    dynamics = prepared_state.dynamics
    dynamics_direction = unpack_dynamics_parameter_vector(dynamics, direction_vector)
    decoder_action_matrix = workflow.evaluator.compute_decoder_best_response_action(
        dynamics=dynamics,
        decoder_template=workflow.decoder_template,
        regularization=workflow.regularization,
        direction=dynamics_direction,
        solve_root=workflow.solve_root,
    )
    decoder = prepared_state.result.decoder
    x_matrix = decoder_parameter_matrix(decoder)
    local_value = 0.0
    for rollout_entry in prepared_state.result.best_response_context.forward_cache.local_rollouts:
        tangent_states = rollout_tangent_from_base_rollout(
            dynamics=dynamics,
            base_rollout=rollout_entry.rollout,
            parameter_action=lambda midpoint_state, midpoint_time, input_function=rollout_entry.input_function: rhs_parameter_action(
                dynamics,
                dynamics_direction,
                midpoint_state,
                midpoint_time,
                input_function=input_function,
            ),
            input_function=rollout_entry.input_function,
            time_integrator=workflow.evaluator.time_integrator,
        )
        observed_tangents = tangent_states[rollout_entry.observation_indices]
        for state, state_tangent, weight in zip(
            rollout_entry.observed_states,
            observed_tangents,
            rollout_entry.observation_weights,
            strict=True,
        ):
            phi = decoder_feature_vector(state, decoder.form)
            dphi = decoder_feature_directional_derivative(state, state_tangent)
            if decoder.form == "V1v":
                dphi = np.concatenate([state_tangent, np.zeros(1, dtype=np.float64)])
            output_tangent = x_matrix.T @ dphi + decoder_action_matrix.T @ phi
            local_value += float(weight) * float(output_tangent @ output_tangent)
    return float(workflow.evaluator.context.allreduce_scalar_sum(local_value))


@timed("goattm.train.compute_dense_vp_metric_matrix")
def compute_dense_vp_metric_matrix(
    prepared_state: ReducedObjectivePreparedState,
    workflow: ReducedObjectiveWorkflow,
) -> np.ndarray:
    dimension = dynamics_parameter_dimension(prepared_state.dynamics)
    basis = np.eye(dimension, dtype=np.float64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    for i in range(dimension):
        diagonal[i] = compute_vp_metric_quadratic(prepared_state, workflow, basis[:, i])
        matrix[i, i] = diagonal[i]
    for i in range(dimension):
        for j in range(i + 1, dimension):
            qij = compute_vp_metric_quadratic(prepared_state, workflow, basis[:, i] + basis[:, j])
            value = 0.5 * (qij - diagonal[i] - diagonal[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return 0.5 * (matrix + matrix.T)


def quotient_vertical_basis(dynamics: DynamicsLike) -> np.ndarray:
    if not isinstance(dynamics, (LinearDynamics, QuadraticDynamics, StabilizedQuadraticDynamics)):
        raise NotImplementedError("quotient vertical basis is implemented for linear/general/stabilized quadratic dynamics.")
    r = dynamics.dimension
    columns = []
    for i in range(r):
        for j in range(i + 1, r):
            k_matrix = np.zeros((r, r), dtype=np.float64)
            k_matrix[i, j] = 1.0
            k_matrix[j, i] = -1.0
            columns.append(_orthogonal_gauge_direction_vector(dynamics, k_matrix))
    if not columns:
        return np.zeros((dynamics_parameter_dimension(dynamics), 0), dtype=np.float64)
    return np.column_stack(columns)


def _orthogonal_gauge_direction_vector(dynamics: DynamicsLike, k_matrix: np.ndarray) -> np.ndarray:
    delta_a = dynamics.a @ k_matrix - k_matrix @ dynamics.a
    delta_b = None if dynamics.b is None else -k_matrix @ dynamics.b
    delta_c = -k_matrix @ dynamics.c
    if isinstance(dynamics, LinearDynamics):
        return _pack_explicit_direction(dynamics, delta_a, np.zeros_like(dynamics.h_matrix), delta_b, delta_c)
    delta_h = _quadratic_gauge_h_matrix(dynamics.h_matrix, k_matrix)
    return _pack_explicit_direction(dynamics, delta_a, delta_h, delta_b, delta_c)


def _quadratic_gauge_h_matrix(h_matrix: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    r = h_matrix.shape[0]

    def q_delta(x: np.ndarray) -> np.ndarray:
        return 2.0 * quadratic_eval(h_matrix, k_matrix @ x, x) - k_matrix @ quadratic_eval(h_matrix, x)

    out = np.zeros_like(h_matrix, dtype=np.float64)
    for i in range(r):
        e_i = np.zeros(r, dtype=np.float64)
        e_i[i] = 1.0
        q_i = q_delta(e_i)
        out[:, lower_triangular_index(i, i)] = q_i
        for j in range(i):
            e_j = np.zeros(r, dtype=np.float64)
            e_j[j] = 1.0
            out[:, lower_triangular_index(i, j)] = q_delta(e_i + e_j) - q_i - q_delta(e_j)
    return out


def _pack_explicit_direction(
    dynamics: DynamicsLike,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray | None,
    delta_c: np.ndarray,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        delta_s, delta_w = _stabilized_a_direction_to_parameters(dynamics, delta_a)
        blocks.extend([delta_s.reshape(-1), delta_w.reshape(-1)])
        blocks.append(compressed_h_to_mu_h(delta_h, dynamics.dimension))
    elif isinstance(dynamics, LinearDynamics):
        blocks.append(delta_a.reshape(-1))
    else:
        blocks.append(delta_a.reshape(-1))
        blocks.append(compressed_h_to_mu_h(delta_h, dynamics.dimension))
    if dynamics.b is not None:
        if delta_b is None:
            raise ValueError("delta_b must be provided when dynamics has an input matrix.")
        blocks.append(delta_b.reshape(-1))
    blocks.append(delta_c.reshape(-1))
    return np.concatenate(blocks, axis=0)


def _stabilized_a_direction_to_parameters(
    dynamics: StabilizedQuadraticDynamics,
    delta_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r = dynamics.dimension
    s_base = s_params_to_matrix(dynamics.s_params.astype(np.float64), r)
    s_size = dynamics.s_params.size
    w_size = dynamics.w_params.size
    basis = np.zeros((r * r, s_size + w_size), dtype=np.float64)
    for idx in range(s_size):
        ds = np.zeros(s_size, dtype=np.float64)
        ds[idx] = 1.0
        ds_matrix = s_params_to_matrix(ds, r)
        da = -(ds_matrix @ s_base.T + s_base @ ds_matrix.T)
        basis[:, idx] = da.reshape(-1)
    for idx in range(w_size):
        dw = np.zeros(w_size, dtype=np.float64)
        dw[idx] = 1.0
        da = w_params_to_matrix(dw, r)
        basis[:, s_size + idx] = da.reshape(-1)
    coeffs, *_ = np.linalg.lstsq(basis, np.asarray(delta_a, dtype=np.float64).reshape(-1), rcond=None)
    residual = basis @ coeffs - np.asarray(delta_a, dtype=np.float64).reshape(-1)
    rel = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(delta_a)))
    if rel > 1e-7:
        raise ValueError(
            "orthogonal gauge direction is not represented accurately in stabilized A parameters; "
            f"relative residual={rel:.3e}."
        )
    return coeffs[:s_size].copy(), coeffs[s_size:].copy()


def euclidean_horizontal_basis(dimension: int, vertical: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    if vertical.shape[1] == 0:
        return np.eye(dimension, dtype=np.float64)
    u, singular_values, _ = np.linalg.svd(vertical, full_matrices=True)
    rank = int(np.sum(singular_values > tolerance * max(vertical.shape) * max(1.0, singular_values[0])))
    return u[:, rank:].copy()


def solve_generalized_trust_region(
    hessian: np.ndarray,
    gradient: np.ndarray,
    metric: np.ndarray,
    radius: float,
) -> np.ndarray:
    metric = 0.5 * (metric + metric.T)
    hessian = 0.5 * (hessian + hessian.T)
    chol = np.linalg.cholesky(metric)
    inv_chol = np.linalg.solve(chol, np.eye(chol.shape[0], dtype=np.float64))
    transformed_hessian = inv_chol @ hessian @ inv_chol.T
    transformed_gradient = inv_chol @ gradient
    y = solve_euclidean_trust_region(transformed_hessian, transformed_gradient, radius)
    return inv_chol.T @ y


def solve_euclidean_trust_region(
    hessian: np.ndarray,
    gradient: np.ndarray,
    radius: float,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    g_hat = eigenvectors.T @ gradient
    lambda_min = float(eigenvalues[0])
    lower = max(0.0, -lambda_min + 1e-12)

    def step_for(lam: float) -> np.ndarray:
        return -(g_hat / (eigenvalues + lam))

    unconstrained = None
    if lambda_min > 0.0:
        candidate = step_for(0.0)
        if np.linalg.norm(candidate) <= radius:
            unconstrained = candidate
    if unconstrained is not None:
        return eigenvectors @ unconstrained

    low = lower
    high = max(1.0, low)
    while np.linalg.norm(step_for(high)) > radius:
        high *= 2.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if np.linalg.norm(step_for(mid)) > radius:
            low = mid
        else:
            high = mid
    return eigenvectors @ step_for(high)


def check_vp_metric_pullback(
    prepared_state: ReducedObjectivePreparedState,
    workflow: ReducedObjectiveWorkflow,
    direction: np.ndarray,
    epsilon: float = 1e-5,
) -> VpMetricCheckResult:
    base_dynamics = prepared_state.dynamics
    base_vector = dynamics_parameter_vector(base_dynamics)
    direction_vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    predicted = compute_vp_metric_quadratic(prepared_state, workflow, direction_vector)

    plus_dynamics = dynamics_from_parameter_vector(base_dynamics, base_vector + epsilon * direction_vector)
    plus_context = workflow.solve_decoder_best_response(plus_dynamics)
    base_decoder = prepared_state.result.decoder
    plus_decoder = plus_context.decoder
    local_value = 0.0
    for base_rollout_entry, plus_rollout_entry in zip(
        prepared_state.result.best_response_context.forward_cache.local_rollouts,
        plus_context.forward_cache.local_rollouts,
        strict=True,
    ):
        for base_state, plus_state, weight in zip(
            base_rollout_entry.observed_states,
            plus_rollout_entry.observed_states,
            base_rollout_entry.observation_weights,
            strict=True,
        ):
            diff = plus_decoder.decode(plus_state) - base_decoder.decode(base_state)
            local_value += float(weight) * float(diff @ diff)
    fd = float(workflow.evaluator.context.allreduce_scalar_sum(local_value)) / (epsilon * epsilon)
    rel = abs(fd - predicted) / max(1.0, abs(fd), abs(predicted))
    return VpMetricCheckResult(
        predicted_quadratic=float(predicted),
        finite_difference_quadratic=float(fd),
        relative_error=float(rel),
    )
