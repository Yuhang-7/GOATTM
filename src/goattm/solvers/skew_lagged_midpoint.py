from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from goattm.core.parametrization import quadratic_features
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.runtime import timed
from goattm.solvers.implicit_midpoint import RolloutResult


DynamicsFactory = Callable[[float], QuadraticDynamics]


@dataclass(frozen=True)
class SkewLaggedMidpointStepInfo:
    success: bool
    iterations: int
    residual_norm: float
    u_next: np.ndarray


def _input_at(input_function: Callable[[float], np.ndarray] | None, time: float) -> np.ndarray | None:
    return None if input_function is None else np.asarray(input_function(time), dtype=np.float64)


def frozen_skew_quadratic_matrix(dynamics: QuadraticDynamics, lag_state: np.ndarray) -> np.ndarray:
    """Return N(lag_state), the skew linearization used for H(m, m) ~= N(lag_state) m."""
    dynamics.validate_state(lag_state, "lag_state")
    action = np.asarray(dynamics.quadratic_bilinear_action_matrix(lag_state), dtype=np.float64)
    if action.shape == (dynamics.dimension, 0):
        return np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    if action.shape[0] == dynamics.dimension and action.shape[1] != dynamics.dimension and np.all(action == 0.0):
        return np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    if action.shape != (dynamics.dimension, dynamics.dimension):
        raise ValueError(
            "quadratic_bilinear_action_matrix must return a square state action matrix for "
            f"skew_lagged_midpoint, got {action.shape}"
        )
    return (2.0 / 3.0) * (action - action.T)


def _accumulate_bilinear_action_matrix_pullback(
    h_matrix: np.ndarray,
    lag_state: np.ndarray,
    action_bar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if h_matrix.ndim != 2:
        raise ValueError(f"h_matrix must be rank-2, got {h_matrix.shape}")
    r = lag_state.shape[0]
    if h_matrix.shape[0] != r:
        raise ValueError(f"h_matrix first dimension must be {r}, got {h_matrix.shape}")
    if action_bar.shape != (r, r):
        raise ValueError(f"action_bar must have shape {(r, r)}, got {action_bar.shape}")

    h_bar = np.zeros_like(h_matrix, dtype=np.float64)
    lag_bar = np.zeros(r, dtype=np.float64)
    for row in range(r):
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h_matrix[row, idx]
                if i == j:
                    h_bar[row, idx] += action_bar[row, i] * lag_state[i]
                    lag_bar[i] += action_bar[row, i] * coeff
                else:
                    h_bar[row, idx] += 0.5 * (
                        action_bar[row, i] * lag_state[j] + action_bar[row, j] * lag_state[i]
                    )
                    lag_bar[j] += 0.5 * action_bar[row, i] * coeff
                    lag_bar[i] += 0.5 * action_bar[row, j] * coeff
                idx += 1
    return h_bar, lag_bar


def _outer_input_gradient(
    dynamics: QuadraticDynamics,
    vector: np.ndarray,
    input_vector: np.ndarray | None,
) -> np.ndarray | None:
    if getattr(dynamics, "b", None) is None:
        return None
    if input_vector is None:
        return np.zeros_like(dynamics.b, dtype=np.float64)
    return np.outer(vector, input_vector)


def _parameter_action_at(
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None,
    state: np.ndarray,
    time: float,
) -> np.ndarray:
    if parameter_action is None:
        return np.zeros_like(state, dtype=np.float64)
    return np.asarray(parameter_action(state, time), dtype=np.float64)


def _parameter_action_jacobian(
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None,
    state: np.ndarray,
    time: float,
    eps: float = 1.0e-7,
) -> np.ndarray:
    r = state.shape[0]
    if parameter_action is None:
        return np.zeros((r, r), dtype=np.float64)
    jacobian = np.zeros((r, r), dtype=np.float64)
    for column in range(r):
        basis = np.zeros(r, dtype=np.float64)
        basis[column] = eps
        plus = np.asarray(parameter_action(state + basis, time), dtype=np.float64)
        minus = np.asarray(parameter_action(state - basis, time), dtype=np.float64)
        jacobian[:, column] = (plus - minus) / (2.0 * eps)
    return jacobian


def _skew_lagged_midpoint_step_tangent(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    u_prev_tangent: np.ndarray,
    dt: float,
    t_prev: float,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    dynamics.validate_state(u_prev, "u_prev")
    dynamics.validate_state(u_prev_tangent, "u_prev_tangent")
    midpoint_time = float(t_prev + 0.5 * dt)
    alpha = 2.0 / float(dt)
    beta = 0.5 * float(dt)
    gamma = 2.0 / 3.0
    identity = np.eye(dynamics.dimension, dtype=np.float64)

    rhs_prev = dynamics.rhs_at_time(u_prev, t_prev, input_function=input_function)
    lag_state = u_prev + beta * rhs_prev
    skew_matrix = frozen_skew_quadratic_matrix(dynamics, lag_state)
    rhs = alpha * u_prev + dynamics.forcing(_input_at(input_function, midpoint_time))
    lhs = alpha * identity - dynamics.a - skew_matrix
    midpoint = np.linalg.solve(lhs, rhs)

    param_jacobian_zero = _parameter_action_jacobian(
        parameter_action,
        np.zeros(dynamics.dimension, dtype=np.float64),
        t_prev,
    )
    param_jacobian_lag = _parameter_action_jacobian(parameter_action, lag_state, t_prev)
    delta_a = param_jacobian_zero
    delta_m_lag = 0.5 * (param_jacobian_lag - delta_a)

    rhs_prev_tangent = (
        dynamics.rhs_jacobian(u_prev) @ u_prev_tangent
        + _parameter_action_at(parameter_action, u_prev, t_prev)
    )
    lag_state_tangent = u_prev_tangent + beta * rhs_prev_tangent
    action_tangent = np.asarray(dynamics.quadratic_bilinear_action_matrix(lag_state_tangent), dtype=np.float64)
    if action_tangent.shape != (dynamics.dimension, dynamics.dimension):
        if np.all(action_tangent == 0.0):
            action_tangent = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
        else:
            raise ValueError(
                "quadratic_bilinear_action_matrix must return a square state action matrix for "
                f"skew_lagged_midpoint tangent, got {action_tangent.shape}"
            )
    action_tangent = action_tangent + delta_m_lag
    delta_skew = gamma * (action_tangent - action_tangent.T)
    delta_lhs = -delta_a - delta_skew
    delta_rhs = alpha * u_prev_tangent + _parameter_action_at(
        parameter_action,
        np.zeros(dynamics.dimension, dtype=np.float64),
        midpoint_time,
    )
    midpoint_tangent = np.linalg.solve(lhs, delta_rhs - delta_lhs @ midpoint)
    return 2.0 * midpoint_tangent - u_prev_tangent


def skew_lagged_midpoint_step_reverse(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    dt: float,
    t_prev: float,
    adjoint_next: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Exact reverse-mode derivative of one skew-lagged midpoint step."""
    dynamics.validate_state(u_prev, "u_prev")
    dynamics.validate_state(adjoint_next, "adjoint_next")
    midpoint_time = float(t_prev + 0.5 * dt)
    input_prev = _input_at(input_function, t_prev)
    input_mid = _input_at(input_function, midpoint_time)

    alpha = 2.0 / float(dt)
    beta = 0.5 * float(dt)
    gamma = 2.0 / 3.0
    identity = np.eye(dynamics.dimension, dtype=np.float64)

    rhs_prev = dynamics.rhs(u_prev, p=input_prev)
    lag_state = u_prev + beta * rhs_prev
    skew_matrix = frozen_skew_quadratic_matrix(dynamics, lag_state)
    forcing_mid = dynamics.forcing(input_mid)
    rhs = alpha * u_prev + forcing_mid
    lhs = alpha * identity - dynamics.a - skew_matrix
    midpoint = np.linalg.solve(lhs, rhs)

    u_bar = -np.asarray(adjoint_next, dtype=np.float64).copy()
    midpoint_bar = 2.0 * np.asarray(adjoint_next, dtype=np.float64)

    solve_bar = np.linalg.solve(lhs.T, midpoint_bar)
    rhs_bar = solve_bar
    lhs_bar = -np.outer(solve_bar, midpoint)

    a_bar = -lhs_bar
    n_bar = -lhs_bar
    action_bar = gamma * (n_bar - n_bar.T)
    h_bar, lag_bar = _accumulate_bilinear_action_matrix_pullback(dynamics.h_matrix, lag_state, action_bar)

    u_bar += alpha * rhs_bar
    c_bar = rhs_bar.copy()
    b_bar = _outer_input_gradient(dynamics, rhs_bar, input_mid)

    u_bar += lag_bar
    rhs_prev_bar = beta * lag_bar
    a_bar += np.outer(rhs_prev_bar, u_prev)
    h_bar += np.outer(rhs_prev_bar, quadratic_features(u_prev))
    c_bar += rhs_prev_bar
    prev_b_bar = _outer_input_gradient(dynamics, rhs_prev_bar, input_prev)
    if b_bar is None:
        b_bar = prev_b_bar
    elif prev_b_bar is not None:
        b_bar += prev_b_bar
    u_bar += dynamics.rhs_jacobian(u_prev).T @ rhs_prev_bar

    return u_bar, a_bar, h_bar, b_bar, c_bar


def solve_skew_lagged_midpoint_step(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    dt: float,
    t_prev: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> SkewLaggedMidpointStepInfo:
    dynamics.validate_state(u_prev, "u_prev")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    midpoint_time = float(t_prev + 0.5 * dt)
    forcing_mid = dynamics.forcing(_input_at(input_function, midpoint_time))
    alpha = 2.0 / float(dt)
    identity = np.eye(dynamics.dimension, dtype=np.float64)
    rhs = alpha * u_prev + forcing_mid

    lag_state = u_prev + 0.5 * dt * dynamics.rhs_at_time(u_prev, t_prev, input_function=input_function)
    if not np.all(np.isfinite(lag_state)):
        return SkewLaggedMidpointStepInfo(False, 0, np.inf, u_prev.copy())
    skew_matrix = frozen_skew_quadratic_matrix(dynamics, lag_state)
    lhs = alpha * identity - dynamics.a - skew_matrix
    try:
        midpoint = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return SkewLaggedMidpointStepInfo(False, 0, np.inf, u_prev.copy())
    if not np.all(np.isfinite(midpoint)):
        return SkewLaggedMidpointStepInfo(False, 1, np.inf, u_prev.copy())
    residual = lhs @ midpoint - rhs
    residual_norm = float(np.linalg.norm(residual))

    u_next = 2.0 * midpoint - u_prev
    if not np.all(np.isfinite(u_next)):
        return SkewLaggedMidpointStepInfo(False, 1, np.inf, u_prev.copy())
    return SkewLaggedMidpointStepInfo(True, 1, residual_norm, u_next)


@timed("goattm.solvers.rollout_skew_lagged_midpoint")
def rollout_skew_lagged_midpoint(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    t_final: float,
    max_dt: float,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> RolloutResult:
    dynamics.validate_state(u0, "u0")
    if max_dt <= 0.0:
        raise ValueError(f"max_dt must be positive, got {max_dt}")
    current_state = u0.copy()
    current_time = 0.0
    states = [current_state.copy()]
    times = [current_time]
    dt_history: list[float] = []

    while current_time < t_final - 1e-14:
        step_dt = min(float(max_dt), float(t_final - current_time))
        info = solve_skew_lagged_midpoint_step(
            dynamics=dynamics,
            u_prev=current_state,
            dt=step_dt,
            t_prev=current_time,
            input_function=input_function,
        )
        if not info.success:
            return RolloutResult(
                success=False,
                accepted_steps=len(dt_history),
                dt_reductions=0,
                newton_failures=0,
                final_time=current_time,
                dt_history=np.asarray(dt_history, dtype=np.float64),
                times=np.asarray(times, dtype=np.float64),
                states=np.stack(states, axis=0),
            )
        current_state = info.u_next
        current_time += step_dt
        dt_history.append(step_dt)
        states.append(current_state.copy())
        times.append(current_time)

    return RolloutResult(
        success=True,
        accepted_steps=len(dt_history),
        dt_reductions=0,
        newton_failures=0,
        final_time=current_time,
        dt_history=np.asarray(dt_history, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        states=np.stack(states, axis=0),
    )


@timed("goattm.solvers.rollout_skew_lagged_midpoint_to_observation_times")
def rollout_skew_lagged_midpoint_to_observation_times(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    observation_times: np.ndarray,
    max_dt: float,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[RolloutResult, np.ndarray]:
    dynamics.validate_state(u0, "u0")
    if observation_times.ndim != 1 or observation_times.shape[0] < 1:
        raise ValueError("observation_times must be one-dimensional with at least one entry.")
    if not np.all(np.diff(observation_times) > 0.0):
        raise ValueError("observation_times must be strictly increasing.")
    if abs(float(observation_times[0])) > 1e-14:
        raise ValueError("observation_times must start at 0.")
    if max_dt <= 0.0:
        raise ValueError(f"max_dt must be positive, got {max_dt}")

    current_state = u0.copy()
    current_time = 0.0
    states = [current_state.copy()]
    times = [current_time]
    dt_history: list[float] = []
    observation_indices = [0]

    for target_time in observation_times[1:]:
        while current_time < float(target_time) - 1e-14:
            step_dt = min(float(max_dt), float(target_time - current_time))
            info = solve_skew_lagged_midpoint_step(
                dynamics=dynamics,
                u_prev=current_state,
                dt=step_dt,
                t_prev=current_time,
                input_function=input_function,
            )
            if not info.success:
                rollout = RolloutResult(
                    success=False,
                    accepted_steps=len(dt_history),
                    dt_reductions=0,
                    newton_failures=0,
                    final_time=current_time,
                    dt_history=np.asarray(dt_history, dtype=np.float64),
                    times=np.asarray(times, dtype=np.float64),
                    states=np.stack(states, axis=0),
                )
                return rollout, np.asarray(observation_indices, dtype=int)
            current_state = info.u_next
            current_time += step_dt
            dt_history.append(step_dt)
            states.append(current_state.copy())
            times.append(current_time)
        observation_indices.append(len(states) - 1)

    rollout = RolloutResult(
        success=True,
        accepted_steps=len(dt_history),
        dt_reductions=0,
        newton_failures=0,
        final_time=current_time,
        dt_history=np.asarray(dt_history, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        states=np.stack(states, axis=0),
    )
    return rollout, np.asarray(observation_indices, dtype=int)


@timed("goattm.solvers.rollout_skew_lagged_midpoint_tangent_from_base_rollout")
def rollout_skew_lagged_midpoint_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    if not base_rollout.success:
        raise RuntimeError("Base rollout must be successful before solving the tangent system.")
    if base_rollout.states.ndim != 2 or base_rollout.times.ndim != 1:
        raise ValueError("base_rollout must contain trajectory arrays.")
    if base_rollout.states.shape[0] != base_rollout.times.shape[0]:
        raise ValueError("base_rollout states/times shape mismatch.")
    if base_rollout.dt_history.shape[0] != base_rollout.states.shape[0] - 1:
        raise ValueError("base_rollout dt_history length is inconsistent with states.")

    tangent_states = np.zeros_like(base_rollout.states, dtype=np.float64)
    for step_idx in range(base_rollout.accepted_steps):
        tangent_states[step_idx + 1] = _skew_lagged_midpoint_step_tangent(
            dynamics=dynamics,
            u_prev=base_rollout.states[step_idx],
            u_prev_tangent=tangent_states[step_idx],
            dt=float(base_rollout.dt_history[step_idx]),
            t_prev=float(base_rollout.times[step_idx]),
            parameter_action=parameter_action,
            input_function=input_function,
        )
    return tangent_states


@timed("goattm.solvers.compute_skew_lagged_midpoint_discrete_adjoint")
def compute_skew_lagged_midpoint_discrete_adjoint(
    dynamics: QuadraticDynamics,
    states: np.ndarray,
    times: np.ndarray,
    dt_history: np.ndarray,
    state_loss_gradients: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError(f"states must have shape (N, r), got {states.shape}")
    if times.shape[0] != states.shape[0]:
        raise ValueError("times and states are inconsistent.")
    if dt_history.shape[0] != states.shape[0] - 1:
        raise ValueError("dt_history and states are inconsistent.")
    if state_loss_gradients.shape != states.shape:
        raise ValueError(f"state_loss_gradients must have shape {states.shape}, got {state_loss_gradients.shape}")

    adjoints = np.zeros_like(states, dtype=np.float64)
    adjoints[-1] = np.asarray(state_loss_gradients[-1], dtype=np.float64)
    for step_idx in range(dt_history.shape[0] - 1, -1, -1):
        state_bar, _, _, _, _ = skew_lagged_midpoint_step_reverse(
            dynamics=dynamics,
            u_prev=states[step_idx],
            dt=float(dt_history[step_idx]),
            t_prev=float(times[step_idx]),
            adjoint_next=adjoints[step_idx + 1],
            input_function=input_function,
        )
        adjoints[step_idx] = np.asarray(state_loss_gradients[step_idx], dtype=np.float64) + state_bar
    return adjoints


@timed("goattm.solvers.compute_skew_lagged_midpoint_incremental_discrete_adjoint")
def compute_skew_lagged_midpoint_incremental_discrete_adjoint(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    base_adjoints: np.ndarray,
    state_loss_gradient_direction: np.ndarray,
    make_perturbed_dynamics: DynamicsFactory,
    input_function: Callable[[float], np.ndarray] | None = None,
    finite_difference_epsilon: float = 1.0e-6,
) -> np.ndarray:
    if not rollout.success:
        raise RuntimeError("Base rollout must be successful before solving the incremental adjoint.")
    if tangent_states.shape != rollout.states.shape:
        raise ValueError(f"tangent_states must have shape {rollout.states.shape}, got {tangent_states.shape}")
    if base_adjoints.shape != rollout.states.shape:
        raise ValueError(f"base_adjoints must have shape {rollout.states.shape}, got {base_adjoints.shape}")
    if state_loss_gradient_direction.shape != rollout.states.shape:
        raise ValueError(
            f"state_loss_gradient_direction must have shape {rollout.states.shape}, "
            f"got {state_loss_gradient_direction.shape}"
        )
    if finite_difference_epsilon <= 0.0:
        raise ValueError(f"finite_difference_epsilon must be positive, got {finite_difference_epsilon}")

    adjoint_tangents = np.zeros_like(base_adjoints, dtype=np.float64)
    adjoint_tangents[-1] = np.asarray(state_loss_gradient_direction[-1], dtype=np.float64)
    eps = float(finite_difference_epsilon)
    dynamics_plus = make_perturbed_dynamics(eps)
    dynamics_minus = make_perturbed_dynamics(-eps)

    for step_idx in range(rollout.accepted_steps - 1, -1, -1):
        state = rollout.states[step_idx]
        state_tangent = tangent_states[step_idx]
        adjoint_next = base_adjoints[step_idx + 1]
        adjoint_next_tangent = adjoint_tangents[step_idx + 1]
        dt = float(rollout.dt_history[step_idx])
        time = float(rollout.times[step_idx])
        state_bar_plus, _, _, _, _ = skew_lagged_midpoint_step_reverse(
            dynamics=dynamics_plus,
            u_prev=state + eps * state_tangent,
            dt=dt,
            t_prev=time,
            adjoint_next=adjoint_next + eps * adjoint_next_tangent,
            input_function=input_function,
        )
        state_bar_minus, _, _, _, _ = skew_lagged_midpoint_step_reverse(
            dynamics=dynamics_minus,
            u_prev=state - eps * state_tangent,
            dt=dt,
            t_prev=time,
            adjoint_next=adjoint_next - eps * adjoint_next_tangent,
            input_function=input_function,
        )
        adjoint_tangents[step_idx] = np.asarray(state_loss_gradient_direction[step_idx], dtype=np.float64)
        adjoint_tangents[step_idx] += (state_bar_plus - state_bar_minus) / (2.0 * eps)
    return adjoint_tangents


@timed("goattm.solvers.accumulate_skew_lagged_midpoint_parameter_gradients")
def accumulate_skew_lagged_midpoint_parameter_gradients(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    adjoints: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    if not rollout.success:
        raise RuntimeError("Base rollout must be successful before accumulating parameter gradients.")
    if adjoints.shape != rollout.states.shape:
        raise ValueError(f"adjoints must have shape {rollout.states.shape}, got {adjoints.shape}")

    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    h_grad = np.zeros_like(dynamics.h_matrix, dtype=np.float64)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    b_grad = None if getattr(dynamics, "b", None) is None else np.zeros_like(dynamics.b, dtype=np.float64)
    for step_idx in range(rollout.accepted_steps):
        _, da, dh, db, dc = skew_lagged_midpoint_step_reverse(
            dynamics=dynamics,
            u_prev=rollout.states[step_idx],
            dt=float(rollout.dt_history[step_idx]),
            t_prev=float(rollout.times[step_idx]),
            adjoint_next=adjoints[step_idx + 1],
            input_function=input_function,
        )
        a_grad += da
        h_grad += dh
        c_grad += dc
        if b_grad is not None and db is not None:
            b_grad += db
    return a_grad, h_grad, b_grad, c_grad


@timed("goattm.solvers.accumulate_skew_lagged_midpoint_parameter_hessian_action_terms")
def accumulate_skew_lagged_midpoint_parameter_hessian_action_terms(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    adjoints: np.ndarray,
    adjoint_tangents: np.ndarray,
    make_perturbed_dynamics: DynamicsFactory,
    input_function: Callable[[float], np.ndarray] | None = None,
    finite_difference_epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    if not rollout.success:
        raise RuntimeError("Base rollout must be successful before accumulating Hessian-action terms.")
    if tangent_states.shape != rollout.states.shape:
        raise ValueError(f"tangent_states must have shape {rollout.states.shape}, got {tangent_states.shape}")
    if adjoints.shape != rollout.states.shape:
        raise ValueError(f"adjoints must have shape {rollout.states.shape}, got {adjoints.shape}")
    if adjoint_tangents.shape != rollout.states.shape:
        raise ValueError(f"adjoint_tangents must have shape {rollout.states.shape}, got {adjoint_tangents.shape}")
    if finite_difference_epsilon <= 0.0:
        raise ValueError(f"finite_difference_epsilon must be positive, got {finite_difference_epsilon}")

    eps = float(finite_difference_epsilon)
    dynamics_plus = make_perturbed_dynamics(eps)
    dynamics_minus = make_perturbed_dynamics(-eps)
    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    delta_a_grad = np.zeros_like(a_grad)
    h_grad = np.zeros_like(dynamics.h_matrix, dtype=np.float64)
    delta_h_grad = np.zeros_like(h_grad)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    delta_c_grad = np.zeros_like(c_grad)
    b_grad = None if getattr(dynamics, "b", None) is None else np.zeros_like(dynamics.b, dtype=np.float64)
    delta_b_grad = None if getattr(dynamics, "b", None) is None else np.zeros_like(dynamics.b, dtype=np.float64)

    for step_idx in range(rollout.accepted_steps):
        state = rollout.states[step_idx]
        state_tangent = tangent_states[step_idx]
        adjoint_next = adjoints[step_idx + 1]
        adjoint_next_tangent = adjoint_tangents[step_idx + 1]
        dt = float(rollout.dt_history[step_idx])
        time = float(rollout.times[step_idx])
        _, da, dh, db, dc = skew_lagged_midpoint_step_reverse(
            dynamics=dynamics,
            u_prev=state,
            dt=dt,
            t_prev=time,
            adjoint_next=adjoint_next,
            input_function=input_function,
        )
        _, da_plus, dh_plus, db_plus, dc_plus = skew_lagged_midpoint_step_reverse(
            dynamics=dynamics_plus,
            u_prev=state + eps * state_tangent,
            dt=dt,
            t_prev=time,
            adjoint_next=adjoint_next + eps * adjoint_next_tangent,
            input_function=input_function,
        )
        _, da_minus, dh_minus, db_minus, dc_minus = skew_lagged_midpoint_step_reverse(
            dynamics=dynamics_minus,
            u_prev=state - eps * state_tangent,
            dt=dt,
            t_prev=time,
            adjoint_next=adjoint_next - eps * adjoint_next_tangent,
            input_function=input_function,
        )
        a_grad += da
        h_grad += dh
        c_grad += dc
        delta_a_grad += (da_plus - da_minus) / (2.0 * eps)
        delta_h_grad += (dh_plus - dh_minus) / (2.0 * eps)
        delta_c_grad += (dc_plus - dc_minus) / (2.0 * eps)
        if b_grad is not None and db is not None and db_plus is not None and db_minus is not None:
            b_grad += db
            delta_b_grad += (db_plus - db_minus) / (2.0 * eps)
    return a_grad, delta_a_grad, h_grad, delta_h_grad, b_grad, delta_b_grad, c_grad, delta_c_grad
