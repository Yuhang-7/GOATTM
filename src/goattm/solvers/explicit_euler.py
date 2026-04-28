from __future__ import annotations

from typing import Callable

import numpy as np

from goattm.core.parametrization import quadratic_features
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.runtime import timed
from goattm.solvers.implicit_midpoint import RolloutResult


def _quadratic_features_directional_derivative(state: np.ndarray, state_tangent: np.ndarray) -> np.ndarray:
    if state.shape != state_tangent.shape:
        raise ValueError(f"state_tangent must have shape {state.shape}, got {state_tangent.shape}")
    quad_state = quadratic_features(state)
    quad_tangent = np.zeros_like(quad_state, dtype=np.float64)
    index = 0
    for i in range(state.shape[0]):
        for j in range(i + 1):
            quad_tangent[index] = state_tangent[i] * state[j] + state[i] * state_tangent[j]
            index += 1
    return quad_tangent


def explicit_euler_step(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    dt: float,
    time: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    dynamics.validate_state(state, "state")
    return state + dt * dynamics.rhs_at_time(state, time, input_function=input_function)


def _validate_base_rollout(base_rollout: RolloutResult) -> None:
    if not base_rollout.success:
        raise RuntimeError("Base rollout must be successful before tangent or adjoint propagation.")
    if base_rollout.times.shape[0] != base_rollout.states.shape[0]:
        raise ValueError("Base rollout times and states are inconsistent.")
    if base_rollout.dt_history.shape[0] != base_rollout.states.shape[0] - 1:
        raise ValueError("Base rollout dt_history is inconsistent with states.")


@timed("goattm.solvers.rollout_explicit_euler")
def rollout_explicit_euler(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    t_final: float,
    max_dt: float,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> RolloutResult:
    dynamics.validate_state(u0, "u0")
    current_state = u0.copy()
    current_time = 0.0
    states = [current_state.copy()]
    times = [current_time]
    dt_history: list[float] = []

    while current_time < t_final - 1e-14:
        step_dt = min(float(max_dt), float(t_final - current_time))
        current_state = explicit_euler_step(
            dynamics=dynamics,
            state=current_state,
            dt=step_dt,
            time=current_time,
            input_function=input_function,
        )
        if not np.all(np.isfinite(current_state)):
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


@timed("goattm.solvers.rollout_explicit_euler_to_observation_times")
def rollout_explicit_euler_to_observation_times(
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

    current_state = u0.copy()
    current_time = 0.0
    states = [current_state.copy()]
    times = [current_time]
    dt_history: list[float] = []
    observation_indices = [0]

    for target_time in observation_times[1:]:
        while current_time < float(target_time) - 1e-14:
            step_dt = min(float(max_dt), float(target_time - current_time))
            current_state = explicit_euler_step(
                dynamics=dynamics,
                state=current_state,
                dt=step_dt,
                time=current_time,
                input_function=input_function,
            )
            if not np.all(np.isfinite(current_state)):
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


@timed("goattm.solvers.rollout_explicit_euler_tangent_from_base_rollout")
def rollout_explicit_euler_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
) -> np.ndarray:
    _validate_base_rollout(base_rollout)
    tangent_states = np.zeros_like(base_rollout.states, dtype=np.float64)
    current_tangent = np.zeros(base_rollout.states.shape[1], dtype=np.float64)
    for step_idx in range(base_rollout.accepted_steps):
        state = base_rollout.states[step_idx]
        time = float(base_rollout.times[step_idx])
        dt = float(base_rollout.dt_history[step_idx])
        forcing = np.zeros_like(current_tangent, dtype=np.float64)
        if parameter_action is not None:
            forcing = np.asarray(parameter_action(state, time), dtype=np.float64)
        current_tangent = current_tangent + dt * (dynamics.rhs_jacobian(state) @ current_tangent + forcing)
        tangent_states[step_idx + 1] = current_tangent.copy()
    return tangent_states


@timed("goattm.solvers.compute_explicit_euler_discrete_adjoint")
def compute_explicit_euler_discrete_adjoint(
    dynamics: QuadraticDynamics,
    states: np.ndarray,
    times: np.ndarray,
    dt_history: np.ndarray,
    state_loss_gradients: np.ndarray,
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
        state = states[step_idx]
        dt = float(dt_history[step_idx])
        jacobian = dynamics.rhs_jacobian(state)
        adjoints[step_idx] = np.asarray(state_loss_gradients[step_idx], dtype=np.float64) + (
            np.eye(dynamics.dimension, dtype=np.float64) + dt * jacobian
        ).T @ adjoints[step_idx + 1]
    return adjoints


@timed("goattm.solvers.accumulate_explicit_euler_parameter_gradients")
def accumulate_explicit_euler_parameter_gradients(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    adjoints: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    _validate_base_rollout(rollout)
    if adjoints.shape != rollout.states.shape:
        raise ValueError(f"adjoints must have shape {rollout.states.shape}, got {adjoints.shape}")

    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    h_grad = np.zeros_like(dynamics.h_matrix)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)
    for step_idx in range(rollout.accepted_steps):
        state = rollout.states[step_idx]
        time = float(rollout.times[step_idx])
        dt = float(rollout.dt_history[step_idx])
        adjoint_next = adjoints[step_idx + 1]
        a_grad += dt * np.outer(adjoint_next, state)
        h_grad += dt * np.outer(adjoint_next, quadratic_features(state))
        c_grad += dt * adjoint_next
        if b_grad is not None and input_function is not None:
            b_grad += dt * np.outer(adjoint_next, np.asarray(input_function(time), dtype=np.float64))
    return a_grad, h_grad, b_grad, c_grad


@timed("goattm.solvers.compute_explicit_euler_incremental_discrete_adjoint")
def compute_explicit_euler_incremental_discrete_adjoint(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    base_adjoints: np.ndarray,
    state_loss_gradient_direction: np.ndarray,
    jacobian_direction: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
) -> np.ndarray:
    _validate_base_rollout(rollout)
    if tangent_states.shape != rollout.states.shape:
        raise ValueError(f"tangent_states must have shape {rollout.states.shape}, got {tangent_states.shape}")
    if base_adjoints.shape != rollout.states.shape:
        raise ValueError(f"base_adjoints must have shape {rollout.states.shape}, got {base_adjoints.shape}")
    if state_loss_gradient_direction.shape != rollout.states.shape:
        raise ValueError(
            f"state_loss_gradient_direction must have shape {rollout.states.shape}, "
            f"got {state_loss_gradient_direction.shape}"
        )

    adjoint_tangents = np.zeros_like(base_adjoints, dtype=np.float64)
    adjoint_tangents[-1] = np.asarray(state_loss_gradient_direction[-1], dtype=np.float64)
    identity = np.eye(dynamics.dimension, dtype=np.float64)
    for step_idx in range(rollout.accepted_steps - 1, -1, -1):
        state = rollout.states[step_idx]
        state_tangent = tangent_states[step_idx]
        time = float(rollout.times[step_idx])
        dt = float(rollout.dt_history[step_idx])
        jacobian = dynamics.rhs_jacobian(state)
        delta_jacobian = np.asarray(jacobian_direction(state, state_tangent, time), dtype=np.float64)
        adjoint_tangents[step_idx] = (
            np.asarray(state_loss_gradient_direction[step_idx], dtype=np.float64)
            + (identity + dt * jacobian).T @ adjoint_tangents[step_idx + 1]
            + dt * delta_jacobian.T @ base_adjoints[step_idx + 1]
        )
    return adjoint_tangents


@timed("goattm.solvers.accumulate_explicit_euler_parameter_hessian_action_terms")
def accumulate_explicit_euler_parameter_hessian_action_terms(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    adjoints: np.ndarray,
    adjoint_tangents: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    _validate_base_rollout(rollout)
    if tangent_states.shape != rollout.states.shape:
        raise ValueError(f"tangent_states must have shape {rollout.states.shape}, got {tangent_states.shape}")
    if adjoints.shape != rollout.states.shape:
        raise ValueError(f"adjoints must have shape {rollout.states.shape}, got {adjoints.shape}")
    if adjoint_tangents.shape != rollout.states.shape:
        raise ValueError(f"adjoint_tangents must have shape {rollout.states.shape}, got {adjoint_tangents.shape}")

    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    delta_a_grad = np.zeros_like(a_grad)
    h_grad = np.zeros_like(dynamics.h_matrix)
    delta_h_grad = np.zeros_like(h_grad)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    delta_c_grad = np.zeros_like(c_grad)
    b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)
    delta_b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)

    for step_idx in range(rollout.accepted_steps):
        state = rollout.states[step_idx]
        state_tangent = tangent_states[step_idx]
        dt = float(rollout.dt_history[step_idx])
        time = float(rollout.times[step_idx])
        adjoint_next = adjoints[step_idx + 1]
        adjoint_tangent_next = adjoint_tangents[step_idx + 1]
        quad_state = quadratic_features(state)
        quad_tangent = _quadratic_features_directional_derivative(state, state_tangent)
        a_grad += dt * np.outer(adjoint_next, state)
        delta_a_grad += dt * (np.outer(adjoint_tangent_next, state) + np.outer(adjoint_next, state_tangent))
        h_grad += dt * np.outer(adjoint_next, quad_state)
        delta_h_grad += dt * (np.outer(adjoint_tangent_next, quad_state) + np.outer(adjoint_next, quad_tangent))
        c_grad += dt * adjoint_next
        delta_c_grad += dt * adjoint_tangent_next
        if b_grad is not None and input_function is not None:
            stage_input = np.asarray(input_function(time), dtype=np.float64)
            b_grad += dt * np.outer(adjoint_next, stage_input)
            delta_b_grad += dt * np.outer(adjoint_tangent_next, stage_input)
    return a_grad, delta_a_grad, h_grad, delta_h_grad, b_grad, delta_b_grad, c_grad, delta_c_grad
