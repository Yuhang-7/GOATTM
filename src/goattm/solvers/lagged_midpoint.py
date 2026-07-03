from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from goattm.core.parametrization import quadratic_features
from goattm.core.quadratic import quadratic_bilinear_action_matrix
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.runtime import timed
from goattm.solvers.implicit_midpoint import RolloutResult
from goattm.solvers.rk4 import _rk4_reverse_step, rk4_step_with_stages


@dataclass(frozen=True)
class LaggedMidpointStepCache:
    previous_state: np.ndarray
    next_state: np.ndarray
    predictor_state: np.ndarray
    linear_operator: np.ndarray
    system_matrix: np.ndarray
    rhs: np.ndarray
    time: float
    dt: float


def _input_at(input_function: Callable[[float], np.ndarray] | None, time: float) -> np.ndarray | None:
    return None if input_function is None else np.asarray(input_function(time), dtype=np.float64)


def _lagged_linear_operator(dynamics: QuadraticDynamics, predictor_state: np.ndarray) -> np.ndarray:
    return dynamics.a + quadratic_bilinear_action_matrix(dynamics.h_matrix, predictor_state)


def _accumulate_bilinear_action_matrix_reverse(
    dynamics: QuadraticDynamics,
    predictor_state: np.ndarray,
    operator_bar: np.ndarray,
    h_grad: np.ndarray,
) -> np.ndarray:
    z_bar = np.zeros_like(predictor_state, dtype=np.float64)
    h_matrix = dynamics.h_matrix
    r = predictor_state.shape[0]
    for row in range(r):
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h_matrix[row, idx]
                if i == j:
                    h_grad[row, idx] += operator_bar[row, i] * predictor_state[i]
                    z_bar[i] += operator_bar[row, i] * coeff
                else:
                    h_grad[row, idx] += 0.5 * (
                        operator_bar[row, i] * predictor_state[j]
                        + operator_bar[row, j] * predictor_state[i]
                    )
                    z_bar[j] += 0.5 * operator_bar[row, i] * coeff
                    z_bar[i] += 0.5 * operator_bar[row, j] * coeff
                idx += 1
    return z_bar


def lagged_midpoint_step_with_cache(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    dt: float,
    time: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, LaggedMidpointStepCache]:
    dynamics.validate_state(state, "state")
    half_dt = 0.5 * dt
    predictor_state, _ = rk4_step_with_stages(
        dynamics=dynamics,
        state=state,
        dt=half_dt,
        time=time,
        input_function=input_function,
    )
    mid_time = time + half_dt
    forcing = dynamics.forcing(_input_at(input_function, mid_time))
    linear_operator = _lagged_linear_operator(dynamics, predictor_state)
    identity = np.eye(dynamics.dimension, dtype=np.float64)
    system_matrix = identity - half_dt * linear_operator
    rhs = (identity + half_dt * linear_operator) @ state + dt * forcing
    next_state = np.linalg.solve(system_matrix, rhs)
    return next_state, LaggedMidpointStepCache(
        previous_state=state.copy(),
        next_state=next_state.copy(),
        predictor_state=predictor_state.copy(),
        linear_operator=linear_operator.copy(),
        system_matrix=system_matrix.copy(),
        rhs=rhs.copy(),
        time=float(time),
        dt=float(dt),
    )


def lagged_midpoint_step(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    dt: float,
    time: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    return lagged_midpoint_step_with_cache(
        dynamics=dynamics,
        state=state,
        dt=dt,
        time=time,
        input_function=input_function,
    )[0]


def _validate_observation_times(observation_times: np.ndarray) -> None:
    if observation_times.ndim != 1 or observation_times.shape[0] < 1:
        raise ValueError("observation_times must be one-dimensional with at least one entry.")
    if not np.all(np.diff(observation_times) > 0.0):
        raise ValueError("observation_times must be strictly increasing.")
    if abs(float(observation_times[0])) > 1e-14:
        raise ValueError("observation_times must start at 0.")


@timed("goattm.solvers.rollout_lagged_midpoint")
def rollout_lagged_midpoint(
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
        current_state = lagged_midpoint_step(
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


@timed("goattm.solvers.rollout_lagged_midpoint_to_observation_times")
def rollout_lagged_midpoint_to_observation_times(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    observation_times: np.ndarray,
    max_dt: float,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[RolloutResult, np.ndarray]:
    dynamics.validate_state(u0, "u0")
    _validate_observation_times(observation_times)
    current_state = u0.copy()
    current_time = 0.0
    states = [current_state.copy()]
    times = [current_time]
    dt_history: list[float] = []
    observation_indices = [0]

    for target_time in observation_times[1:]:
        while current_time < float(target_time) - 1e-14:
            step_dt = min(float(max_dt), float(target_time - current_time))
            current_state = lagged_midpoint_step(
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


def _lagged_midpoint_reverse_step(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    dt: float,
    time: float,
    adjoint_next: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    accumulate_parameters: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    next_state, cache = lagged_midpoint_step_with_cache(
        dynamics=dynamics,
        state=state,
        dt=dt,
        time=time,
        input_function=input_function,
    )
    if not np.allclose(next_state, cache.next_state):
        raise RuntimeError("Internal lagged midpoint cache inconsistency.")

    half_dt = 0.5 * dt
    alpha = np.linalg.solve(cache.system_matrix.T, adjoint_next)
    operator_bar = half_dt * (np.outer(alpha, cache.previous_state) + np.outer(alpha, cache.next_state))
    state_bar = alpha + half_dt * (cache.linear_operator.T @ alpha)
    forcing_bar = dt * alpha

    a_grad = operator_bar.copy()
    h_grad = np.zeros_like(dynamics.h_matrix)
    c_grad = forcing_bar.copy()
    b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)
    mid_time = time + half_dt
    if b_grad is not None and input_function is not None:
        b_grad += np.outer(forcing_bar, np.asarray(input_function(mid_time), dtype=np.float64))

    predictor_bar = _accumulate_bilinear_action_matrix_reverse(
        dynamics=dynamics,
        predictor_state=cache.predictor_state,
        operator_bar=operator_bar,
        h_grad=h_grad,
    )

    _, predictor_stages = rk4_step_with_stages(
        dynamics=dynamics,
        state=state,
        dt=half_dt,
        time=time,
        input_function=input_function,
    )
    predictor_state_bar, pred_a_grad, pred_h_grad, pred_b_grad, pred_c_grad = _rk4_reverse_step(
        dynamics=dynamics,
        stages=predictor_stages,
        dt=half_dt,
        adjoint_next=predictor_bar,
        input_function=input_function,
        accumulate_parameters=accumulate_parameters,
    )
    state_bar += predictor_state_bar
    if accumulate_parameters:
        a_grad += pred_a_grad
        h_grad += pred_h_grad
        c_grad += pred_c_grad
        if b_grad is not None and pred_b_grad is not None:
            b_grad += pred_b_grad
    return state_bar, a_grad, h_grad, b_grad, c_grad


@timed("goattm.solvers.compute_lagged_midpoint_discrete_adjoint")
def compute_lagged_midpoint_discrete_adjoint(
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
        state_bar, _, _, _, _ = _lagged_midpoint_reverse_step(
            dynamics=dynamics,
            state=states[step_idx],
            dt=float(dt_history[step_idx]),
            time=float(times[step_idx]),
            adjoint_next=adjoints[step_idx + 1],
            input_function=input_function,
            accumulate_parameters=False,
        )
        adjoints[step_idx] = np.asarray(state_loss_gradients[step_idx], dtype=np.float64) + state_bar
    return adjoints


@timed("goattm.solvers.accumulate_lagged_midpoint_parameter_gradients")
def accumulate_lagged_midpoint_parameter_gradients(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    adjoints: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    if not rollout.success:
        raise RuntimeError("Cannot accumulate parameter gradients from an unsuccessful rollout.")
    if adjoints.shape != rollout.states.shape:
        raise ValueError(f"adjoints must have shape {rollout.states.shape}, got {adjoints.shape}")

    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    h_grad = np.zeros_like(dynamics.h_matrix)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)

    for step_idx in range(rollout.accepted_steps):
        _, da, dh, db, dc = _lagged_midpoint_reverse_step(
            dynamics=dynamics,
            state=rollout.states[step_idx],
            dt=float(rollout.dt_history[step_idx]),
            time=float(rollout.times[step_idx]),
            adjoint_next=adjoints[step_idx + 1],
            input_function=input_function,
            accumulate_parameters=True,
        )
        a_grad += da
        h_grad += dh
        c_grad += dc
        if b_grad is not None and db is not None:
            b_grad += db
    return a_grad, h_grad, b_grad, c_grad


@timed("goattm.solvers.rollout_lagged_midpoint_tangent_from_base_rollout")
def rollout_lagged_midpoint_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    raise NotImplementedError("Lagged-midpoint tangent rollout is not implemented yet.")
