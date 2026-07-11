from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from goattm.core.parametrization import quadratic_features
from goattm.core.quadratic import quadratic_bilinear_action_matrix
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.runtime import timed
from goattm.solvers.implicit_midpoint import RolloutResult
from goattm.solvers.lagged_midpoint_numba import (
    accumulate_lagged_midpoint_parameter_gradients_presampled_kernel,
    compute_lagged_midpoint_discrete_adjoint_cached_presampled_kernel,
    compute_lagged_midpoint_incremental_discrete_adjoint_cached_presampled_kernel,
    compute_lagged_midpoint_discrete_adjoint_presampled_kernel,
    lagged_midpoint_final_time_grid,
    lagged_midpoint_time_grid,
    presample_lagged_midpoint_inputs,
    rollout_lagged_midpoint_explicit_parameter_tangent_cached_kernel,
    rollout_lagged_midpoint_presampled_kernel,
    rollout_lagged_midpoint_presampled_cached_kernel,
    rollout_lagged_midpoint_explicit_parameter_tangent_presampled_kernel,
)
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


@dataclass(frozen=True)
class LaggedMidpointForwardCache:
    """Per-step data retained so tangent and adjoint solves do not replay forward."""

    predictors: np.ndarray
    stage2: np.ndarray
    stage3: np.ndarray
    stage4: np.ndarray
    linear_operators: np.ndarray
    system_matrices: np.ndarray
    jacobian1: np.ndarray
    jacobian2: np.ndarray
    jacobian3: np.ndarray
    jacobian4: np.ndarray
    feature1: np.ndarray
    feature2: np.ndarray
    feature3: np.ndarray
    feature4: np.ndarray


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
    times_grid, dt_history_grid = lagged_midpoint_final_time_grid(t_final, max_dt)
    fast_rollout = _rollout_lagged_midpoint_presampled_if_available(
        dynamics=dynamics,
        u0=u0,
        times=times_grid,
        dt_history=dt_history_grid,
        input_function=input_function,
    )
    if fast_rollout is not None:
        return fast_rollout

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
    times_grid, dt_history_grid, observation_indices_grid = lagged_midpoint_time_grid(observation_times, max_dt)
    fast_rollout = _rollout_lagged_midpoint_presampled_if_available(
        dynamics=dynamics,
        u0=u0,
        times=times_grid,
        dt_history=dt_history_grid,
        input_function=input_function,
    )
    if fast_rollout is not None:
        if fast_rollout.success:
            return fast_rollout, observation_indices_grid.astype(int, copy=False)
        completed_observation_indices = observation_indices_grid[observation_indices_grid <= fast_rollout.accepted_steps]
        return fast_rollout, completed_observation_indices.astype(int, copy=False)

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


def _rollout_lagged_midpoint_presampled_if_available(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    times: np.ndarray,
    dt_history: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None,
) -> RolloutResult | None:
    presampled = presample_lagged_midpoint_inputs(
        input_function=input_function,
        times=times,
        dt_history=dt_history,
        input_dimension=dynamics.input_dimension,
    )
    if presampled is None:
        return None
    p0_values, pq_values, pm_values = presampled
    b_matrix = (
        np.zeros((dynamics.dimension, 0), dtype=np.float64)
        if dynamics.b is None
        else np.asarray(dynamics.b, dtype=np.float64)
    )
    (
        success,
        accepted,
        states,
        predictors,
        stage2,
        stage3,
        stage4,
        linear_operators,
        system_matrices,
        jacobian1,
        jacobian2,
        jacobian3,
        jacobian4,
        feature1,
        feature2,
        feature3,
        feature4,
    ) = rollout_lagged_midpoint_presampled_cached_kernel(
        np.asarray(dynamics.a, dtype=np.float64),
        np.asarray(dynamics.h_matrix, dtype=np.float64),
        b_matrix,
        np.asarray(dynamics.c, dtype=np.float64),
        np.asarray(u0, dtype=np.float64),
        np.asarray(dt_history, dtype=np.float64),
        p0_values,
        pq_values,
        pm_values,
    )
    accepted_steps = int(accepted)
    stored_states = states[: accepted_steps + 1].copy()
    stored_times = times[: accepted_steps + 1].copy()
    stored_dt_history = dt_history[:accepted_steps].copy()
    solver_cache = None
    if bool(success):
        solver_cache = LaggedMidpointForwardCache(
            predictors=predictors[:accepted_steps].copy(),
            stage2=stage2[:accepted_steps].copy(),
            stage3=stage3[:accepted_steps].copy(),
            stage4=stage4[:accepted_steps].copy(),
            linear_operators=linear_operators[:accepted_steps].copy(),
            system_matrices=system_matrices[:accepted_steps].copy(),
            jacobian1=jacobian1[:accepted_steps].copy(),
            jacobian2=jacobian2[:accepted_steps].copy(),
            jacobian3=jacobian3[:accepted_steps].copy(),
            jacobian4=jacobian4[:accepted_steps].copy(),
            feature1=feature1[:accepted_steps].copy(),
            feature2=feature2[:accepted_steps].copy(),
            feature3=feature3[:accepted_steps].copy(),
            feature4=feature4[:accepted_steps].copy(),
        )
    return RolloutResult(
        success=bool(success),
        accepted_steps=accepted_steps,
        dt_reductions=0,
        newton_failures=0,
        final_time=float(stored_times[-1]) if stored_times.shape[0] else 0.0,
        dt_history=stored_dt_history,
        times=stored_times,
        states=stored_states,
        solver_cache=solver_cache,
    )


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
    forward_cache: LaggedMidpointForwardCache | None = None,
) -> np.ndarray:
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError(f"states must have shape (N, r), got {states.shape}")
    if times.shape[0] != states.shape[0]:
        raise ValueError("times and states are inconsistent.")
    if dt_history.shape[0] != states.shape[0] - 1:
        raise ValueError("dt_history and states are inconsistent.")
    if state_loss_gradients.shape != states.shape:
        raise ValueError(f"state_loss_gradients must have shape {states.shape}, got {state_loss_gradients.shape}")

    fast_adjoint = _compute_lagged_midpoint_discrete_adjoint_presampled_if_available(
        dynamics=dynamics,
        states=states,
        times=times,
        dt_history=dt_history,
        state_loss_gradients=state_loss_gradients,
        input_function=input_function,
        forward_cache=forward_cache,
    )
    if fast_adjoint is not None:
        return fast_adjoint

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


@timed("goattm.solvers.compute_lagged_midpoint_incremental_discrete_adjoint")
def compute_lagged_midpoint_incremental_discrete_adjoint(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    base_adjoints: np.ndarray,
    state_loss_gradient_direction: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray | None,
    delta_c: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    forward_cache: LaggedMidpointForwardCache | None = None,
) -> np.ndarray:
    """Differentiate the lagged-midpoint discrete adjoint recursion.

    This is the exact directional derivative of the discrete reverse pass for
    the supplied trajectory. It reuses the base forward stages and Jacobians;
    no perturbed rollout or finite-difference reverse solve is performed.
    """
    if not rollout.success:
        raise RuntimeError("Cannot compute an incremental adjoint from an unsuccessful rollout.")
    if tangent_states.shape != rollout.states.shape:
        raise ValueError(f"tangent_states must have shape {rollout.states.shape}, got {tangent_states.shape}")
    if base_adjoints.shape != rollout.states.shape:
        raise ValueError(f"base_adjoints must have shape {rollout.states.shape}, got {base_adjoints.shape}")
    if state_loss_gradient_direction.shape != rollout.states.shape:
        raise ValueError(
            "state_loss_gradient_direction must have shape "
            f"{rollout.states.shape}, got {state_loss_gradient_direction.shape}"
        )
    cache = forward_cache if forward_cache is not None else rollout.solver_cache
    if not isinstance(cache, LaggedMidpointForwardCache):
        raise ValueError("A LaggedMidpointForwardCache is required for the incremental adjoint.")
    presampled = presample_lagged_midpoint_inputs(
        input_function=input_function,
        times=rollout.times,
        dt_history=rollout.dt_history,
        input_dimension=dynamics.input_dimension,
    )
    if presampled is None:
        raise ValueError("The input function must support lagged-midpoint pre-sampling.")
    p0_values, pq_values, pm_values = presampled
    b_matrix = _dynamics_b_matrix_for_numba(dynamics)
    delta_b_matrix = np.zeros_like(b_matrix) if delta_b is None else np.asarray(delta_b, dtype=np.float64)
    return compute_lagged_midpoint_incremental_discrete_adjoint_cached_presampled_kernel(
        np.asarray(dynamics.h_matrix, dtype=np.float64),
        np.asarray(delta_a, dtype=np.float64),
        np.asarray(delta_h, dtype=np.float64),
        delta_b_matrix,
        np.asarray(delta_c, dtype=np.float64),
        np.asarray(rollout.states, dtype=np.float64),
        np.asarray(tangent_states, dtype=np.float64),
        np.asarray(rollout.dt_history, dtype=np.float64),
        np.asarray(base_adjoints, dtype=np.float64),
        np.asarray(state_loss_gradient_direction, dtype=np.float64),
        cache.predictors,
        cache.stage2,
        cache.stage3,
        cache.stage4,
        cache.linear_operators,
        cache.system_matrices,
        cache.jacobian1,
        cache.jacobian2,
        cache.jacobian3,
        cache.jacobian4,
        cache.feature1,
        cache.feature2,
        cache.feature3,
        cache.feature4,
        p0_values,
        pq_values,
        pm_values,
    )


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

    fast_gradients = _accumulate_lagged_midpoint_parameter_gradients_presampled_if_available(
        dynamics=dynamics,
        rollout=rollout,
        adjoints=adjoints,
        input_function=input_function,
    )
    if fast_gradients is not None:
        return fast_gradients

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


def _dynamics_b_matrix_for_numba(dynamics: QuadraticDynamics) -> np.ndarray:
    return (
        np.zeros((dynamics.dimension, 0), dtype=np.float64)
        if dynamics.b is None
        else np.asarray(dynamics.b, dtype=np.float64)
    )


def _compute_lagged_midpoint_discrete_adjoint_presampled_if_available(
    dynamics: QuadraticDynamics,
    states: np.ndarray,
    times: np.ndarray,
    dt_history: np.ndarray,
    state_loss_gradients: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None,
    forward_cache: LaggedMidpointForwardCache | None = None,
) -> np.ndarray | None:
    presampled = presample_lagged_midpoint_inputs(
        input_function=input_function,
        times=times,
        dt_history=dt_history,
        input_dimension=dynamics.input_dimension,
    )
    if presampled is None:
        return None
    p0_values, pq_values, pm_values = presampled
    if forward_cache is not None:
        return compute_lagged_midpoint_discrete_adjoint_cached_presampled_kernel(
            np.asarray(dynamics.a, dtype=np.float64),
            np.asarray(dynamics.h_matrix, dtype=np.float64),
            np.asarray(states, dtype=np.float64),
            np.asarray(dt_history, dtype=np.float64),
            np.asarray(state_loss_gradients, dtype=np.float64),
            forward_cache.predictors,
            forward_cache.stage2,
            forward_cache.stage3,
            forward_cache.stage4,
            forward_cache.linear_operators,
            forward_cache.system_matrices,
            forward_cache.jacobian1,
            forward_cache.jacobian2,
            forward_cache.jacobian3,
            forward_cache.jacobian4,
        )
    return compute_lagged_midpoint_discrete_adjoint_presampled_kernel(
        np.asarray(dynamics.a, dtype=np.float64),
        np.asarray(dynamics.h_matrix, dtype=np.float64),
        _dynamics_b_matrix_for_numba(dynamics),
        np.asarray(dynamics.c, dtype=np.float64),
        np.asarray(states, dtype=np.float64),
        np.asarray(dt_history, dtype=np.float64),
        np.asarray(state_loss_gradients, dtype=np.float64),
        p0_values,
        pq_values,
        pm_values,
    )


def _accumulate_lagged_midpoint_parameter_gradients_presampled_if_available(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    adjoints: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray] | None:
    presampled = presample_lagged_midpoint_inputs(
        input_function=input_function,
        times=rollout.times,
        dt_history=rollout.dt_history,
        input_dimension=dynamics.input_dimension,
    )
    if presampled is None:
        return None
    p0_values, pq_values, pm_values = presampled
    a_grad, h_grad, b_grad, c_grad = accumulate_lagged_midpoint_parameter_gradients_presampled_kernel(
        np.asarray(dynamics.a, dtype=np.float64),
        np.asarray(dynamics.h_matrix, dtype=np.float64),
        _dynamics_b_matrix_for_numba(dynamics),
        np.asarray(dynamics.c, dtype=np.float64),
        np.asarray(rollout.states, dtype=np.float64),
        np.asarray(rollout.dt_history, dtype=np.float64),
        np.asarray(adjoints, dtype=np.float64),
        p0_values,
        pq_values,
        pm_values,
    )
    return a_grad, h_grad, None if dynamics.b is None else b_grad, c_grad


@timed("goattm.solvers.rollout_lagged_midpoint_tangent_from_base_rollout")
def rollout_lagged_midpoint_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
    parameter_linear_operator_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    forcing_parameter_action: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    if not base_rollout.success:
        raise RuntimeError("Cannot compute tangent rollout from an unsuccessful base rollout.")
    if base_rollout.states.ndim != 2 or base_rollout.states.shape[0] < 1:
        raise ValueError(f"base_rollout states must have shape (N, r), got {base_rollout.states.shape}")
    if base_rollout.dt_history.shape[0] != base_rollout.states.shape[0] - 1:
        raise ValueError("base_rollout dt_history and states are inconsistent.")

    tangent_states = np.zeros_like(base_rollout.states, dtype=np.float64)
    current_tangent = np.zeros(base_rollout.states.shape[1], dtype=np.float64)
    for step_idx in range(base_rollout.accepted_steps):
        state = base_rollout.states[step_idx]
        next_state = base_rollout.states[step_idx + 1]
        time = float(base_rollout.times[step_idx])
        dt = float(base_rollout.dt_history[step_idx])
        half_dt = 0.5 * dt
        predictor_state, predictor_stages = rk4_step_with_stages(
            dynamics=dynamics,
            state=state,
            dt=half_dt,
            time=time,
            input_function=input_function,
        )
        predictor_tangent = _rk4_tangent_from_stages(
            dynamics=dynamics,
            stages=predictor_stages,
            state_tangent=current_tangent,
            dt=half_dt,
            parameter_action=parameter_action,
        )
        mid_time = time + half_dt
        linear_operator = _lagged_linear_operator(dynamics, predictor_state)
        linear_operator_tangent = quadratic_bilinear_action_matrix(dynamics.h_matrix, predictor_tangent)
        if parameter_linear_operator_action is not None:
            linear_operator_tangent = linear_operator_tangent + np.asarray(
                parameter_linear_operator_action(predictor_state, mid_time),
                dtype=np.float64,
            )
        forcing_tangent = np.zeros(dynamics.dimension, dtype=np.float64)
        if forcing_parameter_action is not None:
            forcing_tangent = np.asarray(forcing_parameter_action(mid_time), dtype=np.float64)

        identity = np.eye(dynamics.dimension, dtype=np.float64)
        system_matrix = identity - half_dt * linear_operator
        rhs_tangent = (
            (identity + half_dt * linear_operator) @ current_tangent
            + half_dt * (linear_operator_tangent @ (state + next_state))
            + dt * forcing_tangent
        )
        current_tangent = np.linalg.solve(system_matrix, rhs_tangent)
        tangent_states[step_idx + 1] = current_tangent.copy()
    return tangent_states


@timed("goattm.solvers.rollout_lagged_midpoint_explicit_parameter_tangent_from_base_rollout")
def rollout_lagged_midpoint_explicit_parameter_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray | None,
    delta_c: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    forward_cache: LaggedMidpointForwardCache | None = None,
) -> np.ndarray | None:
    """Fast lagged-midpoint tangent rollout for explicit parameter directions.

    Returns ``None`` when the input function cannot be pre-sampled, which lets
    callers fall back to the fully generic Python/callback implementation. The
    kernel itself receives only dense arrays, so it stays independent of the
    dynamics parametrization used to produce ``delta_a`` and ``delta_h``.
    """

    if not base_rollout.success:
        raise RuntimeError("Cannot compute tangent rollout from an unsuccessful base rollout.")
    if base_rollout.states.ndim != 2 or base_rollout.states.shape[0] < 1:
        raise ValueError(f"base_rollout states must have shape (N, r), got {base_rollout.states.shape}")
    if base_rollout.dt_history.shape[0] != base_rollout.states.shape[0] - 1:
        raise ValueError("base_rollout dt_history and states are inconsistent.")

    input_dimension = dynamics.input_dimension
    presampled = presample_lagged_midpoint_inputs(
        input_function=input_function,
        times=base_rollout.times,
        dt_history=base_rollout.dt_history,
        input_dimension=input_dimension,
    )
    if presampled is None:
        return None
    p0_values, pq_values, pm_values = presampled

    r = dynamics.dimension
    b_matrix = (
        np.zeros((r, 0), dtype=np.float64)
        if dynamics.b is None
        else np.asarray(dynamics.b, dtype=np.float64)
    )
    delta_b_matrix = (
        np.zeros_like(b_matrix, dtype=np.float64)
        if delta_b is None
        else np.asarray(delta_b, dtype=np.float64)
    )
    if forward_cache is not None:
        return rollout_lagged_midpoint_explicit_parameter_tangent_cached_kernel(
            np.asarray(dynamics.a, dtype=np.float64),
            np.asarray(dynamics.h_matrix, dtype=np.float64),
            np.asarray(delta_a, dtype=np.float64),
            np.asarray(delta_h, dtype=np.float64),
            delta_b_matrix,
            np.asarray(delta_c, dtype=np.float64),
            np.asarray(base_rollout.states, dtype=np.float64),
            np.asarray(base_rollout.dt_history, dtype=np.float64),
            forward_cache.predictors,
            forward_cache.stage2,
            forward_cache.stage3,
            forward_cache.stage4,
            forward_cache.linear_operators,
            forward_cache.system_matrices,
            forward_cache.jacobian1,
            forward_cache.jacobian2,
            forward_cache.jacobian3,
            forward_cache.jacobian4,
            forward_cache.feature1,
            forward_cache.feature2,
            forward_cache.feature3,
            forward_cache.feature4,
            p0_values,
            pq_values,
            pm_values,
        )
    return rollout_lagged_midpoint_explicit_parameter_tangent_presampled_kernel(
        np.asarray(dynamics.a, dtype=np.float64),
        np.asarray(dynamics.h_matrix, dtype=np.float64),
        b_matrix,
        np.asarray(dynamics.c, dtype=np.float64),
        np.asarray(delta_a, dtype=np.float64),
        np.asarray(delta_h, dtype=np.float64),
        delta_b_matrix,
        np.asarray(delta_c, dtype=np.float64),
        np.asarray(base_rollout.states, dtype=np.float64),
        np.asarray(base_rollout.dt_history, dtype=np.float64),
        p0_values,
        pq_values,
        pm_values,
    )


def _rk4_tangent_from_stages(
    dynamics: QuadraticDynamics,
    stages,
    state_tangent: np.ndarray,
    dt: float,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None,
) -> np.ndarray:
    dk1 = _rk4_stage_tangent(dynamics, stages.y1, state_tangent, stages.t1, parameter_action)
    dy2 = state_tangent + 0.5 * dt * dk1
    dk2 = _rk4_stage_tangent(dynamics, stages.y2, dy2, stages.t2, parameter_action)
    dy3 = state_tangent + 0.5 * dt * dk2
    dk3 = _rk4_stage_tangent(dynamics, stages.y3, dy3, stages.t3, parameter_action)
    dy4 = state_tangent + dt * dk3
    dk4 = _rk4_stage_tangent(dynamics, stages.y4, dy4, stages.t4, parameter_action)
    return state_tangent + (dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)


def _rk4_stage_tangent(
    dynamics: QuadraticDynamics,
    stage_state: np.ndarray,
    stage_tangent: np.ndarray,
    stage_time: float,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None,
) -> np.ndarray:
    out = dynamics.rhs_jacobian(stage_state) @ stage_tangent
    if parameter_action is not None:
        out = out + np.asarray(parameter_action(stage_state, stage_time), dtype=np.float64)
    return out
