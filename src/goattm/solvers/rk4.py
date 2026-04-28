from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from goattm.core.parametrization import quadratic_features
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.runtime import timed
from goattm.solvers.explicit_euler import _quadratic_features_directional_derivative
from goattm.solvers.implicit_midpoint import RolloutResult


@dataclass(frozen=True)
class Rk4StepStages:
    k1: np.ndarray
    k2: np.ndarray
    k3: np.ndarray
    k4: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    y3: np.ndarray
    y4: np.ndarray
    t1: float
    t2: float
    t3: float
    t4: float


def _validate_base_rollout(base_rollout: RolloutResult) -> None:
    if not base_rollout.success:
        raise RuntimeError("Base rollout must be successful before tangent or adjoint propagation.")
    if base_rollout.times.shape[0] != base_rollout.states.shape[0]:
        raise ValueError("Base rollout times and states are inconsistent.")
    if base_rollout.dt_history.shape[0] != base_rollout.states.shape[0] - 1:
        raise ValueError("Base rollout dt_history is inconsistent with states.")


def _input_at(input_function: Callable[[float], np.ndarray] | None, time: float) -> np.ndarray | None:
    return None if input_function is None else np.asarray(input_function(time), dtype=np.float64)


def rk4_step_with_stages(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    dt: float,
    time: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, Rk4StepStages]:
    dynamics.validate_state(state, "state")
    t1 = float(time)
    t2 = float(time + 0.5 * dt)
    t3 = t2
    t4 = float(time + dt)
    y1 = state
    k1 = dynamics.rhs(y1, p=_input_at(input_function, t1))
    y2 = state + 0.5 * dt * k1
    k2 = dynamics.rhs(y2, p=_input_at(input_function, t2))
    y3 = state + 0.5 * dt * k2
    k3 = dynamics.rhs(y3, p=_input_at(input_function, t3))
    y4 = state + dt * k3
    k4 = dynamics.rhs(y4, p=_input_at(input_function, t4))
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state, Rk4StepStages(
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        y1=y1,
        y2=y2,
        y3=y3,
        y4=y4,
        t1=t1,
        t2=t2,
        t3=t3,
        t4=t4,
    )


def rk4_step(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    dt: float,
    time: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    return rk4_step_with_stages(dynamics, state, dt, time=time, input_function=input_function)[0]


@timed("goattm.solvers.rollout_rk4")
def rollout_rk4(
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
        current_state = rk4_step(
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


@timed("goattm.solvers.rollout_rk4_to_observation_times")
def rollout_rk4_to_observation_times(
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
            current_state = rk4_step(
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


@timed("goattm.solvers.rollout_rk4_tangent_from_base_rollout")
def rollout_rk4_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    _validate_base_rollout(base_rollout)
    tangent_states = np.zeros_like(base_rollout.states, dtype=np.float64)
    current_tangent = np.zeros(base_rollout.states.shape[1], dtype=np.float64)
    for step_idx in range(base_rollout.accepted_steps):
        state = base_rollout.states[step_idx]
        time = float(base_rollout.times[step_idx])
        dt = float(base_rollout.dt_history[step_idx])
        _, stages = rk4_step_with_stages(dynamics, state, dt, time=time, input_function=input_function)
        dk1 = _rk4_stage_tangent(dynamics, stages.y1, current_tangent, stages.t1, parameter_action)
        dy2 = current_tangent + 0.5 * dt * dk1
        dk2 = _rk4_stage_tangent(dynamics, stages.y2, dy2, stages.t2, parameter_action)
        dy3 = current_tangent + 0.5 * dt * dk2
        dk3 = _rk4_stage_tangent(dynamics, stages.y3, dy3, stages.t3, parameter_action)
        dy4 = current_tangent + dt * dk3
        dk4 = _rk4_stage_tangent(dynamics, stages.y4, dy4, stages.t4, parameter_action)
        current_tangent = current_tangent + (dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)
        tangent_states[step_idx + 1] = current_tangent.copy()
    return tangent_states


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


@timed("goattm.solvers.compute_rk4_discrete_adjoint")
def compute_rk4_discrete_adjoint(
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
        state = states[step_idx]
        time = float(times[step_idx])
        dt = float(dt_history[step_idx])
        adjoints[step_idx] = np.asarray(state_loss_gradients[step_idx], dtype=np.float64) + _rk4_step_state_pullback(
            dynamics=dynamics,
            state=state,
            dt=dt,
            time=time,
            adjoint_next=adjoints[step_idx + 1],
            input_function=input_function,
        )
    return adjoints


def _rk4_step_state_pullback(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    dt: float,
    time: float,
    adjoint_next: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    _, stages = rk4_step_with_stages(dynamics, state, dt, time=time, input_function=input_function)
    state_bar, _, _, _, _ = _rk4_reverse_step(
        dynamics=dynamics,
        stages=stages,
        dt=dt,
        adjoint_next=adjoint_next,
        input_function=input_function,
        accumulate_parameters=False,
    )
    return state_bar


@timed("goattm.solvers.accumulate_rk4_parameter_gradients")
def accumulate_rk4_parameter_gradients(
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
        _, stages = rk4_step_with_stages(dynamics, state, dt, time=time, input_function=input_function)
        _, da, dh, db, dc = _rk4_reverse_step(
            dynamics=dynamics,
            stages=stages,
            dt=dt,
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


def _rk4_reverse_step(
    dynamics: QuadraticDynamics,
    stages: Rk4StepStages,
    dt: float,
    adjoint_next: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    accumulate_parameters: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    state_bar = np.asarray(adjoint_next, dtype=np.float64).copy()
    k1_bar = (dt / 6.0) * adjoint_next
    k2_bar = (dt / 3.0) * adjoint_next
    k3_bar = (dt / 3.0) * adjoint_next
    k4_bar = (dt / 6.0) * adjoint_next

    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    h_grad = np.zeros_like(dynamics.h_matrix)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)

    y4_bar = dynamics.rhs_jacobian(stages.y4).T @ k4_bar
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients(dynamics, stages.y4, stages.t4, k4_bar, a_grad, h_grad, b_grad, c_grad, input_function)
    state_bar += y4_bar
    k3_bar = k3_bar + dt * y4_bar

    y3_bar = dynamics.rhs_jacobian(stages.y3).T @ k3_bar
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients(dynamics, stages.y3, stages.t3, k3_bar, a_grad, h_grad, b_grad, c_grad, input_function)
    state_bar += y3_bar
    k2_bar = k2_bar + 0.5 * dt * y3_bar

    y2_bar = dynamics.rhs_jacobian(stages.y2).T @ k2_bar
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients(dynamics, stages.y2, stages.t2, k2_bar, a_grad, h_grad, b_grad, c_grad, input_function)
    state_bar += y2_bar
    k1_bar = k1_bar + 0.5 * dt * y2_bar

    y1_bar = dynamics.rhs_jacobian(stages.y1).T @ k1_bar
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients(dynamics, stages.y1, stages.t1, k1_bar, a_grad, h_grad, b_grad, c_grad, input_function)
    state_bar += y1_bar
    return state_bar, a_grad, h_grad, b_grad, c_grad


def _accumulate_stage_parameter_gradients(
    dynamics: QuadraticDynamics,
    stage_state: np.ndarray,
    stage_time: float,
    stage_adjoint: np.ndarray,
    a_grad: np.ndarray,
    h_grad: np.ndarray,
    b_grad: np.ndarray | None,
    c_grad: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None,
) -> None:
    a_grad += np.outer(stage_adjoint, stage_state)
    h_grad += np.outer(stage_adjoint, quadratic_features(stage_state))
    c_grad += stage_adjoint
    if b_grad is not None and input_function is not None:
        b_grad += np.outer(stage_adjoint, np.asarray(input_function(stage_time), dtype=np.float64))


@timed("goattm.solvers.compute_rk4_incremental_discrete_adjoint")
def compute_rk4_incremental_discrete_adjoint(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    base_adjoints: np.ndarray,
    state_loss_gradient_direction: np.ndarray,
    jacobian_direction: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
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
    for step_idx in range(rollout.accepted_steps - 1, -1, -1):
        state = rollout.states[step_idx]
        state_tangent = tangent_states[step_idx]
        time = float(rollout.times[step_idx])
        dt = float(rollout.dt_history[step_idx])
        adjoint_tangents[step_idx] = np.asarray(state_loss_gradient_direction[step_idx], dtype=np.float64)
        adjoint_tangents[step_idx] += _rk4_step_state_pullback(
            dynamics=dynamics,
            state=state,
            dt=dt,
            time=time,
            adjoint_next=adjoint_tangents[step_idx + 1],
        )
        adjoint_tangents[step_idx] += _rk4_step_pullback_direction(
            dynamics=dynamics,
            state=state,
            state_tangent=state_tangent,
            dt=dt,
            time=time,
            adjoint_next=base_adjoints[step_idx + 1],
            jacobian_direction=jacobian_direction,
            parameter_action=parameter_action,
            input_function=input_function,
        )
    return adjoint_tangents


def _rk4_step_pullback_direction(
    dynamics: QuadraticDynamics,
    state: np.ndarray,
    state_tangent: np.ndarray,
    dt: float,
    time: float,
    adjoint_next: np.ndarray,
    jacobian_direction: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    _, stages = rk4_step_with_stages(dynamics, state, dt, time=time, input_function=input_function)
    stage_tangents = _rk4_stage_tangents(dynamics, stages, state_tangent, dt, parameter_action)
    _, _, delta_state_bar = _rk4_stage_bars_and_direction(
        dynamics=dynamics,
        stages=stages,
        stage_tangents=stage_tangents,
        dt=dt,
        adjoint_next=adjoint_next,
        adjoint_next_tangent=np.zeros_like(adjoint_next),
        jacobian_direction=jacobian_direction,
    )
    return delta_state_bar


@timed("goattm.solvers.accumulate_rk4_parameter_hessian_action_terms")
def accumulate_rk4_parameter_hessian_action_terms(
    dynamics: QuadraticDynamics,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    adjoints: np.ndarray,
    adjoint_tangents: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    jacobian_direction: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None,
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
        time = float(rollout.times[step_idx])
        dt = float(rollout.dt_history[step_idx])
        _, stages = rk4_step_with_stages(dynamics, state, dt, time=time, input_function=input_function)
        stage_tangents = _rk4_stage_tangents(dynamics, stages, state_tangent, dt, parameter_action)
        da, dh, db, dc, dda, ddh, ddb, ddc = _rk4_parameter_gradient_and_direction_from_stages(
            dynamics=dynamics,
            stages=stages,
            stage_tangents=stage_tangents,
            dt=dt,
            adjoint_next=adjoints[step_idx + 1],
            adjoint_next_tangent=adjoint_tangents[step_idx + 1],
            input_function=input_function,
            jacobian_direction=jacobian_direction,
        )
        a_grad += da
        h_grad += dh
        c_grad += dc
        delta_a_grad += dda
        delta_h_grad += ddh
        delta_c_grad += ddc
        if b_grad is not None and db is not None and delta_b_grad is not None and ddb is not None:
            b_grad += db
            delta_b_grad += ddb
    return a_grad, delta_a_grad, h_grad, delta_h_grad, b_grad, delta_b_grad, c_grad, delta_c_grad


def _rk4_stage_tangents(
    dynamics: QuadraticDynamics,
    stages: Rk4StepStages,
    state_tangent: np.ndarray,
    dt: float,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dy1 = state_tangent
    dk1 = _rk4_stage_tangent(dynamics, stages.y1, dy1, stages.t1, parameter_action)
    dy2 = state_tangent + 0.5 * dt * dk1
    dk2 = _rk4_stage_tangent(dynamics, stages.y2, dy2, stages.t2, parameter_action)
    dy3 = state_tangent + 0.5 * dt * dk2
    dk3 = _rk4_stage_tangent(dynamics, stages.y3, dy3, stages.t3, parameter_action)
    dy4 = state_tangent + dt * dk3
    return dy1, dy2, dy3, dy4


def _rk4_stage_bars_and_direction(
    dynamics: QuadraticDynamics,
    stages: Rk4StepStages,
    stage_tangents: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dt: float,
    adjoint_next: np.ndarray,
    adjoint_next_tangent: np.ndarray,
    jacobian_direction: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    k1_bar = (dt / 6.0) * adjoint_next
    k2_bar = (dt / 3.0) * adjoint_next
    k3_bar = (dt / 3.0) * adjoint_next
    k4_bar = (dt / 6.0) * adjoint_next
    dk1_bar = (dt / 6.0) * adjoint_next_tangent
    dk2_bar = (dt / 3.0) * adjoint_next_tangent
    dk3_bar = (dt / 3.0) * adjoint_next_tangent
    dk4_bar = (dt / 6.0) * adjoint_next_tangent
    delta_state_bar = np.asarray(adjoint_next_tangent, dtype=np.float64).copy()

    y4_bar = dynamics.rhs_jacobian(stages.y4).T @ k4_bar
    dy4_bar = jacobian_direction(stages.y4, stage_tangents[3], stages.t4).T @ k4_bar + dynamics.rhs_jacobian(stages.y4).T @ dk4_bar
    delta_state_bar += dy4_bar
    k3_bar = k3_bar + dt * y4_bar
    dk3_bar = dk3_bar + dt * dy4_bar

    y3_bar = dynamics.rhs_jacobian(stages.y3).T @ k3_bar
    dy3_bar = jacobian_direction(stages.y3, stage_tangents[2], stages.t3).T @ k3_bar + dynamics.rhs_jacobian(stages.y3).T @ dk3_bar
    delta_state_bar += dy3_bar
    k2_bar = k2_bar + 0.5 * dt * y3_bar
    dk2_bar = dk2_bar + 0.5 * dt * dy3_bar

    y2_bar = dynamics.rhs_jacobian(stages.y2).T @ k2_bar
    dy2_bar = jacobian_direction(stages.y2, stage_tangents[1], stages.t2).T @ k2_bar + dynamics.rhs_jacobian(stages.y2).T @ dk2_bar
    delta_state_bar += dy2_bar
    k1_bar = k1_bar + 0.5 * dt * y2_bar
    dk1_bar = dk1_bar + 0.5 * dt * dy2_bar

    dy1_bar = jacobian_direction(stages.y1, stage_tangents[0], stages.t1).T @ k1_bar + dynamics.rhs_jacobian(stages.y1).T @ dk1_bar
    delta_state_bar += dy1_bar
    return (k1_bar, k2_bar, k3_bar, k4_bar), (dk1_bar, dk2_bar, dk3_bar, dk4_bar), delta_state_bar


def _rk4_parameter_gradient_and_direction_from_stages(
    dynamics: QuadraticDynamics,
    stages: Rk4StepStages,
    stage_tangents: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dt: float,
    adjoint_next: np.ndarray,
    adjoint_next_tangent: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    jacobian_direction: Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    if jacobian_direction is None:
        jacobian_direction = lambda state, tangent, time: dynamics.quadratic_jacobian(tangent)  # noqa: E731
    stage_bars, stage_bar_directions, _ = _rk4_stage_bars_and_direction(
        dynamics=dynamics,
        stages=stages,
        stage_tangents=stage_tangents,
        dt=dt,
        adjoint_next=adjoint_next,
        adjoint_next_tangent=adjoint_next_tangent,
        jacobian_direction=jacobian_direction,
    )
    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    delta_a_grad = np.zeros_like(a_grad)
    h_grad = np.zeros_like(dynamics.h_matrix)
    delta_h_grad = np.zeros_like(h_grad)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    delta_c_grad = np.zeros_like(c_grad)
    b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)
    delta_b_grad = None if dynamics.b is None else np.zeros_like(dynamics.b)
    for stage_state, stage_tangent, stage_time, stage_bar, stage_bar_direction in zip(
        (stages.y1, stages.y2, stages.y3, stages.y4),
        stage_tangents,
        (stages.t1, stages.t2, stages.t3, stages.t4),
        stage_bars,
        stage_bar_directions,
        strict=True,
    ):
        zeta = quadratic_features(stage_state)
        zeta_tangent = _quadratic_features_directional_derivative(stage_state, stage_tangent)
        a_grad += np.outer(stage_bar, stage_state)
        delta_a_grad += np.outer(stage_bar_direction, stage_state) + np.outer(stage_bar, stage_tangent)
        h_grad += np.outer(stage_bar, zeta)
        delta_h_grad += np.outer(stage_bar_direction, zeta) + np.outer(stage_bar, zeta_tangent)
        c_grad += stage_bar
        delta_c_grad += stage_bar_direction
        if b_grad is not None and input_function is not None:
            p_stage = np.asarray(input_function(stage_time), dtype=np.float64)
            b_grad += np.outer(stage_bar, p_stage)
            delta_b_grad += np.outer(stage_bar_direction, p_stage)
    return a_grad, h_grad, b_grad, c_grad, delta_a_grad, delta_h_grad, delta_b_grad, delta_c_grad
