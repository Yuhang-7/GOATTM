from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.runtime import timed


@dataclass(frozen=True)
class NewtonSolveInfo:
    success: bool
    iterations: int
    residual_norm: float
    backtracks: int


@dataclass(frozen=True)
class StepResult:
    success: bool
    dt_used: float
    dt_reductions: int
    newton_failures: int
    residual_norm: float
    newton_iterations: int
    u_next: np.ndarray


@dataclass(frozen=True)
class RolloutResult:
    success: bool
    accepted_steps: int
    dt_reductions: int
    newton_failures: int
    final_time: float
    dt_history: np.ndarray
    times: np.ndarray
    states: np.ndarray


def explicit_euler_guess(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    dt: float,
    t_prev: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    dynamics.validate_state(u_prev, "u_prev")
    return u_prev + dt * dynamics.rhs_at_time(u_prev, t_prev, input_function=input_function)


def implicit_midpoint_residual(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    u_next: np.ndarray,
    dt: float,
    t_prev: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
    residual_scale: float = 1.0,
) -> np.ndarray:
    dynamics.validate_state(u_prev, "u_prev")
    dynamics.validate_state(u_next, "u_next")
    if not (0.0 <= residual_scale <= 1.0):
        raise ValueError(f"residual_scale must lie in [0, 1], got {residual_scale}")
    midpoint = 0.5 * (u_prev + u_next)
    midpoint_time = t_prev + 0.5 * dt
    return u_next - u_prev - residual_scale * dt * dynamics.rhs_at_time(midpoint, midpoint_time, input_function=input_function)


def implicit_midpoint_jacobian(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    u_next: np.ndarray,
    dt: float,
    t_prev: float = 0.0,
    residual_scale: float = 1.0,
) -> np.ndarray:
    dynamics.validate_state(u_prev, "u_prev")
    dynamics.validate_state(u_next, "u_next")
    if not (0.0 <= residual_scale <= 1.0):
        raise ValueError(f"residual_scale must lie in [0, 1], got {residual_scale}")
    midpoint = 0.5 * (u_prev + u_next)
    return np.eye(dynamics.dimension) - 0.5 * residual_scale * dt * dynamics.rhs_jacobian(midpoint)


def solve_implicit_midpoint_step(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    dt: float,
    t_prev: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
    guess: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 25,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 8,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, NewtonSolveInfo]:
    dynamics.validate_state(u_prev, "u_prev")
    if not (0.0 <= residual_scale <= 1.0):
        raise ValueError(f"residual_scale must lie in [0, 1], got {residual_scale}")
    if guess is None:
        current = u_prev + residual_scale * dt * dynamics.rhs_at_time(u_prev, t_prev, input_function=input_function)
    else:
        dynamics.validate_state(guess, "guess")
        current = guess.copy()

    total_backtracks = 0
    residual = implicit_midpoint_residual(
        dynamics,
        u_prev,
        current,
        dt,
        t_prev=t_prev,
        input_function=input_function,
        residual_scale=residual_scale,
    )
    residual_norm = float(np.linalg.norm(residual))

    for iteration in range(1, max_iter + 1):
        if not np.isfinite(residual_norm):
            return current, NewtonSolveInfo(False, iteration - 1, residual_norm, total_backtracks)
        if residual_norm <= tol:
            return current, NewtonSolveInfo(True, iteration - 1, residual_norm, total_backtracks)

        jacobian = implicit_midpoint_jacobian(
            dynamics,
            u_prev,
            current,
            dt,
            t_prev=t_prev,
            residual_scale=residual_scale,
        )
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return current, NewtonSolveInfo(False, iteration - 1, residual_norm, total_backtracks)

        damping = 1.0
        accepted = False
        trial = current
        trial_residual = residual
        trial_norm = residual_norm
        for _ in range(max_backtracks + 1):
            trial = current + damping * delta
            trial_residual = implicit_midpoint_residual(
                dynamics,
                u_prev,
                trial,
                dt,
                t_prev=t_prev,
                input_function=input_function,
                residual_scale=residual_scale,
            )
            trial_norm = float(np.linalg.norm(trial_residual))
            if np.isfinite(trial_norm) and trial_norm < residual_norm:
                accepted = True
                break
            damping *= backtrack_factor
            total_backtracks += 1

        if not accepted:
            return current, NewtonSolveInfo(False, iteration, residual_norm, total_backtracks)

        current = trial
        residual = trial_residual
        residual_norm = trial_norm

    return current, NewtonSolveInfo(residual_norm <= tol, max_iter, residual_norm, total_backtracks)


def solve_implicit_midpoint_step_homotopy(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    dt: float,
    t_prev: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
    tol: float = 1e-10,
    max_iter: int = 25,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 8,
    initial_lambda_step: float = 0.1,
    min_lambda_step: float = 1.0e-4,
) -> tuple[np.ndarray, NewtonSolveInfo]:
    """Track the physical implicit-midpoint root from lambda=0 to lambda=1.

    The homotopy residual is
        F_lambda(u) = u - u_prev - lambda * dt * f((u_prev + u) / 2).
    At lambda=0, the root is exactly u_prev.  Stepping lambda upward and
    warm-starting Newton with the previous root makes it much harder for
    Newton to jump to a far-field spurious root of the quadratic equation.
    """
    dynamics.validate_state(u_prev, "u_prev")
    if not (0.0 < initial_lambda_step <= 1.0):
        raise ValueError(f"initial_lambda_step must lie in (0, 1], got {initial_lambda_step}")
    if not (0.0 < min_lambda_step <= initial_lambda_step):
        raise ValueError(
            f"min_lambda_step must lie in (0, initial_lambda_step], got {min_lambda_step}"
        )

    current = u_prev.copy()
    current_lambda = 0.0
    lambda_step = float(initial_lambda_step)
    total_iterations = 0
    total_backtracks = 0
    last_residual_norm = 0.0

    while current_lambda < 1.0 - 1e-14:
        target_lambda = min(1.0, current_lambda + lambda_step)
        candidate, info = solve_implicit_midpoint_step(
            dynamics=dynamics,
            u_prev=u_prev,
            dt=dt,
            t_prev=t_prev,
            input_function=input_function,
            guess=current,
            tol=tol,
            max_iter=max_iter,
            backtrack_factor=backtrack_factor,
            max_backtracks=max_backtracks,
            residual_scale=target_lambda,
        )
        total_iterations += info.iterations
        total_backtracks += info.backtracks
        last_residual_norm = info.residual_norm
        if info.success and np.all(np.isfinite(candidate)):
            current = candidate
            current_lambda = target_lambda
            lambda_step = min(1.0 - current_lambda, initial_lambda_step, 1.25 * lambda_step)
            if lambda_step <= 0.0:
                break
            continue

        lambda_step *= 0.5
        if lambda_step < min_lambda_step:
            return current, NewtonSolveInfo(
                success=False,
                iterations=total_iterations,
                residual_norm=last_residual_norm,
                backtracks=total_backtracks,
            )

    final_residual = implicit_midpoint_residual(
        dynamics=dynamics,
        u_prev=u_prev,
        u_next=current,
        dt=dt,
        t_prev=t_prev,
        input_function=input_function,
        residual_scale=1.0,
    )
    final_norm = float(np.linalg.norm(final_residual))
    return current, NewtonSolveInfo(
        success=np.isfinite(final_norm) and final_norm <= tol,
        iterations=total_iterations,
        residual_norm=final_norm,
        backtracks=total_backtracks,
    )


def solve_implicit_midpoint_step_with_retry(
    dynamics: QuadraticDynamics,
    u_prev: np.ndarray,
    dt_initial: float,
    t_prev: float = 0.0,
    input_function: Callable[[float], np.ndarray] | None = None,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 25,
    use_homotopy_fallback: bool = True,
    homotopy_initial_lambda_step: float = 0.1,
    homotopy_min_lambda_step: float = 1.0e-4,
) -> StepResult:
    dynamics.validate_state(u_prev, "u_prev")
    dt_trial = dt_initial
    dt_reductions = 0
    newton_failures = 0
    guess = explicit_euler_guess(dynamics, u_prev, dt_trial, t_prev=t_prev, input_function=input_function)

    while dt_trial >= dt_min:
        u_next, info = solve_implicit_midpoint_step(
            dynamics=dynamics,
            u_prev=u_prev,
            dt=dt_trial,
            t_prev=t_prev,
            input_function=input_function,
            guess=guess,
            tol=tol,
            max_iter=max_iter,
        )
        if info.success and np.all(np.isfinite(u_next)):
            return StepResult(
                success=True,
                dt_used=dt_trial,
                dt_reductions=dt_reductions,
                newton_failures=newton_failures,
                residual_norm=info.residual_norm,
                newton_iterations=info.iterations,
                u_next=u_next,
            )

        if use_homotopy_fallback:
            u_next, info = solve_implicit_midpoint_step_homotopy(
                dynamics=dynamics,
                u_prev=u_prev,
                dt=dt_trial,
                t_prev=t_prev,
                input_function=input_function,
                tol=tol,
                max_iter=max_iter,
                initial_lambda_step=homotopy_initial_lambda_step,
                min_lambda_step=homotopy_min_lambda_step,
            )
            if info.success:
                return StepResult(
                    success=True,
                    dt_used=dt_trial,
                    dt_reductions=dt_reductions,
                    newton_failures=newton_failures + 1,
                    residual_norm=info.residual_norm,
                    newton_iterations=info.iterations,
                    u_next=u_next,
                )

        newton_failures += 1
        dt_trial *= dt_shrink
        dt_reductions += 1
        guess = explicit_euler_guess(dynamics, u_prev, dt_trial, t_prev=t_prev, input_function=input_function)

    return StepResult(
        success=False,
        dt_used=dt_trial,
        dt_reductions=dt_reductions,
        newton_failures=newton_failures,
        residual_norm=math.inf,
        newton_iterations=max_iter,
        u_next=u_prev.copy(),
    )


def rollout_implicit_midpoint(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    t_final: float,
    dt_initial: float,
    input_function: Callable[[float], np.ndarray] | None = None,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> RolloutResult:
    dynamics.validate_state(u0, "u0")
    current_u = u0.copy()
    current_t = 0.0
    current_dt = dt_initial

    states = [current_u.copy()]
    dt_history: list[float] = []
    times = [current_t]
    accepted_steps = 0
    total_reductions = 0
    total_newton_failures = 0

    while current_t < t_final - 1e-14:
        step_dt = min(current_dt, t_final - current_t)
        step_result = solve_implicit_midpoint_step_with_retry(
            dynamics=dynamics,
            u_prev=current_u,
            dt_initial=step_dt,
            t_prev=current_t,
            input_function=input_function,
            dt_shrink=dt_shrink,
            dt_min=dt_min,
            tol=tol,
            max_iter=max_iter,
        )
        total_reductions += step_result.dt_reductions
        total_newton_failures += step_result.newton_failures

        if not step_result.success:
            return RolloutResult(
                success=False,
                accepted_steps=accepted_steps,
                dt_reductions=total_reductions,
                newton_failures=total_newton_failures,
                final_time=current_t,
                dt_history=np.asarray(dt_history, dtype=float),
                times=np.asarray(times, dtype=float),
                states=np.stack(states, axis=0),
            )

        current_u = step_result.u_next
        current_t += step_result.dt_used
        current_dt = step_result.dt_used
        accepted_steps += 1
        dt_history.append(step_result.dt_used)
        times.append(current_t)
        states.append(current_u.copy())

    return RolloutResult(
        success=True,
        accepted_steps=accepted_steps,
        dt_reductions=total_reductions,
        newton_failures=total_newton_failures,
        final_time=current_t,
        dt_history=np.asarray(dt_history, dtype=float),
        times=np.asarray(times, dtype=float),
        states=np.stack(states, axis=0),
    )


@timed("goattm.solvers.rollout_implicit_midpoint_to_observation_times")
def rollout_implicit_midpoint_to_observation_times(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    observation_times: np.ndarray,
    max_dt: float,
    input_function: Callable[[float], np.ndarray] | None = None,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> tuple[RolloutResult, np.ndarray]:
    dynamics.validate_state(u0, "u0")
    if observation_times.ndim != 1 or observation_times.shape[0] < 2:
        raise ValueError(f"observation_times must have shape (N,), N>=2, got {observation_times.shape}")
    if abs(float(observation_times[0])) > 1e-14:
        raise ValueError("observation_times must start at 0.0.")
    if np.any(np.diff(observation_times) <= 0.0):
        raise ValueError("observation_times must be strictly increasing.")
    if max_dt <= 0.0:
        raise ValueError(f"max_dt must be positive, got {max_dt}")
    if not (0.0 < dt_shrink < 1.0):
        raise ValueError(f"dt_shrink must lie in (0, 1), got {dt_shrink}")

    current_u = u0.copy()
    current_t = 0.0
    current_dt = max_dt

    states = [current_u.copy()]
    times = [current_t]
    dt_history: list[float] = []
    observation_indices = [0]
    accepted_steps = 0
    total_reductions = 0
    total_newton_failures = 0

    for obs_idx in range(1, observation_times.shape[0]):
        target_t = float(observation_times[obs_idx])

        while current_t < target_t - 1e-14:
            step_dt = min(current_dt, target_t - current_t)
            step_result = solve_implicit_midpoint_step_with_retry(
                dynamics=dynamics,
                u_prev=current_u,
                dt_initial=step_dt,
                t_prev=current_t,
                input_function=input_function,
                dt_shrink=dt_shrink,
                dt_min=min(dt_min, step_dt),
                tol=tol,
                max_iter=max_iter,
            )
            total_reductions += step_result.dt_reductions
            total_newton_failures += step_result.newton_failures

            if not step_result.success:
                return (
                    RolloutResult(
                        success=False,
                        accepted_steps=accepted_steps,
                        dt_reductions=total_reductions,
                        newton_failures=total_newton_failures,
                        final_time=current_t,
                        dt_history=np.asarray(dt_history, dtype=float),
                        times=np.asarray(times, dtype=float),
                        states=np.stack(states, axis=0),
                    ),
                    np.asarray(observation_indices, dtype=int),
                )

            current_u = step_result.u_next
            current_t += step_result.dt_used
            current_dt = min(max_dt, step_result.dt_used)
            accepted_steps += 1
            dt_history.append(step_result.dt_used)
            times.append(current_t)
            states.append(current_u.copy())

        if abs(current_t - target_t) > 1e-11:
            raise RuntimeError(
                f"Adaptive rollout failed to land on observation time {target_t}; reached {current_t} instead."
            )
        observation_indices.append(len(states) - 1)

    return (
        RolloutResult(
            success=True,
            accepted_steps=accepted_steps,
            dt_reductions=total_reductions,
            newton_failures=total_newton_failures,
            final_time=current_t,
            dt_history=np.asarray(dt_history, dtype=float),
            times=np.asarray(times, dtype=float),
            states=np.stack(states, axis=0),
        ),
        np.asarray(observation_indices, dtype=int),
    )


@timed("goattm.solvers.rollout_implicit_midpoint_tangent_from_base_rollout")
def rollout_implicit_midpoint_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    parameter_action: Callable[[np.ndarray, float], np.ndarray],
    u0_tangent: np.ndarray | None = None,
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
    if u0_tangent is not None:
        dynamics.validate_state(u0_tangent, "u0_tangent")
        tangent_states[0] = np.asarray(u0_tangent, dtype=np.float64)

    identity = np.eye(dynamics.dimension, dtype=np.float64)
    for step_idx in range(base_rollout.accepted_steps):
        dt = float(base_rollout.dt_history[step_idx])
        u_prev = base_rollout.states[step_idx]
        u_next = base_rollout.states[step_idx + 1]
        midpoint_state = 0.5 * (u_prev + u_next)
        midpoint_time = float(base_rollout.times[step_idx] + 0.5 * dt)
        jacobian = dynamics.rhs_jacobian(midpoint_state)
        lhs = identity - 0.5 * dt * jacobian
        rhs = (identity + 0.5 * dt * jacobian) @ tangent_states[step_idx]
        rhs += dt * np.asarray(parameter_action(midpoint_state, midpoint_time), dtype=np.float64)
        tangent_states[step_idx + 1] = np.linalg.solve(lhs, rhs)

    return tangent_states
