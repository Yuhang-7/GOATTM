from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class NewtonInfo:
    success: bool
    iterations: int
    residual_norm: float
    backtracks: int


@dataclass
class StepInfo:
    success: bool
    dt_used: float
    dt_reductions: int
    newton_failures: int
    residual_norm: float
    newton_iterations: int
    u_next: np.ndarray


@dataclass
class RolloutInfo:
    success: bool
    accepted_steps: int
    dt_reductions: int
    newton_failures: int
    final_time: float
    states: np.ndarray


def generate_dissipative_A(r: int, rng: np.random.Generator, mu: float = 0.5, skew_scale: float = 0.15) -> np.ndarray:
    x = rng.standard_normal((r, r))
    symmetric_negative = -(x @ x.T) / r - mu * np.eye(r)

    y = rng.standard_normal((r, r))
    skew = skew_scale * (y - y.T)
    return symmetric_negative + skew


def symmetrize_last_two_axes(tensor: np.ndarray) -> np.ndarray:
    return 0.5 * (tensor + np.swapaxes(tensor, 1, 2))


def full_symmetrize(tensor: np.ndarray) -> np.ndarray:
    perms = [
        tensor,
        np.transpose(tensor, (0, 2, 1)),
        np.transpose(tensor, (1, 0, 2)),
        np.transpose(tensor, (1, 2, 0)),
        np.transpose(tensor, (2, 0, 1)),
        np.transpose(tensor, (2, 1, 0)),
    ]
    return sum(perms) / 6.0


def generate_energy_preserving_H(r: int, rng: np.random.Generator, scale: float = 0.08) -> np.ndarray:
    raw = scale * rng.standard_normal((r, r, r))
    symmetric_bilinear = symmetrize_last_two_axes(raw)
    fully_symmetric = full_symmetrize(symmetric_bilinear)
    tensor = symmetric_bilinear - fully_symmetric

    # Numerical cleanup: keep symmetry in the last two slots exact.
    return symmetrize_last_two_axes(tensor)


def quadratic_eval(h_tensor: np.ndarray, u: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
    if v is None:
        v = u
    return np.einsum("ijk,j,k->i", h_tensor, u, v)


def quadratic_linearization_matrix(h_tensor: np.ndarray, u: np.ndarray) -> np.ndarray:
    # Matrix M(u) such that M(u) @ v = H(u, v).
    return np.einsum("ijk,j->ik", h_tensor, u)


def midpoint_rhs(a: np.ndarray, h_tensor: np.ndarray, c: np.ndarray, u: np.ndarray) -> np.ndarray:
    return a @ u + quadratic_eval(h_tensor, u) + c


def cn_rhs(a: np.ndarray, h_tensor: np.ndarray, c: np.ndarray, u: np.ndarray) -> np.ndarray:
    return a @ u + quadratic_eval(h_tensor, u) + c


def midpoint_residual(a: np.ndarray, h_tensor: np.ndarray, c: np.ndarray, u_prev: np.ndarray, u_next: np.ndarray, dt: float) -> np.ndarray:
    midpoint = 0.5 * (u_prev + u_next)
    return u_next - u_prev - dt * midpoint_rhs(a, h_tensor, c, midpoint)


def midpoint_jacobian(a: np.ndarray, h_tensor: np.ndarray, u_prev: np.ndarray, u_next: np.ndarray, dt: float) -> np.ndarray:
    midpoint = 0.5 * (u_prev + u_next)
    h_matrix = quadratic_linearization_matrix(h_tensor, midpoint)
    return np.eye(u_prev.size) - 0.5 * dt * a - dt * h_matrix


def cn_residual(a: np.ndarray, h_tensor: np.ndarray, c: np.ndarray, u_prev: np.ndarray, u_next: np.ndarray, dt: float) -> np.ndarray:
    return u_next - u_prev - 0.5 * dt * (cn_rhs(a, h_tensor, c, u_prev) + cn_rhs(a, h_tensor, c, u_next))


def cn_jacobian(a: np.ndarray, h_tensor: np.ndarray, u_prev: np.ndarray, u_next: np.ndarray, dt: float) -> np.ndarray:
    h_matrix = quadratic_linearization_matrix(h_tensor, u_next)
    return np.eye(u_prev.size) - 0.5 * dt * a - dt * h_matrix


def explicit_euler_guess(a: np.ndarray, h_tensor: np.ndarray, c: np.ndarray, u_prev: np.ndarray, dt: float) -> np.ndarray:
    return u_prev + dt * midpoint_rhs(a, h_tensor, c, u_prev)


def newton_raphson_step(
    a: np.ndarray,
    h_tensor: np.ndarray,
    c: np.ndarray,
    u_prev: np.ndarray,
    dt: float,
    residual_fn,
    jacobian_fn,
    guess: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 25,
    backtrack_factor: float = 0.5,
    max_backtracks: int = 8,
) -> tuple[np.ndarray, NewtonInfo]:
    if guess is None:
        current = explicit_euler_guess(a, h_tensor, c, u_prev, dt)
    else:
        current = guess.copy()

    total_backtracks = 0
    residual = residual_fn(a, h_tensor, c, u_prev, current, dt)
    residual_norm = float(np.linalg.norm(residual))

    for iteration in range(1, max_iter + 1):
        if not np.isfinite(residual_norm):
            return current, NewtonInfo(False, iteration - 1, residual_norm, total_backtracks)
        if residual_norm <= tol:
            return current, NewtonInfo(True, iteration - 1, residual_norm, total_backtracks)

        jac = jacobian_fn(a, h_tensor, u_prev, current, dt)
        try:
            delta = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            return current, NewtonInfo(False, iteration - 1, residual_norm, total_backtracks)

        damping = 1.0
        accepted = False
        trial = current
        trial_residual = residual
        trial_norm = residual_norm
        for _ in range(max_backtracks + 1):
            trial = current + damping * delta
            trial_residual = residual_fn(a, h_tensor, c, u_prev, trial, dt)
            trial_norm = float(np.linalg.norm(trial_residual))
            if np.isfinite(trial_norm) and trial_norm < residual_norm:
                accepted = True
                break
            damping *= backtrack_factor
            total_backtracks += 1

        if not accepted:
            return current, NewtonInfo(False, iteration, residual_norm, total_backtracks)

        current = trial
        residual = trial_residual
        residual_norm = trial_norm

    return current, NewtonInfo(residual_norm <= tol, max_iter, residual_norm, total_backtracks)


def solve_step_with_retry(
    a: np.ndarray,
    h_tensor: np.ndarray,
    c: np.ndarray,
    u_prev: np.ndarray,
    dt_initial: float,
    method: str,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> StepInfo:
    if method == "midpoint":
        residual_fn = midpoint_residual
        jacobian_fn = midpoint_jacobian
    elif method == "cn":
        residual_fn = cn_residual
        jacobian_fn = cn_jacobian
    else:
        raise ValueError(f"Unsupported method: {method}")

    dt_trial = dt_initial
    dt_reductions = 0
    newton_failures = 0
    guess = explicit_euler_guess(a, h_tensor, c, u_prev, dt_trial)

    while dt_trial >= dt_min:
        u_next, info = newton_raphson_step(
            a=a,
            h_tensor=h_tensor,
            c=c,
            u_prev=u_prev,
            dt=dt_trial,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            guess=guess,
            tol=tol,
            max_iter=max_iter,
        )
        if info.success:
            return StepInfo(
                success=True,
                dt_used=dt_trial,
                dt_reductions=dt_reductions,
                newton_failures=newton_failures,
                residual_norm=info.residual_norm,
                newton_iterations=info.iterations,
                u_next=u_next,
            )

        newton_failures += 1
        dt_trial *= dt_shrink
        dt_reductions += 1
        guess = explicit_euler_guess(a, h_tensor, c, u_prev, dt_trial)

    return StepInfo(
        success=False,
        dt_used=dt_trial,
        dt_reductions=dt_reductions,
        newton_failures=newton_failures,
        residual_norm=math.inf,
        newton_iterations=max_iter,
        u_next=u_prev.copy(),
    )


def rollout_solver(
    a: np.ndarray,
    h_tensor: np.ndarray,
    c: np.ndarray,
    u0: np.ndarray,
    t_final: float,
    dt_initial: float,
    method: str,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> RolloutInfo:
    current_u = u0.copy()
    current_t = 0.0
    current_dt = dt_initial

    states = [current_u.copy()]
    accepted_steps = 0
    total_reductions = 0
    total_newton_failures = 0

    while current_t < t_final - 1e-14:
        step_dt = min(current_dt, t_final - current_t)
        step_info = solve_step_with_retry(
            a=a,
            h_tensor=h_tensor,
            c=c,
            u_prev=current_u,
            dt_initial=step_dt,
            method=method,
            dt_shrink=dt_shrink,
            dt_min=dt_min,
            tol=tol,
            max_iter=max_iter,
        )
        total_reductions += step_info.dt_reductions
        total_newton_failures += step_info.newton_failures

        if not step_info.success:
            return RolloutInfo(
                success=False,
                accepted_steps=accepted_steps,
                dt_reductions=total_reductions,
                newton_failures=total_newton_failures,
                final_time=current_t,
                states=np.stack(states, axis=0),
            )

        current_u = step_info.u_next
        current_t += step_info.dt_used
        current_dt = step_info.dt_used
        accepted_steps += 1
        states.append(current_u.copy())

    return RolloutInfo(
        success=True,
        accepted_steps=accepted_steps,
        dt_reductions=total_reductions,
        newton_failures=total_newton_failures,
        final_time=current_t,
        states=np.stack(states, axis=0),
    )


def energy_preservation_check(h_tensor: np.ndarray, u: np.ndarray) -> float:
    return float(np.dot(u, quadratic_eval(h_tensor, u)))


def summarize_method(
    method: str,
    r: int,
    rng: np.random.Generator,
    step_trials: int,
    rollout_trials: int,
    c_scale: float,
    t_final: float,
    n_time_steps: int,
) -> dict:
    step_success = 0
    step_failure = 0
    step_reductions = 0
    step_newton_failures = 0
    dt_initial = t_final / n_time_steps

    for _ in range(step_trials):
        a = generate_dissipative_A(r, rng)
        h_tensor = generate_energy_preserving_H(r, rng)
        c = c_scale * rng.standard_normal(r)
        u_prev = 0.2 * rng.standard_normal(r)

        step_info = solve_step_with_retry(
            a=a,
            h_tensor=h_tensor,
            c=c,
            u_prev=u_prev,
            dt_initial=dt_initial,
            method=method,
        )
        if step_info.success:
            step_success += 1
        else:
            step_failure += 1
        step_reductions += step_info.dt_reductions
        step_newton_failures += step_info.newton_failures

    rollout_success = 0
    rollout_failure = 0
    rollout_reductions = 0
    rollout_newton_failures = 0
    rollout_steps = 0

    for _ in range(rollout_trials):
        a = generate_dissipative_A(r, rng)
        h_tensor = generate_energy_preserving_H(r, rng)
        c = c_scale * rng.standard_normal(r)
        u0 = 0.2 * rng.standard_normal(r)

        rollout = rollout_solver(
            a=a,
            h_tensor=h_tensor,
            c=c,
            u0=u0,
            t_final=1.0,
            dt_initial=dt_initial,
            method=method,
        )
        if rollout.success:
            rollout_success += 1
        else:
            rollout_failure += 1
        rollout_reductions += rollout.dt_reductions
        rollout_newton_failures += rollout.newton_failures
        rollout_steps += rollout.accepted_steps

    return {
        "method": method,
        "r": r,
        "step_success": step_success,
        "step_failure": step_failure,
        "step_dt_reductions": step_reductions,
        "step_newton_failures": step_newton_failures,
        "rollout_success": rollout_success,
        "rollout_failure": rollout_failure,
        "rollout_dt_reductions": rollout_reductions,
        "rollout_newton_failures": rollout_newton_failures,
        "rollout_accepted_steps": rollout_steps,
    }


def main() -> int:
    seed = 20260425
    rng = np.random.default_rng(seed)

    dims = [10, 15, 20]
    methods = ["midpoint", "cn"]
    step_trials = 20
    rollout_trials = 10
    c_scale = 1.0
    t_final = 1.0
    n_time_steps = 100
    dt_initial = t_final / n_time_steps

    print("Nonlinear quadratic solver comparison tests")
    print(f"seed={seed}")
    print(f"methods={methods}")
    print(f"step_trials={step_trials}, rollout_trials={rollout_trials}")
    print(f"c_scale={c_scale}, t_final={t_final}, n_time_steps={n_time_steps}, dt_initial={dt_initial}")
    print()
    print(
        "| method | r | step solve success | step solve failure | step dt reductions | "
        "step Newton failures | rollout success | rollout failure | rollout dt reductions | "
        "rollout Newton failures | rollout accepted steps |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for method in methods:
        for r in dims:
            summary = summarize_method(method, r, rng, step_trials, rollout_trials, c_scale, t_final, n_time_steps)
            print(
                f"| {summary['method']} | {summary['r']} | {summary['step_success']} | {summary['step_failure']} | "
                f"{summary['step_dt_reductions']} | {summary['step_newton_failures']} | "
                f"{summary['rollout_success']} | {summary['rollout_failure']} | "
                f"{summary['rollout_dt_reductions']} | {summary['rollout_newton_failures']} | "
                f"{summary['rollout_accepted_steps']} |"
            )

    print()
    print("Energy-preserving spot checks:")
    for r in dims:
        h_tensor = generate_energy_preserving_H(r, rng)
        u = rng.standard_normal(r)
        err = energy_preservation_check(h_tensor, u)
        print(f"r={r}, u^T H(u,u)={err:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
