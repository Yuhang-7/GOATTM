from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class RandomSystem:
    s: np.ndarray
    w: np.ndarray
    a: np.ndarray
    h: np.ndarray
    b: np.ndarray
    c: np.ndarray
    p_frequency: np.ndarray
    p_phase: np.ndarray


@dataclass
class RelaxedRollout:
    success: bool
    times: np.ndarray
    states: np.ndarray
    gammas: np.ndarray
    dt_history: np.ndarray
    rejected_steps: int
    failed_reason: str


def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.ravel()))


def rescale_to_fro_norm(x: np.ndarray, target_norm: float) -> np.ndarray:
    norm = fro_norm(x)
    if norm == 0.0:
        raise ValueError("Cannot rescale a zero array.")
    return x * (target_norm / norm)


def symmetrize_last_two_axes(tensor: np.ndarray) -> np.ndarray:
    return 0.5 * (tensor + np.swapaxes(tensor, 1, 2))


def full_symmetrize(tensor: np.ndarray) -> np.ndarray:
    return (
        tensor
        + np.transpose(tensor, (0, 2, 1))
        + np.transpose(tensor, (1, 0, 2))
        + np.transpose(tensor, (1, 2, 0))
        + np.transpose(tensor, (2, 0, 1))
        + np.transpose(tensor, (2, 1, 0))
    ) / 6.0


def generate_energy_preserving_h(n: int, rng: np.random.Generator, target_norm: float) -> np.ndarray:
    raw = rng.standard_normal((n, n, n))
    symmetric_bilinear = symmetrize_last_two_axes(raw)
    tensor = symmetric_bilinear - full_symmetrize(symmetric_bilinear)
    tensor = symmetrize_last_two_axes(tensor)
    return rescale_to_fro_norm(tensor, target_norm)


def generate_system(
    n: int,
    r: int,
    rng: np.random.Generator,
    target_norm: float,
) -> RandomSystem:
    s0 = rng.standard_normal((n, n))
    y = rng.standard_normal((n, n))
    w0 = y - y.T
    a0 = -(s0 @ s0.T) + w0
    a_scale = target_norm / fro_norm(a0)

    s = np.sqrt(a_scale) * s0
    w = a_scale * w0
    a = -(s @ s.T) + w
    h = generate_energy_preserving_h(n, rng, target_norm=target_norm)
    b = rescale_to_fro_norm(rng.standard_normal((n, r)), target_norm)
    c = rescale_to_fro_norm(rng.standard_normal(n), target_norm)

    p_frequency = rng.uniform(0.5, 2.0, size=r)
    p_phase = rng.uniform(-np.pi, np.pi, size=r)
    return RandomSystem(s=s, w=w, a=a, h=h, b=b, c=c, p_frequency=p_frequency, p_phase=p_phase)


def p_value(system: RandomSystem, t: float) -> np.ndarray:
    return np.sin(system.p_frequency * t + system.p_phase)


def h_eval(h: np.ndarray, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if y is None:
        y = x
    return np.einsum("ijk,j,k->i", h, x, y, optimize=True)


def h_matrix(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("ijk,j->ik", h, x, optimize=True)


def rhs(system: RandomSystem, t: float, x: np.ndarray) -> np.ndarray:
    return system.a @ x + h_eval(system.h, x) + system.b @ p_value(system, t) + system.c


def energy(x: np.ndarray) -> float:
    return 0.5 * float(np.dot(x, x))


def target_energy_increment(system: RandomSystem, t: float, dt: float, x0: np.ndarray, x_star: np.ndarray) -> float:
    s0 = system.s.T @ x0
    s1 = system.s.T @ x_star
    force0 = system.b @ p_value(system, t) + system.c
    force1 = system.b @ p_value(system, t + dt) + system.c
    return 0.5 * dt * (
        -float(np.dot(s0, s0))
        - float(np.dot(s1, s1))
        + float(np.dot(x0, force0))
        + float(np.dot(x_star, force1))
    )


def relaxed_step(system: RandomSystem, t: float, x: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
    n = x.size
    identity = np.eye(n)
    d_matrix = system.s @ system.s.T - system.w
    m = h_matrix(system.h, x)
    left = identity + 0.5 * dt * d_matrix - 0.5 * dt * m
    right_matrix = identity - 0.5 * dt * d_matrix + 0.5 * dt * m
    right = right_matrix @ x + dt * (system.b @ p_value(system, t + 0.5 * dt) + system.c)
    x_star = np.linalg.solve(left, right)

    d = x_star - x
    d_norm_sq = float(np.dot(d, d))
    if d_norm_sq == 0.0:
        return x_star, 1.0

    delta_target = target_energy_increment(system, t, dt, x, x_star)
    gamma = 1.0 + 2.0 * (delta_target - (energy(x_star) - energy(x))) / d_norm_sq
    x_next = x + gamma * d
    return x_next, float(gamma)


def rollout_relaxed(
    system: RandomSystem,
    x0: np.ndarray,
    t_final: float,
    dt_initial: float,
    dt_max: float,
    dt_min: float,
    gamma_tol: float,
    shrink: float,
    grow: float,
    grow_after: int,
    max_attempts: int,
    terminal_tol: float,
    max_accepted_steps: int,
) -> RelaxedRollout:
    t = 0.0
    x = x0.copy()
    dt = dt_initial
    ok_streak = 0
    rejected = 0
    times = [t]
    states = [x.copy()]
    gammas: list[float] = []
    dts: list[float] = []

    while t < t_final - terminal_tol:
        if len(gammas) >= max_accepted_steps:
            return RelaxedRollout(False, np.array(times), np.stack(states), np.array(gammas), np.array(dts), rejected, "max_accepted_steps")
        accepted = False
        for _ in range(max_attempts):
            step_dt = min(dt, t_final - t)
            try:
                x_trial, gamma = relaxed_step(system, t, x, step_dt)
            except np.linalg.LinAlgError:
                gamma = np.nan
                x_trial = x

            if np.isfinite(gamma) and gamma > 0.0 and abs(gamma - 1.0) <= gamma_tol and np.all(np.isfinite(x_trial)):
                t += gamma * step_dt
                x = x_trial
                times.append(t)
                states.append(x.copy())
                gammas.append(gamma)
                dts.append(step_dt)
                ok_streak += 1
                if ok_streak >= grow_after:
                    dt = min(grow * dt, dt_max)
                    ok_streak = 0
                accepted = True
                break

            rejected += 1
            ok_streak = 0
            dt *= shrink
            if dt < dt_min:
                return RelaxedRollout(False, np.array(times), np.stack(states), np.array(gammas), np.array(dts), rejected, "dt_below_min")

        if not accepted:
            return RelaxedRollout(False, np.array(times), np.stack(states), np.array(gammas), np.array(dts), rejected, "max_attempts")

    return RelaxedRollout(True, np.array(times), np.stack(states), np.array(gammas), np.array(dts), rejected, "")


def solve_rk45_reference(
    system: RandomSystem,
    x0: np.ndarray,
    t_end: float,
    max_step: float,
    rtol: float,
    atol: float,
):
    return solve_ivp(
        fun=lambda t, x: rhs(system, t, x),
        t_span=(0.0, t_end),
        y0=x0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
    )


def run_trial(args: argparse.Namespace, rng: np.random.Generator, trial: int) -> dict[str, float | int | str | bool]:
    system = generate_system(args.dimension, args.input_dimension, rng, args.target_norm)
    x0 = args.x0_scale * rng.standard_normal(args.dimension)
    dt_initial = min(args.dt_max, args.t_final / args.initial_steps)

    relaxed = rollout_relaxed(
        system=system,
        x0=x0,
        t_final=args.t_final,
        dt_initial=dt_initial,
        dt_max=args.dt_max,
        dt_min=args.dt_min,
        gamma_tol=args.gamma_tol,
        shrink=args.shrink,
        grow=args.grow,
        grow_after=args.grow_after,
        max_attempts=args.max_attempts,
        terminal_tol=args.terminal_tol,
        max_accepted_steps=args.max_accepted_steps,
    )

    t_end = max(args.t_final, float(relaxed.times[-1]))
    reference = solve_rk45_reference(
        system=system,
        x0=x0,
        t_end=t_end,
        max_step=args.rk45_max_step,
        rtol=args.rk45_rtol,
        atol=args.rk45_atol,
    )

    if relaxed.success and reference.success:
        ref_states = reference.sol(relaxed.times).T
        errors = np.linalg.norm(relaxed.states - ref_states, axis=1)
        ref_norms = np.maximum(np.linalg.norm(ref_states, axis=1), 1.0)
        rel_errors = errors / ref_norms
        max_abs_error = float(np.max(errors))
        final_abs_error = float(errors[-1])
        max_rel_error = float(np.max(rel_errors))
        final_rel_error = float(rel_errors[-1])
    else:
        max_abs_error = np.nan
        final_abs_error = np.nan
        max_rel_error = np.nan
        final_rel_error = np.nan

    h_energy_probe = max(abs(float(np.dot(v, h_eval(system.h, v)))) for v in rng.standard_normal((8, args.dimension)))
    return {
        "trial": trial,
        "success": bool(relaxed.success and reference.success),
        "relaxed_success": bool(relaxed.success),
        "rk45_success": bool(reference.success),
        "failed_reason": relaxed.failed_reason,
        "a_fro": fro_norm(system.a),
        "h_fro": fro_norm(system.h),
        "b_fro": fro_norm(system.b),
        "c_norm": float(np.linalg.norm(system.c)),
        "h_energy_probe_max_abs": h_energy_probe,
        "steps": int(max(relaxed.times.size - 1, 0)),
        "rejected_steps": int(relaxed.rejected_steps),
        "final_time": float(relaxed.times[-1]),
        "gamma_min": float(np.min(relaxed.gammas)) if relaxed.gammas.size else np.nan,
        "gamma_max": float(np.max(relaxed.gammas)) if relaxed.gammas.size else np.nan,
        "dt_min_used": float(np.min(relaxed.dt_history)) if relaxed.dt_history.size else np.nan,
        "dt_max_used": float(np.max(relaxed.dt_history)) if relaxed.dt_history.size else np.nan,
        "rk45_nfev": int(reference.nfev),
        "max_abs_error": max_abs_error,
        "final_abs_error": final_abs_error,
        "max_rel_error": max_rel_error,
        "final_rel_error": final_rel_error,
    }


def print_markdown_table(rows: list[dict[str, float | int | str | bool]]) -> None:
    print("| trial | ok | ||A||F | ||H||F | ||B||F | ||c|| | steps | rejected | gamma range | max abs err | final abs err | max rel err |")
    print("|---:|:---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|")
    for row in rows:
        gamma_range = f"{row['gamma_min']:.4g}..{row['gamma_max']:.4g}"
        print(
            f"| {row['trial']} | {str(row['success'])} | {row['a_fro']:.3g} | {row['h_fro']:.3g} | "
            f"{row['b_fro']:.3g} | {row['c_norm']:.3g} | {row['steps']} | {row['rejected_steps']} | "
            f"{gamma_range} | {row['max_abs_error']:.3e} | {row['final_abs_error']:.3e} | {row['max_rel_error']:.3e} |"
        )


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def rollout_error_summary(system: RandomSystem, relaxed: RelaxedRollout, reference) -> dict[str, float]:
    if relaxed.times.size == 0 or not reference.success:
        return {
            "max_abs_error": np.nan,
            "final_abs_error": np.nan,
            "max_rel_error": np.nan,
            "final_rel_error": np.nan,
        }
    ref_states = reference.sol(relaxed.times).T
    errors = np.linalg.norm(relaxed.states - ref_states, axis=1)
    ref_norms = np.maximum(np.linalg.norm(ref_states, axis=1), 1.0)
    rel_errors = errors / ref_norms
    return {
        "max_abs_error": float(np.max(errors)),
        "final_abs_error": float(errors[-1]),
        "max_rel_error": float(np.max(rel_errors)),
        "final_rel_error": float(rel_errors[-1]),
    }


def make_rollout_for_dt(args: argparse.Namespace, system: RandomSystem, x0: np.ndarray, dt: float, gamma_tol: float) -> RelaxedRollout:
    return rollout_relaxed(
        system=system,
        x0=x0,
        t_final=args.t_final,
        dt_initial=dt,
        dt_max=dt,
        dt_min=args.dt_min,
        gamma_tol=gamma_tol,
        shrink=args.shrink,
        grow=1.0,
        grow_after=10**9,
        max_attempts=args.max_attempts,
        terminal_tol=args.terminal_tol,
        max_accepted_steps=args.max_accepted_steps,
    )


def run_dt_sweep(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    system = generate_system(args.dimension, args.input_dimension, rng, args.target_norm)
    x0 = args.x0_scale * rng.standard_normal(args.dimension)
    dt_values = parse_float_list(args.dt_sweep)

    rows: list[dict[str, float | int | str | bool]] = []
    rollouts: dict[tuple[str, float], RelaxedRollout] = {}
    for controller, gamma_tol in (("controlled", args.gamma_tol), ("forced", args.forced_gamma_tol)):
        for dt in dt_values:
            relaxed = make_rollout_for_dt(args, system, x0, dt, gamma_tol)
            rollouts[(controller, dt)] = relaxed
    reference_t_end = max(args.t_final, max(float(rollout.times[-1]) for rollout in rollouts.values()))
    reference = solve_rk45_reference(
        system=system,
        x0=x0,
        t_end=reference_t_end,
        max_step=args.rk45_max_step,
        rtol=args.rk45_rtol,
        atol=args.rk45_atol,
    )

    for controller, gamma_tol in (("controlled", args.gamma_tol), ("forced", args.forced_gamma_tol)):
        for dt in dt_values:
            relaxed = rollouts[(controller, dt)]
            errors = rollout_error_summary(system, relaxed, reference)
            rows.append(
                {
                    "controller": controller,
                    "requested_dt": dt,
                    "success": bool(relaxed.success and reference.success),
                    "failed_reason": relaxed.failed_reason,
                    "steps": int(max(relaxed.times.size - 1, 0)),
                    "rejected_steps": int(relaxed.rejected_steps),
                    "final_time": float(relaxed.times[-1]),
                    "gamma_min": float(np.min(relaxed.gammas)) if relaxed.gammas.size else np.nan,
                    "gamma_max": float(np.max(relaxed.gammas)) if relaxed.gammas.size else np.nan,
                    "dt_min_used": float(np.min(relaxed.dt_history)) if relaxed.dt_history.size else np.nan,
                    "dt_max_used": float(np.max(relaxed.dt_history)) if relaxed.dt_history.size else np.nan,
                    **errors,
                }
            )

    print("Step-size sweep against tight RK45 reference")
    print(
        f"seed={args.seed}, dimension={args.dimension}, target_norm={args.target_norm}, "
        f"||A||F={fro_norm(system.a):.3g}, ||H||F={fro_norm(system.h):.3g}, "
        f"||B||F={fro_norm(system.b):.3g}, ||c||={np.linalg.norm(system.c):.3g}"
    )
    print("| controller | requested dt | ok | steps | rejected | gamma range | final abs err | max abs err | final rel err |")
    print("|:---|---:|:---:|---:|---:|:---|---:|---:|---:|")
    for row in rows:
        gamma_range = f"{row['gamma_min']:.4g}..{row['gamma_max']:.4g}"
        print(
            f"| {row['controller']} | {row['requested_dt']:.3g} | {row['success']} | {row['steps']} | "
            f"{row['rejected_steps']} | {gamma_range} | {row['final_abs_error']:.3e} | "
            f"{row['max_abs_error']:.3e} | {row['final_rel_error']:.3e} |"
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    controlled = [row for row in rows if row["controller"] == "controlled" and np.isfinite(float(row["final_abs_error"]))]
    forced = [row for row in rows if row["controller"] == "forced" and np.isfinite(float(row["final_abs_error"]))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    for label, subset, marker in (("controlled", controlled, "o"), ("forced", forced, "s")):
        if subset:
            xs = np.array([float(row["requested_dt"]) for row in subset])
            ys = np.array([float(row["final_abs_error"]) for row in subset])
            order = np.argsort(xs)
            ax.loglog(xs[order], ys[order], marker=marker, label=label)
    ax.set_xlabel("requested dt")
    ax.set_ylabel("final absolute error vs RK45")
    ax.set_title("Convergence / large-step error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    t_plot = np.linspace(0.0, args.t_final, 800)
    ref_plot = reference.sol(t_plot).T
    ax.plot(t_plot, np.linalg.norm(ref_plot, axis=1), "k-", linewidth=2, label="RK45 reference")
    selected_dts = [dt_values[0], dt_values[len(dt_values) // 2], dt_values[-1]]
    for dt in selected_dts:
        for controller, linestyle in (("controlled", "-"), ("forced", "--")):
            relaxed = rollouts[(controller, dt)]
            if relaxed.times.size > 1:
                ax.plot(relaxed.times, np.linalg.norm(relaxed.states, axis=1), linestyle, marker=".", markersize=3, label=f"{controller} dt={dt:g}")
    ax.set_xlabel("time")
    ax.set_ylabel("state norm")
    ax.set_title("Trajectory norm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for dt in selected_dts:
        for controller, linestyle in (("controlled", "-"), ("forced", "--")):
            relaxed = rollouts[(controller, dt)]
            if relaxed.times.size > 1:
                ref_states = reference.sol(relaxed.times).T
                err = np.linalg.norm(relaxed.states - ref_states, axis=1)
                ax.semilogy(relaxed.times, np.maximum(err, 1e-16), linestyle, marker=".", markersize=3, label=f"{controller} dt={dt:g}")
    ax.set_xlabel("time")
    ax.set_ylabel("absolute error")
    ax.set_title("Error along accepted times")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for dt in selected_dts:
        for controller, linestyle in (("controlled", "-"), ("forced", "--")):
            relaxed = rollouts[(controller, dt)]
            if relaxed.gammas.size:
                ax.plot(relaxed.times[1:], relaxed.gammas, linestyle, marker=".", markersize=3, label=f"{controller} dt={dt:g}")
    ax.axhline(1.0, color="k", linewidth=1)
    ax.axhline(1.0 + args.gamma_tol, color="gray", linewidth=1, linestyle=":")
    ax.axhline(1.0 - args.gamma_tol, color="gray", linewidth=1, linestyle=":")
    ax.set_xlabel("time")
    ax.set_ylabel("gamma")
    ax.set_title("Relaxation factor")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_out, dpi=180)
    print(f"\nWrote CSV summary to {args.csv_out}")
    print(f"Wrote visualization to {args.plot_out}")

    return 0 if reference.success and controlled else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the relaxed linearly-implicit solver against tight RK45 references.")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--input-dimension", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--target-norm", type=float, default=100.0)
    parser.add_argument("--x0-scale", type=float, default=0.1)
    parser.add_argument("--t-final", type=float, default=0.03)
    parser.add_argument("--dt-max", type=float, default=5e-4)
    parser.add_argument("--initial-steps", type=int, default=100)
    parser.add_argument("--dt-min", type=float, default=1e-10)
    parser.add_argument("--gamma-tol", type=float, default=0.2)
    parser.add_argument("--shrink", type=float, default=0.5)
    parser.add_argument("--grow", type=float, default=1.2)
    parser.add_argument("--grow-after", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--terminal-tol", type=float, default=1e-8)
    parser.add_argument("--max-accepted-steps", type=int, default=100000)
    parser.add_argument("--rk45-max-step", type=float, default=5e-6)
    parser.add_argument("--rk45-rtol", type=float, default=1e-10)
    parser.add_argument("--rk45-atol", type=float, default=1e-12)
    parser.add_argument("--csv-out", type=Path, default=Path("module_test/implicit_explicit_solver/relaxed_solver_rk45_validation.csv"))
    parser.add_argument("--dt-sweep", type=str, default="4e-3,2e-3,1e-3,5e-4,2.5e-4,1.25e-4")
    parser.add_argument("--forced-gamma-tol", type=float, default=5.0)
    parser.add_argument("--plot-out", type=Path, default=Path("module_test/implicit_explicit_solver/relaxed_solver_rk45_validation.png"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dt_sweep:
        return run_dt_sweep(args)

    rng = np.random.default_rng(args.seed)
    rows = [run_trial(args, rng, trial) for trial in range(args.trials)]

    print("Relaxed linearly-implicit solver vs tight RK45 reference")
    print(
        f"seed={args.seed}, dimension={args.dimension}, target_norm={args.target_norm}, "
        f"t_final={args.t_final}, dt_max={args.dt_max}, rk45_max_step={args.rk45_max_step}"
    )
    print_markdown_table(rows)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote CSV summary to {args.csv_out}")

    return 0 if all(bool(row["success"]) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
