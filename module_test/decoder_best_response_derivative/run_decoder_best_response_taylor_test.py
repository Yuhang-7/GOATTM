from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension
from goattm.data.npz_dataset import build_cubic_spline_input_function, load_npz_sample_manifest
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems.decoder_normal_equation import (
    DecoderTikhonovRegularization,
    assemble_decoder_normal_equation_from_npz_dataset,
    decoder_parameter_matrix,
    update_decoder_from_normal_equation,
)
from goattm.problems.qoi_dataset_problem import evaluate_npz_qoi_dataset_loss_and_gradients
from goattm.runtime.distributed import DistributedContext
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times


def stack_decoder_gradient_matrix(decoder_gradients: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack(
        [
            decoder_gradients["v1"].T.astype(np.float64, copy=False),
            decoder_gradients["v2"].T.astype(np.float64, copy=False),
            decoder_gradients["v0"].reshape(1, -1).astype(np.float64, copy=False),
        ]
    )


def perturb_dynamics(
    dynamics: StabilizedQuadraticDynamics,
    direction: dict[str, np.ndarray],
    step: float,
) -> StabilizedQuadraticDynamics:
    return StabilizedQuadraticDynamics(
        s_params=dynamics.s_params + step * direction["s_params"],
        w_params=dynamics.w_params + step * direction["w_params"],
        mu_h=dynamics.mu_h + step * direction["mu_h"],
        b=None if dynamics.b is None else dynamics.b + step * direction["b"],
        c=dynamics.c + step * direction["c"],
    )


def random_direction(dynamics: StabilizedQuadraticDynamics, rng: np.random.Generator) -> dict[str, np.ndarray]:
    direction = {
        "s_params": rng.standard_normal(dynamics.s_params.shape),
        "w_params": rng.standard_normal(dynamics.w_params.shape),
        "mu_h": rng.standard_normal(dynamics.mu_h.shape),
        "c": rng.standard_normal(dynamics.c.shape),
    }
    if dynamics.b is not None:
        direction["b"] = rng.standard_normal(dynamics.b.shape)
    norm = np.sqrt(sum(float(np.sum(value**2)) for value in direction.values()))
    return {key: value / norm for key, value in direction.items()}


def build_dataset(
    root: Path,
    dynamics: StabilizedQuadraticDynamics,
    truth_decoder: QuadraticDecoder,
    sample_count: int,
    observation_times: np.ndarray,
    rng: np.random.Generator,
) -> Path:
    sample_paths: list[str] = []
    sample_ids: list[str] = []

    for sample_idx in range(sample_count):
        u0 = 0.08 * rng.standard_normal(dynamics.dimension)
        input_values = np.column_stack(
            [
                0.2
                + 0.1 * np.sin(2.0 * np.pi * observation_times + 0.3 * sample_idx)
                + 0.05 * np.cos(4.0 * np.pi * observation_times - 0.2 * sample_idx)
            ]
        )
        input_function = build_cubic_spline_input_function(observation_times, input_values)
        rollout, observation_indices = rollout_implicit_midpoint_to_observation_times(
            dynamics=dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=float(observation_times[1] - observation_times[0]),
            input_function=input_function,
            dt_shrink=0.5,
            dt_min=1e-12,
            tol=1e-12,
            max_iter=30,
        )
        states = rollout.states[observation_indices]
        qoi_observations = np.vstack([truth_decoder.decode(state) for state in states])
        sample_path = root / f"sample_{sample_idx}.npz"
        np.savez(
            sample_path,
            sample_id=np.array(f"sample-{sample_idx}"),
            observation_times=observation_times,
            u0=u0,
            qoi_observations=qoi_observations,
            input_times=observation_times,
            input_values=input_values,
        )
        sample_paths.append(sample_path.name)
        sample_ids.append(f"sample-{sample_idx}")

    manifest_path = root / "manifest.npz"
    np.savez(
        manifest_path,
        sample_paths=np.asarray(sample_paths, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
    )
    return manifest_path


def total_decoder_gradient_matrix(
    dynamics: StabilizedQuadraticDynamics,
    decoder: QuadraticDecoder,
    manifest_path: Path,
    max_dt: float,
    regularization: DecoderTikhonovRegularization,
) -> tuple[np.ndarray, object]:
    result = evaluate_npz_qoi_dataset_loss_and_gradients(
        dynamics=dynamics,
        decoder=decoder,
        manifest=manifest_path,
        max_dt=max_dt,
        context=DistributedContext(),
        dt_shrink=0.5,
        dt_min=1e-12,
        tol=1e-12,
        max_iter=30,
    )
    data_grad = stack_decoder_gradient_matrix(result.decoder_gradients)
    total_grad = data_grad + regularization.diagonal(dynamics.dimension)[:, None] * decoder_parameter_matrix(decoder)
    return total_grad, result


def fit_loglog_slope(eps_values: np.ndarray, error_values: np.ndarray, eps_min: float, eps_max: float) -> float:
    mask = (eps_values >= eps_min) & (eps_values <= eps_max) & (error_values > 0.0)
    x = np.log10(eps_values[mask])
    y = np.log10(error_values[mask])
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    rng = np.random.default_rng(20260427)
    r = 3
    dq = 2
    observation_times = np.linspace(0.0, 0.24, 7, dtype=np.float64)
    max_dt = float(observation_times[1] - observation_times[0])

    base_dynamics = StabilizedQuadraticDynamics(
        s_params=np.array([0.45, 0.05, 0.02, 0.38, -0.01, 0.28], dtype=np.float64),
        w_params=np.array([0.08, -0.04, 0.06], dtype=np.float64),
        mu_h=0.03 * rng.standard_normal(mu_h_dimension(r)),
        b=np.array([[0.35], [-0.12], [0.18]], dtype=np.float64),
        c=np.array([0.04, -0.015, 0.02], dtype=np.float64),
    )
    truth_decoder = QuadraticDecoder(
        v1=0.25 * rng.standard_normal((dq, r)),
        v2=0.15 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
        v0=0.1 * rng.standard_normal(dq),
    )
    template_decoder = QuadraticDecoder(
        v1=np.zeros_like(truth_decoder.v1),
        v2=np.zeros_like(truth_decoder.v2),
        v0=np.zeros_like(truth_decoder.v0),
    )
    regularization = DecoderTikhonovRegularization(
        coeff_v1=float(10.0 ** rng.uniform(-4.5, -3.0)),
        coeff_v2=float(10.0 ** rng.uniform(-4.5, -3.0)),
        coeff_v0=float(10.0 ** rng.uniform(-4.5, -3.0)),
    )
    direction = random_direction(base_dynamics, rng)
    fd_step = 1e-6

    with tempfile.TemporaryDirectory(prefix="goattm_best_response_") as tmpdir:
        manifest_path = build_dataset(
            root=Path(tmpdir),
            dynamics=base_dynamics,
            truth_decoder=truth_decoder,
            sample_count=6,
            observation_times=observation_times,
            rng=rng,
        )
        manifest = load_npz_sample_manifest(manifest_path)

        base_result = update_decoder_from_normal_equation(
            dynamics=base_dynamics,
            decoder_template=template_decoder,
            manifest=manifest,
            max_dt=max_dt,
            regularization=regularization,
            context=DistributedContext(),
            dt_shrink=0.5,
            dt_min=1e-12,
            tol=1e-12,
            max_iter=30,
        )
        x0 = decoder_parameter_matrix(base_result.decoder)
        h_ff = base_result.system.regularized_global_normal_matrix

        grad_plus, plus_eval = total_decoder_gradient_matrix(
            dynamics=perturb_dynamics(base_dynamics, direction, fd_step),
            decoder=base_result.decoder,
            manifest_path=manifest_path,
            max_dt=max_dt,
            regularization=regularization,
        )
        grad_minus, minus_eval = total_decoder_gradient_matrix(
            dynamics=perturb_dynamics(base_dynamics, direction, -fd_step),
            decoder=base_result.decoder,
            manifest_path=manifest_path,
            max_dt=max_dt,
            regularization=regularization,
        )
        for dataset_eval in (plus_eval, minus_eval):
            if any(item.rollout.dt_reductions != 0 for item in dataset_eval.local_sample_results):
                raise RuntimeError("Taylor test expects a fixed accepted-step pattern, but dt reduction occurred.")

        h_fg_action = (grad_plus - grad_minus) / (2.0 * fd_step)
        dx = -np.linalg.solve(h_ff, h_fg_action)

        eps_values = np.logspace(-8, -2, 25)
        zero_order_errors = np.zeros_like(eps_values)
        first_order_errors = np.zeros_like(eps_values)

        for idx, eps in enumerate(eps_values):
            perturbed_result = update_decoder_from_normal_equation(
                dynamics=perturb_dynamics(base_dynamics, direction, eps),
                decoder_template=template_decoder,
                manifest=manifest,
                max_dt=max_dt,
                regularization=regularization,
                context=DistributedContext(),
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter=30,
            )
            x_eps = decoder_parameter_matrix(perturbed_result.decoder)
            zero_order_errors[idx] = np.linalg.norm(x_eps - x0)
            first_order_errors[idx] = np.linalg.norm(x_eps - x0 - eps * dx)

        zero_slope = fit_loglog_slope(eps_values, zero_order_errors, 1e-6, 1e-3)
        first_slope = fit_loglog_slope(eps_values, first_order_errors, 1e-6, 1e-3)

        if not (0.85 <= zero_slope <= 1.15):
            raise AssertionError(f"Zero-order Taylor slope should be about 1, got {zero_slope:.3f}")
        if not (1.70 <= first_slope <= 2.30):
            raise AssertionError(f"First-order Taylor slope should be about 2, got {first_slope:.3f}")

        output_dir = ROOT / "module_test" / "output_plots" / "decoder_best_response_taylor"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "decoder_best_response_taylor.png"
        data_path = output_dir / "decoder_best_response_taylor_data.npz"

        plt.figure(figsize=(8.5, 5.5))
        plt.loglog(eps_values, zero_order_errors, "o-", label=r"$\|f^*(g+\epsilon dg)-f^*(g)\|$")
        plt.loglog(
            eps_values,
            first_order_errors,
            "o-",
            label=r"$\|f^*(g+\epsilon dg)-f^*(g)-\epsilon Df^*(g)[dg]\|$",
        )
        plt.loglog(eps_values, zero_order_errors[-1] * (eps_values / eps_values[-1]), "--", label="reference slope 1")
        plt.loglog(
            eps_values,
            max(first_order_errors[-1], 1e-30) * (eps_values / eps_values[-1]) ** 2,
            "--",
            label="reference slope 2",
        )
        plt.xlabel("eps")
        plt.ylabel("error norm")
        plt.title("Decoder best-response Taylor test via Schur-complement derivative")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

        np.savez(
            data_path,
            eps_values=eps_values,
            zero_order_errors=zero_order_errors,
            first_order_errors=first_order_errors,
            zero_slope=np.array(zero_slope),
            first_slope=np.array(first_slope),
            fd_step=np.array(fd_step),
            regularization=np.array(
                [regularization.coeff_v1, regularization.coeff_v2, regularization.coeff_v0],
                dtype=np.float64,
            ),
        )

    print("Decoder best-response derivative Taylor test passed.")
    print(f"zero_order_slope={zero_slope:.6f}")
    print(f"first_order_slope={first_slope:.6f}")
    print(f"plot={plot_path}")
    print(f"data={data_path}")


if __name__ == "__main__":
    main()
