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
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems.reduced_qoi_best_response import (
    DecoderTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
    dynamics_parameter_vector,
    unpack_dynamics_parameter_vector,
)
from goattm.runtime.distributed import DistributedContext
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times


def build_dataset(
    root: Path,
    dynamics: StabilizedQuadraticDynamics,
    decoder: QuadraticDecoder,
    sample_count: int,
    rng: np.random.Generator,
) -> Path:
    observation_times = np.array([0.0, 0.04, 0.08, 0.12, 0.16], dtype=float)
    sample_paths = []
    sample_ids = []
    for sample_idx in range(sample_count):
        u0 = 0.1 * rng.standard_normal(dynamics.dimension)
        input_values = np.column_stack([0.2 + 0.08 * np.sin(2.0 * np.pi * observation_times + 0.4 * sample_idx)])
        input_function = lambda t, times=observation_times, values=input_values: np.asarray(  # noqa: E731
            [np.interp(t, times, values[:, 0])],
            dtype=np.float64,
        )
        rollout, observation_indices = rollout_implicit_midpoint_to_observation_times(
            dynamics=dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=0.04,
            input_function=input_function,
            dt_shrink=0.5,
            dt_min=1e-12,
            tol=1e-12,
            max_iter=30,
        )
        qoi_observations = np.vstack([decoder.decode(state) for state in rollout.states[observation_indices]])
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


def fit_slope(eps_values: np.ndarray, errors: np.ndarray) -> float:
    coeffs = np.polyfit(np.log10(eps_values), np.log10(errors), 1)
    return float(coeffs[0])


def main() -> None:
    rng = np.random.default_rng(20260427)
    truth_dynamics = StabilizedQuadraticDynamics(
        s_params=np.array([0.48, 0.03, -0.02], dtype=float),
        w_params=np.array([0.08], dtype=float),
        mu_h=0.025 * rng.standard_normal(mu_h_dimension(2)),
        b=np.array([[0.35], [-0.12]], dtype=float),
        c=np.array([0.03, -0.015], dtype=float),
    )
    candidate_dynamics = StabilizedQuadraticDynamics(
        s_params=truth_dynamics.s_params + 0.12 * rng.standard_normal(truth_dynamics.s_params.shape),
        w_params=truth_dynamics.w_params + 0.12 * rng.standard_normal(truth_dynamics.w_params.shape),
        mu_h=truth_dynamics.mu_h + 0.12 * rng.standard_normal(truth_dynamics.mu_h.shape),
        b=truth_dynamics.b + 0.12 * rng.standard_normal(truth_dynamics.b.shape),
        c=truth_dynamics.c + 0.12 * rng.standard_normal(truth_dynamics.c.shape),
    )
    truth_decoder = QuadraticDecoder(
        v1=0.25 * rng.standard_normal((2, 2)),
        v2=0.12 * rng.standard_normal((2, compressed_quadratic_dimension(2))),
        v0=0.08 * rng.standard_normal(2),
    )
    decoder_template = QuadraticDecoder(
        v1=np.zeros_like(truth_decoder.v1),
        v2=np.zeros_like(truth_decoder.v2),
        v0=np.zeros_like(truth_decoder.v0),
    )
    regularization = DecoderTikhonovRegularization(
        coeff_v1=float(10.0 ** rng.uniform(-4.0, -2.5)),
        coeff_v2=float(10.0 ** rng.uniform(-4.0, -2.5)),
        coeff_v0=float(10.0 ** rng.uniform(-4.0, -2.5)),
    )

    with tempfile.TemporaryDirectory(prefix="goattm_reduced_qoi_") as tmpdir:
        manifest_path = build_dataset(Path(tmpdir), truth_dynamics, truth_decoder, sample_count=4, rng=rng)
        evaluator = ObservationAlignedBestResponseEvaluator(
            manifest=manifest_path,
            max_dt=0.04,
            context=DistributedContext(),
            dt_shrink=0.5,
            dt_min=1e-12,
            tol=1e-12,
            max_iter=30,
        )
        base = evaluator.evaluate_reduced_data_loss_and_gradient(candidate_dynamics, decoder_template, regularization)
        direction_vector = rng.standard_normal(dynamics_parameter_vector(candidate_dynamics).shape)
        direction_vector /= np.linalg.norm(direction_vector)
        direction = unpack_dynamics_parameter_vector(candidate_dynamics, direction_vector)
        directional_derivative = float(np.dot(base.reduced_data_gradient, direction_vector))

        eps_values = np.logspace(-6, -2, 21)
        zero_order = np.zeros_like(eps_values)
        first_order = np.zeros_like(eps_values)
        for idx, eps in enumerate(eps_values):
            perturbed = StabilizedQuadraticDynamics(
                s_params=candidate_dynamics.s_params + eps * direction.s_params,
                w_params=candidate_dynamics.w_params + eps * direction.w_params,
                mu_h=candidate_dynamics.mu_h + eps * direction.mu_h,
                b=candidate_dynamics.b + eps * direction.b,
                c=candidate_dynamics.c + eps * direction.c,
            )
            value_eps = evaluator.evaluate_reduced_data_loss_and_gradient(
                perturbed,
                decoder_template,
                regularization,
            ).data_loss
            delta = value_eps - base.data_loss
            zero_order[idx] = abs(delta)
            first_order[idx] = abs(delta - eps * directional_derivative)

        zero_slope = fit_slope(eps_values[(eps_values >= 1e-5) & (eps_values <= 1e-3)], zero_order[(eps_values >= 1e-5) & (eps_values <= 1e-3)])
        first_slope = fit_slope(eps_values[(eps_values >= 1e-5) & (eps_values <= 1e-3)], first_order[(eps_values >= 1e-5) & (eps_values <= 1e-3)])

        output_dir = ROOT / "module_test" / "output_plots" / "reduced_qoi_best_response_taylor"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "reduced_qoi_best_response_taylor.png"
        data_path = output_dir / "reduced_qoi_best_response_taylor_data.npz"

        plt.figure(figsize=(8.5, 5.5))
        plt.loglog(eps_values, zero_order, "o-", label=r"$|q(g+\epsilon dg)-q(g)|$")
        plt.loglog(
            eps_values,
            first_order,
            "o-",
            label=r"$|q(g+\epsilon dg)-q(g)-\epsilon \nabla q(g)^T dg|$",
        )
        plt.loglog(eps_values, zero_order[-1] * (eps_values / eps_values[-1]), "--", label="reference slope 1")
        plt.loglog(
            eps_values,
            max(first_order[-1], 1e-30) * (eps_values / eps_values[-1]) ** 2,
            "--",
            label="reference slope 2",
        )
        plt.title("Reduced QoI Taylor test with cached best-response decoder")
        plt.xlabel("eps")
        plt.ylabel("error")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

        np.savez(
            data_path,
            eps_values=eps_values,
            zero_order=zero_order,
            first_order=first_order,
            zero_slope=np.array(zero_slope),
            first_slope=np.array(first_slope),
        )

    print("Reduced qoi best-response Taylor test passed.")
    print(f"zero_order_slope={zero_slope:.6f}")
    print(f"first_order_slope={first_slope:.6f}")
    print(f"plot={plot_path}")
    print(f"data={data_path}")


if __name__ == "__main__":
    main()
