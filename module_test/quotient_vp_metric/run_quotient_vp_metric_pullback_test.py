from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension
from goattm.core.quadratic import quadratic_eval
from goattm.data.npz_dataset import build_cubic_spline_input_function
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems.reduced_qoi_best_response import (
    DecoderTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
    dynamics_parameter_vector,
    rhs_parameter_action,
    unpack_dynamics_parameter_vector,
)
from goattm.runtime.distributed import DistributedContext
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times
from goattm.train.quotient_trust_region import (
    check_vp_metric_pullback,
    quotient_vertical_basis,
)


def build_dataset(
    root: Path,
    dynamics: StabilizedQuadraticDynamics,
    decoder: QuadraticDecoder,
    sample_count: int,
    observation_times: np.ndarray,
    rng: np.random.Generator,
) -> Path:
    sample_paths: list[str] = []
    sample_ids: list[str] = []
    for sample_idx in range(sample_count):
        u0 = 0.06 * rng.standard_normal(dynamics.dimension)
        input_values = np.column_stack(
            [
                0.18
                + 0.08 * np.sin(2.0 * np.pi * observation_times + 0.25 * sample_idx)
                + 0.03 * np.cos(4.0 * np.pi * observation_times - 0.15 * sample_idx)
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
        if not rollout.success:
            raise RuntimeError("truth rollout failed while building quotient VP metric test data.")
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


def fit_loglog_slope(eps_values: np.ndarray, error_values: np.ndarray) -> float:
    mask = (error_values > 0.0) & np.isfinite(error_values)
    return float(np.polyfit(np.log10(eps_values[mask]), np.log10(error_values[mask]), 1)[0])


def assert_stabilized_vertical_basis_reshapes_correctly(
    dynamics: StabilizedQuadraticDynamics,
    rng: np.random.Generator,
) -> float:
    vertical = quotient_vertical_basis(dynamics)
    if vertical.shape[1] == 0:
        return 0.0
    max_relative_error = 0.0
    input_function = lambda _t: np.array([0.31], dtype=np.float64)  # noqa: E731
    column = 0
    for i in range(dynamics.dimension):
        for j in range(i + 1, dynamics.dimension):
            k_matrix = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
            k_matrix[i, j] = 1.0
            k_matrix[j, i] = -1.0
            delta_a = dynamics.a @ k_matrix - k_matrix @ dynamics.a
            delta_b = -k_matrix @ dynamics.b
            delta_c = -k_matrix @ dynamics.c
            direction = unpack_dynamics_parameter_vector(dynamics, vertical[:, column])
            for _ in range(8):
                state = 0.1 * rng.standard_normal(dynamics.dimension)
                action = rhs_parameter_action(dynamics, direction, state, 0.2, input_function=input_function)
                p_value = input_function(0.2)
                expected = (
                    delta_a @ state
                    + 2.0 * quadratic_eval(dynamics.h_matrix, k_matrix @ state, state)
                    - k_matrix @ quadratic_eval(dynamics.h_matrix, state)
                    + delta_b @ p_value
                    + delta_c
                )
                scale = max(1.0, float(np.linalg.norm(expected)))
                if not np.all(np.isfinite(action)):
                    raise AssertionError("vertical direction produced a non-finite RHS action.")
                max_relative_error = max(max_relative_error, float(np.linalg.norm(action - expected) / scale))
            column += 1
    return max_relative_error


def main() -> None:
    rng = np.random.default_rng(20260528)
    r = 3
    dq = 3
    truth_dynamics = StabilizedQuadraticDynamics(
        s_params=np.array([0.40, 0.04, -0.02, 0.36, 0.03, 0.31], dtype=np.float64),
        w_params=np.array([0.08, -0.04, 0.05], dtype=np.float64),
        mu_h=0.015 * rng.standard_normal(mu_h_dimension(r)),
        b=np.array([[0.22], [-0.10], [0.06]], dtype=np.float64),
        c=np.array([0.02, -0.01, 0.015], dtype=np.float64),
    )
    candidate_dynamics = StabilizedQuadraticDynamics(
        s_params=truth_dynamics.s_params + 0.05 * rng.standard_normal(truth_dynamics.s_params.shape),
        w_params=truth_dynamics.w_params + 0.05 * rng.standard_normal(truth_dynamics.w_params.shape),
        mu_h=truth_dynamics.mu_h + 0.05 * rng.standard_normal(truth_dynamics.mu_h.shape),
        b=truth_dynamics.b + 0.05 * rng.standard_normal(truth_dynamics.b.shape),
        c=truth_dynamics.c + 0.05 * rng.standard_normal(truth_dynamics.c.shape),
    )
    truth_decoder = QuadraticDecoder(
        v1=0.30 * rng.standard_normal((dq, r)),
        v2=0.10 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
        v0=0.04 * rng.standard_normal(dq),
    )
    decoder_template = QuadraticDecoder(
        v1=np.zeros_like(truth_decoder.v1),
        v2=np.zeros_like(truth_decoder.v2),
        v0=np.zeros_like(truth_decoder.v0),
    )
    regularization = DecoderTikhonovRegularization(coeff_v1=1e-5, coeff_v2=1e-5, coeff_v0=1e-5)
    observation_times = np.linspace(0.0, 0.24, 7)

    with tempfile.TemporaryDirectory(prefix="goattm_quotient_vp_metric_") as tmpdir:
        manifest_path = build_dataset(
            Path(tmpdir),
            dynamics=truth_dynamics,
            decoder=truth_decoder,
            sample_count=5,
            observation_times=observation_times,
            rng=rng,
        )
        evaluator = ObservationAlignedBestResponseEvaluator(
            manifest=manifest_path,
            max_dt=float(observation_times[1] - observation_times[0]),
            context=DistributedContext(),
            time_integrator="implicit_midpoint",
            dt_shrink=0.5,
            dt_min=1e-12,
            tol=1e-12,
            max_iter=30,
        )
        workflow = evaluator.build_reduced_objective_workflow(
            decoder_template=decoder_template,
            regularization=regularization,
        )
        prepared = workflow.prepare(candidate_dynamics)
        direction = rng.standard_normal(dynamics_parameter_vector(candidate_dynamics).shape)
        direction /= np.linalg.norm(direction)
        eps_values = np.array([2e-5, 1e-5, 5e-6, 2.5e-6], dtype=np.float64)
        relative_errors = []
        records = []
        for eps in eps_values:
            check = check_vp_metric_pullback(prepared, workflow, direction, epsilon=float(eps))
            relative_errors.append(check.relative_error)
            records.append(
                {
                    "epsilon": float(eps),
                    "predicted_quadratic": check.predicted_quadratic,
                    "finite_difference_quadratic": check.finite_difference_quadratic,
                    "relative_error": check.relative_error,
                }
            )
        relative_errors_array = np.asarray(relative_errors, dtype=np.float64)
        slope = fit_loglog_slope(eps_values, relative_errors_array)
        reshape_scale = assert_stabilized_vertical_basis_reshapes_correctly(candidate_dynamics, rng)

    summary = {
        "test": "quotient_vp_metric_pullback",
        "dynamics": "StabilizedQuadraticDynamics",
        "decoder": "QuadraticDecoder(V1V2v)",
        "decoder_tikhonov": {
            "coeff_v1": regularization.coeff_v1,
            "coeff_v2": regularization.coeff_v2,
            "coeff_v0": regularization.coeff_v0,
        },
        "records": records,
        "relative_error_slope": slope,
        "max_stabilized_vertical_action_scale": reshape_scale,
    }
    print(json.dumps(summary, indent=2))
    if relative_errors_array[-1] > 5e-3:
        raise AssertionError(f"G_vp pullback relative error too large: {relative_errors_array[-1]:.3e}")
    if slope < 0.75:
        raise AssertionError(f"G_vp pullback relative error slope too small: {slope:.3f}")


if __name__ == "__main__":
    main()
