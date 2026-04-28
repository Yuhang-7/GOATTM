from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension  # noqa: E402
from goattm.data import load_npz_sample_manifest  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.problems import (  # noqa: E402
    DecoderTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
    dynamics_from_parameter_vector,
    dynamics_parameter_vector,
)
from goattm.runtime import FunctionTimer, use_function_timer  # noqa: E402
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times  # noqa: E402


PHASE_ALIASES = {
    "forward_solve": "goattm.solvers.rollout_implicit_midpoint_to_observation_times",
    "adjoint_solve": "goattm.losses.compute_midpoint_discrete_adjoint",
    "normal_solve": "goattm.problems.solve_decoder_linear_system",
    "incremental_forward_solve": "goattm.solvers.rollout_implicit_midpoint_tangent_from_base_rollout",
    "incremental_normal_solve": "goattm.problems.solve_decoder_best_response_action_matrix",
    "reduced_gradient_total": "goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_reduced_objective_and_gradient",
}


def build_dataset(root: Path, sample_count: int = 100) -> tuple[Path, StabilizedQuadraticDynamics, QuadraticDecoder]:
    rng = np.random.default_rng(33017)
    truth_dynamics = StabilizedQuadraticDynamics(
        s_params=np.array([0.46, -0.07, 0.035], dtype=float),
        w_params=np.array([0.11], dtype=float),
        mu_h=0.03 * rng.standard_normal(mu_h_dimension(2)),
        b=np.array([[0.33], [-0.09]], dtype=float),
        c=np.array([0.025, -0.012], dtype=float),
    )
    truth_decoder = QuadraticDecoder(
        v1=0.22 * rng.standard_normal((2, 2)),
        v2=0.10 * rng.standard_normal((2, compressed_quadratic_dimension(2))),
        v0=0.05 * rng.standard_normal(2),
    )
    observation_times = np.array([0.0, 0.04, 0.08, 0.12, 0.16], dtype=float)
    sample_paths: list[str] = []
    sample_ids: list[str] = []
    for sample_idx in range(sample_count):
        u0 = 0.1 * rng.standard_normal(truth_dynamics.dimension)
        input_values = np.column_stack([0.18 + 0.10 * np.sin(2.0 * np.pi * observation_times + 0.07 * sample_idx)])
        input_function = lambda t, times=observation_times, values=input_values: np.asarray(  # noqa: E731
            [np.interp(t, times, values[:, 0])],
            dtype=np.float64,
        )
        rollout, observation_indices = rollout_implicit_midpoint_to_observation_times(
            dynamics=truth_dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=0.04,
            input_function=input_function,
            dt_shrink=0.5,
            dt_min=1e-12,
            tol=1e-12,
            max_iter=30,
        )
        qoi_observations = np.vstack([truth_decoder.decode(state) for state in rollout.states[observation_indices]])
        sample_path = root / f"sample_{sample_idx:03d}.npz"
        np.savez(
            sample_path,
            sample_id=np.array(f"sample-{sample_idx:03d}"),
            observation_times=observation_times,
            u0=u0,
            qoi_observations=qoi_observations,
            input_times=observation_times,
            input_values=input_values,
        )
        sample_paths.append(sample_path.name)
        sample_ids.append(f"sample-{sample_idx:03d}")

    manifest_path = root / "manifest.npz"
    np.savez(
        manifest_path,
        sample_paths=np.asarray(sample_paths, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
    )
    return manifest_path, truth_dynamics, truth_decoder


def build_initial_dynamics(truth_dynamics: StabilizedQuadraticDynamics) -> StabilizedQuadraticDynamics:
    rng = np.random.default_rng(33018)
    return StabilizedQuadraticDynamics(
        s_params=truth_dynamics.s_params + 0.05 * rng.standard_normal(truth_dynamics.s_params.shape),
        w_params=truth_dynamics.w_params + 0.05 * rng.standard_normal(truth_dynamics.w_params.shape),
        mu_h=truth_dynamics.mu_h + 0.04 * rng.standard_normal(truth_dynamics.mu_h.shape),
        b=truth_dynamics.b + 0.04 * rng.standard_normal(truth_dynamics.b.shape),
        c=truth_dynamics.c + 0.04 * rng.standard_normal(truth_dynamics.c.shape),
    )


def run_taylor_workload(
    manifest_path: Path,
    initial_dynamics: StabilizedQuadraticDynamics,
    decoder_template: QuadraticDecoder,
    regularization: DecoderTikhonovRegularization,
    eps_values: np.ndarray,
    timer: FunctionTimer | None,
) -> np.ndarray:
    evaluator = ObservationAlignedBestResponseEvaluator(
        manifest=load_npz_sample_manifest(manifest_path),
        max_dt=0.04,
        dt_shrink=0.5,
        dt_min=1e-12,
        tol=1e-12,
        max_iter=30,
    )
    base_result = evaluator.evaluate_reduced_objective_and_gradient(
        dynamics=initial_dynamics,
        decoder_template=decoder_template,
        regularization=regularization,
    )
    base_vector = dynamics_parameter_vector(initial_dynamics)
    direction_rng = np.random.default_rng(33019)
    direction = direction_rng.standard_normal(base_vector.shape[0])
    direction /= np.linalg.norm(direction)

    residuals = np.zeros_like(eps_values, dtype=np.float64)
    for eps_idx, eps in enumerate(eps_values):
        shifted_dynamics = dynamics_from_parameter_vector(initial_dynamics, base_vector + eps * direction)
        shifted_result = evaluator.evaluate_reduced_objective_and_gradient(
            dynamics=shifted_dynamics,
            decoder_template=decoder_template,
            regularization=regularization,
        )
        residuals[eps_idx] = (
            shifted_result.objective_value
            - base_result.objective_value
            - eps * float(np.dot(base_result.gradient, direction))
        ) / eps
    return residuals


def extract_phase_summary(timer: FunctionTimer) -> dict[str, object]:
    timer_json = timer.to_json_ready()
    records = {
        record["name"]: record
        for record in timer_json["records"]  # type: ignore[index]
    }
    phase_summary: dict[str, object] = {}
    for phase_name, timer_name in PHASE_ALIASES.items():
        record = records.get(timer_name)
        if record is None:
            phase_summary[phase_name] = None
        else:
            phase_summary[phase_name] = {
                "timer_name": timer_name,
                "call_count": int(record["call_count"]),
                "total_seconds": float(record["total_seconds"]),
                "average_seconds": float(record["average_seconds"]),
            }
    return phase_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one phase-timing workload for GOATTM.")
    parser.add_argument("--output-json", required=True, help="Where to write the benchmark JSON.")
    parser.add_argument("--sample-count", type=int, default=100, help="Number of samples in the synthetic dataset.")
    args = parser.parse_args()

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    eps_values = np.logspace(-9, -2, 8)
    regularization = DecoderTikhonovRegularization(coeff_v1=1e-3, coeff_v2=2e-3, coeff_v0=1e-3)
    numba_disabled = os.environ.get("GOATTM_DISABLE_NUMBA", "").strip().lower() in {"1", "true", "yes", "on"}

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path, truth_dynamics, truth_decoder = build_dataset(Path(tmpdir), sample_count=args.sample_count)
        initial_dynamics = build_initial_dynamics(truth_dynamics)
        decoder_template = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )

        warmup_timer = FunctionTimer()
        with use_function_timer(warmup_timer):
            run_taylor_workload(
                manifest_path=manifest_path,
                initial_dynamics=initial_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
                eps_values=np.array([1e-4], dtype=np.float64),
                timer=warmup_timer,
            )

        timer = FunctionTimer()
        start = time.perf_counter()
        with use_function_timer(timer):
            residuals = run_taylor_workload(
                manifest_path=manifest_path,
                initial_dynamics=initial_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
                eps_values=eps_values,
                timer=timer,
            )
        wall_seconds = time.perf_counter() - start

    payload = {
        "sample_count": args.sample_count,
        "numba_disabled": numba_disabled,
        "eps_values": eps_values.tolist(),
        "residuals": residuals.tolist(),
        "wall_seconds": wall_seconds,
        "phase_summary": extract_phase_summary(timer),
        "full_timer": timer.to_json_ready(),
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
