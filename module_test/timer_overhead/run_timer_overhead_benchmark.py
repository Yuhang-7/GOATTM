from __future__ import annotations

import json
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


OUTPUT_DIR = ROOT / "module_test" / "output_plots" / "timer_overhead"


def build_dataset(root: Path, sample_count: int = 100) -> tuple[Path, StabilizedQuadraticDynamics, QuadraticDecoder]:
    rng = np.random.default_rng(29017)
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
    rng = np.random.default_rng(29018)
    return StabilizedQuadraticDynamics(
        s_params=truth_dynamics.s_params + 0.05 * rng.standard_normal(truth_dynamics.s_params.shape),
        w_params=truth_dynamics.w_params + 0.05 * rng.standard_normal(truth_dynamics.w_params.shape),
        mu_h=truth_dynamics.mu_h + 0.04 * rng.standard_normal(truth_dynamics.mu_h.shape),
        b=truth_dynamics.b + 0.04 * rng.standard_normal(truth_dynamics.b.shape),
        c=truth_dynamics.c + 0.04 * rng.standard_normal(truth_dynamics.c.shape),
    )


def run_reduced_qoi_taylor_workload(
    manifest_path: Path,
    initial_dynamics: StabilizedQuadraticDynamics,
    decoder_template: QuadraticDecoder,
    regularization: DecoderTikhonovRegularization,
    eps_values: np.ndarray,
    enable_timer: bool,
) -> tuple[float, np.ndarray, FunctionTimer | None]:
    evaluator = ObservationAlignedBestResponseEvaluator(
        manifest=load_npz_sample_manifest(manifest_path),
        max_dt=0.04,
        dt_shrink=0.5,
        dt_min=1e-12,
        tol=1e-12,
        max_iter=30,
    )
    timer = FunctionTimer() if enable_timer else None

    start = time.perf_counter()
    if timer is None:
        base_result = evaluator.evaluate_reduced_data_loss_and_gradient(
            dynamics=initial_dynamics,
            decoder_template=decoder_template,
            regularization=regularization,
        )
    else:
        with use_function_timer(timer):
            base_result = evaluator.evaluate_reduced_data_loss_and_gradient(
                dynamics=initial_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
            )
    base_vector = dynamics_parameter_vector(initial_dynamics)
    direction_rng = np.random.default_rng(29019)
    direction = direction_rng.standard_normal(base_vector.shape[0])
    direction /= np.linalg.norm(direction)

    residuals = np.zeros_like(eps_values, dtype=np.float64)
    for eps_idx, eps in enumerate(eps_values):
        shifted_dynamics = dynamics_from_parameter_vector(initial_dynamics, base_vector + eps * direction)
        if timer is None:
            shifted_result = evaluator.evaluate_reduced_data_loss_and_gradient(
                dynamics=shifted_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
            )
        else:
            with use_function_timer(timer):
                shifted_result = evaluator.evaluate_reduced_data_loss_and_gradient(
                    dynamics=shifted_dynamics,
                    decoder_template=decoder_template,
                    regularization=regularization,
                )
        residuals[eps_idx] = (
            shifted_result.data_loss
            - base_result.data_loss
            - eps * float(np.dot(base_result.reduced_data_gradient, direction))
        ) / eps

    return time.perf_counter() - start, residuals, timer


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eps_values = np.logspace(-9, -2, 8)
    regularization = DecoderTikhonovRegularization(coeff_v1=1e-3, coeff_v2=2e-3, coeff_v0=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path, truth_dynamics, truth_decoder = build_dataset(Path(tmpdir), sample_count=100)
        initial_dynamics = build_initial_dynamics(truth_dynamics)
        decoder_template = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )

        disabled_times: list[float] = []
        enabled_times: list[float] = []
        disabled_residuals = None
        enabled_residuals = None
        latest_timer = None

        for _ in range(3):
            elapsed, residuals, _ = run_reduced_qoi_taylor_workload(
                manifest_path=manifest_path,
                initial_dynamics=initial_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
                eps_values=eps_values,
                enable_timer=False,
            )
            disabled_times.append(elapsed)
            disabled_residuals = residuals

        for _ in range(3):
            elapsed, residuals, timer = run_reduced_qoi_taylor_workload(
                manifest_path=manifest_path,
                initial_dynamics=initial_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
                eps_values=eps_values,
                enable_timer=True,
            )
            enabled_times.append(elapsed)
            enabled_residuals = residuals
            latest_timer = timer

    if latest_timer is None or disabled_residuals is None or enabled_residuals is None:
        raise RuntimeError("Benchmark did not produce timing data.")

    timer_json_path = latest_timer.write_json(OUTPUT_DIR / "enabled_timer_summary.json")
    timer_text_path = latest_timer.write_text(OUTPUT_DIR / "enabled_timer_summary.txt")
    np.savez(
        OUTPUT_DIR / "reduced_qoi_taylor_residuals.npz",
        eps_values=eps_values,
        disabled_residuals=disabled_residuals,
        enabled_residuals=enabled_residuals,
        disabled_times=np.asarray(disabled_times, dtype=np.float64),
        enabled_times=np.asarray(enabled_times, dtype=np.float64),
    )

    disabled_mean = float(np.mean(disabled_times))
    enabled_mean = float(np.mean(enabled_times))
    overhead_seconds = enabled_mean - disabled_mean
    overhead_percent = 100.0 * overhead_seconds / disabled_mean if disabled_mean > 0.0 else float("nan")
    benchmark_record = {
        "sample_count": 100,
        "eps_values": eps_values.tolist(),
        "disabled_times_seconds": disabled_times,
        "enabled_times_seconds": enabled_times,
        "disabled_mean_seconds": disabled_mean,
        "enabled_mean_seconds": enabled_mean,
        "overhead_seconds": overhead_seconds,
        "overhead_percent": overhead_percent,
        "timer_json_path": str(timer_json_path),
        "timer_text_path": str(timer_text_path),
    }
    benchmark_path = OUTPUT_DIR / "timer_overhead_benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps(benchmark_record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
