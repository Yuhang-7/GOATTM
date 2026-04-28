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
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.problems import (  # noqa: E402
    DecoderTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
)
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times  # noqa: E402


OUTPUT_DIR = ROOT / "module_test" / "output_plots" / "reduced_qoi_best_response"


def build_dataset(
    root: Path,
    sample_count: int,
) -> tuple[Path, StabilizedQuadraticDynamics, StabilizedQuadraticDynamics, QuadraticDecoder, QuadraticDecoder]:
    rng = np.random.default_rng(43017)
    truth_dynamics = StabilizedQuadraticDynamics(
        s_params=np.array([0.46, -0.07, 0.035], dtype=float),
        w_params=np.array([0.11], dtype=float),
        mu_h=0.03 * rng.standard_normal(mu_h_dimension(2)),
        b=np.array([[0.33], [-0.09]], dtype=float),
        c=np.array([0.025, -0.012], dtype=float),
    )
    candidate_dynamics = StabilizedQuadraticDynamics(
        s_params=truth_dynamics.s_params + 0.05 * rng.standard_normal(truth_dynamics.s_params.shape),
        w_params=truth_dynamics.w_params + 0.05 * rng.standard_normal(truth_dynamics.w_params.shape),
        mu_h=truth_dynamics.mu_h + 0.04 * rng.standard_normal(truth_dynamics.mu_h.shape),
        b=truth_dynamics.b + 0.04 * rng.standard_normal(truth_dynamics.b.shape),
        c=truth_dynamics.c + 0.04 * rng.standard_normal(truth_dynamics.c.shape),
    )
    truth_decoder = QuadraticDecoder(
        v1=0.22 * rng.standard_normal((2, 2)),
        v2=0.10 * rng.standard_normal((2, compressed_quadratic_dimension(2))),
        v0=0.05 * rng.standard_normal(2),
    )
    decoder_template = QuadraticDecoder(
        v1=np.zeros_like(truth_decoder.v1),
        v2=np.zeros_like(truth_decoder.v2),
        v0=np.zeros_like(truth_decoder.v0),
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
    return manifest_path, truth_dynamics, candidate_dynamics, truth_decoder, decoder_template


def benchmark_once(
    manifest_path: Path,
    candidate_dynamics: StabilizedQuadraticDynamics,
    decoder_template: QuadraticDecoder,
    regularization: DecoderTikhonovRegularization,
) -> tuple[float, float]:
    evaluator_old = ObservationAlignedBestResponseEvaluator(
        manifest=manifest_path,
        max_dt=0.04,
        dt_shrink=0.5,
        dt_min=1e-12,
        tol=1e-12,
        max_iter=30,
    )
    start = time.perf_counter()
    old_result = evaluator_old.evaluate_reduced_data_loss_and_gradient(candidate_dynamics, decoder_template, regularization)
    old_seconds = time.perf_counter() - start

    evaluator_new = ObservationAlignedBestResponseEvaluator(
        manifest=manifest_path,
        max_dt=0.04,
        dt_shrink=0.5,
        dt_min=1e-12,
        tol=1e-12,
        max_iter=30,
    )
    start = time.perf_counter()
    new_result = evaluator_new.evaluate_reduced_objective_and_gradient(candidate_dynamics, decoder_template, regularization)
    new_seconds = time.perf_counter() - start

    if not np.all(np.isfinite(old_result.reduced_data_gradient)):
        raise RuntimeError("Old reduced-data gradient produced non-finite values.")
    if not np.all(np.isfinite(new_result.reduced_objective_gradient)):
        raise RuntimeError("New reduced-objective gradient produced non-finite values.")
    return old_seconds, new_seconds


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path, _, candidate_dynamics, _, decoder_template = build_dataset(Path(tmpdir), sample_count=100)
        regularization = DecoderTikhonovRegularization(coeff_v1=1e-3, coeff_v2=2e-3, coeff_v0=1e-3)
        old_times: list[float] = []
        new_times: list[float] = []
        for _ in range(3):
            old_seconds, new_seconds = benchmark_once(
                manifest_path=manifest_path,
                candidate_dynamics=candidate_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
            )
            old_times.append(old_seconds)
            new_times.append(new_seconds)

    payload = {
        "sample_count": 100,
        "old_reduced_data_gradient_seconds": old_times,
        "new_reduced_objective_gradient_seconds": new_times,
        "old_mean_seconds": float(np.mean(old_times)),
        "new_mean_seconds": float(np.mean(new_times)),
        "speedup_ratio_old_over_new": float(np.mean(old_times) / np.mean(new_times)),
    }
    output_path = OUTPUT_DIR / "reduced_gradient_speed_compare.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
