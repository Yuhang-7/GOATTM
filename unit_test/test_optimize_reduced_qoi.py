from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.data import build_cubic_spline_input_function  # noqa: E402
from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.problems import DecoderTikhonovRegularization  # noqa: E402
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times  # noqa: E402
from goattm.train import AdamUpdaterConfig, ReducedQoiTrainerConfig, optimize_reduced_qoi_from_manifest  # noqa: E402


class OptimizeReducedQoiTest(unittest.TestCase):
    def test_optimize_helper_supports_seed_split_and_no_test_split(self) -> None:
        rng = np.random.default_rng(8201)
        truth_dynamics = StabilizedQuadraticDynamics(
            s_params=np.array([0.41, -0.03, 0.015], dtype=float),
            w_params=np.array([0.06], dtype=float),
            mu_h=0.02 * rng.standard_normal(mu_h_dimension(2)),
            b=np.array([[0.22], [-0.10]], dtype=float),
            c=np.array([0.02, -0.01], dtype=float),
        )
        initial_dynamics = StabilizedQuadraticDynamics(
            s_params=truth_dynamics.s_params + 0.03 * rng.standard_normal(truth_dynamics.s_params.shape),
            w_params=truth_dynamics.w_params + 0.03 * rng.standard_normal(truth_dynamics.w_params.shape),
            mu_h=truth_dynamics.mu_h + 0.03 * rng.standard_normal(truth_dynamics.mu_h.shape),
            b=truth_dynamics.b + 0.03 * rng.standard_normal(truth_dynamics.b.shape),
            c=truth_dynamics.c + 0.03 * rng.standard_normal(truth_dynamics.c.shape),
        )
        truth_decoder = QuadraticDecoder(
            v1=0.20 * rng.standard_normal((2, 2)),
            v2=0.09 * rng.standard_normal((2, compressed_quadratic_dimension(2))),
            v0=0.05 * rng.standard_normal(2),
        )
        decoder_template = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )
        regularization = DecoderTikhonovRegularization(coeff_v1=1e-3, coeff_v2=1e-3, coeff_v0=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_dataset(root, truth_dynamics, truth_decoder, rng)
            trainer_config = ReducedQoiTrainerConfig(
                output_dir=root / "runs_seed_split",
                max_iterations=1,
                checkpoint_every=1,
                log_every=1,
                test_every=1,
                adam=AdamUpdaterConfig(learning_rate=1e-2),
            )
            run_with_test = optimize_reduced_qoi_from_manifest(
                manifest=manifest_path,
                initial_dynamics=initial_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
                max_dt=0.04,
                trainer_config=trainer_config,
                sample_seed=31,
                train_fraction=0.67,
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter_newton=30,
            )
            self.assertIsNotNone(run_with_test.test_manifest)
            self.assertEqual(len(run_with_test.train_manifest.sample_ids), 4)
            self.assertEqual(len(run_with_test.test_manifest.sample_ids), 2)
            self.assertIsNotNone(run_with_test.result.final_snapshot.test_data_loss)

            no_test_config = ReducedQoiTrainerConfig(
                output_dir=root / "runs_no_test",
                max_iterations=1,
                checkpoint_every=1,
                log_every=1,
                test_every=1,
            )
            run_no_test = optimize_reduced_qoi_from_manifest(
                manifest=manifest_path,
                initial_dynamics=initial_dynamics,
                decoder_template=decoder_template,
                regularization=regularization,
                max_dt=0.04,
                trainer_config=no_test_config,
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter_newton=30,
            )
            self.assertIsNone(run_no_test.test_manifest)
            self.assertEqual(len(run_no_test.train_manifest.sample_ids), 6)
            self.assertIsNone(run_no_test.result.final_snapshot.test_data_loss)

    def _write_dataset(
        self,
        root: Path,
        dynamics: StabilizedQuadraticDynamics,
        decoder: QuadraticDecoder,
        rng: np.random.Generator,
    ) -> Path:
        observation_times = np.array([0.0, 0.04, 0.08, 0.12, 0.16], dtype=float)
        sample_paths: list[str] = []
        sample_ids: list[str] = []
        for sample_idx in range(6):
            u0 = 0.1 * rng.standard_normal(dynamics.dimension)
            input_values = np.column_stack([0.18 + 0.10 * np.sin(2.0 * np.pi * observation_times + 0.2 * sample_idx)])
            input_function = build_cubic_spline_input_function(observation_times, input_values)
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


if __name__ == "__main__":
    unittest.main()
