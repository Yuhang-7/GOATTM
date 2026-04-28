from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension  # noqa: E402
from goattm.data import make_npz_train_test_split  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.train import NewtonActionUpdaterConfig, ReducedQoiTrainer, ReducedQoiTrainerConfig  # noqa: E402
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization  # noqa: E402
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times  # noqa: E402


class ReducedQoiTrainerTest(unittest.TestCase):
    def test_trainer_runs_and_writes_structured_logs_and_checkpoints(self) -> None:
        rng = np.random.default_rng(8101)
        truth_dynamics = StabilizedQuadraticDynamics(
            s_params=np.array([0.48, 0.03, -0.02], dtype=float),
            w_params=np.array([0.08], dtype=float),
            mu_h=0.025 * rng.standard_normal(mu_h_dimension(2)),
            b=np.array([[0.35], [-0.12]], dtype=float),
            c=np.array([0.03, -0.015], dtype=float),
        )
        initial_dynamics = StabilizedQuadraticDynamics(
            s_params=truth_dynamics.s_params + 0.05 * rng.standard_normal(truth_dynamics.s_params.shape),
            w_params=truth_dynamics.w_params + 0.05 * rng.standard_normal(truth_dynamics.w_params.shape),
            mu_h=truth_dynamics.mu_h + 0.05 * rng.standard_normal(truth_dynamics.mu_h.shape),
            b=truth_dynamics.b + 0.05 * rng.standard_normal(truth_dynamics.b.shape),
            c=truth_dynamics.c + 0.05 * rng.standard_normal(truth_dynamics.c.shape),
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
        regularization = DecoderTikhonovRegularization(coeff_v1=1e-3, coeff_v2=2e-3, coeff_v0=1e-3)
        dynamics_regularization = DynamicsTikhonovRegularization(
            coeff_s=1e-5,
            coeff_w=2e-5,
            coeff_mu_h=3e-5,
            coeff_b=4e-5,
            coeff_c=5e-5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = self._write_dataset(root, truth_dynamics, truth_decoder, rng)
            split = make_npz_train_test_split(manifest_path, sample_seed=19, train_fraction=0.67)
            output_dir = root / "train_output"

            trainer = ReducedQoiTrainer(
                train_manifest=split.train_manifest,
                test_manifest=split.test_manifest,
                decoder_template=decoder_template,
                regularization=regularization,
                max_dt=0.04,
                config=ReducedQoiTrainerConfig(
                    output_dir=output_dir,
                    max_iterations=3,
                    checkpoint_every=1,
                    log_every=1,
                    test_every=1,
                ),
                dynamics_regularization=dynamics_regularization,
                preprocess_record={"applied": True, "pipeline": "unit-test", "normalization": "zscore"},
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter_newton=30,
            )
            result = trainer.train(initial_dynamics)

            self.assertTrue(result.metrics_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.latest_checkpoint_path.exists())
            self.assertTrue(result.best_checkpoint_path.exists())
            self.assertTrue(result.timing_json_path.exists())
            self.assertTrue(result.timing_summary_path.exists())
            self.assertTrue(result.stdout_log_path.exists())
            self.assertTrue(result.stderr_log_path.exists())
            self.assertTrue(result.output_dir.exists())
            self.assertEqual(result.output_dir.parent, output_dir)
            self.assertTrue(result.output_dir.name.startswith("train_run_"))

            config_path = result.output_dir / "config.json"
            split_path = result.output_dir / "split.json"
            preprocess_path = result.output_dir / "preprocess.json"
            initial_parameters_path = result.output_dir / "initial_parameters.npz"
            self.assertTrue(config_path.exists())
            self.assertTrue(split_path.exists())
            self.assertTrue(preprocess_path.exists())
            self.assertTrue(initial_parameters_path.exists())

            metrics_lines = result.metrics_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(metrics_lines), 4)
            first_record = json.loads(metrics_lines[0])
            self.assertIn("train_objective", first_record)
            self.assertIn("test_data_loss", first_record)
            self.assertIn("train_dynamics_regularization_loss", first_record)

            config_record = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config_record["optimizer"], "adam")
            self.assertEqual(config_record["max_iterations"], 3)
            self.assertTrue(config_record["enable_function_timing"])
            self.assertAlmostEqual(config_record["dynamics_regularization"]["coeff_s"], dynamics_regularization.coeff_s)

            split_record = json.loads(split_path.read_text(encoding="utf-8"))
            self.assertEqual(set(split_record["train_sample_ids"]), set(split.train_manifest.sample_ids))
            self.assertEqual(set(split_record["test_sample_ids"]), set(split.test_manifest.sample_ids))
            preprocess_record = json.loads(preprocess_path.read_text(encoding="utf-8"))
            self.assertTrue(preprocess_record["applied"])
            self.assertEqual(preprocess_record["pipeline"], "unit-test")

            with np.load(initial_parameters_path, allow_pickle=True) as initial_parameters:
                self.assertIn("decoder_template_v1", initial_parameters.files)
                self.assertIn("decoder_template_v2", initial_parameters.files)
                self.assertIn("decoder_template_v0", initial_parameters.files)
                self.assertIn("s_params", initial_parameters.files)
                self.assertIn("w_params", initial_parameters.files)
                self.assertIn("mu_h", initial_parameters.files)
                self.assertIn("b_matrix", initial_parameters.files)
                self.assertIn("c_vector", initial_parameters.files)

            with np.load(result.latest_checkpoint_path, allow_pickle=True) as checkpoint:
                self.assertIn("decoder_v1", checkpoint.files)
                self.assertIn("decoder_v2", checkpoint.files)
                self.assertIn("decoder_v0", checkpoint.files)
                self.assertIn("s_params", checkpoint.files)
                self.assertIn("w_params", checkpoint.files)
                self.assertIn("mu_h", checkpoint.files)
                self.assertIn("b_matrix", checkpoint.files)
                self.assertIn("c_vector", checkpoint.files)

            timing_record = json.loads(result.timing_json_path.read_text(encoding="utf-8"))
            self.assertGreater(len(timing_record["records"]), 0)
            self.assertTrue(any(record["name"] == "goattm.train.ReducedQoiTrainer.train" for record in timing_record["records"]))
            stdout_text = result.stdout_log_path.read_text(encoding="utf-8")
            self.assertIn("Starting GOATTM training run", stdout_text)
            self.assertIn("[iter 0000]", stdout_text)
            self.assertIn("Completed GOATTM training run", stdout_text)
            stderr_text = result.stderr_log_path.read_text(encoding="utf-8")
            self.assertEqual(stderr_text, "")

            self.assertLessEqual(result.best_snapshot.test_data_loss, result.final_snapshot.test_data_loss)

    def test_newton_trainer_accepts_action_and_explicit_hessian_modes(self) -> None:
        rng = np.random.default_rng(8102)
        truth_dynamics = StabilizedQuadraticDynamics(
            s_params=np.array([0.42, -0.02, 0.01], dtype=float),
            w_params=np.array([0.05], dtype=float),
            mu_h=0.02 * rng.standard_normal(mu_h_dimension(2)),
            b=np.array([[0.18], [-0.09]], dtype=float),
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
            v1=0.18 * rng.standard_normal((2, 2)),
            v2=0.08 * rng.standard_normal((2, compressed_quadratic_dimension(2))),
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
            split = make_npz_train_test_split(manifest_path, sample_seed=23, train_fraction=0.67)

            for mode in ("action", "explicit"):
                trainer = ReducedQoiTrainer(
                    train_manifest=split.train_manifest,
                    test_manifest=None,
                    decoder_template=decoder_template,
                    regularization=regularization,
                    max_dt=0.04,
                    config=ReducedQoiTrainerConfig(
                        output_dir=root / f"train_output_{mode}",
                        max_iterations=1,
                        checkpoint_every=1,
                        log_every=1,
                        optimizer="newton_action",
                        newton_action=NewtonActionUpdaterConfig(
                            hessian_mode=mode,
                            damping=1e-3,
                            cg_tolerance=1e-8,
                            cg_max_iterations=20,
                        ),
                    ),
                    dt_shrink=0.5,
                    dt_min=1e-12,
                    tol=1e-12,
                    max_iter_newton=30,
                )
                result = trainer.train(initial_dynamics)
                self.assertTrue(np.isfinite(result.final_snapshot.objective_value))
                self.assertTrue(result.metrics_path.exists())

    def _write_dataset(
        self,
        root: Path,
        dynamics: StabilizedQuadraticDynamics,
        decoder: QuadraticDecoder,
        rng: np.random.Generator,
    ) -> Path:
        observation_times = np.array([0.0, 0.04, 0.08, 0.12, 0.16], dtype=float)
        sample_paths = []
        sample_ids = []
        for sample_idx in range(6):
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


if __name__ == "__main__":
    unittest.main()
