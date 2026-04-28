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

from goattm.core.parametrization import s_params_to_matrix  # noqa: E402
from goattm.data.npz_dataset import NpzQoiSample, NpzSampleManifest, load_npz_qoi_sample, save_npz_qoi_sample  # noqa: E402
from goattm.preprocess.opinf_initialization import (  # noqa: E402
    OpInfLatentEmbeddingConfig,
    initialize_reduced_model_via_opinf,
)


class OpInfInitializationTest(unittest.TestCase):
    def test_opinf_initializer_builds_latent_manifests_and_model_from_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(20260428)
            rank = 2
            dq = 3
            final_time = 2.0
            observation_times = np.linspace(0.0, final_time, 41, dtype=float)
            theta = 0.7
            basis, _ = np.linalg.qr(rng.standard_normal((dq, rank)))

            train_paths = []
            train_ids = []
            initial_conditions = [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([0.8, -0.4]),
                np.array([-0.5, 0.7]),
            ]
            for idx, latent_u0 in enumerate(initial_conditions):
                latent = np.vstack([
                    _damped_rotation_solution(theta, t / final_time, latent_u0) for t in observation_times
                ])
                qoi = latent @ basis.T
                sample = NpzQoiSample(
                    sample_id=f"train-{idx}",
                    observation_times=observation_times,
                    u0=qoi[0].copy(),
                    qoi_observations=qoi,
                    metadata={"kind": "raw"},
                )
                path = raw_root / f"train_{idx}.npz"
                save_npz_qoi_sample(path, sample)
                train_paths.append(path)
                train_ids.append(sample.sample_id)

            manifest = NpzSampleManifest(root_dir=raw_root, sample_paths=tuple(train_paths), sample_ids=tuple(train_ids))
            result = initialize_reduced_model_via_opinf(
                train_manifest=manifest,
                output_dir=root / "opinf_init",
                rank=rank,
                apply_normalization=False,
                time_rescale_to_unit_interval=True,
                max_dt=0.05,
            )

            self.assertEqual(result.decoder.v1.shape, (dq, rank))
            self.assertEqual(result.dynamics.dimension, rank)
            self.assertAlmostEqual(result.time_scale, final_time)
            self.assertTrue(result.time_rescaled_to_unit_interval)
            self.assertLess(result.regression_relative_residual, 1.0e-1)
            self.assertTrue(result.validation_success)
            self.assertGreaterEqual(result.validation_attempt_count, 1)
            self.assertTrue(result.validation_log_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.latent_train_manifest_path.exists())
            np.testing.assert_allclose(s_params_to_matrix(result.dynamics.s_params, rank), np.eye(rank), atol=1e-12)

            log_lines = result.validation_log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(log_lines), result.validation_attempt_count)
            self.assertTrue(json.loads(log_lines[-1])["success"])

            latent_sample = load_npz_qoi_sample(result.latent_train_manifest.absolute_paths()[0])
            self.assertEqual(latent_sample.u0.shape, (rank,))
            self.assertAlmostEqual(float(latent_sample.observation_times[-1]), 1.0)
            self.assertIn("latent_trajectory", latent_sample.metadata)
            self.assertIn("time_scale", latent_sample.metadata)

    def test_opinf_initializer_supports_qoi_augmentation_when_rank_exceeds_output_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(20260429)
            dq = 3
            rank = 5
            observation_times = np.linspace(0.0, 1.0, 21, dtype=float)

            train_paths = []
            train_ids = []
            for idx in range(4):
                p = rng.uniform(-1.0, 1.0, size=observation_times.shape[0])
                qoi = np.column_stack(
                    [
                        np.sin((1.0 + j) * observation_times + 0.2 * idx) + 0.1 * (j + 1) * p
                        for j in range(dq)
                    ]
                )
                sample = NpzQoiSample(
                    sample_id=f"aug-{idx}",
                    observation_times=observation_times,
                    u0=qoi[0].copy(),
                    qoi_observations=qoi,
                    input_times=observation_times,
                    input_values=p[:, None],
                )
                path = raw_root / f"aug_{idx}.npz"
                save_npz_qoi_sample(path, sample)
                train_paths.append(path)
                train_ids.append(sample.sample_id)

            manifest = NpzSampleManifest(root_dir=raw_root, sample_paths=tuple(train_paths), sample_ids=tuple(train_ids))
            result = initialize_reduced_model_via_opinf(
                train_manifest=manifest,
                output_dir=root / "opinf_aug",
                rank=rank,
                apply_normalization=False,
                time_rescale_to_unit_interval=True,
                latent_embedding=OpInfLatentEmbeddingConfig(
                    mode="qoi_augmentation",
                    augmentation_seed=7,
                    augmentation_scale=0.25,
                ),
            )

            self.assertEqual(result.decoder.v1.shape, (dq, rank))
            self.assertEqual(result.dynamics.dimension, rank)
            self.assertEqual(result.decoder_basis.shape, (dq, rank))
            self.assertTrue(result.validation_success)
            np.testing.assert_allclose(result.decoder_basis[:, :dq], np.eye(dq), atol=1e-12)
            self.assertTrue(np.any(np.abs(result.decoder_basis[:, dq:]) > 0.0))


def _damped_rotation_solution(theta: float, t: float, u0: np.ndarray) -> np.ndarray:
    c = np.cos(theta * t)
    s = np.sin(theta * t)
    rotation = np.array([[c, s], [-s, c]], dtype=float)
    return np.exp(-t) * (rotation @ u0)


if __name__ == "__main__":
    unittest.main()
