from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.data.npz_dataset import (  # noqa: E402
    NpzSampleManifest,
    load_npz_qoi_sample,
    save_npz_qoi_sample,
)
from goattm.preprocess.normalization import (  # noqa: E402
    compute_training_normalization_stats,
    materialize_normalized_train_test_split,
)


class DatasetNormalizationTest(unittest.TestCase):
    def test_training_stats_and_materialized_split_use_training_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)

            train_samples = [
                (
                    "train-a",
                    np.array([[1.0, 10.0], [3.0, 14.0]], dtype=float),
                    np.array([[2.0], [4.0]], dtype=float),
                ),
                (
                    "train-b",
                    np.array([[5.0, 18.0], [7.0, 22.0]], dtype=float),
                    np.array([[6.0], [8.0]], dtype=float),
                ),
            ]
            test_samples = [
                (
                    "test-a",
                    np.array([[9.0, 26.0], [11.0, 30.0]], dtype=float),
                    np.array([[10.0], [12.0]], dtype=float),
                )
            ]

            train_paths = []
            train_ids = []
            for idx, (sample_id, qoi, inputs) in enumerate(train_samples):
                path = raw_root / f"train_{idx}.npz"
                save_npz_qoi_sample(
                    path,
                    sample=_build_sample(sample_id, qoi, inputs),
                )
                train_paths.append(path)
                train_ids.append(sample_id)

            test_paths = []
            test_ids = []
            for idx, (sample_id, qoi, inputs) in enumerate(test_samples):
                path = raw_root / f"test_{idx}.npz"
                save_npz_qoi_sample(
                    path,
                    sample=_build_sample(sample_id, qoi, inputs),
                )
                test_paths.append(path)
                test_ids.append(sample_id)

            train_manifest = NpzSampleManifest(root_dir=raw_root, sample_paths=tuple(train_paths), sample_ids=tuple(train_ids))
            test_manifest = NpzSampleManifest(root_dir=raw_root, sample_paths=tuple(test_paths), sample_ids=tuple(test_ids))

            stats = compute_training_normalization_stats(train_manifest)
            np.testing.assert_allclose(stats.qoi_mean, np.array([4.0, 16.0]))
            np.testing.assert_allclose(stats.qoi_std, np.array([3.0, 6.0]) / 0.9)
            np.testing.assert_allclose(stats.qoi_centered_max_abs, np.array([3.0, 6.0]))
            np.testing.assert_allclose(stats.input_mean, np.array([5.0]))
            np.testing.assert_allclose(stats.input_std, np.array([3.0]) / 0.9)
            np.testing.assert_allclose(stats.input_centered_max_abs, np.array([3.0]))
            self.assertEqual(stats.scale_mode, "max_abs")
            self.assertAlmostEqual(stats.target_max_abs, 0.9)

            artifacts = materialize_normalized_train_test_split(
                train_manifest=train_manifest,
                test_manifest=test_manifest,
                output_dir=root / "normalized",
            )

            normalized_train_arrays = []
            normalized_train_inputs = []
            for path in artifacts.train_manifest.absolute_paths():
                sample = load_npz_qoi_sample(path)
                normalized_train_arrays.append(sample.qoi_observations)
                normalized_train_inputs.append(sample.input_values)
                self.assertEqual(sample.metadata["split"], "train")
            train_qoi = np.concatenate(normalized_train_arrays, axis=0)
            train_inputs = np.concatenate(normalized_train_inputs, axis=0)
            np.testing.assert_allclose(np.mean(train_qoi, axis=0), np.zeros(2), atol=1e-12)
            np.testing.assert_allclose(np.max(np.abs(train_qoi), axis=0), 0.9 * np.ones(2), atol=1e-12)
            np.testing.assert_allclose(np.mean(train_inputs, axis=0), np.zeros(1), atol=1e-12)
            np.testing.assert_allclose(np.max(np.abs(train_inputs), axis=0), 0.9 * np.ones(1), atol=1e-12)

            normalized_test = load_npz_qoi_sample(artifacts.test_manifest.absolute_paths()[0])
            self.assertEqual(normalized_test.metadata["split"], "test")
            expected_test_qoi = (test_samples[0][1] - stats.qoi_mean[None, :]) / stats.qoi_std[None, :]
            expected_test_input = (test_samples[0][2] - stats.input_mean[None, :]) / stats.input_std[None, :]
            np.testing.assert_allclose(normalized_test.qoi_observations, expected_test_qoi)
            np.testing.assert_allclose(normalized_test.input_values, expected_test_input)

    def test_zero_variance_dimension_uses_unit_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "single.npz"
            save_npz_qoi_sample(
                path,
                sample=_build_sample(
                    "constant-dim",
                    np.array([[3.0, 1.0], [3.0, 5.0]], dtype=float),
                    np.array([[2.0], [2.0]], dtype=float),
                ),
            )
            manifest = NpzSampleManifest(root_dir=root, sample_paths=(path,), sample_ids=("constant-dim",))
            stats = compute_training_normalization_stats(manifest)
            np.testing.assert_allclose(stats.qoi_mean, np.array([3.0, 3.0]))
            np.testing.assert_allclose(stats.qoi_std, np.array([1.0, 2.0 / 0.9]))
            np.testing.assert_allclose(stats.qoi_centered_max_abs, np.array([0.0, 2.0]))
            np.testing.assert_allclose(stats.input_mean, np.array([2.0]))
            np.testing.assert_allclose(stats.input_std, np.array([1.0]))
            np.testing.assert_allclose(stats.input_centered_max_abs, np.array([0.0]))


def _build_sample(sample_id: str, qoi_observations: np.ndarray, input_values: np.ndarray):
    from goattm.data.npz_dataset import NpzQoiSample  # noqa: E402

    observation_times = np.linspace(0.0, 0.1 * (qoi_observations.shape[0] - 1), qoi_observations.shape[0], dtype=float)
    input_times = np.linspace(0.0, 0.1 * (input_values.shape[0] - 1), input_values.shape[0], dtype=float)
    return NpzQoiSample(
        sample_id=sample_id,
        observation_times=observation_times,
        u0=np.array([0.0, 0.0], dtype=float),
        qoi_observations=qoi_observations,
        input_times=input_times,
        input_values=input_values,
        metadata={"origin": "unit-test"},
    )


if __name__ == "__main__":
    unittest.main()
