from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.data.npz_dataset import (  # noqa: E402
    build_cubic_spline_input_function,
    make_npz_train_test_split,
    build_piecewise_linear_input_function,
    load_npz_qoi_sample,
    load_npz_sample_manifest,
)


class NpzDatasetTest(unittest.TestCase):
    def test_build_piecewise_linear_input_function_interpolates_componentwise(self) -> None:
        times = np.array([0.0, 0.5, 1.0], dtype=float)
        values = np.array([[0.0, 1.0], [1.0, -1.0], [0.5, 0.5]], dtype=float)
        input_function = build_piecewise_linear_input_function(times, values)
        np.testing.assert_allclose(input_function(0.25), np.array([0.5, 0.0]))

    def test_build_cubic_spline_input_function_matches_scipy_reference(self) -> None:
        times = np.array([0.0, 0.2, 0.7, 1.0], dtype=float)
        values = np.array(
            [
                [0.0, 1.0],
                [0.4, 0.2],
                [-0.1, -0.3],
                [0.3, 0.7],
            ],
            dtype=float,
        )
        input_function = build_cubic_spline_input_function(times, values)
        reference = CubicSpline(times, values, axis=0, bc_type="natural", extrapolate=True)

        for t in [0.0, 0.15, 0.45, 0.85, 1.0]:
            np.testing.assert_allclose(input_function(t), np.asarray(reference(t), dtype=float))

    def test_load_sample_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_path = root / "sample0.npz"
            manifest_path = root / "manifest.npz"

            np.savez(
                sample_path,
                sample_id=np.array("sample-0"),
                observation_times=np.array([0.0, 0.1, 0.2], dtype=float),
                u0=np.array([1.0, -0.5], dtype=float),
                qoi_observations=np.array([[0.0], [0.1], [0.2]], dtype=float),
                input_times=np.array([0.0, 0.1, 0.2], dtype=float),
                input_values=np.array([[1.0], [1.5], [2.0]], dtype=float),
                split=np.array("train"),
            )
            np.savez(
                manifest_path,
                sample_paths=np.array(["sample0.npz"], dtype=object),
                sample_ids=np.array(["sample-0"], dtype=object),
            )

            sample = load_npz_qoi_sample(sample_path)
            manifest = load_npz_sample_manifest(manifest_path)
            reference = CubicSpline(
                np.array([0.0, 0.1, 0.2], dtype=float),
                np.array([[1.0], [1.5], [2.0]], dtype=float),
                axis=0,
                bc_type="natural",
                extrapolate=True,
            )

            self.assertEqual(sample.sample_id, "sample-0")
            self.assertEqual(sample.metadata["split"], "train")
            np.testing.assert_allclose(sample.build_input_function()(0.05), np.asarray(reference(0.05), dtype=float))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest.entries_for_rank(rank=0, size=1)[0][0], "sample-0")

    def test_manifest_partitions_samples_by_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.npz"
            np.savez(
                manifest_path,
                sample_paths=np.array(["a.npz", "b.npz", "c.npz", "d.npz"], dtype=object),
                sample_ids=np.array(["a", "b", "c", "d"], dtype=object),
            )
            manifest = load_npz_sample_manifest(manifest_path)
            rank0 = manifest.entries_for_rank(rank=0, size=2)
            rank1 = manifest.entries_for_rank(rank=1, size=2)
            self.assertEqual([sample_id for sample_id, _ in rank0], ["a", "c"])
            self.assertEqual([sample_id for sample_id, _ in rank1], ["b", "d"])

    def test_explicit_train_test_split_accepts_one_side_and_infers_the_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.npz"
            np.savez(
                manifest_path,
                sample_paths=np.array(["a.npz", "b.npz", "c.npz", "d.npz"], dtype=object),
                sample_ids=np.array(["a", "b", "c", "d"], dtype=object),
            )
            split = make_npz_train_test_split(manifest_path, train_sample_ids=["a", "c"])
            self.assertEqual(split.train_manifest.sample_ids, ("a", "c"))
            self.assertEqual(split.test_manifest.sample_ids, ("b", "d"))
            self.assertIsNone(split.sample_seed)

    def test_seeded_train_test_split_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.npz"
            np.savez(
                manifest_path,
                sample_paths=np.array(["a.npz", "b.npz", "c.npz", "d.npz", "e.npz"], dtype=object),
                sample_ids=np.array(["a", "b", "c", "d", "e"], dtype=object),
            )
            split_a = make_npz_train_test_split(manifest_path, sample_seed=17, train_fraction=0.6)
            split_b = make_npz_train_test_split(manifest_path, sample_seed=17, train_fraction=0.6)
            self.assertEqual(split_a.train_manifest.sample_ids, split_b.train_manifest.sample_ids)
            self.assertEqual(split_a.test_manifest.sample_ids, split_b.test_manifest.sample_ids)
            self.assertEqual(len(split_a.train_manifest), 3)
            self.assertEqual(len(split_a.test_manifest), 2)

    def test_explicit_split_must_cover_manifest_exactly_and_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.npz"
            np.savez(
                manifest_path,
                sample_paths=np.array(["a.npz", "b.npz", "c.npz"], dtype=object),
                sample_ids=np.array(["a", "b", "c"], dtype=object),
            )
            with self.assertRaises(ValueError):
                make_npz_train_test_split(manifest_path, train_sample_ids=["a", "b"], test_sample_ids=["b", "c"])
            with self.assertRaises(ValueError):
                make_npz_train_test_split(manifest_path, train_sample_ids=["a"], test_sample_ids=["b"])


if __name__ == "__main__":
    unittest.main()
