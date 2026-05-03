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
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.preprocess.opinf_initialization import (  # noqa: E402
    OpInfLatentEmbeddingConfig,
    _assemble_dynamics_fit_system,
    _build_skew_symmetric_basis,
    _compressed_h_basis_tensor,
    _midpoint_regression_arrays,
    initialize_reduced_model_via_opinf,
)
from goattm.preprocess.constrained_least_squares import build_energy_preserving_compressed_h_basis  # noqa: E402
from goattm.runtime.distributed import DistributedContext  # noqa: E402


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

    def test_oldgoam_mode_initializes_general_quadratic_dynamics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)

            rank = 2
            dq = 2
            observation_times = np.linspace(0.0, 1.0, 31, dtype=float)
            basis = np.eye(dq, rank, dtype=float)

            train_paths = []
            train_ids = []
            initial_conditions = (
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([0.75, -0.35]),
                np.array([-0.45, 0.8]),
            )
            for idx, latent_u0 in enumerate(initial_conditions):
                latent = np.vstack([_damped_rotation_solution(0.4, t, latent_u0) for t in observation_times])
                qoi = latent @ basis.T
                sample = NpzQoiSample(
                    sample_id=f"oldgoam-{idx}",
                    observation_times=observation_times,
                    u0=qoi[0].copy(),
                    qoi_observations=qoi,
                )
                path = raw_root / f"oldgoam_{idx}.npz"
                save_npz_qoi_sample(path, sample)
                train_paths.append(path)
                train_ids.append(sample.sample_id)

            manifest = NpzSampleManifest(root_dir=raw_root, sample_paths=tuple(train_paths), sample_ids=tuple(train_ids))
            result = initialize_reduced_model_via_opinf(
                train_manifest=manifest,
                output_dir=root / "opinf_oldgoam",
                rank=rank,
                apply_normalization=False,
                time_rescale_to_unit_interval=True,
                max_dt=0.05,
                dynamic_form="AHBc",
                oldgoam_mode=True,
            )

            self.assertIsInstance(result.dynamics, QuadraticDynamics)
            self.assertTrue(result.oldgoam_mode)
            self.assertTrue(result.as_preprocess_record()["oldgoam_mode"])
            self.assertTrue(result.validation_success)
            self.assertLess(result.regression_relative_residual, 1.0e-1)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["oldgoam_mode"])

    def test_dynamics_fit_assembly_matches_full_design_matrix_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rng = np.random.default_rng(20260430)
            rank = 4
            input_dim = 2
            observation_times = np.linspace(0.0, 1.0, 17, dtype=float)

            sample_paths = []
            sample_ids = []
            for sample_index in range(3):
                latent = rng.normal(size=(observation_times.shape[0], rank))
                inputs = rng.normal(size=(observation_times.shape[0], input_dim))
                sample = NpzQoiSample(
                    sample_id=f"latent-{sample_index}",
                    observation_times=observation_times,
                    u0=latent[0].copy(),
                    qoi_observations=latent.copy(),
                    input_times=observation_times,
                    input_values=inputs,
                    metadata={"latent_trajectory": latent},
                )
                path = root / f"latent_{sample_index}.npz"
                save_npz_qoi_sample(path, sample)
                sample_paths.append(path)
                sample_ids.append(sample.sample_id)

            manifest = NpzSampleManifest(root_dir=root, sample_paths=tuple(sample_paths), sample_ids=tuple(sample_ids))
            w_basis = _build_skew_symmetric_basis(rank)
            h_basis_tensor = _compressed_h_basis_tensor(build_energy_preserving_compressed_h_basis(rank), rank)
            fixed_a_matrix = rng.normal(size=(rank, rank))

            actual_normal, actual_rhs, actual_target_sumsq, actual_step_count = _assemble_dynamics_fit_system(
                manifest=manifest,
                rank=rank,
                w_basis=w_basis,
                h_basis_tensor=h_basis_tensor,
                context=DistributedContext(),
                fixed_a_matrix=fixed_a_matrix,
            )

            expected_normal, expected_rhs, expected_target_sumsq, expected_step_count = _reference_full_design_assembly(
                manifest=manifest,
                rank=rank,
                w_basis=w_basis,
                h_basis_tensor=h_basis_tensor,
                fixed_a_matrix=fixed_a_matrix,
            )

            np.testing.assert_allclose(actual_normal, expected_normal, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=1e-11, atol=1e-11)
            self.assertAlmostEqual(actual_target_sumsq, expected_target_sumsq, places=11)
            self.assertEqual(actual_step_count, expected_step_count)


def _damped_rotation_solution(theta: float, t: float, u0: np.ndarray) -> np.ndarray:
    c = np.cos(theta * t)
    s = np.sin(theta * t)
    rotation = np.array([[c, s], [-s, c]], dtype=float)
    return np.exp(-t) * (rotation @ u0)


def _reference_full_design_assembly(
    manifest: NpzSampleManifest,
    rank: int,
    w_basis: np.ndarray,
    h_basis_tensor: np.ndarray,
    fixed_a_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    input_dim = load_npz_qoi_sample(manifest.absolute_paths()[0]).input_dimension
    nw = w_basis.shape[2]
    nh = h_basis_tensor.shape[2]
    feature_dim = nw + nh + rank * input_dim + rank
    normal = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    rhs = np.zeros(feature_dim, dtype=np.float64)
    target_sumsq = 0.0
    step_count = 0

    for _, sample_path in manifest.entries_for_rank(0, 1):
        sample = load_npz_qoi_sample(sample_path)
        latent = np.asarray(sample.metadata["latent_trajectory"], dtype=np.float64)
        u_mid, du, p_mid = _midpoint_regression_arrays(sample, latent)
        adjusted_target = du - u_mid @ fixed_a_matrix.T
        phi = _reference_full_design_features(u_mid, p_mid, w_basis, h_basis_tensor)
        phi_2d = phi.reshape((-1, feature_dim))
        normal += phi_2d.T @ phi_2d
        rhs += phi_2d.T @ adjusted_target.reshape(-1)
        target_sumsq += float(np.sum(du * du))
        step_count += int(u_mid.shape[0])
    return normal, rhs, target_sumsq, step_count


def _reference_full_design_features(
    u: np.ndarray,
    p: np.ndarray | None,
    w_basis: np.ndarray,
    h_basis_tensor: np.ndarray,
) -> np.ndarray:
    sample_count, rank = u.shape
    nw = w_basis.shape[2]
    nh = h_basis_tensor.shape[2]
    input_dim = 0 if p is None else p.shape[1]
    total_dim = nw + nh + rank * input_dim + rank
    phi = np.zeros((sample_count, rank, total_dim), dtype=np.float64)
    if nw:
        phi[:, :, :nw] = np.einsum("abn,mb->man", w_basis, u, optimize=True)
    zeta = np.empty((sample_count, rank * (rank + 1) // 2), dtype=np.float64)
    idx = 0
    for i in range(rank):
        for j in range(i + 1):
            zeta[:, idx] = u[:, i] * u[:, j]
            idx += 1
    if nh:
        phi[:, :, nw : nw + nh] = np.einsum("asn,ms->man", h_basis_tensor, zeta, optimize=True)
    offset = nw + nh
    if p is not None:
        identity = np.eye(rank, dtype=np.float64)
        for input_index in range(input_dim):
            phi[:, :, offset + input_index * rank : offset + (input_index + 1) * rank] = (
                p[:, input_index, None, None] * identity[None, :, :]
            )
        offset += rank * input_dim
    phi[:, :, offset:] = np.eye(rank, dtype=np.float64)[None, :, :]
    return phi


if __name__ == "__main__":
    unittest.main()
