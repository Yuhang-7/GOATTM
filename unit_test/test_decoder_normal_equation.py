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

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension  # noqa: E402
from goattm.data.npz_dataset import load_npz_qoi_sample, load_npz_sample_manifest  # noqa: E402
from goattm.losses.qoi_loss import rollout_qoi_loss_and_gradients_from_observations  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.problems.decoder_normal_equation import (  # noqa: E402
    DecoderTikhonovRegularization,
    assemble_decoder_normal_equation_from_npz_dataset,
    decoder_parameter_matrix,
    update_decoder_from_normal_equation,
)
from goattm.runtime.distributed import DistributedContext  # noqa: E402


class _FakeReductionBroadcastComm:
    def __init__(
        self,
        rank: int,
        size: int,
        reduced_arrays: list[np.ndarray],
        reduced_scalars: list[int | float],
        broadcast_value: np.ndarray,
    ) -> None:
        self._rank = rank
        self._size = size
        self._reduced_arrays = [np.asarray(value, dtype=np.float64) for value in reduced_arrays]
        self._reduced_scalars = list(reduced_scalars)
        self._broadcast_value = np.asarray(broadcast_value, dtype=np.float64)

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def allreduce(self, value, op=None):
        if isinstance(value, np.ndarray):
            if not self._reduced_arrays:
                raise RuntimeError("No array reduction values remain in fake communicator.")
            return self._reduced_arrays.pop(0).copy()
        if not self._reduced_scalars:
            raise RuntimeError("No scalar reduction values remain in fake communicator.")
        return self._reduced_scalars.pop(0)

    def Allreduce(self, sendbuf, recvbuf, op=None):
        if not self._reduced_arrays:
            raise RuntimeError("No array reduction values remain in fake communicator.")
        recvbuf[...] = self._reduced_arrays.pop(0)

    def bcast(self, value, root: int = 0):
        return self._broadcast_value.copy()


class DecoderNormalEquationTest(unittest.TestCase):
    def test_unregularized_decoder_recovery_matches_ground_truth(self) -> None:
        rng = np.random.default_rng(1234)
        r = 2
        dq = 3
        dynamics = QuadraticDynamics(
            a=np.zeros((r, r), dtype=float),
            mu_h=np.zeros(mu_h_dimension(r), dtype=float),
            c=np.zeros(r, dtype=float),
        )
        true_decoder = QuadraticDecoder(
            v1=0.4 * rng.standard_normal((dq, r)),
            v2=0.3 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.2 * rng.standard_normal(dq),
        )
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(true_decoder.v1),
            v2=np.zeros_like(true_decoder.v2),
            v0=np.zeros_like(true_decoder.v0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = self._write_constant_state_dataset(root=Path(tmpdir), decoder=true_decoder, r=r, rng=rng)
            solve_result = update_decoder_from_normal_equation(
                dynamics=dynamics,
                decoder_template=template_decoder,
                manifest=manifest,
                max_dt=0.2,
                regularization=DecoderTikhonovRegularization(),
                context=DistributedContext(),
            )

        np.testing.assert_allclose(solve_result.decoder.v1, true_decoder.v1, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(solve_result.decoder.v2, true_decoder.v2, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(solve_result.decoder.v0, true_decoder.v0, atol=1e-11, rtol=1e-11)

    def test_regularized_decoder_stationarity_matches_data_gradient_plus_tikhonov_term(self) -> None:
        rng = np.random.default_rng(2037)
        r = 3
        dq = 2
        dynamics = QuadraticDynamics(
            a=0.15 * rng.standard_normal((r, r)),
            mu_h=0.05 * rng.standard_normal(mu_h_dimension(r)),
            c=0.05 * rng.standard_normal(r),
            b=np.array([[0.4], [-0.1], [0.2]], dtype=float),
        )
        template_decoder = QuadraticDecoder(
            v1=np.zeros((dq, r), dtype=float),
            v2=np.zeros((dq, compressed_quadratic_dimension(r)), dtype=float),
            v0=np.zeros(dq, dtype=float),
        )
        regularization = DecoderTikhonovRegularization(
            coeff_v1=float(10.0 ** rng.uniform(-4.0, -1.0)),
            coeff_v2=float(10.0 ** rng.uniform(-4.0, -1.0)),
            coeff_v0=float(10.0 ** rng.uniform(-4.0, -1.0)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = self._write_random_dynamics_dataset(
                root=root,
                dynamics=dynamics,
                dq=dq,
                rng=rng,
                sample_count=6,
            )
            solve_result = update_decoder_from_normal_equation(
                dynamics=dynamics,
                decoder_template=template_decoder,
                manifest=manifest,
                max_dt=0.04,
                regularization=regularization,
                context=DistributedContext(),
            )

            data_v1_grad = np.zeros_like(solve_result.decoder.v1)
            data_v2_grad = np.zeros_like(solve_result.decoder.v2)
            data_v0_grad = np.zeros_like(solve_result.decoder.v0)

            for sample_path in manifest.absolute_paths():
                sample = load_npz_qoi_sample(sample_path)
                result = rollout_qoi_loss_and_gradients_from_observations(
                    dynamics=dynamics,
                    decoder=solve_result.decoder,
                    u0=sample.u0,
                    observation_times=sample.observation_times,
                    max_dt=0.04,
                    qoi_observations=sample.qoi_observations,
                    input_function=sample.build_input_function(),
                )
                data_v1_grad += result.decoder_partials.v1_grad
                data_v2_grad += result.decoder_partials.v2_grad
                data_v0_grad += result.decoder_partials.v0_grad

        total_v1_grad = data_v1_grad + 2.0 * regularization.coeff_v1 * solve_result.decoder.v1
        total_v2_grad = data_v2_grad + 2.0 * regularization.coeff_v2 * solve_result.decoder.v2
        total_v0_grad = data_v0_grad + 2.0 * regularization.coeff_v0 * solve_result.decoder.v0

        total_grad_norm = np.sqrt(
            np.sum(total_v1_grad**2) + np.sum(total_v2_grad**2) + np.sum(total_v0_grad**2)
        )
        self.assertLess(total_grad_norm, 1e-8)

    def test_fake_mpi_nonroot_rank_receives_same_decoder_solution(self) -> None:
        rng = np.random.default_rng(88)
        r = 2
        dq = 2
        dynamics = QuadraticDynamics(
            a=np.zeros((r, r), dtype=float),
            mu_h=np.zeros(mu_h_dimension(r), dtype=float),
            c=np.zeros(r, dtype=float),
        )
        true_decoder = QuadraticDecoder(
            v1=0.3 * rng.standard_normal((dq, r)),
            v2=0.2 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.1 * rng.standard_normal(dq),
        )
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(true_decoder.v1),
            v2=np.zeros_like(true_decoder.v2),
            v0=np.zeros_like(true_decoder.v0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = self._write_constant_state_dataset(root=Path(tmpdir), decoder=true_decoder, r=r, rng=rng)
            serial_result = update_decoder_from_normal_equation(
                dynamics=dynamics,
                decoder_template=template_decoder,
                manifest=manifest,
                max_dt=0.2,
                regularization=DecoderTikhonovRegularization(),
                context=DistributedContext(),
            )
            serial_system = assemble_decoder_normal_equation_from_npz_dataset(
                dynamics=dynamics,
                decoder_template=template_decoder,
                manifest=manifest,
                max_dt=0.2,
                regularization=DecoderTikhonovRegularization(),
                context=DistributedContext(),
            )

            fake_comm = _FakeReductionBroadcastComm(
                rank=1,
                size=2,
                reduced_arrays=[serial_system.global_normal_matrix, serial_system.global_rhs],
                reduced_scalars=[serial_system.global_sample_count, serial_system.global_observation_count],
                broadcast_value=decoder_parameter_matrix(serial_result.decoder),
            )
            fake_context = DistributedContext.from_comm(fake_comm)
            fake_result = update_decoder_from_normal_equation(
                dynamics=dynamics,
                decoder_template=template_decoder,
                manifest=manifest,
                max_dt=0.2,
                regularization=DecoderTikhonovRegularization(),
                context=fake_context,
                solve_root=0,
            )

        np.testing.assert_allclose(fake_result.decoder.v1, serial_result.decoder.v1)
        np.testing.assert_allclose(fake_result.decoder.v2, serial_result.decoder.v2)
        np.testing.assert_allclose(fake_result.decoder.v0, serial_result.decoder.v0)

    def _write_constant_state_dataset(
        self,
        root: Path,
        decoder: QuadraticDecoder,
        r: int,
        rng: np.random.Generator,
        sample_count: int = 10,
    ):
        sample_paths: list[str] = []
        sample_ids: list[str] = []
        observation_times = np.array([0.0, 1.0], dtype=float)

        states: list[np.ndarray] = []
        while len(states) < sample_count:
            candidate = rng.uniform(-0.9, 0.9, size=r)
            states.append(candidate)
            if len(states) >= decoder_parameter_matrix(decoder).shape[0]:
                feature_matrix = np.vstack([self._feature(state) for state in states])
                if np.linalg.matrix_rank(feature_matrix) == feature_matrix.shape[1]:
                    break
        while len(states) < sample_count:
            states.append(rng.uniform(-0.9, 0.9, size=r))

        for sample_idx, u0 in enumerate(states):
            q = decoder.decode(u0)
            qoi_observations = np.vstack([q, q])
            sample_path = root / f"sample_{sample_idx}.npz"
            np.savez(
                sample_path,
                sample_id=np.array(f"sample-{sample_idx}"),
                observation_times=observation_times,
                u0=u0,
                qoi_observations=qoi_observations,
            )
            sample_paths.append(sample_path.name)
            sample_ids.append(f"sample-{sample_idx}")

        manifest_path = root / "manifest.npz"
        np.savez(
            manifest_path,
            sample_paths=np.asarray(sample_paths, dtype=object),
            sample_ids=np.asarray(sample_ids, dtype=object),
        )
        return load_npz_sample_manifest(manifest_path)

    def _write_random_dynamics_dataset(
        self,
        root: Path,
        dynamics: QuadraticDynamics,
        dq: int,
        rng: np.random.Generator,
        sample_count: int,
    ):
        from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times  # noqa: E402

        sample_paths: list[str] = []
        sample_ids: list[str] = []
        observation_times = np.array([0.0, 0.04, 0.08, 0.12, 0.16], dtype=float)
        probe_decoder = QuadraticDecoder(
            v1=0.3 * rng.standard_normal((dq, dynamics.dimension)),
            v2=0.2 * rng.standard_normal((dq, compressed_quadratic_dimension(dynamics.dimension))),
            v0=0.15 * rng.standard_normal(dq),
        )

        for sample_idx in range(sample_count):
            u0 = 0.15 * rng.standard_normal(dynamics.dimension)
            input_values = np.column_stack(
                [0.2 + 0.3 * np.sin(2.0 * np.pi * observation_times + 0.3 * sample_idx)]
            )
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
            )
            states = rollout.states[observation_indices]
            qoi_observations = np.vstack([probe_decoder.decode(state) for state in states])
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
        return load_npz_sample_manifest(manifest_path)

    def _feature(self, state: np.ndarray) -> np.ndarray:
        zeta = np.array([state[0] ** 2, state[1] * state[0], state[1] ** 2], dtype=float)
        return np.concatenate([state, zeta, np.array([1.0])])


if __name__ == "__main__":
    unittest.main()
