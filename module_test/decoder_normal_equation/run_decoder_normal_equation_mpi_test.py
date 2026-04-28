from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.problems.decoder_normal_equation import (
    DecoderTikhonovRegularization,
    assemble_decoder_normal_equation_from_npz_dataset,
    decoder_parameter_matrix,
    update_decoder_from_normal_equation,
)
from goattm.runtime.distributed import DistributedContext
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times


def build_dataset(root: Path, dynamics: QuadraticDynamics, decoder: QuadraticDecoder, sample_count: int) -> Path:
    rng = np.random.default_rng(20260427)
    observation_times = np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20], dtype=np.float64)
    sample_paths: list[str] = []
    sample_ids: list[str] = []

    for sample_idx in range(sample_count):
        u0 = 0.12 * rng.standard_normal(dynamics.dimension)
        input_values = np.column_stack(
            [
                0.25
                + 0.15 * np.sin(2.0 * np.pi * observation_times + 0.25 * sample_idx)
                + 0.05 * np.cos(4.0 * np.pi * observation_times - 0.1 * sample_idx)
            ]
        )
        input_function = lambda t, grid=observation_times, values=input_values: np.asarray(  # noqa: E731
            [np.interp(t, grid, values[:, 0])],
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
        qoi_observations = np.vstack([decoder.decode(state) for state in states])

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


def main() -> None:
    comm = MPI.COMM_WORLD
    context = DistributedContext.from_comm(comm)
    rank = context.rank
    size = context.size

    r = 3
    dq = 2
    sample_count = max(12, 2 * size)
    dynamics = QuadraticDynamics(
        a=np.array(
            [
                [-0.55, 0.08, 0.03],
                [-0.12, -0.35, 0.06],
                [0.04, -0.09, -0.42],
            ],
            dtype=np.float64,
        ),
        mu_h=np.array([0.03, -0.04, 0.02, 0.05, -0.01, 0.025, -0.015, 0.035], dtype=np.float64),
        c=np.array([0.05, -0.02, 0.03], dtype=np.float64),
        b=np.array([[0.45], [-0.15], [0.25]], dtype=np.float64),
    )
    if dynamics.mu_h.shape[0] != mu_h_dimension(r):
        raise AssertionError("Hard-coded mu_h size is inconsistent with r.")

    true_decoder = QuadraticDecoder(
        v1=np.array([[0.4, -0.2, 0.1], [0.15, 0.25, -0.35]], dtype=np.float64),
        v2=np.array(
            [
                [0.06, -0.03, 0.04, 0.02, -0.01, 0.05],
                [-0.02, 0.05, -0.04, 0.03, 0.01, -0.06],
            ],
            dtype=np.float64,
        ),
        v0=np.array([0.12, -0.08], dtype=np.float64),
    )
    if true_decoder.v2.shape[1] != compressed_quadratic_dimension(r):
        raise AssertionError("Hard-coded decoder quadratic width is inconsistent with r.")

    template_decoder = QuadraticDecoder(
        v1=np.zeros_like(true_decoder.v1),
        v2=np.zeros_like(true_decoder.v2),
        v0=np.zeros_like(true_decoder.v0),
    )
    regularization = DecoderTikhonovRegularization(coeff_v1=3e-4, coeff_v2=7e-4, coeff_v0=2e-4)

    with tempfile.TemporaryDirectory(prefix="goattm_decoder_mpi_") as tmpdir:
        dataset_root = Path(tmpdir)
        manifest_path = build_dataset(dataset_root, dynamics, true_decoder, sample_count) if rank == 0 else None
        manifest_path = Path(comm.bcast(str(manifest_path) if manifest_path is not None else None, root=0))
        comm.Barrier()

        distributed_result = update_decoder_from_normal_equation(
            dynamics=dynamics,
            decoder_template=template_decoder,
            manifest=manifest_path,
            max_dt=0.04,
            regularization=regularization,
            context=context,
            solve_root=0,
        )
        distributed_matrix = decoder_parameter_matrix(distributed_result.decoder)
        gathered = comm.allgather(distributed_matrix)
        for matrix in gathered[1:]:
            np.testing.assert_allclose(matrix, gathered[0], atol=1e-12, rtol=1e-12)

        distributed_stationarity = (
            distributed_result.system.regularized_global_normal_matrix @ distributed_result.solution_matrix
            - distributed_result.system.global_rhs
        )
        np.testing.assert_allclose(distributed_stationarity, 0.0, atol=1e-10, rtol=1e-10)

        if rank == 0:
            serial_context = DistributedContext()
            serial_system = assemble_decoder_normal_equation_from_npz_dataset(
                dynamics=dynamics,
                decoder_template=template_decoder,
                manifest=manifest_path,
                max_dt=0.04,
                regularization=regularization,
                context=serial_context,
            )
            serial_result = update_decoder_from_normal_equation(
                dynamics=dynamics,
                decoder_template=template_decoder,
                manifest=manifest_path,
                max_dt=0.04,
                regularization=regularization,
                context=serial_context,
            )

            np.testing.assert_allclose(
                distributed_result.system.global_normal_matrix,
                serial_system.global_normal_matrix,
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                distributed_result.system.global_rhs,
                serial_system.global_rhs,
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(distributed_matrix, decoder_parameter_matrix(serial_result.decoder), atol=1e-12, rtol=1e-12)
            print("MPI decoder normal-equation test passed.")
            print(f"ranks={size}, samples={sample_count}")
            print(f"global_sample_count={distributed_result.system.global_sample_count}")
            print(f"global_observation_count={distributed_result.system.global_observation_count}")
            print(f"decoder_norm={np.linalg.norm(distributed_matrix):.6e}")
            print(f"stationarity_norm={np.linalg.norm(distributed_stationarity):.6e}")

        comm.Barrier()


if __name__ == "__main__":
    main()
