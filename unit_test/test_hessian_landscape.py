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

from goattm.analysis.hessian_landscape import (  # noqa: E402
    HessianLandscapeConfig,
    run_four_hessian_landscape_from_checkpoint,
)
from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension  # noqa: E402
from goattm.data import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.runtime.distributed import DistributedContext  # noqa: E402
from goattm.solvers import rollout_to_observation_times  # noqa: E402


class HessianLandscapeTest(unittest.TestCase):
    def test_checkpoint_four_case_hessian_landscape_writes_outputs(self) -> None:
        rng = np.random.default_rng(20260710)
        r, dq = 2, 1
        qdim = compressed_quadratic_dimension(r)
        dynamics = QuadraticDynamics(
            a=-0.2 * np.eye(r) + 0.01 * rng.standard_normal((r, r)),
            mu_h=0.01 * rng.standard_normal(mu_h_dimension(r)),
            c=0.01 * rng.standard_normal(r),
        )
        decoder = QuadraticDecoder(
            v1=0.1 * rng.standard_normal((dq, r)),
            v2=0.02 * rng.standard_normal((dq, qdim)),
            v0=0.01 * rng.standard_normal(dq),
        )
        times = np.linspace(0.0, 0.2, 5)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_paths = []
            sample_ids = []
            for sample_idx in range(2):
                u0 = 0.05 * rng.standard_normal(r)
                rollout, observation_indices = rollout_to_observation_times(
                    dynamics=dynamics,
                    u0=u0,
                    observation_times=times,
                    max_dt=0.05,
                    time_integrator="lagged_midpoint",
                )
                qoi = np.vstack([decoder.decode(state) for state in rollout.states[observation_indices]])
                sample_id = f"sample_{sample_idx:03d}"
                sample_path = root / f"{sample_id}.npz"
                save_npz_qoi_sample(sample_path, NpzQoiSample(sample_id, times, u0, qoi))
                sample_paths.append(sample_path)
                sample_ids.append(sample_id)

            manifest = NpzSampleManifest(root_dir=root, sample_paths=tuple(sample_paths), sample_ids=tuple(sample_ids))
            manifest_path = root / "manifest.npz"
            save_npz_sample_manifest(manifest_path, manifest)
            checkpoint_path = root / "checkpoint.npz"
            np.savez(
                checkpoint_path,
                dynamics_type=np.asarray("general"),
                a_matrix=dynamics.a,
                mu_h=dynamics.mu_h,
                c_vector=dynamics.c,
                decoder_v1=decoder.v1,
                decoder_v2=decoder.v2,
                decoder_v0=decoder.v0,
                decoder_form=np.asarray(decoder.form),
            )

            output_dir = root / "hessian_landscape"
            result = run_four_hessian_landscape_from_checkpoint(
                manifest_path=manifest_path,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                config=HessianLandscapeConfig(
                    max_dt=0.05,
                    k=1,
                    which_values=("LA",),
                    eigsh_tol=1.0e-7,
                    random_seed=123,
                ),
            )

            self.assertEqual(
                {case.case for case in result.cases},
                {"general_joint_gn", "general_varpro_gn", "energy_joint_gn", "energy_varpro_gn"},
            )
            for case in result.cases:
                self.assertGreater(case.dimension, 1)
                self.assertEqual(len(case.eigensolves), 1)
                self.assertEqual(case.eigensolves[0].eigenvalues.shape, (1,))
                self.assertTrue(np.all(np.isfinite(case.eigensolves[0].eigenvalues)))
            if DistributedContext.from_comm().rank == 0:
                self.assertTrue((output_dir / "summary.json").exists())
                self.assertTrue((output_dir / "eigenvalues.npz").exists())


if __name__ == "__main__":
    unittest.main()
