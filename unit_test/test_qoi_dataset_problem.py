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
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.problems.qoi_dataset_problem import evaluate_npz_qoi_dataset_loss_and_gradients  # noqa: E402
from goattm.runtime.distributed import DistributedContext  # noqa: E402


class QoiDatasetProblemTest(unittest.TestCase):
    def test_dataset_loss_and_gradients_match_sum_of_local_sample_results(self) -> None:
        rng = np.random.default_rng(811)
        r = 2
        dq = 2

        dynamics = StabilizedQuadraticDynamics(
            s_params=np.array([0.6, 0.1, 0.2], dtype=float),
            w_params=np.array([0.15], dtype=float),
            mu_h=0.05 * rng.standard_normal(mu_h_dimension(r)),
            b=np.array([[0.8], [-0.2]], dtype=float),
            c=np.array([0.1, -0.05], dtype=float),
        )
        decoder = QuadraticDecoder(
            v1=0.2 * rng.standard_normal((dq, r)),
            v2=0.05 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.1 * rng.standard_normal(dq),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.npz"
            sample_paths = []
            sample_ids = []

            for sample_idx in range(2):
                observation_times = np.array([0.0, 0.05, 0.1, 0.15], dtype=float)
                u0 = 0.1 * rng.standard_normal(r)
                input_values = np.column_stack([0.3 + 0.1 * sample_idx + 0.05 * observation_times])
                qoi_observations = rng.standard_normal((observation_times.shape[0], dq))
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

            np.savez(
                manifest_path,
                sample_paths=np.asarray(sample_paths, dtype=object),
                sample_ids=np.asarray(sample_ids, dtype=object),
            )

            manifest = load_npz_sample_manifest(manifest_path)
            dataset_result = evaluate_npz_qoi_dataset_loss_and_gradients(
                dynamics=dynamics,
                decoder=decoder,
                manifest=manifest,
                max_dt=0.05,
                context=DistributedContext(),
            )

            expected_loss = 0.0
            expected_v1 = np.zeros_like(decoder.v1)
            expected_v2 = np.zeros_like(decoder.v2)
            expected_v0 = np.zeros_like(decoder.v0)
            expected_dyn = {key: np.zeros_like(value) for key, value in dataset_result.dynamics_gradients.items()}

            for sample_path in manifest.absolute_paths():
                sample = load_npz_qoi_sample(sample_path)
                sample_result = rollout_qoi_loss_and_gradients_from_observations(
                    dynamics=dynamics,
                    decoder=decoder,
                    u0=sample.u0,
                    observation_times=sample.observation_times,
                    max_dt=0.05,
                    qoi_observations=sample.qoi_observations,
                    input_function=sample.build_input_function(),
                )
                expected_loss += sample_result.loss
                expected_v1 += sample_result.decoder_partials.v1_grad
                expected_v2 += sample_result.decoder_partials.v2_grad
                expected_v0 += sample_result.decoder_partials.v0_grad
                for key, value in sample_result.dynamics_gradients.items():
                    expected_dyn[key] += value

            self.assertEqual(dataset_result.local_sample_count, 2)
            self.assertEqual(dataset_result.global_sample_count, 2)
            np.testing.assert_allclose(dataset_result.total_loss, expected_loss)
            np.testing.assert_allclose(dataset_result.decoder_gradients["v1"], expected_v1)
            np.testing.assert_allclose(dataset_result.decoder_gradients["v2"], expected_v2)
            np.testing.assert_allclose(dataset_result.decoder_gradients["v0"], expected_v0)
            for key, value in expected_dyn.items():
                np.testing.assert_allclose(dataset_result.dynamics_gradients[key], value)


if __name__ == "__main__":
    unittest.main()
