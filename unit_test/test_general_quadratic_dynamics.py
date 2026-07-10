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
from goattm.core.quadratic import quadratic_bilinear_action_matrix  # noqa: E402
from goattm.models.general_quadratic_dynamics import GeneralQuadraticDynamics  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.problems.reduced_qoi_best_response import (  # noqa: E402
    DecoderTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
    dynamics_from_parameter_vector,
    dynamics_parameter_vector,
    rhs_parameter_action,
    unpack_dynamics_parameter_vector,
)
from goattm.runtime.distributed import DistributedContext  # noqa: E402
from goattm.solvers.time_integration import rollout_tangent_from_base_rollout, rollout_to_observation_times  # noqa: E402


class GeneralQuadraticDynamicsTest(unittest.TestCase):
    def test_general_and_energy_quadratic_match_lagged_midpoint_and_qoi(self) -> None:
        rng = np.random.default_rng(9101)
        r, dp, dq = 4, 2, 3
        a = -0.15 * np.eye(r) + 0.03 * rng.standard_normal((r, r))
        mu_h = 0.02 * rng.standard_normal(mu_h_dimension(r))
        b = 0.05 * rng.standard_normal((r, dp))
        c = 0.01 * rng.standard_normal(r)
        energy_dynamics = QuadraticDynamics(a=a, mu_h=mu_h, b=b, c=c)
        general_dynamics = GeneralQuadraticDynamics(a=a, h_matrix=energy_dynamics.h_matrix.copy(), b=b, c=c)
        decoder = QuadraticDecoder(
            v1=0.2 * rng.standard_normal((dq, r)),
            v2=0.05 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.01 * rng.standard_normal(dq),
        )
        observation_times = np.linspace(0.0, 0.24, 7)
        u0 = 0.04 * rng.standard_normal(r)

        def input_function(t: float) -> np.ndarray:
            return np.array([0.2 + 0.03 * np.sin(2.0 * np.pi * t), -0.1 + 0.02 * t], dtype=np.float64)

        energy_rollout, energy_indices = rollout_to_observation_times(
            dynamics=energy_dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=0.04,
            input_function=input_function,
            time_integrator="lagged_midpoint",
        )
        general_rollout, general_indices = rollout_to_observation_times(
            dynamics=general_dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=0.04,
            input_function=input_function,
            time_integrator="lagged_midpoint",
        )

        np.testing.assert_array_equal(general_indices, energy_indices)
        np.testing.assert_allclose(general_rollout.times, energy_rollout.times, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(general_rollout.states, energy_rollout.states, atol=1e-13, rtol=1e-13)

        energy_qoi = np.vstack([decoder.decode(state) for state in energy_rollout.states[energy_indices]])
        general_qoi = np.vstack([decoder.decode(state) for state in general_rollout.states[general_indices]])
        np.testing.assert_allclose(general_qoi, energy_qoi, atol=1e-13, rtol=1e-13)

    def test_general_quadratic_varpro_first_order_taylor_with_normal_equation(self) -> None:
        rng = np.random.default_rng(9102)
        r, dp, dq = 3, 1, 2
        truth_dynamics = GeneralQuadraticDynamics(
            a=-0.12 * np.eye(r) + 0.02 * rng.standard_normal((r, r)),
            h_matrix=0.015 * rng.standard_normal((r, compressed_quadratic_dimension(r))),
            b=np.array([[0.15], [-0.08], [0.04]], dtype=np.float64),
            c=0.01 * rng.standard_normal(r),
        )
        candidate_dynamics = GeneralQuadraticDynamics(
            a=truth_dynamics.a + 0.02 * rng.standard_normal(truth_dynamics.a.shape),
            h_matrix=truth_dynamics.h_matrix + 0.02 * rng.standard_normal(truth_dynamics.h_matrix.shape),
            b=truth_dynamics.b + 0.02 * rng.standard_normal(truth_dynamics.b.shape),
            c=truth_dynamics.c + 0.02 * rng.standard_normal(truth_dynamics.c.shape),
        )
        truth_decoder = QuadraticDecoder(
            v1=0.18 * rng.standard_normal((dq, r)),
            v2=0.04 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.02 * rng.standard_normal(dq),
        )
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )
        regularization = DecoderTikhonovRegularization(coeff_v1=2e-5, coeff_v2=3e-5, coeff_v0=4e-5)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_dataset(Path(tmpdir), truth_dynamics, truth_decoder, sample_count=4, rng=rng)
            evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=manifest_path,
                max_dt=0.04,
                time_integrator="lagged_midpoint",
                context=DistributedContext(),
            )
            base = evaluator.evaluate_goam_reduced_objective_and_gradient(
                candidate_dynamics,
                template_decoder,
                regularization,
            )
            self.assertIn("h_matrix", base.dataset_result.dynamics_gradients)

            direction = rng.standard_normal(dynamics_parameter_vector(candidate_dynamics).shape)
            direction /= np.linalg.norm(direction)
            directional_derivative = float(np.dot(base.reduced_objective_gradient, direction))
            base_vector = dynamics_parameter_vector(candidate_dynamics)
            eps_values = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4], dtype=np.float64)
            zero_order = []
            first_order = []
            for eps in eps_values:
                perturbed = dynamics_from_parameter_vector(candidate_dynamics, base_vector + eps * direction)
                value_eps = evaluator.evaluate_goam_reduced_objective_and_gradient(
                    perturbed,
                    template_decoder,
                    regularization,
                ).objective_value
                delta = value_eps - base.objective_value
                zero_order.append(abs(delta))
                first_order.append(abs(delta - eps * directional_derivative))

            zero_slope = self._fit_slope(eps_values, np.asarray(zero_order, dtype=np.float64))
            first_slope = self._fit_slope(eps_values, np.asarray(first_order, dtype=np.float64))
            self.assertGreaterEqual(zero_slope, 0.80)
            self.assertLessEqual(zero_slope, 1.20)
            self.assertGreaterEqual(first_slope, 1.60)
            self.assertLessEqual(first_slope, 2.40)

    def test_lagged_midpoint_tangent_matches_finite_difference_for_general_quadratic(self) -> None:
        rng = np.random.default_rng(9105)
        r, dp = 3, 1
        dynamics = GeneralQuadraticDynamics(
            a=-0.10 * np.eye(r) + 0.02 * rng.standard_normal((r, r)),
            h_matrix=0.015 * rng.standard_normal((r, compressed_quadratic_dimension(r))),
            b=np.array([[0.12], [-0.04], [0.03]], dtype=np.float64),
            c=0.01 * rng.standard_normal(r),
        )
        observation_times = np.linspace(0.0, 0.16, 5)
        u0 = 0.04 * rng.standard_normal(r)

        def input_function(t: float) -> np.ndarray:
            return np.array([0.2 + 0.03 * np.sin(2.0 * np.pi * t)], dtype=np.float64)

        base_rollout, _ = rollout_to_observation_times(
            dynamics=dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=0.04,
            input_function=input_function,
            time_integrator="lagged_midpoint",
        )
        direction_vector = rng.standard_normal(dynamics_parameter_vector(dynamics).shape)
        direction_vector /= np.linalg.norm(direction_vector)
        direction = unpack_dynamics_parameter_vector(dynamics, direction_vector)
        delta_a = direction.a
        delta_h = direction.h_matrix

        def parameter_action(state: np.ndarray, time: float) -> np.ndarray:
            return rhs_parameter_action(dynamics, direction, state, time, input_function=input_function)

        def parameter_linear_operator_action(predictor_state: np.ndarray, _time: float) -> np.ndarray:
            return delta_a + quadratic_bilinear_action_matrix(delta_h, predictor_state)

        def forcing_parameter_action(time: float) -> np.ndarray:
            return direction.c + direction.b @ input_function(time)

        tangent = rollout_tangent_from_base_rollout(
            dynamics=dynamics,
            base_rollout=base_rollout,
            parameter_action=parameter_action,
            input_function=input_function,
            time_integrator="lagged_midpoint",
            parameter_linear_operator_action=parameter_linear_operator_action,
            forcing_parameter_action=forcing_parameter_action,
        )

        eps_values = np.array([1e-6, 3e-6, 1e-5, 3e-5], dtype=np.float64)
        errors = []
        base_vector = dynamics_parameter_vector(dynamics)
        for eps in eps_values:
            perturbed = dynamics_from_parameter_vector(dynamics, base_vector + eps * direction_vector)
            perturbed_rollout, _ = rollout_to_observation_times(
                dynamics=perturbed,
                u0=u0,
                observation_times=observation_times,
                max_dt=0.04,
                input_function=input_function,
                time_integrator="lagged_midpoint",
            )
            finite_difference = (perturbed_rollout.states - base_rollout.states) / eps
            errors.append(float(np.linalg.norm(finite_difference - tangent)))

        slope = self._fit_slope(eps_values, np.asarray(errors, dtype=np.float64))
        self.assertGreaterEqual(slope, 0.80)
        self.assertLessEqual(slope, 1.20)

    def test_general_quadratic_varpro_gauss_newton_action_is_symmetric_and_matches_qform(self) -> None:
        rng = np.random.default_rng(9103)
        dynamics, decoder, manifest_path, regularization, tmpdir = self._build_general_varpro_fixture(rng)
        self.addCleanup(tmpdir.cleanup)
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(decoder.v1),
            v2=np.zeros_like(decoder.v2),
            v0=np.zeros_like(decoder.v0),
        )
        evaluator = ObservationAlignedBestResponseEvaluator(
            manifest=manifest_path,
            max_dt=0.04,
            time_integrator="lagged_midpoint",
            context=DistributedContext(),
        )
        workflow = evaluator.build_reduced_objective_workflow(
            decoder_template=template_decoder,
            regularization=regularization,
        )
        prepared = workflow.prepare(dynamics)
        v = rng.standard_normal(dynamics_parameter_vector(dynamics).shape)
        w = rng.standard_normal(dynamics_parameter_vector(dynamics).shape)
        v /= np.linalg.norm(v)
        w /= np.linalg.norm(w)

        counts_before = evaluator.solve_count_record()
        gv = workflow.evaluate_gauss_newton_hessian_action_from_prepared_state(prepared, v)
        counts_after_v = evaluator.solve_count_record()
        gw = workflow.evaluate_gauss_newton_hessian_action_from_prepared_state(prepared, w)
        counts_after_w = evaluator.solve_count_record()
        sample_count = prepared.result.best_response_context.forward_cache.global_sample_count

        self.assertEqual(counts_after_v["tangent_forward"] - counts_before["tangent_forward"], sample_count)
        self.assertEqual(counts_after_w["tangent_forward"] - counts_after_v["tangent_forward"], sample_count)

        np.testing.assert_allclose(float(np.dot(v, gv.action)), gv.quadratic_form, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(float(np.dot(v, gw.action)), float(np.dot(w, gv.action)), rtol=1e-6, atol=1e-8)
        self.assertGreaterEqual(gv.quadratic_form, -1e-12)

    def test_energy_and_general_quadratic_joint_gn_actions_match_when_h_matches(self) -> None:
        rng = np.random.default_rng(9104)
        r, dp, dq = 3, 1, 2
        a = -0.10 * np.eye(r) + 0.02 * rng.standard_normal((r, r))
        mu_h = 0.012 * rng.standard_normal(mu_h_dimension(r))
        b = 0.03 * rng.standard_normal((r, dp))
        c = 0.01 * rng.standard_normal(r)
        energy_dynamics = QuadraticDynamics(a=a, mu_h=mu_h, b=b, c=c)
        general_dynamics = GeneralQuadraticDynamics(a=a, h_matrix=energy_dynamics.h_matrix.copy(), b=b, c=c)
        decoder = QuadraticDecoder(
            v1=0.12 * rng.standard_normal((dq, r)),
            v2=0.03 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.01 * rng.standard_normal(dq),
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_dataset(Path(tmp), general_dynamics, decoder, sample_count=3, rng=rng)
            evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=manifest_path,
                max_dt=0.04,
                time_integrator="lagged_midpoint",
                context=DistributedContext(),
            )
            energy_direction = rng.standard_normal(dynamics_parameter_vector(energy_dynamics).shape)
            general_direction_obj = self._energy_direction_to_general_direction(energy_dynamics, general_dynamics, energy_direction)
            decoder_direction = rng.standard_normal(decoder.v1.size + decoder.v2.size + decoder.v0.size)
            energy_joint_direction = np.concatenate([energy_direction, decoder_direction])
            general_joint_direction = np.concatenate([general_direction_obj, decoder_direction])

            energy_action = evaluator.evaluate_joint_gauss_newton_hessian_action(
                energy_dynamics,
                decoder,
                energy_joint_direction,
            )
            general_action = evaluator.evaluate_joint_gauss_newton_hessian_action(
                general_dynamics,
                decoder,
                general_joint_direction,
            )

            mapped_general_dynamics_action = self._general_action_to_energy_action(
                energy_dynamics,
                general_dynamics,
                general_action.dynamics_action,
            )
            np.testing.assert_allclose(mapped_general_dynamics_action, energy_action.dynamics_action, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(general_action.decoder_action_matrix, energy_action.decoder_action_matrix, rtol=1e-10, atol=1e-12)

    def _write_dataset(
        self,
        root: Path,
        dynamics: GeneralQuadraticDynamics,
        decoder: QuadraticDecoder,
        sample_count: int,
        rng: np.random.Generator,
    ) -> Path:
        observation_times = np.linspace(0.0, 0.20, 6)
        sample_paths: list[str] = []
        sample_ids: list[str] = []
        for sample_idx in range(sample_count):
            u0 = 0.04 * rng.standard_normal(dynamics.dimension)
            input_values = np.column_stack([0.2 + 0.04 * np.sin(2.0 * np.pi * observation_times + 0.3 * sample_idx)])
            input_function = self._linear_input_function(observation_times, input_values)
            rollout, observation_indices = rollout_to_observation_times(
                dynamics=dynamics,
                u0=u0,
                observation_times=observation_times,
                max_dt=0.04,
                input_function=input_function,
                time_integrator="lagged_midpoint",
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

    def _build_general_varpro_fixture(self, rng: np.random.Generator):
        r, dq = 3, 2
        truth_dynamics = GeneralQuadraticDynamics(
            a=-0.12 * np.eye(r) + 0.02 * rng.standard_normal((r, r)),
            h_matrix=0.015 * rng.standard_normal((r, compressed_quadratic_dimension(r))),
            b=np.array([[0.15], [-0.08], [0.04]], dtype=np.float64),
            c=0.01 * rng.standard_normal(r),
        )
        candidate_dynamics = GeneralQuadraticDynamics(
            a=truth_dynamics.a + 0.02 * rng.standard_normal(truth_dynamics.a.shape),
            h_matrix=truth_dynamics.h_matrix + 0.02 * rng.standard_normal(truth_dynamics.h_matrix.shape),
            b=truth_dynamics.b + 0.02 * rng.standard_normal(truth_dynamics.b.shape),
            c=truth_dynamics.c + 0.02 * rng.standard_normal(truth_dynamics.c.shape),
        )
        truth_decoder = QuadraticDecoder(
            v1=0.18 * rng.standard_normal((dq, r)),
            v2=0.04 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.02 * rng.standard_normal(dq),
        )
        regularization = DecoderTikhonovRegularization(coeff_v1=2e-5, coeff_v2=3e-5, coeff_v0=4e-5)
        tmpdir = tempfile.TemporaryDirectory()
        manifest_path = self._write_dataset(Path(tmpdir.name), truth_dynamics, truth_decoder, sample_count=4, rng=rng)
        return candidate_dynamics, truth_decoder, manifest_path, regularization, tmpdir

    @staticmethod
    def _energy_direction_to_general_direction(
        energy_dynamics: QuadraticDynamics,
        general_dynamics: GeneralQuadraticDynamics,
        direction_vector: np.ndarray,
    ) -> np.ndarray:
        from goattm.core.parametrization import mu_h_to_compressed_h
        from goattm.problems.reduced_qoi_best_response import unpack_dynamics_parameter_vector

        direction = unpack_dynamics_parameter_vector(energy_dynamics, direction_vector)
        h_direction = mu_h_to_compressed_h(direction.mu_h, energy_dynamics.dimension)
        blocks = [direction.a.reshape(-1), h_direction.reshape(-1)]
        if general_dynamics.b is not None:
            blocks.append(direction.b.reshape(-1))
        blocks.append(direction.c)
        return np.concatenate(blocks)

    @staticmethod
    def _general_action_to_energy_action(
        energy_dynamics: QuadraticDynamics,
        general_dynamics: GeneralQuadraticDynamics,
        action_vector: np.ndarray,
    ) -> np.ndarray:
        from goattm.core.parametrization import compressed_h_gradient_to_mu_h
        from goattm.problems.reduced_qoi_best_response import unpack_dynamics_parameter_vector

        action = unpack_dynamics_parameter_vector(general_dynamics, action_vector)
        blocks = [
            action.a.reshape(-1),
            compressed_h_gradient_to_mu_h(action.h_matrix, energy_dynamics.dimension),
        ]
        if energy_dynamics.b is not None:
            blocks.append(action.b.reshape(-1))
        blocks.append(action.c)
        return np.concatenate(blocks)

    @staticmethod
    def _linear_input_function(times: np.ndarray, values: np.ndarray):
        def input_function(t: float) -> np.ndarray:
            return np.array([np.interp(t, times, values[:, j]) for j in range(values.shape[1])], dtype=np.float64)

        return input_function

    @staticmethod
    def _fit_slope(eps_values: np.ndarray, errors: np.ndarray) -> float:
        mask = (errors > 0.0) & np.isfinite(errors)
        return float(np.polyfit(np.log10(eps_values[mask]), np.log10(errors[mask]), 1)[0])


if __name__ == "__main__":
    unittest.main()
