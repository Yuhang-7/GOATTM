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
    rollout_dynamics_parameter_tangent_from_base_rollout,
    unpack_dynamics_parameter_vector,
)
from goattm.runtime.distributed import DistributedContext  # noqa: E402
from goattm.solvers.time_integration import rollout_tangent_from_base_rollout, rollout_to_observation_times  # noqa: E402
from goattm.train.quotient_trust_region import quotient_vertical_basis  # noqa: E402


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

    def test_exact_varpro_hessian_action_has_reduced_objective_taylor_model(self) -> None:
        rng = np.random.default_rng(9109)
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
            time_integrator="explicit_euler",
            context=DistributedContext(),
        )
        workflow = evaluator.build_reduced_objective_workflow(
            decoder_template=template_decoder,
            regularization=regularization,
        )
        prepared = workflow.prepare(dynamics)
        base_vector = dynamics_parameter_vector(dynamics)
        direction = rng.standard_normal(base_vector.shape)
        direction /= np.linalg.norm(direction)
        action = workflow.evaluate_exact_varpro_hessian_action_from_prepared_state(prepared, direction)
        first_derivative = float(np.dot(prepared.gradient, direction))
        second_derivative = float(np.dot(direction, action.action))

        eps_values = np.array([1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4], dtype=np.float64)
        first_order_remainders = []
        second_order_remainders = []
        for eps in eps_values:
            perturbed = dynamics_from_parameter_vector(dynamics, base_vector + eps * direction)
            value = workflow.evaluate_objective(perturbed)
            first_order_remainders.append(abs(value - prepared.objective_value - eps * first_derivative))
            second_order_remainders.append(
                abs(value - prepared.objective_value - eps * first_derivative - 0.5 * eps * eps * second_derivative)
            )

        first_order_slope = self._fit_slope(eps_values, np.asarray(first_order_remainders, dtype=np.float64))
        self.assertGreaterEqual(first_order_slope, 1.80)
        self.assertLessEqual(first_order_slope, 2.20)
        ratios = np.asarray(second_order_remainders, dtype=np.float64) / np.asarray(first_order_remainders, dtype=np.float64)
        self.assertLess(float(np.max(ratios)), 2.0e-3)

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
        gv = workflow.evaluate_linearized_varpro_hessian_action_from_prepared_state(prepared, v)
        counts_after_v = evaluator.solve_count_record()
        gw = workflow.evaluate_linearized_varpro_hessian_action_from_prepared_state(prepared, w)
        counts_after_w = evaluator.solve_count_record()
        sample_count = prepared.result.best_response_context.forward_cache.global_sample_count

        self.assertEqual(counts_after_v["tangent_forward"] - counts_before["tangent_forward"], sample_count)
        self.assertEqual(counts_after_w["tangent_forward"] - counts_after_v["tangent_forward"], sample_count)

        np.testing.assert_allclose(float(np.dot(v, gv.action)), gv.quadratic_form, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(float(np.dot(v, gw.action)), float(np.dot(w, gv.action)), rtol=1e-6, atol=1e-8)
        self.assertGreaterEqual(gv.quadratic_form, -1e-12)

    def test_linearized_varpro_hessian_action_has_frozen_schur_taylor_model(self) -> None:
        rng = np.random.default_rng(9110)
        dynamics, decoder, manifest_path, _, tmpdir = self._build_general_varpro_fixture(rng)
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
            regularization=DecoderTikhonovRegularization(),
        )
        prepared = workflow.prepare(dynamics)
        direction = rng.standard_normal(dynamics_parameter_vector(dynamics).shape)
        direction /= np.linalg.norm(direction)

        action = workflow.evaluate_linearized_varpro_hessian_action_from_prepared_state(prepared, direction)
        residual = self._weighted_residual_vector_from_best_response(prepared.result.best_response_context)
        tangent = self._varpro_residual_tangent_vector(prepared, direction, action.decoder_direction, "lagged_midpoint")
        model_value = 0.5 * float(np.dot(residual, residual))
        model_gradient_action = float(np.dot(residual, tangent))
        model_hessian_action = float(np.dot(tangent, tangent))

        np.testing.assert_allclose(model_gradient_action, float(np.dot(prepared.gradient, direction)), rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(model_hessian_action, float(np.dot(direction, action.action)), rtol=1e-6, atol=1e-12)
        for eps in (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4):
            shifted_model_value = 0.5 * float(np.dot(residual + eps * tangent, residual + eps * tangent))
            taylor_value = model_value + eps * model_gradient_action + 0.5 * eps * eps * model_hessian_action
            np.testing.assert_allclose(shifted_model_value, taylor_value, rtol=1e-12, atol=1e-16)

    def test_general_quadratic_gn_varpro_action_is_symmetric_and_matches_qform(self) -> None:
        rng = np.random.default_rng(9106)
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
        gv = workflow.evaluate_gn_varpro_hessian_action_from_prepared_state(prepared, v)
        counts_after_v = evaluator.solve_count_record()
        gw = workflow.evaluate_gn_varpro_hessian_action_from_prepared_state(prepared, w)
        counts_after_w = evaluator.solve_count_record()
        sample_count = prepared.result.best_response_context.forward_cache.global_sample_count

        self.assertEqual(counts_after_v["tangent_forward"] - counts_before["tangent_forward"], sample_count)
        self.assertEqual(counts_after_w["tangent_forward"] - counts_after_v["tangent_forward"], sample_count)
        self.assertEqual(counts_after_v["adjoint"] - counts_before["adjoint"], sample_count)
        self.assertEqual(counts_after_w["adjoint"] - counts_after_v["adjoint"], sample_count)

        np.testing.assert_allclose(float(np.dot(v, gv.action)), gv.quadratic_form, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(float(np.dot(v, gw.action)), float(np.dot(w, gv.action)), rtol=1e-6, atol=1e-8)
        self.assertGreaterEqual(gv.quadratic_form, -1e-12)

    def test_general_quadratic_gn_varpro_action_matches_residual_finite_difference(self) -> None:
        rng = np.random.default_rng(9107)
        dynamics, decoder, manifest_path, _, tmpdir = self._build_general_varpro_fixture(rng)
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
            regularization=DecoderTikhonovRegularization(),
        )
        prepared = workflow.prepare(dynamics)
        base_vector = dynamics_parameter_vector(dynamics)
        v = rng.standard_normal(base_vector.shape)
        w = rng.standard_normal(base_vector.shape)
        v /= np.linalg.norm(v)
        w /= np.linalg.norm(w)

        hv = workflow.evaluate_gn_varpro_hessian_action_from_prepared_state(prepared, v)
        hw = workflow.evaluate_gn_varpro_hessian_action_from_prepared_state(prepared, w)
        residual = self._weighted_residual_vector_from_best_response(prepared.result.best_response_context)
        jv = self._varpro_residual_tangent_vector(prepared, v, hv.decoder_direction, "lagged_midpoint")
        jw = self._varpro_residual_tangent_vector(prepared, w, hw.decoder_direction, "lagged_midpoint")

        np.testing.assert_allclose(float(np.dot(v, hv.action)), float(np.dot(jv, jv)), rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(float(np.dot(w, hv.action)), float(np.dot(jw, jv)), rtol=1e-6, atol=1e-12)

        eps_values = np.array([1.0e-3, 3.0e-4, 1.0e-4], dtype=np.float64)
        residual_taylor_errors = []
        for eps in eps_values:
            plus = dynamics_from_parameter_vector(dynamics, base_vector + eps * v)
            residual_plus = self._reduced_residual_vector(workflow, plus)
            residual_taylor_errors.append(float(np.linalg.norm(residual_plus - residual - eps * jv)))
        residual_taylor_slope = self._fit_slope(eps_values, np.asarray(residual_taylor_errors, dtype=np.float64))
        self.assertGreaterEqual(residual_taylor_slope, 1.70)
        self.assertLessEqual(residual_taylor_slope, 2.30)

        eps = 3.0e-4
        def finite_difference_residual(direction: np.ndarray) -> np.ndarray:
            plus = dynamics_from_parameter_vector(dynamics, base_vector + eps * direction)
            minus = dynamics_from_parameter_vector(dynamics, base_vector - eps * direction)
            return (
                self._reduced_residual_vector(workflow, plus)
                - self._reduced_residual_vector(workflow, minus)
            ) / (2.0 * eps)

        jv_fd = finite_difference_residual(v)
        jw_fd = finite_difference_residual(w)
        np.testing.assert_allclose(float(np.dot(jv, jv)), float(np.dot(jv_fd, jv_fd)), rtol=2e-3, atol=1e-12)
        np.testing.assert_allclose(float(np.dot(jw, jv)), float(np.dot(jw_fd, jv_fd)), rtol=2e-3, atol=1e-12)

    def test_quadratic_gn_varpro_gauge_vertical_direction_is_null_with_zero_initial_state(self) -> None:
        rng = np.random.default_rng(9108)
        r, dq = 3, 2
        dynamics = QuadraticDynamics(
            a=-0.10 * np.eye(r) + 0.02 * rng.standard_normal((r, r)),
            mu_h=0.015 * rng.standard_normal(mu_h_dimension(r)),
            b=np.array([[0.15], [-0.08], [0.04]], dtype=np.float64),
            c=0.01 * rng.standard_normal(r),
        )
        decoder = QuadraticDecoder(
            v1=0.18 * rng.standard_normal((dq, r)),
            v2=0.04 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.02 * rng.standard_normal(dq),
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_zero_initial_dataset(Path(tmp), dynamics, decoder, sample_count=8, rng=rng)
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
                regularization=DecoderTikhonovRegularization(),
            )
            prepared = workflow.prepare(dynamics)
            vertical = quotient_vertical_basis(dynamics)
            self.assertGreater(vertical.shape[1], 0)
            for _ in range(20):
                coeffs = rng.standard_normal(vertical.shape[1])
                direction = vertical @ coeffs
                direction = direction / np.linalg.norm(direction)
                action = workflow.evaluate_gn_varpro_hessian_action_from_prepared_state(prepared, direction)
                self.assertLess(abs(action.quadratic_form), 1e-14)
                self.assertLess(abs(float(np.dot(direction, action.action))), 1e-14)

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

    def _write_zero_initial_dataset(
        self,
        root: Path,
        dynamics: QuadraticDynamics,
        decoder: QuadraticDecoder,
        sample_count: int,
        rng: np.random.Generator,
    ) -> Path:
        observation_times = np.linspace(0.0, 0.20, 6)
        sample_paths: list[str] = []
        sample_ids: list[str] = []
        for sample_idx in range(sample_count):
            u0 = np.zeros(dynamics.dimension, dtype=np.float64)
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

    @staticmethod
    def _weighted_residual_vector_from_best_response(best_response) -> np.ndarray:
        parts = []
        for rollout_entry in best_response.forward_cache.local_rollouts:
            for state, target, weight in zip(
                rollout_entry.observed_states,
                rollout_entry.sample.qoi_observations,
                rollout_entry.observation_weights,
                strict=True,
            ):
                parts.append(np.sqrt(float(weight)) * (best_response.decoder.decode(state) - target))
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts, axis=0)

    @classmethod
    def _reduced_residual_vector(cls, workflow, dynamics) -> np.ndarray:
        result = workflow.evaluate_objective_and_gradient(dynamics)
        return cls._weighted_residual_vector_from_best_response(result.best_response_context)

    @staticmethod
    def _varpro_residual_tangent_vector(prepared_state, direction: np.ndarray, decoder_direction: QuadraticDecoder, time_integrator: str) -> np.ndarray:
        dynamics_direction = unpack_dynamics_parameter_vector(prepared_state.dynamics, direction)
        best_response = prepared_state.result.best_response_context
        parts = []
        for rollout_entry in best_response.forward_cache.local_rollouts:
            tangent_states = rollout_dynamics_parameter_tangent_from_base_rollout(
                dynamics=prepared_state.dynamics,
                direction=dynamics_direction,
                base_rollout=rollout_entry.rollout,
                input_function=rollout_entry.input_function,
                time_integrator=time_integrator,
            )
            observed_tangents = tangent_states[rollout_entry.observation_indices]
            for state, state_tangent, weight in zip(
                rollout_entry.observed_states,
                observed_tangents,
                rollout_entry.observation_weights,
                strict=True,
            ):
                tangent = best_response.decoder.jacobian(state) @ state_tangent + decoder_direction.decode(state)
                parts.append(np.sqrt(float(weight)) * tangent)
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts, axis=0)

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
