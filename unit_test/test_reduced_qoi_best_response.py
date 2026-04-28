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
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.problems.reduced_qoi_best_response import (  # noqa: E402
    DecoderTikhonovRegularization,
    DynamicsTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
    dynamics_from_parameter_vector,
    dynamics_parameter_vector,
    unpack_dynamics_parameter_vector,
)
from goattm.runtime.distributed import DistributedContext  # noqa: E402
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times  # noqa: E402


class ReducedQoiBestResponseTest(unittest.TestCase):
    def test_forward_and_best_response_cache_reuse_same_object_for_same_dynamics(self) -> None:
        rng = np.random.default_rng(7001)
        truth_dynamics, truth_decoder = self._build_truth_problem(rng)
        candidate_dynamics = self._perturb_dynamics(truth_dynamics, scale=0.08, rng=rng)
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )
        regularization = DecoderTikhonovRegularization(coeff_v1=1e-3, coeff_v2=2e-3, coeff_v0=3e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_dataset(Path(tmpdir), truth_dynamics, truth_decoder, rng)
            evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=manifest_path,
                max_dt=0.04,
                context=DistributedContext(),
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter=30,
            )
            forward_a = evaluator.get_forward_rollouts(candidate_dynamics)
            forward_b = evaluator.get_forward_rollouts(candidate_dynamics)
            self.assertIs(forward_a, forward_b)

            best_a = evaluator.solve_decoder_best_response(candidate_dynamics, template_decoder, regularization)
            best_b = evaluator.solve_decoder_best_response(candidate_dynamics, template_decoder, regularization)
            self.assertIs(best_a, best_b)

            workflow = evaluator.build_reduced_objective_workflow(
                decoder_template=template_decoder,
                regularization=regularization,
            )
            workflow_result = workflow.evaluate_objective_and_gradient(candidate_dynamics)
            direct_result = evaluator.evaluate_reduced_objective_and_gradient(
                candidate_dynamics,
                template_decoder,
                regularization,
            )
            self.assertTrue(np.allclose(workflow_result.gradient, direct_result.gradient))
            self.assertAlmostEqual(workflow.evaluate_objective(candidate_dynamics), direct_result.objective_value)
            explicit_hessian = workflow.evaluate_explicit_hessian(candidate_dynamics).hessian
            self.assertEqual(explicit_hessian.shape, (workflow_result.gradient.shape[0], workflow_result.gradient.shape[0]))
            self.assertTrue(np.all(np.isfinite(explicit_hessian)))

            direction = np.ones_like(workflow_result.gradient)
            hess_action = workflow.lossfunction_hessian_action_wrt_mug(candidate_dynamics, direction)
            self.assertEqual(hess_action.shape, workflow_result.gradient.shape)
            self.assertTrue(np.all(np.isfinite(hess_action)))
            self.assertTrue(np.allclose(hess_action, explicit_hessian @ direction, atol=1e-8, rtol=1e-8))

            direction = rng.standard_normal(workflow_result.gradient.shape[0])
            direction /= np.linalg.norm(direction)
            base_gradient = workflow_result.gradient
            hessian_action = workflow.lossfunction_hessian_action_wrt_mug(candidate_dynamics, direction)
            eps_values = np.array([1e-4, 3e-4, 1e-3, 3e-3], dtype=float)
            errors = []
            base_vector = dynamics_parameter_vector(candidate_dynamics)
            for eps in eps_values:
                perturbed = dynamics_from_parameter_vector(candidate_dynamics, base_vector + eps * direction)
                grad_eps = workflow.evaluate_gradient(perturbed)
                errors.append(np.linalg.norm(grad_eps - base_gradient - eps * hessian_action))
            slope = self._fit_slope(eps_values, np.asarray(errors, dtype=float))
            self.assertGreaterEqual(slope, 1.70)
            self.assertLessEqual(slope, 2.30)

    def test_reduced_qoi_gradient_with_best_response_decoder_passes_taylor_test(self) -> None:
        rng = np.random.default_rng(7002)
        truth_dynamics, truth_decoder = self._build_truth_problem(rng)
        candidate_dynamics = self._perturb_dynamics(truth_dynamics, scale=0.12, rng=rng)
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )
        regularization = DecoderTikhonovRegularization(
            coeff_v1=float(10.0 ** rng.uniform(-4.0, -2.5)),
            coeff_v2=float(10.0 ** rng.uniform(-4.0, -2.5)),
            coeff_v0=float(10.0 ** rng.uniform(-4.0, -2.5)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_dataset(Path(tmpdir), truth_dynamics, truth_decoder, rng)
            evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=manifest_path,
                max_dt=0.04,
                context=DistributedContext(),
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter=30,
            )
            base = evaluator.evaluate_reduced_data_loss_and_gradient(candidate_dynamics, template_decoder, regularization)
            for rollout_entry in base.best_response_context.forward_cache.local_rollouts:
                self.assertEqual(rollout_entry.rollout.dt_reductions, 0)

            direction_vector = rng.standard_normal(dynamics_parameter_vector(candidate_dynamics).shape)
            direction_vector /= np.linalg.norm(direction_vector)
            direction = unpack_dynamics_parameter_vector(candidate_dynamics, direction_vector)
            directional_derivative = float(np.dot(base.reduced_data_gradient, direction_vector))

            eps_values = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4], dtype=float)
            zero_order = []
            first_order = []
            for eps in eps_values:
                perturbed = StabilizedQuadraticDynamics(
                    s_params=candidate_dynamics.s_params + eps * direction.s_params,
                    w_params=candidate_dynamics.w_params + eps * direction.w_params,
                    mu_h=candidate_dynamics.mu_h + eps * direction.mu_h,
                    b=None if candidate_dynamics.b is None else candidate_dynamics.b + eps * direction.b,
                    c=candidate_dynamics.c + eps * direction.c,
                )
                value_eps = evaluator.evaluate_reduced_data_loss_and_gradient(perturbed, template_decoder, regularization).data_loss
                delta = value_eps - base.data_loss
                zero_order.append(abs(delta))
                first_order.append(abs(delta - eps * directional_derivative))

            zero_order = np.asarray(zero_order, dtype=float)
            first_order = np.asarray(first_order, dtype=float)
            zero_slope = self._fit_slope(eps_values, zero_order)
            first_slope = self._fit_slope(eps_values, first_order)

            self.assertGreaterEqual(zero_slope, 0.85)
            self.assertLessEqual(zero_slope, 1.15)
            self.assertGreaterEqual(first_slope, 1.70)
            self.assertLessEqual(first_slope, 2.30)

    def test_goam_reduced_objective_gradient_passes_taylor_test(self) -> None:
        rng = np.random.default_rng(7003)
        truth_dynamics, truth_decoder = self._build_truth_problem(rng)
        candidate_dynamics = self._perturb_dynamics(truth_dynamics, scale=0.12, rng=rng)
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )
        regularization = DecoderTikhonovRegularization(
            coeff_v1=float(10.0 ** rng.uniform(-4.0, -2.5)),
            coeff_v2=float(10.0 ** rng.uniform(-4.0, -2.5)),
            coeff_v0=float(10.0 ** rng.uniform(-4.0, -2.5)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_dataset(Path(tmpdir), truth_dynamics, truth_decoder, rng)
            evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=manifest_path,
                max_dt=0.04,
                context=DistributedContext(),
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter=30,
            )
            base = evaluator.evaluate_goam_reduced_objective_and_gradient(candidate_dynamics, template_decoder, regularization)
            direction_vector = rng.standard_normal(dynamics_parameter_vector(candidate_dynamics).shape)
            direction_vector /= np.linalg.norm(direction_vector)
            direction = unpack_dynamics_parameter_vector(candidate_dynamics, direction_vector)
            directional_derivative = float(np.dot(base.reduced_objective_gradient, direction_vector))

            eps_values = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4], dtype=float)
            zero_order = []
            first_order = []
            for eps in eps_values:
                perturbed = StabilizedQuadraticDynamics(
                    s_params=candidate_dynamics.s_params + eps * direction.s_params,
                    w_params=candidate_dynamics.w_params + eps * direction.w_params,
                    mu_h=candidate_dynamics.mu_h + eps * direction.mu_h,
                    b=None if candidate_dynamics.b is None else candidate_dynamics.b + eps * direction.b,
                    c=candidate_dynamics.c + eps * direction.c,
                )
                value_eps = evaluator.evaluate_goam_reduced_objective_and_gradient(
                    perturbed,
                    template_decoder,
                    regularization,
                ).objective_value
                delta = value_eps - base.objective_value
                zero_order.append(abs(delta))
                first_order.append(abs(delta - eps * directional_derivative))

            zero_order = np.asarray(zero_order, dtype=float)
            first_order = np.asarray(first_order, dtype=float)
            zero_slope = self._fit_slope(eps_values, zero_order)
            first_slope = self._fit_slope(eps_values, first_order)

            self.assertGreaterEqual(zero_slope, 0.85)
            self.assertLessEqual(zero_slope, 1.15)
            self.assertGreaterEqual(first_slope, 1.70)
            self.assertLessEqual(first_slope, 2.30)

    def test_goam_reduced_objective_with_dynamics_regularization_passes_taylor_and_hessian_checks(self) -> None:
        rng = np.random.default_rng(7004)
        truth_dynamics, truth_decoder = self._build_truth_problem(rng)
        candidate_dynamics = self._perturb_dynamics(truth_dynamics, scale=0.08, rng=rng)
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )
        regularization = DecoderTikhonovRegularization(coeff_v1=1e-4, coeff_v2=2e-4, coeff_v0=1e-4)
        dynamics_regularization = DynamicsTikhonovRegularization(
            coeff_s=3e-4,
            coeff_w=2e-4,
            coeff_mu_h=4e-4,
            coeff_b=5e-4,
            coeff_c=6e-4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_dataset(Path(tmpdir), truth_dynamics, truth_decoder, rng)
            evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=manifest_path,
                max_dt=0.04,
                context=DistributedContext(),
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter=30,
            )
            workflow = evaluator.build_reduced_objective_workflow(
                decoder_template=template_decoder,
                regularization=regularization,
                dynamics_regularization=dynamics_regularization,
            )
            base = workflow.evaluate_objective_and_gradient(candidate_dynamics)
            direction = rng.standard_normal(base.gradient.shape[0])
            direction /= np.linalg.norm(direction)
            hess_action = workflow.lossfunction_hessian_action_wrt_mug(candidate_dynamics, direction)
            directional_derivative = float(np.dot(base.gradient, direction))

            eps_values = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4], dtype=float)
            first_order = []
            second_order = []
            base_vector = dynamics_parameter_vector(candidate_dynamics)
            for eps in eps_values:
                perturbed = dynamics_from_parameter_vector(candidate_dynamics, base_vector + eps * direction)
                value_eps = workflow.evaluate_objective(perturbed)
                grad_eps = workflow.evaluate_gradient(perturbed)
                delta = value_eps - base.objective_value
                first_order.append(abs(delta - eps * directional_derivative))
                second_order.append(np.linalg.norm(grad_eps - base.gradient - eps * hess_action))

            first_slope = self._fit_slope(eps_values, np.asarray(first_order, dtype=float))
            second_slope = self._fit_slope(eps_values, np.asarray(second_order, dtype=float))
            self.assertGreaterEqual(first_slope, 1.70)
            self.assertLessEqual(first_slope, 2.30)
            self.assertGreaterEqual(second_slope, 1.70)
            self.assertLessEqual(second_slope, 2.30)

    def _build_truth_problem(
        self,
        rng: np.random.Generator,
    ) -> tuple[StabilizedQuadraticDynamics, QuadraticDecoder]:
        dynamics = StabilizedQuadraticDynamics(
            s_params=np.array([0.48, 0.03, -0.02], dtype=float),
            w_params=np.array([0.08], dtype=float),
            mu_h=0.025 * rng.standard_normal(mu_h_dimension(2)),
            b=np.array([[0.35], [-0.12]], dtype=float),
            c=np.array([0.03, -0.015], dtype=float),
        )
        decoder = QuadraticDecoder(
            v1=0.25 * rng.standard_normal((2, 2)),
            v2=0.12 * rng.standard_normal((2, compressed_quadratic_dimension(2))),
            v0=0.08 * rng.standard_normal(2),
        )
        return dynamics, decoder

    def _perturb_dynamics(
        self,
        dynamics: StabilizedQuadraticDynamics,
        scale: float,
        rng: np.random.Generator,
    ) -> StabilizedQuadraticDynamics:
        return StabilizedQuadraticDynamics(
            s_params=dynamics.s_params + scale * rng.standard_normal(dynamics.s_params.shape),
            w_params=dynamics.w_params + scale * rng.standard_normal(dynamics.w_params.shape),
            mu_h=dynamics.mu_h + scale * rng.standard_normal(dynamics.mu_h.shape),
            b=None if dynamics.b is None else dynamics.b + scale * rng.standard_normal(dynamics.b.shape),
            c=dynamics.c + scale * rng.standard_normal(dynamics.c.shape),
        )

    def _write_dataset(
        self,
        root: Path,
        dynamics: StabilizedQuadraticDynamics,
        decoder: QuadraticDecoder,
        rng: np.random.Generator,
    ) -> Path:
        observation_times = np.array([0.0, 0.04, 0.08, 0.12, 0.16], dtype=float)
        sample_paths = []
        sample_ids = []
        for sample_idx in range(4):
            u0 = 0.1 * rng.standard_normal(dynamics.dimension)
            input_values = np.column_stack(
                [0.2 + 0.08 * np.sin(2.0 * np.pi * observation_times + 0.4 * sample_idx)]
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
                dt_shrink=0.5,
                dt_min=1e-12,
                tol=1e-12,
                max_iter=30,
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

    def _fit_slope(self, eps_values: np.ndarray, error_values: np.ndarray) -> float:
        coeffs = np.polyfit(np.log10(eps_values), np.log10(error_values), 1)
        return float(coeffs[0])


if __name__ == "__main__":
    unittest.main()
