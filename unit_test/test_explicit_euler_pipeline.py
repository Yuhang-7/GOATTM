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

from goattm.core.parametrization import compressed_quadratic_dimension, compressed_h_to_mu_h, mu_h_dimension  # noqa: E402
from goattm.core.quadratic import quadratic_jacobian_matrix  # noqa: E402
from goattm.data.npz_dataset import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest  # noqa: E402
from goattm.losses.qoi_loss import rollout_qoi_loss_and_gradients_from_observations  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.problems.reduced_qoi_best_response import (  # noqa: E402
    DecoderTikhonovRegularization,
    DynamicsTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
    dynamics_from_parameter_vector,
    dynamics_parameter_vector,
)
from goattm.runtime.distributed import DistributedContext  # noqa: E402
from goattm.solvers import (  # noqa: E402
    compute_explicit_euler_discrete_adjoint,
    compute_explicit_euler_incremental_discrete_adjoint,
    rollout_explicit_euler,
    rollout_explicit_euler_tangent_from_base_rollout,
    rollout_explicit_euler_to_observation_times,
)


class ExplicitEulerPipelineTest(unittest.TestCase):
    def test_tangent_and_incremental_adjoint_match_finite_difference(self) -> None:
        rng = np.random.default_rng(8401)
        r = 2
        dp = 1
        a = np.array([[-0.2, 0.1], [-0.1, -0.15]], dtype=float)
        h_matrix = np.array(
            [
                [0.05, -0.03, 0.01],
                [-0.02, 0.04, -0.015],
            ],
            dtype=float,
        )
        b = np.array([[0.2], [-0.1]], dtype=float)
        c = np.array([0.03, -0.04], dtype=float)
        dynamics = QuadraticDynamics(
            a=a,
            mu_h=compressed_h_to_mu_h(h_matrix, r),
            b=b,
            c=c,
        )
        delta_a = rng.standard_normal(a.shape) * 0.05
        delta_h_matrix = rng.standard_normal(h_matrix.shape) * 0.03
        delta_b = rng.standard_normal(b.shape) * 0.05
        delta_c = rng.standard_normal(c.shape) * 0.05
        direction = QuadraticDynamics(
            a=delta_a,
            mu_h=compressed_h_to_mu_h(delta_h_matrix, r),
            b=delta_b,
            c=delta_c,
        )
        u0 = np.array([0.1, -0.2], dtype=float)
        t_final = 0.1
        dt = 0.01

        def input_function(t: float) -> np.ndarray:
            return np.array([0.3 - 0.05 * t], dtype=float)

        base_rollout = rollout_explicit_euler(
            dynamics=dynamics,
            u0=u0,
            t_final=t_final,
            max_dt=dt,
            input_function=input_function,
        )
        self.assertTrue(base_rollout.success)

        tangent_states = rollout_explicit_euler_tangent_from_base_rollout(
            dynamics=dynamics,
            base_rollout=base_rollout,
            parameter_action=lambda state, time: direction.a @ state + direction.h_matrix @ np.array(  # noqa: E731
                [state[0] * state[0], state[1] * state[0], state[1] * state[1]],
                dtype=float,
            ) + direction.b @ input_function(time) + direction.c,
        )

        eps_values = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3], dtype=float)
        tangent_errors = []
        for eps in eps_values:
            perturbed_plus = QuadraticDynamics(
                a=a + eps * delta_a,
                mu_h=dynamics.mu_h + eps * direction.mu_h,
                b=b + eps * delta_b,
                c=c + eps * delta_c,
            )
            perturbed_minus = QuadraticDynamics(
                a=a - eps * delta_a,
                mu_h=dynamics.mu_h - eps * direction.mu_h,
                b=b - eps * delta_b,
                c=c - eps * delta_c,
            )
            rollout_plus = rollout_explicit_euler(
                dynamics=perturbed_plus,
                u0=u0,
                t_final=t_final,
                max_dt=dt,
                input_function=input_function,
            )
            rollout_minus = rollout_explicit_euler(
                dynamics=perturbed_minus,
                u0=u0,
                t_final=t_final,
                max_dt=dt,
                input_function=input_function,
            )
            tangent_errors.append(np.linalg.norm((rollout_plus.states - rollout_minus.states) / (2.0 * eps) - tangent_states))
        self.assertLess(max(tangent_errors), 1e-8)

        state_loss_gradients = base_rollout.states.copy()
        base_adjoints = compute_explicit_euler_discrete_adjoint(
            dynamics=dynamics,
            states=base_rollout.states,
            times=base_rollout.times,
            dt_history=base_rollout.dt_history,
            state_loss_gradients=state_loss_gradients,
        )
        adjoint_tangents = compute_explicit_euler_incremental_discrete_adjoint(
            dynamics=dynamics,
            rollout=base_rollout,
            tangent_states=tangent_states,
            base_adjoints=base_adjoints,
            state_loss_gradient_direction=tangent_states,
            jacobian_direction=lambda state, state_tangent, time: direction.a + quadratic_jacobian_matrix(dynamics.h_matrix, state_tangent) + quadratic_jacobian_matrix(direction.h_matrix, state),  # noqa: E731
        )

        adjoint_errors = []
        for eps in eps_values:
            perturbed_plus = QuadraticDynamics(
                a=a + eps * delta_a,
                mu_h=dynamics.mu_h + eps * direction.mu_h,
                b=b + eps * delta_b,
                c=c + eps * delta_c,
            )
            perturbed_minus = QuadraticDynamics(
                a=a - eps * delta_a,
                mu_h=dynamics.mu_h - eps * direction.mu_h,
                b=b - eps * delta_b,
                c=c - eps * delta_c,
            )
            rollout_plus = rollout_explicit_euler(
                dynamics=perturbed_plus,
                u0=u0,
                t_final=t_final,
                max_dt=dt,
                input_function=input_function,
            )
            rollout_minus = rollout_explicit_euler(
                dynamics=perturbed_minus,
                u0=u0,
                t_final=t_final,
                max_dt=dt,
                input_function=input_function,
            )
            adjoints_plus = compute_explicit_euler_discrete_adjoint(
                dynamics=perturbed_plus,
                states=rollout_plus.states,
                times=rollout_plus.times,
                dt_history=rollout_plus.dt_history,
                state_loss_gradients=rollout_plus.states,
            )
            adjoints_minus = compute_explicit_euler_discrete_adjoint(
                dynamics=perturbed_minus,
                states=rollout_minus.states,
                times=rollout_minus.times,
                dt_history=rollout_minus.dt_history,
                state_loss_gradients=rollout_minus.states,
            )
            adjoint_errors.append(np.linalg.norm((adjoints_plus - adjoints_minus) / (2.0 * eps) - adjoint_tangents))
        self.assertLess(max(adjoint_errors), 1e-7)

    def test_reduced_objective_first_and_second_order_taylor_with_explicit_euler(self) -> None:
        rng = np.random.default_rng(8402)
        truth_dynamics, truth_decoder = self._build_truth_problem(rng)
        candidate = self._perturb_dynamics(truth_dynamics, scale=0.05, rng=rng)
        template_decoder = QuadraticDecoder(
            v1=np.zeros_like(truth_decoder.v1),
            v2=np.zeros_like(truth_decoder.v2),
            v0=np.zeros_like(truth_decoder.v0),
        )
        regularization = DecoderTikhonovRegularization(coeff_v1=1e-6, coeff_v2=1e-6, coeff_v0=1e-6)
        dynamics_regularization = DynamicsTikhonovRegularization(
            coeff_s=1e-4,
            coeff_w=1e-4,
            coeff_mu_h=1e-4,
            coeff_b=1e-4,
            coeff_c=1e-4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_dataset(Path(tmpdir), truth_dynamics, truth_decoder, rng)
            evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=manifest_path,
                max_dt=0.01,
                context=DistributedContext(),
                time_integrator="explicit_euler",
                dt_shrink=0.5,
                dt_min=1e-8,
                tol=1e-10,
                max_iter=20,
            )
            workflow = evaluator.build_reduced_objective_workflow(
                decoder_template=template_decoder,
                regularization=regularization,
                dynamics_regularization=dynamics_regularization,
            )
            base = workflow.evaluate_objective_and_gradient(candidate)
            direction = rng.standard_normal(base.gradient.shape[0])
            direction /= np.linalg.norm(direction)
            directional_derivative = float(np.dot(base.gradient, direction))
            hessian_action = workflow.evaluate_hessian_action(candidate, direction).action
            quadratic_term = float(np.dot(direction, hessian_action))
            base_vector = dynamics_parameter_vector(candidate)

            eps_values = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3], dtype=float)
            first_errors = []
            second_errors = []
            for eps in eps_values:
                perturbed = dynamics_from_parameter_vector(candidate, base_vector + eps * direction)
                value_eps = workflow.evaluate_objective(perturbed)
                delta = value_eps - base.objective_value
                first_errors.append(abs(delta - eps * directional_derivative))
                second_errors.append(abs(delta - eps * directional_derivative - 0.5 * eps * eps * quadratic_term))

            first_slope = self._fit_slope(eps_values, np.asarray(first_errors, dtype=float))
            second_slope = self._fit_slope(eps_values, np.asarray(second_errors, dtype=float))
            self.assertGreaterEqual(first_slope, 1.70)
            self.assertLessEqual(first_slope, 2.30)
            self.assertGreaterEqual(second_slope, 2.60)
            self.assertLessEqual(second_slope, 3.40)

    def _build_truth_problem(self, rng: np.random.Generator) -> tuple[StabilizedQuadraticDynamics, QuadraticDecoder]:
        r = 2
        dq = 2
        dp = 1
        s_params = np.array([0.22, 0.0, 0.18], dtype=float)
        w_params = np.array([0.14], dtype=float)
        mu_h = 0.04 * rng.standard_normal(mu_h_dimension(r))
        b = np.array([[0.18], [-0.12]], dtype=float)
        c = np.array([0.02, -0.03], dtype=float)
        dynamics = StabilizedQuadraticDynamics(
            s_params=s_params,
            w_params=w_params,
            mu_h=mu_h,
            b=b,
            c=c,
        )
        decoder = QuadraticDecoder(
            v1=np.array([[1.0, 0.2], [-0.15, 0.9]], dtype=float),
            v2=0.02 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=np.array([0.05, -0.02], dtype=float),
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
        sample_paths = []
        sample_ids = []
        observation_times = np.linspace(0.0, 0.1, 6, dtype=float)
        for sample_index in range(3):
            u0 = 0.08 * rng.standard_normal(dynamics.dimension)

            def input_function(t: float, offset=float(sample_index)) -> np.ndarray:
                return np.array([0.2 + 0.1 * np.cos(1.7 * t + offset)], dtype=float)

            rollout, observation_indices = rollout_explicit_euler_to_observation_times(
                dynamics=dynamics,
                u0=u0,
                observation_times=observation_times,
                max_dt=0.01,
                input_function=input_function,
            )
            qoi_observations = np.stack([decoder.decode(state) for state in rollout.states[observation_indices]], axis=0)
            sample = NpzQoiSample(
                sample_id=f"sample-{sample_index:03d}",
                observation_times=observation_times,
                u0=u0,
                qoi_observations=qoi_observations,
            )
            sample_path = root / f"sample_{sample_index:03d}.npz"
            save_npz_qoi_sample(sample_path, sample)
            sample_paths.append(sample_path)
            sample_ids.append(sample.sample_id)
        manifest = NpzSampleManifest(root_dir=root, sample_paths=tuple(sample_paths), sample_ids=tuple(sample_ids))
        manifest_path = root / "manifest.npz"
        save_npz_sample_manifest(manifest_path, manifest)
        return manifest_path

    def _fit_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        mask = (x > 0.0) & (y > 0.0)
        coeff = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
        return float(coeff[0])


if __name__ == "__main__":
    unittest.main()
