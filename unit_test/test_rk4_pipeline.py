from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import (  # noqa: E402
    compressed_h_to_mu_h,
    compressed_quadratic_dimension,
    mu_h_dimension,
    skew_symmetric_dimension,
    upper_triangular_dimension,
)
from goattm.core.quadratic import quadratic_jacobian_matrix  # noqa: E402
from goattm.losses.qoi_loss import rollout_qoi_loss_and_gradients  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402
from goattm.solvers.rk4 import (  # noqa: E402
    compute_rk4_discrete_adjoint,
    compute_rk4_incremental_discrete_adjoint,
    rollout_rk4,
    rollout_rk4_tangent_from_base_rollout,
)


class Rk4PipelineTest(unittest.TestCase):
    def test_rk4_tangent_and_incremental_adjoint_match_finite_difference(self) -> None:
        rng = np.random.default_rng(2026042801)
        r = 2
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
        dynamics = QuadraticDynamics(a=a, mu_h=compressed_h_to_mu_h(h_matrix, r), b=b, c=c)

        delta_a = 0.05 * rng.standard_normal(a.shape)
        raw_delta_h = 0.03 * rng.standard_normal(h_matrix.shape)
        delta_b = 0.05 * rng.standard_normal(b.shape)
        delta_c = 0.05 * rng.standard_normal(c.shape)
        direction = QuadraticDynamics(
            a=delta_a,
            mu_h=compressed_h_to_mu_h(raw_delta_h, r),
            b=delta_b,
            c=delta_c,
        )

        u0 = np.array([0.1, -0.2], dtype=float)

        def input_function(t: float) -> np.ndarray:
            return np.array([0.3 - 0.05 * t], dtype=float)

        def parameter_action(state: np.ndarray, time: float) -> np.ndarray:
            return direction.rhs(state, p=input_function(time))

        def jacobian_direction(state: np.ndarray, state_tangent: np.ndarray, time: float) -> np.ndarray:
            _ = time
            return (
                direction.a
                + quadratic_jacobian_matrix(dynamics.h_matrix, state_tangent)
                + quadratic_jacobian_matrix(direction.h_matrix, state)
            )

        base_rollout = rollout_rk4(
            dynamics=dynamics,
            u0=u0,
            t_final=0.1,
            max_dt=0.01,
            input_function=input_function,
        )
        self.assertTrue(base_rollout.success)

        tangent_states = rollout_rk4_tangent_from_base_rollout(
            dynamics=dynamics,
            base_rollout=base_rollout,
            parameter_action=parameter_action,
            input_function=input_function,
        )

        state_loss_gradients = base_rollout.states.copy()
        base_adjoints = compute_rk4_discrete_adjoint(
            dynamics=dynamics,
            states=base_rollout.states,
            times=base_rollout.times,
            dt_history=base_rollout.dt_history,
            state_loss_gradients=state_loss_gradients,
            input_function=input_function,
        )
        adjoint_tangents = compute_rk4_incremental_discrete_adjoint(
            dynamics=dynamics,
            rollout=base_rollout,
            tangent_states=tangent_states,
            base_adjoints=base_adjoints,
            state_loss_gradient_direction=tangent_states,
            jacobian_direction=jacobian_direction,
            parameter_action=parameter_action,
            input_function=input_function,
        )

        eps_values = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3], dtype=float)
        tangent_errors = []
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
            rollout_plus = rollout_rk4(perturbed_plus, u0, t_final=0.1, max_dt=0.01, input_function=input_function)
            rollout_minus = rollout_rk4(perturbed_minus, u0, t_final=0.1, max_dt=0.01, input_function=input_function)
            tangent_errors.append(np.linalg.norm((rollout_plus.states - rollout_minus.states) / (2.0 * eps) - tangent_states))

            adjoints_plus = compute_rk4_discrete_adjoint(
                dynamics=perturbed_plus,
                states=rollout_plus.states,
                times=rollout_plus.times,
                dt_history=rollout_plus.dt_history,
                state_loss_gradients=rollout_plus.states,
                input_function=input_function,
            )
            adjoints_minus = compute_rk4_discrete_adjoint(
                dynamics=perturbed_minus,
                states=rollout_minus.states,
                times=rollout_minus.times,
                dt_history=rollout_minus.dt_history,
                state_loss_gradients=rollout_minus.states,
                input_function=input_function,
            )
            adjoint_errors.append(np.linalg.norm((adjoints_plus - adjoints_minus) / (2.0 * eps) - adjoint_tangents))

        self.assertLess(max(tangent_errors), 1e-10)
        self.assertLess(max(adjoint_errors), 1e-6)

    def test_rk4_qoi_discrete_adjoint_gradient_passes_taylor_test(self) -> None:
        rng = np.random.default_rng(2026042802)
        r = 2
        dq = 1
        dp = 1
        s_params = 0.15 * rng.standard_normal(upper_triangular_dimension(r))
        w_params = 0.1 * rng.standard_normal(skew_symmetric_dimension(r))
        mu_h = 0.05 * rng.standard_normal(mu_h_dimension(r))
        b = 0.1 * rng.standard_normal((r, dp))
        c = 0.1 * rng.standard_normal(r)
        decoder = QuadraticDecoder(
            v1=0.2 * rng.standard_normal((dq, r)),
            v2=0.05 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.1 * rng.standard_normal(dq),
        )
        u0 = 0.1 * rng.standard_normal(r)
        qoi_observations = rng.standard_normal((4, dq))

        def input_function(t: float) -> np.ndarray:
            return np.array([0.3 + 0.1 * t], dtype=float)

        def build_dynamics(
            s_vec: np.ndarray,
            w_vec: np.ndarray,
            mu_vec: np.ndarray,
            b_mat: np.ndarray,
            c_vec: np.ndarray,
        ) -> StabilizedQuadraticDynamics:
            return StabilizedQuadraticDynamics(
                s_params=s_vec,
                w_params=w_vec,
                mu_h=mu_vec,
                b=b_mat,
                c=c_vec,
            )

        result = rollout_qoi_loss_and_gradients(
            dynamics=build_dynamics(s_params, w_params, mu_h, b, c),
            decoder=decoder,
            u0=u0,
            t_final=0.15,
            dt_initial=0.05,
            qoi_observations=qoi_observations,
            input_function=input_function,
            time_integrator="rk4",
        )

        ds = rng.standard_normal(s_params.shape)
        dw = rng.standard_normal(w_params.shape)
        dmu = rng.standard_normal(mu_h.shape)
        db = rng.standard_normal(b.shape)
        dc = rng.standard_normal(c.shape)
        directional_derivative = (
            float(np.dot(result.dynamics_gradients["s_params"], ds))
            + float(np.dot(result.dynamics_gradients["w_params"], dw))
            + float(np.dot(result.dynamics_gradients["mu_h"], dmu))
            + float(np.sum(result.dynamics_gradients["b"] * db))
            + float(np.dot(result.dynamics_gradients["c"], dc))
        )

        eps_values = np.array([1e-3, 5e-4, 2.5e-4, 1.25e-4], dtype=float)
        zero_order = []
        first_order = []
        for eps in eps_values:
            trial = rollout_qoi_loss_and_gradients(
                dynamics=build_dynamics(
                    s_params + eps * ds,
                    w_params + eps * dw,
                    mu_h + eps * dmu,
                    b + eps * db,
                    c + eps * dc,
                ),
                decoder=decoder,
                u0=u0,
                t_final=0.15,
                dt_initial=0.05,
                qoi_observations=qoi_observations,
                input_function=input_function,
                time_integrator="rk4",
            )
            delta = trial.loss - result.loss
            zero_order.append(abs(delta))
            first_order.append(abs(delta - eps * directional_derivative))

        zero_rates = np.asarray(zero_order[:-1]) / np.asarray(zero_order[1:])
        first_rates = np.asarray(first_order[:-1]) / np.asarray(first_order[1:])
        self.assertTrue(np.all(zero_rates > 1.8))
        self.assertTrue(np.all(zero_rates < 2.2))
        self.assertTrue(np.all(first_rates > 3.5))


if __name__ == "__main__":
    unittest.main()
