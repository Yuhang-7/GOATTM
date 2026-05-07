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
    compressed_quadratic_dimension,
    compressed_h_gradient_to_skew_cp,
    skew_cp_direction_to_compressed_h,
    skew_cp_parameter_action,
    skew_cp_quadratic_eval,
    skew_cp_quadratic_eval_batch,
    skew_cp_quadratic_jacobian_matrix,
    skew_cp_to_compressed_h,
)
from goattm.core.quadratic import quadratic_eval, quadratic_jacobian_matrix  # noqa: E402
from goattm.losses.qoi_loss import rollout_qoi_loss_and_gradients  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.skew_cp_quadratic_dynamics import SkewCPQuadraticDynamics  # noqa: E402
from goattm.problems.reduced_qoi_best_response import (  # noqa: E402
    DynamicsTikhonovRegularization,
    dynamics_from_parameter_vector,
    dynamics_parameter_vector,
    dynamics_regularization_gradient_vector,
    pack_dynamics_gradient_vector,
    rhs_parameter_action,
    unpack_dynamics_parameter_vector,
)


def _random_factors(rng: np.random.Generator, d: int = 4, rank: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        rng.standard_normal((d, rank)),
        rng.standard_normal((d, rank)),
        rng.standard_normal((d, rank)),
    )


class SkewCPParametrizationTest(unittest.TestCase):
    def test_quadratic_eval_matches_materialized_h_matrix(self) -> None:
        rng = np.random.default_rng(701)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        state = rng.standard_normal(u_matrix.shape[0])
        h_matrix = skew_cp_to_compressed_h(u_matrix, v_matrix, z_matrix)

        expected = quadratic_eval(h_matrix, state)
        actual = skew_cp_quadratic_eval(u_matrix, v_matrix, z_matrix, state)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_skew_cp_quadratic_is_energy_preserving(self) -> None:
        rng = np.random.default_rng(702)
        u_matrix, v_matrix, z_matrix = _random_factors(rng, d=5, rank=4)

        for _ in range(12):
            state = rng.standard_normal(u_matrix.shape[0])
            defect = float(np.dot(state, skew_cp_quadratic_eval(u_matrix, v_matrix, z_matrix, state)))
            self.assertAlmostEqual(defect, 0.0, places=12)

    def test_batch_eval_matches_loop(self) -> None:
        rng = np.random.default_rng(703)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        states = rng.standard_normal((7, u_matrix.shape[0]))

        expected = np.vstack(
            [skew_cp_quadratic_eval(u_matrix, v_matrix, z_matrix, state) for state in states]
        )
        actual = skew_cp_quadratic_eval_batch(u_matrix, v_matrix, z_matrix, states)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_jacobian_matches_materialized_h_matrix_and_finite_difference(self) -> None:
        rng = np.random.default_rng(704)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        state = rng.standard_normal(u_matrix.shape[0])
        direction = rng.standard_normal(u_matrix.shape[0])
        eps = 1e-7
        h_matrix = skew_cp_to_compressed_h(u_matrix, v_matrix, z_matrix)

        jacobian = skew_cp_quadratic_jacobian_matrix(u_matrix, v_matrix, z_matrix, state)
        materialized = quadratic_jacobian_matrix(h_matrix, state)
        finite_diff = (
            skew_cp_quadratic_eval(u_matrix, v_matrix, z_matrix, state + eps * direction)
            - skew_cp_quadratic_eval(u_matrix, v_matrix, z_matrix, state - eps * direction)
        ) / (2.0 * eps)

        np.testing.assert_allclose(jacobian, materialized, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jacobian @ direction, finite_diff, rtol=1e-6, atol=1e-7)

    def test_directional_h_matrix_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(705)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        du_matrix, dv_matrix, dz_matrix = _random_factors(rng)
        eps = 1e-7

        analytic = skew_cp_direction_to_compressed_h(
            u_matrix,
            v_matrix,
            z_matrix,
            du_matrix,
            dv_matrix,
            dz_matrix,
        )
        finite_diff = (
            skew_cp_to_compressed_h(
                u_matrix + eps * du_matrix,
                v_matrix + eps * dv_matrix,
                z_matrix + eps * dz_matrix,
            )
            - skew_cp_to_compressed_h(
                u_matrix - eps * du_matrix,
                v_matrix - eps * dv_matrix,
                z_matrix - eps * dz_matrix,
            )
        ) / (2.0 * eps)

        np.testing.assert_allclose(analytic, finite_diff, rtol=1e-6, atol=1e-7)

    def test_parameter_action_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(706)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        du_matrix, dv_matrix, dz_matrix = _random_factors(rng)
        state = rng.standard_normal(u_matrix.shape[0])
        eps = 1e-7

        analytic = skew_cp_parameter_action(
            u_matrix,
            v_matrix,
            z_matrix,
            du_matrix,
            dv_matrix,
            dz_matrix,
            state,
        )
        finite_diff = (
            skew_cp_quadratic_eval(
                u_matrix + eps * du_matrix,
                v_matrix + eps * dv_matrix,
                z_matrix + eps * dz_matrix,
                state,
            )
            - skew_cp_quadratic_eval(
                u_matrix - eps * du_matrix,
                v_matrix - eps * dv_matrix,
                z_matrix - eps * dz_matrix,
                state,
            )
        ) / (2.0 * eps)

        np.testing.assert_allclose(analytic, finite_diff, rtol=1e-6, atol=1e-7)

    def test_compressed_h_gradient_pullback_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(707)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        h_grad = rng.standard_normal(skew_cp_to_compressed_h(u_matrix, v_matrix, z_matrix).shape)
        eps = 1e-7

        analytic_u, analytic_v, analytic_z = compressed_h_gradient_to_skew_cp(
            h_grad,
            u_matrix,
            v_matrix,
            z_matrix,
        )

        def objective(
            u_value: np.ndarray,
            v_value: np.ndarray,
            z_value: np.ndarray,
        ) -> float:
            h_matrix = skew_cp_to_compressed_h(u_value, v_value, z_value)
            return float(np.sum(h_grad * h_matrix))

        finite_u = np.zeros_like(u_matrix)
        finite_v = np.zeros_like(v_matrix)
        finite_z = np.zeros_like(z_matrix)
        for array_name, target in (("u", finite_u), ("v", finite_v), ("z", finite_z)):
            source = {"u": u_matrix, "v": v_matrix, "z": z_matrix}[array_name]
            for index in np.ndindex(source.shape):
                perturb = np.zeros_like(source)
                perturb[index] = eps
                u_plus, v_plus, z_plus = u_matrix, v_matrix, z_matrix
                u_minus, v_minus, z_minus = u_matrix, v_matrix, z_matrix
                if array_name == "u":
                    u_plus = u_matrix + perturb
                    u_minus = u_matrix - perturb
                elif array_name == "v":
                    v_plus = v_matrix + perturb
                    v_minus = v_matrix - perturb
                else:
                    z_plus = z_matrix + perturb
                    z_minus = z_matrix - perturb
                target[index] = (
                    objective(u_plus, v_plus, z_plus) - objective(u_minus, v_minus, z_minus)
                ) / (2.0 * eps)

        np.testing.assert_allclose(analytic_u, finite_u, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(analytic_v, finite_v, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(analytic_z, finite_z, rtol=1e-6, atol=1e-7)

    def test_dynamics_parameter_vector_round_trip_and_regularization_gradient(self) -> None:
        rng = np.random.default_rng(708)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        d = u_matrix.shape[0]
        input_dim = 2
        dynamics = SkewCPQuadraticDynamics(
            a=rng.standard_normal((d, d)),
            skew_u=u_matrix,
            skew_v=v_matrix,
            skew_z=z_matrix,
            b=rng.standard_normal((d, input_dim)),
            c=rng.standard_normal(d),
        )

        vector = dynamics_parameter_vector(dynamics)
        recovered = dynamics_from_parameter_vector(dynamics, vector)
        self.assertIsInstance(recovered, SkewCPQuadraticDynamics)
        np.testing.assert_allclose(recovered.a, dynamics.a)
        np.testing.assert_allclose(recovered.skew_u, dynamics.skew_u)
        np.testing.assert_allclose(recovered.skew_v, dynamics.skew_v)
        np.testing.assert_allclose(recovered.skew_z, dynamics.skew_z)
        np.testing.assert_allclose(recovered.b, dynamics.b)
        np.testing.assert_allclose(recovered.c, dynamics.c)

        regularization = DynamicsTikhonovRegularization(coeff_a=1e-3, coeff_mu_h=2e-3, coeff_b=3e-3, coeff_c=4e-3)
        gradient = dynamics_regularization_gradient_vector(dynamics, regularization)
        self.assertEqual(gradient.shape, vector.shape)

    def test_rhs_parameter_action_matches_vector_finite_difference(self) -> None:
        rng = np.random.default_rng(709)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        d = u_matrix.shape[0]
        input_dim = 2
        dynamics = SkewCPQuadraticDynamics(
            a=rng.standard_normal((d, d)),
            skew_u=u_matrix,
            skew_v=v_matrix,
            skew_z=z_matrix,
            b=rng.standard_normal((d, input_dim)),
            c=rng.standard_normal(d),
        )
        vector = dynamics_parameter_vector(dynamics)
        delta = rng.standard_normal(vector.shape)
        direction = unpack_dynamics_parameter_vector(dynamics, delta)
        state = rng.standard_normal(d)
        forcing = rng.standard_normal(input_dim)
        eps = 1e-7

        def input_function(_: float) -> np.ndarray:
            return forcing

        analytic = rhs_parameter_action(dynamics, direction, state, 0.25, input_function=input_function)
        plus = dynamics_from_parameter_vector(dynamics, vector + eps * delta)
        minus = dynamics_from_parameter_vector(dynamics, vector - eps * delta)
        finite_diff = (plus.rhs(state, forcing) - minus.rhs(state, forcing)) / (2.0 * eps)

        np.testing.assert_allclose(analytic, finite_diff, rtol=1e-6, atol=1e-7)

    def test_pack_dynamics_gradient_vector_uses_skew_factor_blocks(self) -> None:
        rng = np.random.default_rng(710)
        u_matrix, v_matrix, z_matrix = _random_factors(rng)
        d = u_matrix.shape[0]
        dynamics = SkewCPQuadraticDynamics(
            a=rng.standard_normal((d, d)),
            skew_u=u_matrix,
            skew_v=v_matrix,
            skew_z=z_matrix,
            c=rng.standard_normal(d),
        )
        gradients = {
            "a": rng.standard_normal(dynamics.a.shape),
            "skew_u": rng.standard_normal(dynamics.skew_u.shape),
            "skew_v": rng.standard_normal(dynamics.skew_v.shape),
            "skew_z": rng.standard_normal(dynamics.skew_z.shape),
            "c": rng.standard_normal(dynamics.c.shape),
        }
        vector = pack_dynamics_gradient_vector(dynamics, gradients)

        expected_size = (
            dynamics.a.size
            + dynamics.skew_u.size
            + dynamics.skew_v.size
            + dynamics.skew_z.size
            + dynamics.c.size
        )
        self.assertEqual(vector.shape, (expected_size,))

    def test_rollout_qoi_loss_skew_cp_gradient_passes_taylor_test(self) -> None:
        rng = np.random.default_rng(711)
        d = 2
        rank = 2
        output_dim = 1
        input_dim = 1
        skew_u, skew_v, skew_z = _random_factors(rng, d=d, rank=rank)
        dynamics = SkewCPQuadraticDynamics(
            a=0.08 * rng.standard_normal((d, d)),
            skew_u=0.06 * skew_u,
            skew_v=0.06 * skew_v,
            skew_z=0.06 * skew_z,
            b=0.05 * rng.standard_normal((d, input_dim)),
            c=0.05 * rng.standard_normal(d),
        )
        decoder = QuadraticDecoder(
            v1=0.1 * rng.standard_normal((output_dim, d)),
            v2=0.04 * rng.standard_normal((output_dim, compressed_quadratic_dimension(d))),
            v0=0.02 * rng.standard_normal(output_dim),
        )
        u0 = 0.05 * rng.standard_normal(d)
        qoi_observations = rng.standard_normal((4, output_dim))

        def input_function(t: float) -> np.ndarray:
            return np.array([0.2 + 0.1 * t], dtype=np.float64)

        result = rollout_qoi_loss_and_gradients(
            dynamics=dynamics,
            decoder=decoder,
            u0=u0,
            t_final=0.12,
            dt_initial=0.04,
            qoi_observations=qoi_observations,
            input_function=input_function,
        )

        da = rng.standard_normal(dynamics.a.shape)
        du = rng.standard_normal(dynamics.skew_u.shape)
        dv = rng.standard_normal(dynamics.skew_v.shape)
        dz = rng.standard_normal(dynamics.skew_z.shape)
        db = rng.standard_normal(dynamics.b.shape)
        dc = rng.standard_normal(dynamics.c.shape)
        directional_derivative = (
            float(np.sum(result.dynamics_gradients["a"] * da))
            + float(np.sum(result.dynamics_gradients["skew_u"] * du))
            + float(np.sum(result.dynamics_gradients["skew_v"] * dv))
            + float(np.sum(result.dynamics_gradients["skew_z"] * dz))
            + float(np.sum(result.dynamics_gradients["b"] * db))
            + float(np.dot(result.dynamics_gradients["c"], dc))
        )

        def loss_at(eps: float) -> float:
            perturbed = SkewCPQuadraticDynamics(
                a=dynamics.a + eps * da,
                skew_u=dynamics.skew_u + eps * du,
                skew_v=dynamics.skew_v + eps * dv,
                skew_z=dynamics.skew_z + eps * dz,
                b=dynamics.b + eps * db,
                c=dynamics.c + eps * dc,
            )
            return rollout_qoi_loss_and_gradients(
                dynamics=perturbed,
                decoder=decoder,
                u0=u0,
                t_final=0.12,
                dt_initial=0.04,
                qoi_observations=qoi_observations,
                input_function=input_function,
            ).loss

        step_sizes = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
        zero_order_errors = []
        first_order_errors = []
        for eps in step_sizes:
            trial_loss = loss_at(eps)
            zero_order_errors.append(abs(trial_loss - result.loss))
            first_order_errors.append(abs(trial_loss - result.loss - eps * directional_derivative))

        self.assertGreater(zero_order_errors[0], zero_order_errors[-1])
        self.assertGreater(first_order_errors[0], first_order_errors[-1])
        self.assertLess(first_order_errors[-1], 0.15 * zero_order_errors[-1])


if __name__ == "__main__":
    unittest.main()
