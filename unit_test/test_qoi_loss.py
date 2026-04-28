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
    mu_h_dimension,
    skew_symmetric_dimension,
    upper_triangular_dimension,
)
from goattm.losses.qoi_loss import (  # noqa: E402
    qoi_trajectory_loss,
    qoi_trajectory_loss_and_partials,
    rollout_qoi_loss_and_gradients,
    trapezoidal_rule_weights_from_times,
)
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402


class QoiLossTest(unittest.TestCase):
    def test_qoi_trajectory_loss_matches_manual_trapezoid_formula(self) -> None:
        rng = np.random.default_rng(71)
        r = 3
        dq = 2
        times = np.array([0.0, 0.2, 0.5, 0.9], dtype=float)
        states = rng.standard_normal((times.size, r))
        decoder = QuadraticDecoder(
            v1=rng.standard_normal((dq, r)),
            v2=rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=rng.standard_normal(dq),
        )
        qoi_obs = rng.standard_normal((times.size, dq))

        weights = trapezoidal_rule_weights_from_times(times)
        manual = 0.0
        for n in range(times.size):
            residual = decoder.decode(states[n]) - qoi_obs[n]
            manual += 0.5 * weights[n] * float(np.dot(residual, residual))

        self.assertAlmostEqual(qoi_trajectory_loss(states, decoder, qoi_obs, times), manual)

    def test_decoder_loss_partials_match_finite_difference(self) -> None:
        rng = np.random.default_rng(72)
        r = 3
        dq = 2
        times = np.array([0.0, 0.1, 0.3], dtype=float)
        states = rng.standard_normal((times.size, r))
        decoder = QuadraticDecoder(
            v1=rng.standard_normal((dq, r)),
            v2=rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=rng.standard_normal(dq),
        )
        qoi_obs = rng.standard_normal((times.size, dq))
        partials = qoi_trajectory_loss_and_partials(states, decoder, qoi_obs, times)
        eps = 1e-7

        def loss_fn(v1: np.ndarray, v2: np.ndarray, v0: np.ndarray) -> float:
            test_decoder = QuadraticDecoder(v1=v1, v2=v2, v0=v0)
            return qoi_trajectory_loss(states, test_decoder, qoi_obs, times)

        finite_v1 = np.zeros_like(decoder.v1)
        for i in range(decoder.v1.shape[0]):
            for j in range(decoder.v1.shape[1]):
                perturb = np.zeros_like(decoder.v1)
                perturb[i, j] = eps
                finite_v1[i, j] = (
                    loss_fn(decoder.v1 + perturb, decoder.v2, decoder.v0)
                    - loss_fn(decoder.v1 - perturb, decoder.v2, decoder.v0)
                ) / (2.0 * eps)

        finite_v2 = np.zeros_like(decoder.v2)
        for i in range(decoder.v2.shape[0]):
            for j in range(decoder.v2.shape[1]):
                perturb = np.zeros_like(decoder.v2)
                perturb[i, j] = eps
                finite_v2[i, j] = (
                    loss_fn(decoder.v1, decoder.v2 + perturb, decoder.v0)
                    - loss_fn(decoder.v1, decoder.v2 - perturb, decoder.v0)
                ) / (2.0 * eps)

        finite_v0 = np.zeros_like(decoder.v0)
        for i in range(decoder.v0.size):
            perturb = np.zeros_like(decoder.v0)
            perturb[i] = eps
            finite_v0[i] = (
                loss_fn(decoder.v1, decoder.v2, decoder.v0 + perturb)
                - loss_fn(decoder.v1, decoder.v2, decoder.v0 - perturb)
            ) / (2.0 * eps)

        np.testing.assert_allclose(partials.v1_grad, finite_v1, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(partials.v2_grad, finite_v2, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(partials.v0_grad, finite_v0, rtol=1e-5, atol=1e-7)

    def test_qoi_adjoint_gradient_passes_taylor_test(self) -> None:
        rng = np.random.default_rng(7201)
        r = 2
        dq = 2
        dp = 1
        s_params = 0.12 * rng.standard_normal(upper_triangular_dimension(r))
        w_params = 0.08 * rng.standard_normal(skew_symmetric_dimension(r))
        mu_h = 0.04 * rng.standard_normal(mu_h_dimension(r))
        b = 0.08 * rng.standard_normal((r, dp))
        c = 0.08 * rng.standard_normal(r)
        decoder = QuadraticDecoder(
            v1=0.15 * rng.standard_normal((dq, r)),
            v2=0.04 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=0.08 * rng.standard_normal(dq),
        )
        u0 = 0.08 * rng.standard_normal(r)
        t_final = 0.15
        dt_initial = 0.05
        qoi_observations = rng.standard_normal((4, dq))

        def input_function(t: float) -> np.ndarray:
            return np.array([0.25 - 0.05 * t], dtype=float)

        def loss_from_dynamics(
            s_vec: np.ndarray,
            w_vec: np.ndarray,
            mu_vec: np.ndarray,
            b_mat: np.ndarray,
            c_vec: np.ndarray,
        ) -> float:
            dynamics = StabilizedQuadraticDynamics(
                s_params=s_vec,
                w_params=w_vec,
                mu_h=mu_vec,
                b=b_mat,
                c=c_vec,
            )
            return rollout_qoi_loss_and_gradients(
                dynamics=dynamics,
                decoder=decoder,
                u0=u0,
                t_final=t_final,
                dt_initial=dt_initial,
                qoi_observations=qoi_observations,
                input_function=input_function,
            ).loss

        result = rollout_qoi_loss_and_gradients(
            dynamics=StabilizedQuadraticDynamics(
                s_params=s_params,
                w_params=w_params,
                mu_h=mu_h,
                b=b,
                c=c,
            ),
            decoder=decoder,
            u0=u0,
            t_final=t_final,
            dt_initial=dt_initial,
            qoi_observations=qoi_observations,
            input_function=input_function,
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

        base_loss = result.loss
        step_sizes = [1e-1, 5e-2, 2.5e-2, 1.25e-2]
        zero_order_errors: list[float] = []
        first_order_errors: list[float] = []

        for eps in step_sizes:
            trial_loss = loss_from_dynamics(
                s_params + eps * ds,
                w_params + eps * dw,
                mu_h + eps * dmu,
                b + eps * db,
                c + eps * dc,
            )
            zero_order_errors.append(abs(trial_loss - base_loss))
            first_order_errors.append(abs(trial_loss - base_loss - eps * directional_derivative))

        self.assertEqual(result.adjoints.shape[0], result.rollout.states.shape[0])
        self.assertGreater(float(np.linalg.norm(result.adjoints[1:])), 0.0)
        self.assertGreater(zero_order_errors[0], zero_order_errors[-1])
        self.assertGreater(first_order_errors[0], first_order_errors[-1])

        zero_order_rates = [
            zero_order_errors[i] / zero_order_errors[i + 1] for i in range(len(zero_order_errors) - 1)
        ]
        first_order_rates = [
            first_order_errors[i] / first_order_errors[i + 1] for i in range(len(first_order_errors) - 1)
        ]

        for rate in zero_order_rates:
            self.assertGreater(rate, 1.7)
            self.assertLess(rate, 2.3)
        self.assertGreater(first_order_rates[0], 3.2)
        self.assertGreater(first_order_rates[1], 3.2)
        self.assertLess(first_order_errors[-1], 0.1 * zero_order_errors[-1])

    def test_rollout_loss_gradients_match_finite_difference(self) -> None:
        rng = np.random.default_rng(73)
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
        t_final = 0.15
        dt_initial = 0.05
        qoi_observations = rng.standard_normal((4, dq))

        def input_function(t: float) -> np.ndarray:
            return np.array([0.3 + 0.1 * t], dtype=float)

        dynamics = StabilizedQuadraticDynamics(
            s_params=s_params,
            w_params=w_params,
            mu_h=mu_h,
            b=b,
            c=c,
        )
        result = rollout_qoi_loss_and_gradients(
            dynamics=dynamics,
            decoder=decoder,
            u0=u0,
            t_final=t_final,
            dt_initial=dt_initial,
            qoi_observations=qoi_observations,
            input_function=input_function,
        )
        eps = 1e-7

        def loss_from_params(
            s_vec: np.ndarray,
            w_vec: np.ndarray,
            mu_vec: np.ndarray,
            b_mat: np.ndarray,
            c_vec: np.ndarray,
            v1: np.ndarray,
            v2: np.ndarray,
            v0: np.ndarray,
        ) -> float:
            dyn = StabilizedQuadraticDynamics(
                s_params=s_vec,
                w_params=w_vec,
                mu_h=mu_vec,
                b=b_mat,
                c=c_vec,
            )
            dec = QuadraticDecoder(v1=v1, v2=v2, v0=v0)
            return rollout_qoi_loss_and_gradients(
                dynamics=dyn,
                decoder=dec,
                u0=u0,
                t_final=t_final,
                dt_initial=dt_initial,
                qoi_observations=qoi_observations,
                input_function=input_function,
            ).loss

        def finite_difference_vector(base: np.ndarray, build_loss) -> np.ndarray:
            out = np.zeros_like(base)
            for i in range(base.size):
                perturb = np.zeros_like(base)
                perturb[i] = eps
                out[i] = (build_loss(base + perturb) - build_loss(base - perturb)) / (2.0 * eps)
            return out

        finite_s = finite_difference_vector(
            s_params,
            lambda val: loss_from_params(val, w_params, mu_h, b, c, decoder.v1, decoder.v2, decoder.v0),
        )
        finite_w = finite_difference_vector(
            w_params,
            lambda val: loss_from_params(s_params, val, mu_h, b, c, decoder.v1, decoder.v2, decoder.v0),
        )
        finite_mu = finite_difference_vector(
            mu_h,
            lambda val: loss_from_params(s_params, w_params, val, b, c, decoder.v1, decoder.v2, decoder.v0),
        )
        finite_c = finite_difference_vector(
            c,
            lambda val: loss_from_params(s_params, w_params, mu_h, b, val, decoder.v1, decoder.v2, decoder.v0),
        )
        finite_v0 = finite_difference_vector(
            decoder.v0,
            lambda val: loss_from_params(s_params, w_params, mu_h, b, c, decoder.v1, decoder.v2, val),
        )

        finite_b = np.zeros_like(b)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                perturb = np.zeros_like(b)
                perturb[i, j] = eps
                finite_b[i, j] = (
                    loss_from_params(s_params, w_params, mu_h, b + perturb, c, decoder.v1, decoder.v2, decoder.v0)
                    - loss_from_params(
                        s_params, w_params, mu_h, b - perturb, c, decoder.v1, decoder.v2, decoder.v0
                    )
                ) / (2.0 * eps)

        finite_v1 = np.zeros_like(decoder.v1)
        for i in range(decoder.v1.shape[0]):
            for j in range(decoder.v1.shape[1]):
                perturb = np.zeros_like(decoder.v1)
                perturb[i, j] = eps
                finite_v1[i, j] = (
                    loss_from_params(s_params, w_params, mu_h, b, c, decoder.v1 + perturb, decoder.v2, decoder.v0)
                    - loss_from_params(
                        s_params, w_params, mu_h, b, c, decoder.v1 - perturb, decoder.v2, decoder.v0
                    )
                ) / (2.0 * eps)

        finite_v2 = np.zeros_like(decoder.v2)
        for i in range(decoder.v2.shape[0]):
            for j in range(decoder.v2.shape[1]):
                perturb = np.zeros_like(decoder.v2)
                perturb[i, j] = eps
                finite_v2[i, j] = (
                    loss_from_params(s_params, w_params, mu_h, b, c, decoder.v1, decoder.v2 + perturb, decoder.v0)
                    - loss_from_params(
                        s_params, w_params, mu_h, b, c, decoder.v1, decoder.v2 - perturb, decoder.v0
                    )
                ) / (2.0 * eps)

        np.testing.assert_allclose(result.dynamics_gradients["s_params"], finite_s, rtol=3e-4, atol=3e-6)
        np.testing.assert_allclose(result.dynamics_gradients["w_params"], finite_w, rtol=3e-4, atol=3e-6)
        np.testing.assert_allclose(result.dynamics_gradients["mu_h"], finite_mu, rtol=4e-4, atol=4e-6)
        np.testing.assert_allclose(result.dynamics_gradients["b"], finite_b, rtol=4e-4, atol=4e-6)
        np.testing.assert_allclose(result.dynamics_gradients["c"], finite_c, rtol=4e-4, atol=4e-6)
        np.testing.assert_allclose(result.decoder_partials.v1_grad, finite_v1, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result.decoder_partials.v2_grad, finite_v2, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result.decoder_partials.v0_grad, finite_v0, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
