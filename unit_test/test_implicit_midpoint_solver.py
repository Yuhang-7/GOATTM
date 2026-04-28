from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import mu_h_dimension  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.solvers.implicit_midpoint import (  # noqa: E402
    implicit_midpoint_residual,
    rollout_implicit_midpoint,
    rollout_implicit_midpoint_to_observation_times,
    solve_implicit_midpoint_step,
    solve_implicit_midpoint_step_with_retry,
)


class ImplicitMidpointSolverTest(unittest.TestCase):
    def test_single_linear_step_matches_closed_form(self) -> None:
        a = np.diag(np.array([-1.0, -2.0]))
        mu_h = np.zeros(mu_h_dimension(2))
        c = np.array([0.5, -0.25])
        u_prev = np.array([1.0, -0.5])
        dt = 0.1

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c)
        u_next, info = solve_implicit_midpoint_step(dynamics, u_prev, dt)

        lhs = np.eye(2) - 0.5 * dt * a
        rhs = (np.eye(2) + 0.5 * dt * a) @ u_prev + dt * c
        expected = np.linalg.solve(lhs, rhs)

        self.assertTrue(info.success)
        np.testing.assert_allclose(u_next, expected, rtol=1e-12, atol=1e-12)

    def test_quadratic_step_satisfies_midpoint_residual(self) -> None:
        rng = np.random.default_rng(31)
        a = -0.5 * np.eye(3)
        mu_h = 0.05 * rng.standard_normal(mu_h_dimension(3))
        c = 0.2 * rng.standard_normal(3)
        u_prev = 0.2 * rng.standard_normal(3)

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c)
        u_next, info = solve_implicit_midpoint_step(dynamics, u_prev, dt=0.05)

        residual = implicit_midpoint_residual(dynamics, u_prev, u_next, dt=0.05)
        self.assertTrue(info.success)
        self.assertLess(np.linalg.norm(residual), 1e-10)

    def test_rollout_reaches_final_time_with_expected_step_count(self) -> None:
        a = -0.3 * np.eye(2)
        mu_h = np.zeros(mu_h_dimension(2))
        c = np.array([0.1, -0.2])
        u0 = np.array([0.25, -0.1])

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c)
        result = rollout_implicit_midpoint(dynamics, u0=u0, t_final=1.0, dt_initial=0.01)

        self.assertTrue(result.success)
        self.assertEqual(result.accepted_steps, 100)
        self.assertEqual(result.states.shape, (101, 2))
        self.assertEqual(result.dt_history.shape, (100,))
        self.assertEqual(result.times.shape, (101,))
        np.testing.assert_allclose(result.times[0], 0.0)
        np.testing.assert_allclose(result.times[-1], 1.0)
        np.testing.assert_allclose(result.dt_history, 0.01)
        np.testing.assert_allclose(np.diff(result.times), result.dt_history)
        self.assertAlmostEqual(result.final_time, 1.0, places=12)

    def test_retry_path_reports_failure_when_newton_is_disabled(self) -> None:
        rng = np.random.default_rng(32)
        a = -0.2 * np.eye(3)
        mu_h = 0.1 * rng.standard_normal(mu_h_dimension(3))
        c = 0.5 * rng.standard_normal(3)
        u_prev = 0.2 * rng.standard_normal(3)

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c)
        result = solve_implicit_midpoint_step_with_retry(
            dynamics,
            u_prev=u_prev,
            dt_initial=0.1,
            dt_shrink=0.8,
            dt_min=0.02,
            max_iter=0,
        )

        self.assertFalse(result.success)
        self.assertGreaterEqual(result.dt_reductions, 1)
        self.assertGreaterEqual(result.newton_failures, 1)

    def test_rollout_supports_time_dependent_input_function(self) -> None:
        a = -0.4 * np.eye(2)
        mu_h = np.zeros(mu_h_dimension(2))
        b = np.array([[1.0], [0.5]])
        c = np.array([0.0, 0.1])
        u0 = np.array([0.0, 0.0])

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c, b=b)

        def input_function(t: float) -> np.ndarray:
            return np.array([1.0 + 0.2 * t])

        result = rollout_implicit_midpoint(
            dynamics,
            u0=u0,
            t_final=0.2,
            dt_initial=0.01,
            input_function=input_function,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.accepted_steps, 20)
        self.assertEqual(result.states.shape, (21, 2))

    def test_rollout_to_observation_times_lands_exactly_on_requested_grid(self) -> None:
        a = -0.3 * np.eye(2)
        mu_h = np.zeros(mu_h_dimension(2))
        c = np.array([0.05, -0.02])
        u0 = np.array([0.1, -0.05])
        observation_times = np.array([0.0, 0.03, 0.08, 0.15], dtype=float)

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c)
        result, observation_indices = rollout_implicit_midpoint_to_observation_times(
            dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=0.05,
        )

        self.assertTrue(result.success)
        np.testing.assert_allclose(result.times[observation_indices], observation_times)
        self.assertEqual(observation_indices.shape, observation_times.shape)
        self.assertGreaterEqual(result.accepted_steps, observation_times.size - 1)


if __name__ == "__main__":
    unittest.main()
