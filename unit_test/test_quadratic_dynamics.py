from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import mu_h_dimension, quadratic_features  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402


class QuadraticDynamicsTest(unittest.TestCase):
    def test_rhs_matches_manual_formula(self) -> None:
        rng = np.random.default_rng(21)
        a = rng.standard_normal((3, 3))
        mu_h = rng.standard_normal(mu_h_dimension(3))
        c = rng.standard_normal(3)
        u = rng.standard_normal(3)

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c)
        expected = a @ u + dynamics.h_matrix @ quadratic_features(u) + c
        np.testing.assert_allclose(dynamics.rhs(u), expected)

    def test_rhs_jacobian_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(22)
        a = rng.standard_normal((4, 4))
        mu_h = rng.standard_normal(mu_h_dimension(4))
        c = rng.standard_normal(4)
        u = rng.standard_normal(4)
        direction = rng.standard_normal(4)
        eps = 1e-7

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c)
        fd = (dynamics.rhs(u + eps * direction) - dynamics.rhs(u - eps * direction)) / (2.0 * eps)
        jacobian_action = dynamics.rhs_jacobian(u) @ direction
        np.testing.assert_allclose(jacobian_action, fd, rtol=1e-5, atol=1e-7)

    def test_invalid_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            QuadraticDynamics(
                a=np.zeros((2, 3)),
                mu_h=np.zeros(mu_h_dimension(2)),
                c=np.zeros(2),
            )
        with self.assertRaises(ValueError):
            QuadraticDynamics(
                a=np.zeros((3, 3)),
                mu_h=np.zeros(2),
                c=np.zeros(3),
            )
        with self.assertRaises(ValueError):
            QuadraticDynamics(
                a=np.zeros((3, 3)),
                mu_h=np.zeros(mu_h_dimension(3)),
                c=np.zeros(3),
                b=np.zeros((2, 2)),
            )

    def test_rhs_with_input_matches_manual_formula(self) -> None:
        rng = np.random.default_rng(23)
        a = rng.standard_normal((3, 3))
        mu_h = rng.standard_normal(mu_h_dimension(3))
        b = rng.standard_normal((3, 2))
        c = rng.standard_normal(3)
        u = rng.standard_normal(3)
        p = rng.standard_normal(2)

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c, b=b)
        expected = a @ u + dynamics.h_matrix @ quadratic_features(u) + b @ p + c
        np.testing.assert_allclose(dynamics.rhs(u, p=p), expected)

    def test_rhs_skips_input_term_when_p_is_missing(self) -> None:
        rng = np.random.default_rng(24)
        a = rng.standard_normal((3, 3))
        mu_h = rng.standard_normal(mu_h_dimension(3))
        b = rng.standard_normal((3, 2))
        c = rng.standard_normal(3)
        u = rng.standard_normal(3)

        dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c, b=b)
        expected = a @ u + dynamics.h_matrix @ quadratic_features(u) + c
        np.testing.assert_allclose(dynamics.rhs(u), expected)


if __name__ == "__main__":
    unittest.main()
