from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import mu_h_dimension, mu_h_to_compressed_h, quadratic_features  # noqa: E402
from goattm.core.quadratic import quadratic_bilinear_action_matrix, quadratic_eval, quadratic_jacobian_matrix  # noqa: E402


class QuadraticCoreTest(unittest.TestCase):
    def test_quadratic_eval_matches_manual_contraction(self) -> None:
        rng = np.random.default_rng(11)
        h_matrix = rng.standard_normal((3, 6))
        u = rng.standard_normal(3)
        v = rng.standard_normal(3)

        expected = quadratic_bilinear_action_matrix(h_matrix, u) @ v
        actual = quadratic_eval(h_matrix, u, v)
        np.testing.assert_allclose(actual, expected)

    def test_quadratic_eval_matches_compressed_quadratic_features(self) -> None:
        rng = np.random.default_rng(12)
        h_matrix = rng.standard_normal((4, 10))
        u = rng.standard_normal(4)

        expected = h_matrix @ quadratic_features(u)
        actual = quadratic_eval(h_matrix, u)
        np.testing.assert_allclose(actual, expected)

    def test_bilinear_action_matrix_matches_bilinear_eval(self) -> None:
        rng = np.random.default_rng(13)
        h_matrix = rng.standard_normal((4, 10))
        u = rng.standard_normal(4)
        v = rng.standard_normal(4)

        matrix_action = quadratic_bilinear_action_matrix(h_matrix, u) @ v
        direct_action = quadratic_eval(h_matrix, u, v)
        np.testing.assert_allclose(matrix_action, direct_action)

    def test_quadratic_jacobian_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(14)
        h_matrix = rng.standard_normal((5, 15))
        u = rng.standard_normal(5)
        direction = rng.standard_normal(5)
        eps = 1e-7

        fd = (quadratic_eval(h_matrix, u + eps * direction) - quadratic_eval(h_matrix, u - eps * direction)) / (2.0 * eps)
        jacobian_action = quadratic_jacobian_matrix(h_matrix, u) @ direction
        np.testing.assert_allclose(jacobian_action, fd, rtol=1e-5, atol=1e-7)

    def test_energy_preserving_parametrized_h_satisfies_u_h_u_u_zero(self) -> None:
        rng = np.random.default_rng(15)
        r = 6
        h_matrix = mu_h_to_compressed_h(rng.standard_normal(mu_h_dimension(r)), r)

        for _ in range(10):
            u = rng.standard_normal(r)
            defect = float(np.dot(u, quadratic_eval(h_matrix, u)))
            self.assertAlmostEqual(defect, 0.0, places=12)

    def test_energy_preserving_parametrized_h_satisfies_polarization_identity(self) -> None:
        rng = np.random.default_rng(16)
        r = 6
        h_matrix = mu_h_to_compressed_h(rng.standard_normal(mu_h_dimension(r)), r)

        for _ in range(10):
            s = rng.standard_normal(r)
            w = rng.standard_normal(r)
            lhs = 2.0 * float(np.dot(w, quadratic_eval(h_matrix, s, w))) + float(np.dot(s, quadratic_eval(h_matrix, w)))
            self.assertAlmostEqual(lhs, 0.0, places=12)

    def test_energy_preserving_parametrized_h_satisfies_companion_identity(self) -> None:
        rng = np.random.default_rng(17)
        r = 6
        h_matrix = mu_h_to_compressed_h(rng.standard_normal(mu_h_dimension(r)), r)

        for _ in range(10):
            s = rng.standard_normal(r)
            w = rng.standard_normal(r)
            lhs = 2.0 * float(np.dot(s, quadratic_eval(h_matrix, s, w))) + float(np.dot(w, quadratic_eval(h_matrix, s)))
            self.assertAlmostEqual(lhs, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
