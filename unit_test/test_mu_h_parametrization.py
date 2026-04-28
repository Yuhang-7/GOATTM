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
    compressed_h_gradient_to_mu_h,
    compressed_h_to_mu_h,
    compressed_quadratic_dimension,
    mu_h_dimension,
    mu_h_to_compressed_h,
    quadratic_features,
)
from goattm.core.quadratic import quadratic_eval  # noqa: E402


class MuHParametrizationTest(unittest.TestCase):
    def test_mu_h_dimension_matches_formula(self) -> None:
        for r in range(2, 8):
            self.assertEqual(mu_h_dimension(r), r * (r - 1) * (r + 1) // 3)

    def test_mu_h_to_compressed_h_has_expected_shape(self) -> None:
        rng = np.random.default_rng(41)
        r = 5
        mu_h = rng.standard_normal(mu_h_dimension(r))
        h_matrix = mu_h_to_compressed_h(mu_h, r)
        self.assertEqual(h_matrix.shape, (r, compressed_quadratic_dimension(r)))

    def test_compressed_h_to_mu_h_inverts_parametrization(self) -> None:
        rng = np.random.default_rng(42)
        r = 6
        mu_h = rng.standard_normal(mu_h_dimension(r))
        h_matrix = mu_h_to_compressed_h(mu_h, r)
        recovered = compressed_h_to_mu_h(h_matrix, r)
        np.testing.assert_allclose(recovered, mu_h)

    def test_mu_h_parametrization_is_energy_preserving(self) -> None:
        rng = np.random.default_rng(43)
        r = 6
        mu_h = rng.standard_normal(mu_h_dimension(r))
        h_matrix = mu_h_to_compressed_h(mu_h, r)

        for _ in range(10):
            u = rng.standard_normal(r)
            defect = float(np.dot(u, quadratic_eval(h_matrix, u)))
            self.assertAlmostEqual(defect, 0.0, places=12)

    def test_mu_h_parametrization_satisfies_polarization_identity(self) -> None:
        rng = np.random.default_rng(44)
        r = 6
        mu_h = rng.standard_normal(mu_h_dimension(r))
        h_matrix = mu_h_to_compressed_h(mu_h, r)

        for _ in range(10):
            s = rng.standard_normal(r)
            w = rng.standard_normal(r)
            lhs = 2.0 * float(np.dot(w, quadratic_eval(h_matrix, s, w))) + float(np.dot(s, quadratic_eval(h_matrix, w)))
            self.assertAlmostEqual(lhs, 0.0, places=12)

    def test_compressed_h_gradient_to_mu_h_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(45)
        r = 5
        mu_h = rng.standard_normal(mu_h_dimension(r))
        u = rng.standard_normal(r)
        g = rng.standard_normal(r)
        eps = 1e-7

        zeta = quadratic_features(u)
        h_grad = np.outer(g, zeta)
        analytic = compressed_h_gradient_to_mu_h(h_grad, r)

        def objective(mu_h_vec: np.ndarray) -> float:
            h_matrix = mu_h_to_compressed_h(mu_h_vec, r)
            return float(np.dot(g, quadratic_eval(h_matrix, u)))

        finite_diff = np.zeros_like(mu_h)
        for i in range(mu_h.size):
            perturb = np.zeros_like(mu_h)
            perturb[i] = eps
            finite_diff[i] = (objective(mu_h + perturb) - objective(mu_h - perturb)) / (2.0 * eps)

        np.testing.assert_allclose(analytic, finite_diff, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
