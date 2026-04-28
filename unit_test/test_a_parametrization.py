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
    a_from_stabilized_params,
    a_gradient_to_stabilized_params,
    s_params_to_matrix,
    skew_symmetric_dimension,
    upper_triangular_dimension,
    w_params_to_matrix,
)


class AParametrizationTest(unittest.TestCase):
    def test_param_dimensions_match_closed_form(self) -> None:
        for r in range(2, 8):
            self.assertEqual(upper_triangular_dimension(r), r * (r + 1) // 2)
            self.assertEqual(skew_symmetric_dimension(r), r * (r - 1) // 2)

    def test_s_and_w_matrices_have_expected_structure(self) -> None:
        r = 4
        s_params = np.arange(upper_triangular_dimension(r), dtype=float)
        w_params = np.arange(skew_symmetric_dimension(r), dtype=float) + 1.0

        s_matrix = s_params_to_matrix(s_params, r)
        w_matrix = w_params_to_matrix(w_params, r)

        np.testing.assert_allclose(np.tril(s_matrix, k=-1), 0.0)
        np.testing.assert_allclose(w_matrix + w_matrix.T, 0.0)

    def test_a_parametrization_has_negative_semidefinite_symmetric_part(self) -> None:
        rng = np.random.default_rng(61)
        r = 5
        s_params = rng.standard_normal(upper_triangular_dimension(r))
        w_params = rng.standard_normal(skew_symmetric_dimension(r))

        a_matrix = a_from_stabilized_params(s_params, w_params, r)
        sym_part = 0.5 * (a_matrix + a_matrix.T)
        eigvals = np.linalg.eigvalsh(sym_part)
        self.assertLessEqual(np.max(eigvals), 1e-12)

    def test_a_gradient_pullback_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(62)
        r = 4
        s_params = rng.standard_normal(upper_triangular_dimension(r))
        w_params = rng.standard_normal(skew_symmetric_dimension(r))
        test_matrix = rng.standard_normal((r, r))
        eps = 1e-7

        analytic_s, analytic_w = a_gradient_to_stabilized_params(test_matrix, s_params, w_params, r)

        def objective(s_vec: np.ndarray, w_vec: np.ndarray) -> float:
            a_matrix = a_from_stabilized_params(s_vec, w_vec, r)
            return float(np.sum(test_matrix * a_matrix))

        finite_s = np.zeros_like(s_params)
        for i in range(s_params.size):
            perturb = np.zeros_like(s_params)
            perturb[i] = eps
            finite_s[i] = (
                objective(s_params + perturb, w_params) - objective(s_params - perturb, w_params)
            ) / (2.0 * eps)

        finite_w = np.zeros_like(w_params)
        for i in range(w_params.size):
            perturb = np.zeros_like(w_params)
            perturb[i] = eps
            finite_w[i] = (
                objective(s_params, w_params + perturb) - objective(s_params, w_params - perturb)
            ) / (2.0 * eps)

        np.testing.assert_allclose(analytic_s, finite_s, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(analytic_w, finite_w, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
