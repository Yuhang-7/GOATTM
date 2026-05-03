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
from goattm.problems import (  # noqa: E402
    DynamicsTikhonovRegularization,
    dynamics_from_parameter_vector,
    dynamics_parameter_vector,
    dynamics_regularization_gradient_vector,
    dynamics_regularization_loss,
    spectral_abscissa_softplus_gradient_matrix,
    spectral_abscissa_softplus_penalty,
    symmetric_part_largest_eigenvalue,
)


class SpectralRegularizationTest(unittest.TestCase):
    def test_symmetric_part_largest_eigenvalue_matches_eigvalsh(self) -> None:
        rng = np.random.default_rng(8101)
        a_matrix = rng.normal(size=(4, 4))
        symmetric_part = 0.5 * (a_matrix + a_matrix.T)

        actual = symmetric_part_largest_eigenvalue(a_matrix)
        expected = np.linalg.eigvalsh(symmetric_part)[-1]

        self.assertAlmostEqual(actual, float(expected), places=13)

    def test_spectral_penalty_matrix_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(8102)
        a_matrix = np.array(
            [
                [0.8, 0.2, -0.1],
                [0.4, -0.5, 0.3],
                [0.1, -0.2, -0.9],
            ],
            dtype=np.float64,
        )
        direction = rng.normal(size=a_matrix.shape)
        coefficient = 0.7
        alpha = 0.1

        gradient = spectral_abscissa_softplus_gradient_matrix(a_matrix, coefficient=coefficient, alpha=alpha)
        h = 1.0e-6
        value_plus = spectral_abscissa_softplus_penalty(a_matrix + h * direction, coefficient=coefficient, alpha=alpha)
        value_minus = spectral_abscissa_softplus_penalty(a_matrix - h * direction, coefficient=coefficient, alpha=alpha)
        finite_difference = (value_plus - value_minus) / (2.0 * h)
        directional_derivative = float(np.sum(gradient * direction))

        self.assertAlmostEqual(directional_derivative, finite_difference, places=7)

    def test_default_spectral_coefficient_is_deactivated(self) -> None:
        dynamics = _build_general_quadratic_dynamics()
        baseline = DynamicsTikhonovRegularization(coeff_a=0.3, coeff_mu_h=0.2, coeff_c=0.1)
        default_spectral = DynamicsTikhonovRegularization(
            coeff_a=0.3,
            coeff_mu_h=0.2,
            coeff_c=0.1,
            coeff_spectral_abscissa=0.0,
        )

        self.assertAlmostEqual(
            dynamics_regularization_loss(dynamics, baseline),
            dynamics_regularization_loss(dynamics, default_spectral),
            places=14,
        )
        np.testing.assert_allclose(
            dynamics_regularization_gradient_vector(dynamics, baseline),
            dynamics_regularization_gradient_vector(dynamics, default_spectral),
            atol=0.0,
            rtol=0.0,
        )

    def test_spectral_regularization_gradient_vector_matches_finite_difference(self) -> None:
        dynamics = _build_general_quadratic_dynamics()
        regularization = DynamicsTikhonovRegularization(
            coeff_spectral_abscissa=0.6,
            spectral_abscissa_alpha=0.05,
        )
        base_vector = dynamics_parameter_vector(dynamics)
        direction = np.linspace(-0.4, 0.3, base_vector.shape[0], dtype=np.float64)
        direction /= np.linalg.norm(direction)

        gradient = dynamics_regularization_gradient_vector(dynamics, regularization)
        h = 1.0e-6
        plus = dynamics_from_parameter_vector(dynamics, base_vector + h * direction)
        minus = dynamics_from_parameter_vector(dynamics, base_vector - h * direction)
        value_plus = dynamics_regularization_loss(plus, regularization)
        value_minus = dynamics_regularization_loss(minus, regularization)
        finite_difference = (value_plus - value_minus) / (2.0 * h)

        self.assertAlmostEqual(float(np.dot(gradient, direction)), finite_difference, places=7)


def _build_general_quadratic_dynamics() -> QuadraticDynamics:
    rank = 3
    a_matrix = np.array(
        [
            [0.3, 0.1, -0.2],
            [0.2, -0.4, 0.05],
            [-0.1, 0.15, -0.6],
        ],
        dtype=np.float64,
    )
    mu_h = np.linspace(-0.2, 0.2, mu_h_dimension(rank), dtype=np.float64)
    c = np.array([0.05, -0.03, 0.02], dtype=np.float64)
    b = np.array([[0.1, -0.2], [0.05, 0.04], [-0.03, 0.07]], dtype=np.float64)
    return QuadraticDynamics(a=a_matrix, mu_h=mu_h, b=b, c=c)


if __name__ == "__main__":
    unittest.main()
