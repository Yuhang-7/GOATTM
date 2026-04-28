from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.problems.decoder_normal_equation import DecoderTikhonovRegularization  # noqa: E402


def solve_matrix_normal_equation(phi: np.ndarray, targets: np.ndarray, weights: np.ndarray, reg_diag: np.ndarray) -> np.ndarray:
    gram = np.zeros((phi.shape[1], phi.shape[1]), dtype=np.float64)
    rhs = np.zeros((phi.shape[1], targets.shape[1]), dtype=np.float64)
    for row, target, weight in zip(phi, targets, weights):
        gram += weight * np.outer(row, row)
        rhs += weight * np.outer(row, target)
    return np.linalg.solve(gram + np.diag(reg_diag), rhs)


def solve_augmented_least_squares(phi: np.ndarray, targets: np.ndarray, weights: np.ndarray, reg_diag: np.ndarray) -> np.ndarray:
    weighted_design = np.sqrt(weights)[:, None] * phi
    weighted_targets = np.sqrt(weights)[:, None] * targets
    augmented_design = np.vstack([weighted_design, np.diag(np.sqrt(reg_diag))])
    augmented_targets = np.vstack([weighted_targets, np.zeros((phi.shape[1], targets.shape[1]), dtype=np.float64)])
    solution, *_ = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)
    return solution


def main() -> None:
    rng = np.random.default_rng(77)
    sample_count = 12
    latent_dim = 3
    quad_dim = latent_dim * (latent_dim + 1) // 2
    feature_dim = latent_dim + quad_dim + 1
    output_dim = 4

    phi = rng.standard_normal((sample_count, feature_dim))
    targets = rng.standard_normal((sample_count, output_dim))
    weights = 0.1 + rng.random(sample_count)
    regularization = DecoderTikhonovRegularization(coeff_v1=2e-3, coeff_v2=3e-3, coeff_v0=4e-3)
    reg_diag = regularization.diagonal(latent_dim)

    normal_solution = solve_matrix_normal_equation(phi, targets, weights, reg_diag)
    augmented_solution = solve_augmented_least_squares(phi, targets, weights, reg_diag)
    stationarity_residual = (
        sum(weight * np.outer(row, row) for row, weight in zip(phi, weights)) + np.diag(reg_diag)
    ) @ normal_solution - sum(weight * np.outer(row, target) for row, target, weight in zip(phi, targets, weights))

    np.testing.assert_allclose(normal_solution, augmented_solution, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(stationarity_residual, 0.0, atol=1e-11, rtol=1e-11)

    print("Decoder normal-equation module test passed.")
    print(f"feature_dim={feature_dim}, output_dim={output_dim}, samples={sample_count}")
    print(f"solution_norm={np.linalg.norm(normal_solution):.6e}")
    print(f"stationarity_residual_norm={np.linalg.norm(stationarity_residual):.6e}")


if __name__ == "__main__":
    main()
