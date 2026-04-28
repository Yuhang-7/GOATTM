from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.parametrization import compressed_quadratic_dimension, mu_h_dimension


@dataclass(frozen=True)
class ConstrainedMatrixLeastSquaresResult:
    coefficients: np.ndarray
    solution_matrix: np.ndarray
    design_matrix: np.ndarray
    objective_value: float
    residual_norm: float


def build_energy_preserving_compressed_h_basis(r: int) -> np.ndarray:
    s = compressed_quadratic_dimension(r)
    mu_dim = mu_h_dimension(r)
    basis = np.zeros((r * s, mu_dim), dtype=np.float64)

    # Fill the linear map mu_h -> vec_F(H) directly.  This avoids repeatedly
    # invoking the scalar parametrization routine on every MPI rank.
    for k in range(r):
        fk = (k * (k - 1) * (k + 1)) // 3

        for j in range(k):
            for i in range(j):
                muind1 = fk + i * k + j
                muind2 = fk + j * k + i

                basis[_fortran_flat_index(i, _compressed_quadratic_index(k, j), r), muind1] = 1.0
                basis[_fortran_flat_index(k, _compressed_quadratic_index(j, i), r), muind1] = -1.0

                basis[_fortran_flat_index(j, _compressed_quadratic_index(k, i), r), muind2] = 1.0
                basis[_fortran_flat_index(k, _compressed_quadratic_index(j, i), r), muind2] = -1.0

            muind = fk + j * k + j
            basis[_fortran_flat_index(j, _compressed_quadratic_index(k, j), r), muind] = 1.0
            basis[_fortran_flat_index(k, _compressed_quadratic_index(j, j), r), muind] = -1.0

        for i in range(k):
            muind = fk + k * k + i
            basis[_fortran_flat_index(i, _compressed_quadratic_index(k, k), r), muind] = 1.0
            basis[_fortran_flat_index(k, _compressed_quadratic_index(k, i), r), muind] = -1.0
    return basis


def _compressed_quadratic_index(i: int, j: int) -> int:
    return (i * (i + 1)) // 2 + j


def _fortran_flat_index(row: int, column: int, row_count: int) -> int:
    return row + column * row_count


def solve_basis_constrained_matrix_least_squares(
    regressor: np.ndarray,
    target: np.ndarray,
    basis_matrix: np.ndarray,
    regularization: float = 0.0,
) -> ConstrainedMatrixLeastSquaresResult:
    regressor = np.asarray(regressor, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    basis_matrix = np.asarray(basis_matrix, dtype=np.float64)

    if regressor.ndim != 2:
        raise ValueError(f"regressor must be rank-2, got shape {regressor.shape}")
    if target.ndim != 2:
        raise ValueError(f"target must be rank-2, got shape {target.shape}")
    row_dim, sample_count = target.shape
    feature_dim, reg_sample_count = regressor.shape
    if reg_sample_count != sample_count:
        raise ValueError("regressor and target must share the same sample dimension.")
    if basis_matrix.ndim != 2 or basis_matrix.shape[0] != row_dim * feature_dim:
        raise ValueError(
            "basis_matrix must have shape (row_dim * feature_dim, n_coeff), "
            f"got {basis_matrix.shape} for row_dim={row_dim}, feature_dim={feature_dim}"
        )
    if regularization < 0.0:
        raise ValueError(f"regularization must be nonnegative, got {regularization}")

    design_matrix = np.kron(regressor.T, np.eye(row_dim, dtype=np.float64)) @ basis_matrix
    target_vector = target.reshape(-1, order="F")
    normal_matrix = design_matrix.T @ design_matrix
    if regularization > 0.0:
        normal_matrix = normal_matrix + float(regularization) * np.eye(normal_matrix.shape[0], dtype=np.float64)
    rhs = design_matrix.T @ target_vector
    coefficients = np.linalg.solve(normal_matrix, rhs)
    solution_vector = basis_matrix @ coefficients
    solution_matrix = solution_vector.reshape((row_dim, feature_dim), order="F")
    residual = solution_matrix @ regressor - target
    objective_value = 0.5 * float(np.sum(residual * residual))
    if regularization > 0.0:
        objective_value += 0.5 * float(regularization) * float(coefficients @ coefficients)
    return ConstrainedMatrixLeastSquaresResult(
        coefficients=coefficients,
        solution_matrix=solution_matrix,
        design_matrix=design_matrix,
        objective_value=objective_value,
        residual_norm=float(np.linalg.norm(residual)),
    )
