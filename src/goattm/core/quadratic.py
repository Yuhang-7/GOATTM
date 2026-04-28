from __future__ import annotations

import os
import numpy as np

from goattm.core.parametrization import quadratic_features

if os.environ.get("GOATTM_DISABLE_NUMBA", "").strip().lower() in {"1", "true", "yes", "on"}:
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator
else:
    try:
        from numba import njit
    except ImportError:  # pragma: no cover
        def njit(*args, **kwargs):  # type: ignore[misc]
            def decorator(func):
                return func

            return decorator


def _infer_latent_dimension_from_compressed_width(s: int) -> int:
    disc = 1 + 8 * s
    root = int(np.sqrt(disc))
    if root * root != disc:
        raise ValueError(f"Compressed quadratic width {s} is not triangular.")
    r = (-1 + root) // 2
    if r * (r + 1) // 2 != s:
        raise ValueError(f"Compressed quadratic width {s} is not triangular.")
    return r


def _validate_compressed_h_matrix(h_matrix: np.ndarray) -> tuple[int, int]:
    if h_matrix.ndim != 2:
        raise ValueError(f"h_matrix must be rank-2, got shape {h_matrix.shape}")
    q, s = h_matrix.shape
    r = _infer_latent_dimension_from_compressed_width(s)
    return r, q


def _validate_state_vector(u: np.ndarray, expected_dim: int, name: str) -> None:
    if u.ndim != 1 or u.shape[0] != expected_dim:
        raise ValueError(f"{name} must have shape ({expected_dim},), got {u.shape}")


@njit(cache=True)
def _quadratic_eval_kernel(h_matrix: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r = h_matrix.shape[0]
    out = np.zeros(r, dtype=np.float64)
    for row in range(r):
        acc = 0.0
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h_matrix[row, idx]
                if i == j:
                    acc += coeff * u[i] * v[j]
                else:
                    acc += 0.5 * coeff * (u[i] * v[j] + u[j] * v[i])
                idx += 1
        out[row] = acc
    return out


@njit(cache=True)
def _bilinear_action_matrix_kernel(h_matrix: np.ndarray, u: np.ndarray) -> np.ndarray:
    q = h_matrix.shape[0]
    r = u.shape[0]
    out = np.zeros((q, r), dtype=np.float64)
    for row in range(q):
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h_matrix[row, idx]
                if i == j:
                    out[row, i] += coeff * u[i]
                else:
                    half = 0.5 * coeff
                    out[row, i] += half * u[j]
                    out[row, j] += half * u[i]
                idx += 1
    return out


def quadratic_eval(h_matrix: np.ndarray, u: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
    """Evaluate the symmetric bilinear map H(u, v) from the compressed matrix representation."""
    r, _ = _validate_compressed_h_matrix(h_matrix)
    _validate_state_vector(u, r, "u")
    if v is None:
        v = u
    _validate_state_vector(v, r, "v")
    return _quadratic_eval_kernel(h_matrix, u, v)


def quadratic_bilinear_action_matrix(h_matrix: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return M(u) such that M(u) @ v = H(u, v)."""
    r, _ = _validate_compressed_h_matrix(h_matrix)
    _validate_state_vector(u, r, "u")
    return _bilinear_action_matrix_kernel(h_matrix, u)


def quadratic_jacobian_matrix(h_matrix: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Return J_quad(u) such that J_quad(u) @ delta is the directional derivative
    of H(u, u) in direction delta.
    """
    return 2.0 * quadratic_bilinear_action_matrix(h_matrix, u)


def energy_preserving_defect(h_matrix: np.ndarray, u: np.ndarray) -> float:
    _validate_state_vector(u, _validate_compressed_h_matrix(h_matrix)[0], "u")
    return float(np.dot(u, quadratic_eval(h_matrix, u)))
