from __future__ import annotations

import os
import numpy as np

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


@njit(cache=True)
def compressed_quadratic_dimension(r: int) -> int:
    return (r * (r + 1)) // 2


@njit(cache=True)
def upper_triangular_dimension(r: int) -> int:
    return (r * (r + 1)) // 2


@njit(cache=True)
def skew_symmetric_dimension(r: int) -> int:
    return (r * (r - 1)) // 2


@njit(cache=True)
def mu_h_dimension(r: int) -> int:
    return (r * (r - 1) * (r + 1)) // 3


@njit(cache=True)
def lower_triangular_index(i: int, j: int) -> int:
    return (i * (i + 1)) // 2 + j


@njit(cache=True)
def quadratic_features(u: np.ndarray) -> np.ndarray:
    r = u.shape[0]
    s = compressed_quadratic_dimension(r)
    out = np.zeros(s, dtype=u.dtype)
    for i in range(r):
        base = (i * (i + 1)) // 2
        for j in range(i + 1):
            out[base + j] = u[i] * u[j]
    return out


@njit(cache=True)
def s_params_to_matrix(s_params: np.ndarray, r: int) -> np.ndarray:
    s_matrix = np.zeros((r, r), dtype=np.float64)
    idx = 0
    for i in range(r):
        for j in range(i, r):
            s_matrix[i, j] = s_params[idx]
            idx += 1
    return s_matrix


@njit(cache=True)
def w_params_to_matrix(w_params: np.ndarray, r: int) -> np.ndarray:
    w_matrix = np.zeros((r, r), dtype=np.float64)
    idx = 0
    for i in range(r):
        for j in range(i + 1, r):
            value = w_params[idx]
            w_matrix[i, j] = value
            w_matrix[j, i] = -value
            idx += 1
    return w_matrix


@njit(cache=True)
def a_from_stabilized_params(s_params: np.ndarray, w_params: np.ndarray, r: int) -> np.ndarray:
    s_matrix = s_params_to_matrix(s_params, r)
    w_matrix = w_params_to_matrix(w_params, r)
    return -s_matrix @ s_matrix.T + w_matrix


@njit(cache=True)
def mu_h_to_compressed_h(mu_h: np.ndarray, r: int) -> np.ndarray:
    s = compressed_quadratic_dimension(r)
    h_matrix = np.zeros((r, s), dtype=np.float64)

    for k in range(r):
        fk = (k * (k - 1) * (k + 1)) // 3

        for j in range(k):
            for i in range(j):
                muind1 = fk + i * k + j
                muind2 = fk + j * k + i

                h_matrix[i, lower_triangular_index(k, j)] = mu_h[muind1]
                h_matrix[j, lower_triangular_index(k, i)] = mu_h[muind2]
                h_matrix[k, lower_triangular_index(j, i)] = -mu_h[muind1] - mu_h[muind2]

            muind = fk + j * k + j
            h_matrix[j, lower_triangular_index(k, j)] = mu_h[muind]
            h_matrix[k, lower_triangular_index(j, j)] = -mu_h[muind]

        for i in range(k):
            muind = fk + k * k + i
            h_matrix[i, lower_triangular_index(k, k)] = mu_h[muind]
            h_matrix[k, lower_triangular_index(k, i)] = -mu_h[muind]

    return h_matrix


@njit(cache=True)
def compressed_h_to_mu_h(h_matrix: np.ndarray, r: int) -> np.ndarray:
    mu_h = np.zeros(mu_h_dimension(r), dtype=np.float64)

    for k in range(r):
        fk = (k * (k - 1) * (k + 1)) // 3
        for j in range(k):
            for i in range(j):
                muind1 = fk + i * k + j
                muind2 = fk + j * k + i
                mu_h[muind1] = h_matrix[i, lower_triangular_index(k, j)]
                mu_h[muind2] = h_matrix[j, lower_triangular_index(k, i)]

            muind = fk + j * k + j
            mu_h[muind] = h_matrix[j, lower_triangular_index(k, j)]

        for i in range(k):
            muind = fk + k * k + i
            mu_h[muind] = h_matrix[i, lower_triangular_index(k, k)]

    return mu_h


@njit(cache=True)
def compressed_h_gradient_to_mu_h(h_grad: np.ndarray, r: int) -> np.ndarray:
    dmu_h = np.zeros(mu_h_dimension(r), dtype=np.float64)

    for k in range(r):
        fk = (k * (k - 1) * (k + 1)) // 3
        for j in range(k):
            for i in range(j):
                muind1 = fk + i * k + j
                muind2 = fk + j * k + i

                dmu_h[muind1] += h_grad[i, lower_triangular_index(k, j)]
                dmu_h[muind1] -= h_grad[k, lower_triangular_index(j, i)]

                dmu_h[muind2] += h_grad[j, lower_triangular_index(k, i)]
                dmu_h[muind2] -= h_grad[k, lower_triangular_index(j, i)]

            muind = fk + j * k + j
            dmu_h[muind] += h_grad[j, lower_triangular_index(k, j)]
            dmu_h[muind] -= h_grad[k, lower_triangular_index(j, j)]

        for i in range(k):
            muind = fk + k * k + i
            dmu_h[muind] += h_grad[i, lower_triangular_index(k, k)]
            dmu_h[muind] -= h_grad[k, lower_triangular_index(k, i)]

    return dmu_h


def a_gradient_to_stabilized_params(
    a_grad: np.ndarray,
    s_params: np.ndarray,
    w_params: np.ndarray,
    r: int,
) -> tuple[np.ndarray, np.ndarray]:
    s_matrix = s_params_to_matrix(s_params.astype(np.float64), r)
    grad_s_matrix = -(a_grad + a_grad.T) @ s_matrix

    grad_s_params = np.zeros_like(s_params, dtype=np.float64)
    idx = 0
    for i in range(r):
        for j in range(i, r):
            grad_s_params[idx] = grad_s_matrix[i, j]
            idx += 1

    grad_w_params = np.zeros_like(w_params, dtype=np.float64)
    idx = 0
    for i in range(r):
        for j in range(i + 1, r):
            grad_w_params[idx] = a_grad[i, j] - a_grad[j, i]
            idx += 1

    return grad_s_params, grad_w_params
