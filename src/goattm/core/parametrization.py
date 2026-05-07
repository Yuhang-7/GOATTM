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


def _validate_skew_cp_factors(u_matrix: np.ndarray, v_matrix: np.ndarray, z_matrix: np.ndarray) -> tuple[int, int]:
    if u_matrix.ndim != 2:
        raise ValueError(f"u_matrix must be rank-2, got shape {u_matrix.shape}")
    if v_matrix.shape != u_matrix.shape:
        raise ValueError(f"v_matrix must have shape {u_matrix.shape}, got {v_matrix.shape}")
    if z_matrix.shape != u_matrix.shape:
        raise ValueError(f"z_matrix must have shape {u_matrix.shape}, got {z_matrix.shape}")
    return u_matrix.shape


def skew_cp_quadratic_eval(
    u_matrix: np.ndarray,
    v_matrix: np.ndarray,
    z_matrix: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Evaluate the skew-CP quadratic term without materializing the dense tensor."""
    d, _ = _validate_skew_cp_factors(u_matrix, v_matrix, z_matrix)
    if state.ndim != 1 or state.shape[0] != d:
        raise ValueError(f"state must have shape ({d},), got {state.shape}")
    alpha = u_matrix.T @ state
    beta = v_matrix.T @ state
    gamma = z_matrix.T @ state
    return 2.0 * (u_matrix @ (gamma * beta) - v_matrix @ (gamma * alpha))


def skew_cp_quadratic_eval_batch(
    u_matrix: np.ndarray,
    v_matrix: np.ndarray,
    z_matrix: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Evaluate the skew-CP quadratic term for states with shape (n, d)."""
    d, _ = _validate_skew_cp_factors(u_matrix, v_matrix, z_matrix)
    if states.ndim != 2 or states.shape[1] != d:
        raise ValueError(f"states must have shape (n, {d}), got {states.shape}")
    alpha = states @ u_matrix
    beta = states @ v_matrix
    gamma = states @ z_matrix
    return 2.0 * ((gamma * beta) @ u_matrix.T - (gamma * alpha) @ v_matrix.T)


def skew_cp_quadratic_jacobian_matrix(
    u_matrix: np.ndarray,
    v_matrix: np.ndarray,
    z_matrix: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Return the Jacobian of the skew-CP quadratic term at a state."""
    d, _ = _validate_skew_cp_factors(u_matrix, v_matrix, z_matrix)
    if state.ndim != 1 or state.shape[0] != d:
        raise ValueError(f"state must have shape ({d},), got {state.shape}")
    alpha = u_matrix.T @ state
    beta = v_matrix.T @ state
    gamma = z_matrix.T @ state
    return 2.0 * (
        (u_matrix * beta.reshape(1, -1)) @ z_matrix.T
        + (u_matrix * gamma.reshape(1, -1)) @ v_matrix.T
        - (v_matrix * alpha.reshape(1, -1)) @ z_matrix.T
        - (v_matrix * gamma.reshape(1, -1)) @ u_matrix.T
    )


def skew_cp_parameter_action(
    u_matrix: np.ndarray,
    v_matrix: np.ndarray,
    z_matrix: np.ndarray,
    du_matrix: np.ndarray,
    dv_matrix: np.ndarray,
    dz_matrix: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Directional derivative of the skew-CP quadratic term with respect to its factors."""
    d, _ = _validate_skew_cp_factors(u_matrix, v_matrix, z_matrix)
    _validate_skew_cp_factors(du_matrix, dv_matrix, dz_matrix)
    if du_matrix.shape != u_matrix.shape:
        raise ValueError(f"direction factors must have shape {u_matrix.shape}, got {du_matrix.shape}")
    if state.ndim != 1 or state.shape[0] != d:
        raise ValueError(f"state must have shape ({d},), got {state.shape}")

    alpha = u_matrix.T @ state
    beta = v_matrix.T @ state
    gamma = z_matrix.T @ state
    d_alpha = du_matrix.T @ state
    d_beta = dv_matrix.T @ state
    d_gamma = dz_matrix.T @ state
    return 2.0 * (
        du_matrix @ (gamma * beta)
        + u_matrix @ (d_gamma * beta + gamma * d_beta)
        - dv_matrix @ (gamma * alpha)
        - v_matrix @ (d_gamma * alpha + gamma * d_alpha)
    )


def skew_cp_to_compressed_h(
    u_matrix: np.ndarray,
    v_matrix: np.ndarray,
    z_matrix: np.ndarray,
) -> np.ndarray:
    """Materialize the compressed symmetric quadratic matrix represented by skew-CP factors."""
    d, rank = _validate_skew_cp_factors(u_matrix, v_matrix, z_matrix)
    h_matrix = np.zeros((d, compressed_quadratic_dimension(d)), dtype=np.float64)
    for a in range(rank):
        u_col = np.asarray(u_matrix[:, a], dtype=np.float64)
        v_col = np.asarray(v_matrix[:, a], dtype=np.float64)
        z_col = np.asarray(z_matrix[:, a], dtype=np.float64)
        idx = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    coeff = 2.0 * z_col[i] * (u_col * v_col[i] - v_col * u_col[i])
                else:
                    coeff = 2.0 * (
                        z_col[i] * (u_col * v_col[j] - v_col * u_col[j])
                        + z_col[j] * (u_col * v_col[i] - v_col * u_col[i])
                    )
                h_matrix[:, idx] += coeff
                idx += 1
    return h_matrix


def skew_cp_direction_to_compressed_h(
    u_matrix: np.ndarray,
    v_matrix: np.ndarray,
    z_matrix: np.ndarray,
    du_matrix: np.ndarray,
    dv_matrix: np.ndarray,
    dz_matrix: np.ndarray,
) -> np.ndarray:
    """Materialize the compressed H directional derivative for skew-CP factor directions."""
    d, rank = _validate_skew_cp_factors(u_matrix, v_matrix, z_matrix)
    _validate_skew_cp_factors(du_matrix, dv_matrix, dz_matrix)
    if du_matrix.shape != u_matrix.shape:
        raise ValueError(f"direction factors must have shape {u_matrix.shape}, got {du_matrix.shape}")
    h_matrix = np.zeros((d, compressed_quadratic_dimension(d)), dtype=np.float64)
    for a in range(rank):
        u_col = np.asarray(u_matrix[:, a], dtype=np.float64)
        v_col = np.asarray(v_matrix[:, a], dtype=np.float64)
        z_col = np.asarray(z_matrix[:, a], dtype=np.float64)
        du_col = np.asarray(du_matrix[:, a], dtype=np.float64)
        dv_col = np.asarray(dv_matrix[:, a], dtype=np.float64)
        dz_col = np.asarray(dz_matrix[:, a], dtype=np.float64)
        idx = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    coeff = 2.0 * (
                        dz_col[i] * (u_col * v_col[i] - v_col * u_col[i])
                        + z_col[i] * (du_col * v_col[i] + u_col * dv_col[i] - dv_col * u_col[i] - v_col * du_col[i])
                    )
                else:
                    coeff = 2.0 * (
                        dz_col[i] * (u_col * v_col[j] - v_col * u_col[j])
                        + z_col[i] * (du_col * v_col[j] + u_col * dv_col[j] - dv_col * u_col[j] - v_col * du_col[j])
                        + dz_col[j] * (u_col * v_col[i] - v_col * u_col[i])
                        + z_col[j] * (du_col * v_col[i] + u_col * dv_col[i] - dv_col * u_col[i] - v_col * du_col[i])
                    )
                h_matrix[:, idx] += coeff
                idx += 1
    return h_matrix


def compressed_h_gradient_to_skew_cp(
    h_grad: np.ndarray,
    u_matrix: np.ndarray,
    v_matrix: np.ndarray,
    z_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull back a compressed-H gradient to skew-CP factor gradients."""
    d, rank = _validate_skew_cp_factors(u_matrix, v_matrix, z_matrix)
    if h_grad.shape != (d, compressed_quadratic_dimension(d)):
        raise ValueError(f"h_grad must have shape ({d}, {compressed_quadratic_dimension(d)}), got {h_grad.shape}")

    n_grad = np.zeros((d, d, d), dtype=np.float64)
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                n_grad[:, i, i] += 2.0 * h_grad[:, idx]
            else:
                n_grad[:, i, j] += 2.0 * h_grad[:, idx]
                n_grad[:, j, i] += 2.0 * h_grad[:, idx]
            idx += 1

    du = np.zeros_like(u_matrix, dtype=np.float64)
    dv = np.zeros_like(v_matrix, dtype=np.float64)
    dz = np.zeros_like(z_matrix, dtype=np.float64)
    for a in range(rank):
        u_col = np.asarray(u_matrix[:, a], dtype=np.float64)
        v_col = np.asarray(v_matrix[:, a], dtype=np.float64)
        z_col = np.asarray(z_matrix[:, a], dtype=np.float64)
        for l in range(d):
            du[l, a] = (
                np.sum(n_grad[l, :, :] * (v_col[:, None] * z_col[None, :]))
                - np.sum(n_grad[:, l, :] * (v_col[:, None] * z_col[None, :]))
            )
            dv[l, a] = (
                np.sum(n_grad[:, l, :] * (u_col[:, None] * z_col[None, :]))
                - np.sum(n_grad[l, :, :] * (u_col[:, None] * z_col[None, :]))
            )
            dz[l, a] = np.sum(n_grad[:, :, l] * (u_col[:, None] * v_col[None, :] - v_col[:, None] * u_col[None, :]))
    return du, dv, dz
