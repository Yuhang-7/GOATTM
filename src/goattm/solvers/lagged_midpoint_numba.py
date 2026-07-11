from __future__ import annotations

import os
from typing import Callable

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


def lagged_midpoint_time_grid(observation_times: np.ndarray, max_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = [float(observation_times[0])]
    dt_history: list[float] = []
    observation_indices = [0]
    current_time = float(observation_times[0])
    for target in observation_times[1:]:
        target_time = float(target)
        while current_time < target_time - 1e-14:
            step_dt = min(float(max_dt), target_time - current_time)
            current_time += step_dt
            dt_history.append(step_dt)
            times.append(current_time)
        observation_indices.append(len(times) - 1)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(dt_history, dtype=np.float64),
        np.asarray(observation_indices, dtype=np.int64),
    )


def lagged_midpoint_final_time_grid(t_final: float, max_dt: float) -> tuple[np.ndarray, np.ndarray]:
    times = [0.0]
    dt_history: list[float] = []
    current_time = 0.0
    while current_time < float(t_final) - 1e-14:
        step_dt = min(float(max_dt), float(t_final) - current_time)
        current_time += step_dt
        dt_history.append(step_dt)
        times.append(current_time)
    return np.asarray(times, dtype=np.float64), np.asarray(dt_history, dtype=np.float64)


def presample_lagged_midpoint_inputs(
    input_function: Callable[[float], np.ndarray] | None,
    times: np.ndarray,
    dt_history: np.ndarray,
    input_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if dt_history.shape[0] == 0:
        empty = np.zeros((0, int(input_dimension)), dtype=np.float64)
        return empty, empty, empty
    if input_function is None:
        zeros = np.zeros((dt_history.shape[0], int(input_dimension)), dtype=np.float64)
        return zeros, zeros.copy(), zeros.copy()
    sampler = getattr(input_function, "sample_lagged_midpoint_inputs", None)
    if sampler is not None:
        p0, pq, pm = sampler(times, dt_history)
        return np.asarray(p0, dtype=np.float64), np.asarray(pq, dtype=np.float64), np.asarray(pm, dtype=np.float64)
    return None


@njit(cache=True)
def _quadratic_features_numba(u: np.ndarray) -> np.ndarray:
    r = u.shape[0]
    out = np.zeros((r * (r + 1)) // 2, dtype=np.float64)
    idx = 0
    for i in range(r):
        for j in range(i + 1):
            out[idx] = u[i] * u[j]
            idx += 1
    return out


@njit(cache=True)
def _rhs_numba(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    r = u.shape[0]
    out = np.zeros(r, dtype=np.float64)
    for row in range(r):
        acc = c[row]
        for j in range(r):
            acc += a[row, j] * u[j]
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                acc += h[row, idx] * u[i] * u[j]
                idx += 1
        for j in range(p.shape[0]):
            acc += b[row, j] * p[j]
        out[row] = acc
    return out


@njit(cache=True)
def _rhs_parameter_action_numba(delta_a: np.ndarray, delta_h: np.ndarray, delta_b: np.ndarray, delta_c: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    r = u.shape[0]
    out = np.zeros(r, dtype=np.float64)
    for row in range(r):
        acc = delta_c[row]
        for j in range(r):
            acc += delta_a[row, j] * u[j]
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                acc += delta_h[row, idx] * u[i] * u[j]
                idx += 1
        for j in range(p.shape[0]):
            acc += delta_b[row, j] * p[j]
        out[row] = acc
    return out


@njit(cache=True)
def _rhs_parameter_action_from_features_numba(
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    u: np.ndarray,
    quadratic_feature: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    return delta_c + delta_a @ u + delta_h @ quadratic_feature + delta_b @ p


@njit(cache=True)
def _rhs_jacobian_action_numba(a: np.ndarray, h: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r = u.shape[0]
    out = np.zeros(r, dtype=np.float64)
    for row in range(r):
        acc = 0.0
        for col in range(r):
            acc += a[row, col] * v[col]
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h[row, idx]
                if i == j:
                    acc += 2.0 * coeff * u[i] * v[i]
                else:
                    acc += coeff * (u[j] * v[i] + u[i] * v[j])
                idx += 1
        out[row] = acc
    return out


@njit(cache=True)
def _rhs_jacobian_transpose_action_numba(a: np.ndarray, h: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r = u.shape[0]
    out = np.zeros(r, dtype=np.float64)
    for row in range(r):
        for col in range(r):
            out[col] += a[row, col] * v[row]
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h[row, idx]
                if i == j:
                    out[i] += 2.0 * coeff * u[i] * v[row]
                else:
                    out[i] += coeff * u[j] * v[row]
                    out[j] += coeff * u[i] * v[row]
                idx += 1
    return out


@njit(cache=True)
def _rhs_jacobian_matrix_numba(a: np.ndarray, h: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Materialize the quadratic RHS Jacobian once for cache reuse."""
    r = u.shape[0]
    out = a.copy()
    for row in range(r):
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h[row, idx]
                if i == j:
                    out[row, i] += 2.0 * coeff * u[i]
                else:
                    out[row, i] += coeff * u[j]
                    out[row, j] += coeff * u[i]
                idx += 1
    return out


@njit(cache=True)
def _rhs_jacobian_direction_transpose_action_numba(
    h: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    state: np.ndarray,
    state_tangent: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Apply the transpose directional state-Jacobian derivative."""
    r = state.shape[0]
    out = np.zeros(r, dtype=np.float64)
    for row in range(r):
        for col in range(r):
            out[col] += delta_a[row, col] * vector[row]
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = delta_h[row, idx] * state[j] + h[row, idx] * state_tangent[j]
                if i == j:
                    out[i] += 2.0 * coeff * vector[row]
                else:
                    out[i] += coeff * vector[row]
                    other_coeff = delta_h[row, idx] * state[i] + h[row, idx] * state_tangent[i]
                    out[j] += other_coeff * vector[row]
                idx += 1
    return out


@njit(cache=True)
def _bilinear_action_numba(h: np.ndarray, u: np.ndarray) -> np.ndarray:
    q = h.shape[0]
    r = u.shape[0]
    out = np.zeros((q, r), dtype=np.float64)
    for row in range(q):
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h[row, idx]
                if i == j:
                    out[row, i] += coeff * u[i]
                else:
                    half = 0.5 * coeff
                    out[row, i] += half * u[j]
                    out[row, j] += half * u[i]
                idx += 1
    return out


@njit(cache=True)
def _rk4_half_predictor_numba(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, state: np.ndarray, dt: float, p0: np.ndarray, pq: np.ndarray, pm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_dt = 0.5 * dt
    k1 = _rhs_numba(a, h, b, c, state, p0)
    y2 = state + 0.5 * half_dt * k1
    k2 = _rhs_numba(a, h, b, c, y2, pq)
    y3 = state + 0.5 * half_dt * k2
    k3 = _rhs_numba(a, h, b, c, y3, pq)
    y4 = state + half_dt * k3
    k4 = _rhs_numba(a, h, b, c, y4, pm)
    predictor = state + (half_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return predictor, y2, y3, y4


@njit(cache=True)
def _rk4_half_tangent_numba(
    a: np.ndarray,
    h: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    state: np.ndarray,
    state_tangent: np.ndarray,
    dt: float,
    p0: np.ndarray,
    pq: np.ndarray,
    pm: np.ndarray,
) -> np.ndarray:
    half_dt = 0.5 * dt
    k1 = _rhs_numba(a, h, b, c, state, p0)
    # The tangent stages need the base RK stage states; recomputing them here
    # keeps this kernel independent from Python-side stage-cache objects.
    y2 = state + 0.5 * half_dt * k1
    k2_base = _rhs_numba(a, h, b, c, y2, pq)
    y3 = state + 0.5 * half_dt * k2_base
    k3_base = _rhs_numba(a, h, b, c, y3, pq)
    y4 = state + half_dt * k3_base

    dk1 = _rhs_jacobian_action_numba(a, h, state, state_tangent) + _rhs_parameter_action_numba(delta_a, delta_h, delta_b, delta_c, state, p0)
    dy2 = state_tangent + 0.5 * half_dt * dk1
    dk2 = _rhs_jacobian_action_numba(a, h, y2, dy2) + _rhs_parameter_action_numba(delta_a, delta_h, delta_b, delta_c, y2, pq)
    dy3 = state_tangent + 0.5 * half_dt * dk2
    dk3 = _rhs_jacobian_action_numba(a, h, y3, dy3) + _rhs_parameter_action_numba(delta_a, delta_h, delta_b, delta_c, y3, pq)
    dy4 = state_tangent + half_dt * dk3
    dk4 = _rhs_jacobian_action_numba(a, h, y4, dy4) + _rhs_parameter_action_numba(delta_a, delta_h, delta_b, delta_c, y4, pm)
    return state_tangent + (half_dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)


@njit(cache=True)
def _rk4_half_tangent_from_cached_stages_numba(
    a: np.ndarray,
    h: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    state: np.ndarray,
    state_tangent: np.ndarray,
    dt: float,
    p0: np.ndarray,
    pq: np.ndarray,
    pm: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    y4: np.ndarray,
) -> np.ndarray:
    """Propagate the RK4 predictor tangent using cached base stages."""
    half_dt = 0.5 * dt
    dk1 = _rhs_jacobian_action_numba(a, h, state, state_tangent) + _rhs_parameter_action_numba(
        delta_a, delta_h, delta_b, delta_c, state, p0
    )
    dy2 = state_tangent + 0.5 * half_dt * dk1
    dk2 = _rhs_jacobian_action_numba(a, h, y2, dy2) + _rhs_parameter_action_numba(
        delta_a, delta_h, delta_b, delta_c, y2, pq
    )
    dy3 = state_tangent + 0.5 * half_dt * dk2
    dk3 = _rhs_jacobian_action_numba(a, h, y3, dy3) + _rhs_parameter_action_numba(
        delta_a, delta_h, delta_b, delta_c, y3, pq
    )
    dy4 = state_tangent + half_dt * dk3
    dk4 = _rhs_jacobian_action_numba(a, h, y4, dy4) + _rhs_parameter_action_numba(
        delta_a, delta_h, delta_b, delta_c, y4, pm
    )
    return state_tangent + (half_dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)


@njit(cache=True)
def _rk4_half_tangent_from_cached_jacobians_numba(
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    state: np.ndarray,
    state_tangent: np.ndarray,
    dt: float,
    p0: np.ndarray,
    pq: np.ndarray,
    pm: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    y4: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
    feature1: np.ndarray,
    feature2: np.ndarray,
    feature3: np.ndarray,
    feature4: np.ndarray,
) -> np.ndarray:
    """Propagate the predictor tangent using cached stage Jacobians."""
    predictor_tangent, _, _, _ = _rk4_half_tangent_stages_from_cached_jacobians_numba(
        delta_a, delta_h, delta_b, delta_c, state, state_tangent, dt,
        p0, pq, pm, y2, y3, y4,
        jacobian1, jacobian2, jacobian3, jacobian4,
        feature1, feature2, feature3, feature4,
    )
    return predictor_tangent


@njit(cache=True)
def _rk4_half_tangent_stages_from_cached_jacobians_numba(
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    state: np.ndarray,
    state_tangent: np.ndarray,
    dt: float,
    p0: np.ndarray,
    pq: np.ndarray,
    pm: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    y4: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
    feature1: np.ndarray,
    feature2: np.ndarray,
    feature3: np.ndarray,
    feature4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return predictor and RK stage tangents from cached base quantities."""
    half_dt = 0.5 * dt
    dk1 = jacobian1 @ state_tangent + _rhs_parameter_action_from_features_numba(
        delta_a, delta_h, delta_b, delta_c, state, feature1, p0
    )
    dy2 = state_tangent + 0.5 * half_dt * dk1
    dk2 = jacobian2 @ dy2 + _rhs_parameter_action_from_features_numba(
        delta_a, delta_h, delta_b, delta_c, y2, feature2, pq
    )
    dy3 = state_tangent + 0.5 * half_dt * dk2
    dk3 = jacobian3 @ dy3 + _rhs_parameter_action_from_features_numba(
        delta_a, delta_h, delta_b, delta_c, y3, feature3, pq
    )
    dy4 = state_tangent + half_dt * dk3
    dk4 = jacobian4 @ dy4 + _rhs_parameter_action_from_features_numba(
        delta_a, delta_h, delta_b, delta_c, y4, feature4, pm
    )
    predictor_tangent = state_tangent + (half_dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)
    return predictor_tangent, dy2, dy3, dy4


@njit(cache=True)
def _lagged_midpoint_step_numba(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, state: np.ndarray, dt: float, p0: np.ndarray, pq: np.ndarray, pm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictor, y2, y3, y4 = _rk4_half_predictor_numba(a, h, b, c, state, dt, p0, pq, pm)
    linear_operator = a + _bilinear_action_numba(h, predictor)
    r = state.shape[0]
    half_dt = 0.5 * dt
    system = np.eye(r, dtype=np.float64) - half_dt * linear_operator
    rhs = np.zeros(r, dtype=np.float64)
    forcing = c.copy()
    for row in range(r):
        for j in range(pm.shape[0]):
            forcing[row] += b[row, j] * pm[j]
        rhs[row] = state[row] + dt * forcing[row]
        for col in range(r):
            rhs[row] += half_dt * linear_operator[row, col] * state[col]
    next_state = np.linalg.solve(system, rhs)
    jacobian1 = _rhs_jacobian_matrix_numba(a, h, state)
    jacobian2 = _rhs_jacobian_matrix_numba(a, h, y2)
    jacobian3 = _rhs_jacobian_matrix_numba(a, h, y3)
    jacobian4 = _rhs_jacobian_matrix_numba(a, h, y4)
    return next_state, predictor, y2, y3, y4, linear_operator, system, jacobian1, jacobian2, jacobian3, jacobian4


@njit(cache=True)
def rollout_lagged_midpoint_presampled_kernel(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, u0: np.ndarray, dt_history: np.ndarray, p0_values: np.ndarray, pq_values: np.ndarray, pm_values: np.ndarray) -> tuple[bool, int, np.ndarray]:
    n_steps = dt_history.shape[0]
    r = u0.shape[0]
    states = np.zeros((n_steps + 1, r), dtype=np.float64)
    states[0, :] = u0
    accepted = 0
    success = True
    for step in range(n_steps):
        next_state, _, _, _, _, _, _, _, _, _, _ = _lagged_midpoint_step_numba(a, h, b, c, states[step], dt_history[step], p0_values[step], pq_values[step], pm_values[step])
        finite = True
        for i in range(r):
            if not np.isfinite(next_state[i]):
                finite = False
        if not finite:
            success = False
            break
        states[step + 1, :] = next_state
        accepted += 1
    return success, accepted, states


@njit(cache=True)
def rollout_lagged_midpoint_presampled_cached_kernel(
    a: np.ndarray,
    h: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u0: np.ndarray,
    dt_history: np.ndarray,
    p0_values: np.ndarray,
    pq_values: np.ndarray,
    pm_values: np.ndarray,
) -> tuple[bool, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out and retain the stages needed by reverse/tangent kernels."""
    n_steps = dt_history.shape[0]
    r = u0.shape[0]
    states = np.zeros((n_steps + 1, r), dtype=np.float64)
    predictors = np.zeros((n_steps, r), dtype=np.float64)
    stage2 = np.zeros((n_steps, r), dtype=np.float64)
    stage3 = np.zeros((n_steps, r), dtype=np.float64)
    stage4 = np.zeros((n_steps, r), dtype=np.float64)
    linear_operators = np.zeros((n_steps, r, r), dtype=np.float64)
    system_matrices = np.zeros((n_steps, r, r), dtype=np.float64)
    jacobian1 = np.zeros((n_steps, r, r), dtype=np.float64)
    jacobian2 = np.zeros((n_steps, r, r), dtype=np.float64)
    jacobian3 = np.zeros((n_steps, r, r), dtype=np.float64)
    jacobian4 = np.zeros((n_steps, r, r), dtype=np.float64)
    feature_dimension = h.shape[1]
    feature1 = np.zeros((n_steps, feature_dimension), dtype=np.float64)
    feature2 = np.zeros((n_steps, feature_dimension), dtype=np.float64)
    feature3 = np.zeros((n_steps, feature_dimension), dtype=np.float64)
    feature4 = np.zeros((n_steps, feature_dimension), dtype=np.float64)
    states[0, :] = u0
    accepted = 0
    success = True
    for step in range(n_steps):
        next_state, predictor, y2, y3, y4, linear_operator, system, j1, j2, j3, j4 = _lagged_midpoint_step_numba(
            a, h, b, c, states[step], dt_history[step], p0_values[step], pq_values[step], pm_values[step]
        )
        finite = True
        for i in range(r):
            if not np.isfinite(next_state[i]):
                finite = False
        if not finite:
            success = False
            break
        states[step + 1, :] = next_state
        predictors[step, :] = predictor
        stage2[step, :] = y2
        stage3[step, :] = y3
        stage4[step, :] = y4
        linear_operators[step, :, :] = linear_operator
        system_matrices[step, :, :] = system
        jacobian1[step, :, :] = j1
        jacobian2[step, :, :] = j2
        jacobian3[step, :, :] = j3
        jacobian4[step, :, :] = j4
        feature1[step, :] = _quadratic_features_numba(states[step])
        feature2[step, :] = _quadratic_features_numba(y2)
        feature3[step, :] = _quadratic_features_numba(y3)
        feature4[step, :] = _quadratic_features_numba(y4)
        accepted += 1
    return success, accepted, states, predictors, stage2, stage3, stage4, linear_operators, system_matrices, jacobian1, jacobian2, jacobian3, jacobian4, feature1, feature2, feature3, feature4


@njit(cache=True)
def rollout_lagged_midpoint_explicit_parameter_tangent_presampled_kernel(
    a: np.ndarray,
    h: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    states: np.ndarray,
    dt_history: np.ndarray,
    p0_values: np.ndarray,
    pq_values: np.ndarray,
    pm_values: np.ndarray,
) -> np.ndarray:
    n_steps = dt_history.shape[0]
    r = states.shape[1]
    tangent_states = np.zeros_like(states)
    current_tangent = np.zeros(r, dtype=np.float64)
    identity = np.eye(r, dtype=np.float64)
    for step in range(n_steps):
        state = states[step]
        next_state = states[step + 1]
        dt = dt_history[step]
        half_dt = 0.5 * dt
        predictor, _, _, _ = _rk4_half_predictor_numba(a, h, b, c, state, dt, p0_values[step], pq_values[step], pm_values[step])
        predictor_tangent = _rk4_half_tangent_numba(
        a,
        h,
        b,
        c,
        delta_a,
        delta_h,
            delta_b,
            delta_c,
            state,
            current_tangent,
            dt,
            p0_values[step],
            pq_values[step],
            pm_values[step],
        )
        linear_operator = a + _bilinear_action_numba(h, predictor)
        linear_operator_tangent = _bilinear_action_numba(h, predictor_tangent) + delta_a + _bilinear_action_numba(delta_h, predictor)

        forcing_tangent = delta_c.copy()
        for row in range(r):
            for col in range(pm_values.shape[1]):
                forcing_tangent[row] += delta_b[row, col] * pm_values[step, col]

        system = identity - half_dt * linear_operator
        rhs_tangent = (identity + half_dt * linear_operator) @ current_tangent
        rhs_tangent += half_dt * (linear_operator_tangent @ (state + next_state))
        rhs_tangent += dt * forcing_tangent
        current_tangent = np.linalg.solve(system, rhs_tangent)
        tangent_states[step + 1, :] = current_tangent
    return tangent_states


@njit(cache=True)
def rollout_lagged_midpoint_explicit_parameter_tangent_cached_kernel(
    a: np.ndarray,
    h: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    states: np.ndarray,
    dt_history: np.ndarray,
    predictors: np.ndarray,
    stage2: np.ndarray,
    stage3: np.ndarray,
    stage4: np.ndarray,
    linear_operators: np.ndarray,
    system_matrices: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
    feature1: np.ndarray,
    feature2: np.ndarray,
    feature3: np.ndarray,
    feature4: np.ndarray,
    p0_values: np.ndarray,
    pq_values: np.ndarray,
    pm_values: np.ndarray,
) -> np.ndarray:
    """Tangent rollout reusing the base RK4 stages and linear systems."""
    n_steps = dt_history.shape[0]
    r = states.shape[1]
    tangent_states = np.zeros_like(states)
    current_tangent = np.zeros(r, dtype=np.float64)
    for step in range(n_steps):
        state = states[step]
        next_state = states[step + 1]
        dt = dt_history[step]
        half_dt = 0.5 * dt
        predictor_tangent = _rk4_half_tangent_from_cached_jacobians_numba(
            delta_a, delta_h, delta_b, delta_c, state, current_tangent, dt,
            p0_values[step], pq_values[step], pm_values[step],
            stage2[step], stage3[step], stage4[step],
            jacobian1[step], jacobian2[step], jacobian3[step], jacobian4[step],
            feature1[step], feature2[step], feature3[step], feature4[step],
        )
        linear_operator = linear_operators[step]
        linear_operator_tangent = (
            _bilinear_action_numba(h, predictor_tangent)
            + delta_a
            + _bilinear_action_numba(delta_h, predictors[step])
        )
        forcing_tangent = delta_c.copy()
        for row in range(r):
            for col in range(pm_values.shape[1]):
                forcing_tangent[row] += delta_b[row, col] * pm_values[step, col]
        rhs_tangent = current_tangent + half_dt * (linear_operator @ current_tangent)
        rhs_tangent += half_dt * (linear_operator_tangent @ (state + next_state))
        rhs_tangent += dt * forcing_tangent
        current_tangent = np.linalg.solve(system_matrices[step], rhs_tangent)
        tangent_states[step + 1, :] = current_tangent
    return tangent_states


@njit(cache=True)
def _accumulate_bilinear_reverse_numba(h: np.ndarray, predictor: np.ndarray, operator_bar: np.ndarray, h_grad: np.ndarray) -> np.ndarray:
    r = predictor.shape[0]
    z_bar = np.zeros(r, dtype=np.float64)
    for row in range(r):
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h[row, idx]
                if i == j:
                    h_grad[row, idx] += operator_bar[row, i] * predictor[i]
                    z_bar[i] += operator_bar[row, i] * coeff
                else:
                    h_grad[row, idx] += 0.5 * (operator_bar[row, i] * predictor[j] + operator_bar[row, j] * predictor[i])
                    z_bar[j] += 0.5 * operator_bar[row, i] * coeff
                    z_bar[i] += 0.5 * operator_bar[row, j] * coeff
                idx += 1
    return z_bar


@njit(cache=True)
def _bilinear_reverse_state_numba(h: np.ndarray, predictor: np.ndarray, operator_bar: np.ndarray) -> np.ndarray:
    """Reverse only the state dependence of the bilinear operator."""
    r = predictor.shape[0]
    z_bar = np.zeros(r, dtype=np.float64)
    for row in range(r):
        idx = 0
        for i in range(r):
            for j in range(i + 1):
                coeff = h[row, idx]
                if i == j:
                    z_bar[i] += operator_bar[row, i] * coeff
                else:
                    z_bar[j] += 0.5 * operator_bar[row, i] * coeff
                    z_bar[i] += 0.5 * operator_bar[row, j] * coeff
                idx += 1
    return z_bar


@njit(cache=True)
def _accumulate_stage_parameter_gradients_numba(stage_state: np.ndarray, stage_adjoint: np.ndarray, stage_input: np.ndarray, a_grad: np.ndarray, h_grad: np.ndarray, b_grad: np.ndarray, c_grad: np.ndarray) -> None:
    r = stage_state.shape[0]
    zeta = _quadratic_features_numba(stage_state)
    for row in range(r):
        c_grad[row] += stage_adjoint[row]
        for col in range(r):
            a_grad[row, col] += stage_adjoint[row] * stage_state[col]
        for col in range(zeta.shape[0]):
            h_grad[row, col] += stage_adjoint[row] * zeta[col]
        for col in range(stage_input.shape[0]):
            b_grad[row, col] += stage_adjoint[row] * stage_input[col]


@njit(cache=True)
def _rk4_half_reverse_numba(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, state: np.ndarray, dt: float, p0: np.ndarray, pq: np.ndarray, pm: np.ndarray, predictor_bar: np.ndarray, accumulate_parameters: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_dt = 0.5 * dt
    predictor, y2, y3, y4 = _rk4_half_predictor_numba(a, h, b, c, state, dt, p0, pq, pm)
    state_bar = predictor_bar.copy()
    k1_bar = (half_dt / 6.0) * predictor_bar
    k2_bar = (half_dt / 3.0) * predictor_bar
    k3_bar = (half_dt / 3.0) * predictor_bar
    k4_bar = (half_dt / 6.0) * predictor_bar
    r = state.shape[0]
    a_grad = np.zeros((r, r), dtype=np.float64)
    h_grad = np.zeros_like(h)
    b_grad = np.zeros_like(b)
    c_grad = np.zeros(r, dtype=np.float64)

    y4_bar = _rhs_jacobian_transpose_action_numba(a, h, y4, k4_bar)
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients_numba(y4, k4_bar, pm, a_grad, h_grad, b_grad, c_grad)
    state_bar += y4_bar
    k3_bar = k3_bar + half_dt * y4_bar

    y3_bar = _rhs_jacobian_transpose_action_numba(a, h, y3, k3_bar)
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients_numba(y3, k3_bar, pq, a_grad, h_grad, b_grad, c_grad)
    state_bar += y3_bar
    k2_bar = k2_bar + 0.5 * half_dt * y3_bar

    y2_bar = _rhs_jacobian_transpose_action_numba(a, h, y2, k2_bar)
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients_numba(y2, k2_bar, pq, a_grad, h_grad, b_grad, c_grad)
    state_bar += y2_bar
    k1_bar = k1_bar + 0.5 * half_dt * y2_bar

    y1_bar = _rhs_jacobian_transpose_action_numba(a, h, state, k1_bar)
    if accumulate_parameters:
        _accumulate_stage_parameter_gradients_numba(state, k1_bar, p0, a_grad, h_grad, b_grad, c_grad)
    state_bar += y1_bar
    return state_bar, a_grad, h_grad, b_grad, c_grad


@njit(cache=True)
def _rk4_half_reverse_state_cached_numba(
    state: np.ndarray,
    dt: float,
    predictor_bar: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
) -> np.ndarray:
    """Reverse the predictor using cached RK4 stage states and no parameter gradients."""
    half_dt = 0.5 * dt
    state_bar = predictor_bar.copy()
    k1_bar = (half_dt / 6.0) * predictor_bar
    k2_bar = (half_dt / 3.0) * predictor_bar
    k3_bar = (half_dt / 3.0) * predictor_bar
    k4_bar = (half_dt / 6.0) * predictor_bar

    y4_bar = jacobian4.T @ k4_bar
    state_bar += y4_bar
    k3_bar = k3_bar + half_dt * y4_bar

    y3_bar = jacobian3.T @ k3_bar
    state_bar += y3_bar
    k2_bar = k2_bar + 0.5 * half_dt * y3_bar

    y2_bar = jacobian2.T @ k2_bar
    state_bar += y2_bar
    k1_bar = k1_bar + 0.5 * half_dt * y2_bar

    y1_bar = jacobian1.T @ k1_bar
    state_bar += y1_bar
    return state_bar


@njit(cache=True)
def _rk4_half_reverse_state_direction_cached_numba(
    h: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    state: np.ndarray,
    state_tangent: np.ndarray,
    dt: float,
    predictor_bar: np.ndarray,
    predictor_bar_tangent: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    y4: np.ndarray,
    y2_tangent: np.ndarray,
    y3_tangent: np.ndarray,
    y4_tangent: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
) -> np.ndarray:
    """Differentiate the cached RK4 reverse state action."""
    half_dt = 0.5 * dt
    k1_bar = (half_dt / 6.0) * predictor_bar
    k2_bar = (half_dt / 3.0) * predictor_bar
    k3_bar = (half_dt / 3.0) * predictor_bar
    k4_bar = (half_dt / 6.0) * predictor_bar
    delta_k1_bar = (half_dt / 6.0) * predictor_bar_tangent
    delta_k2_bar = (half_dt / 3.0) * predictor_bar_tangent
    delta_k3_bar = (half_dt / 3.0) * predictor_bar_tangent
    delta_k4_bar = (half_dt / 6.0) * predictor_bar_tangent
    state_bar_tangent = predictor_bar_tangent.copy()

    y4_bar = jacobian4.T @ k4_bar
    delta_y4_bar = jacobian4.T @ delta_k4_bar + _rhs_jacobian_direction_transpose_action_numba(
        h, delta_a, delta_h, y4, y4_tangent, k4_bar
    )
    state_bar_tangent += delta_y4_bar
    k3_bar += half_dt * y4_bar
    delta_k3_bar += half_dt * delta_y4_bar

    y3_bar = jacobian3.T @ k3_bar
    delta_y3_bar = jacobian3.T @ delta_k3_bar + _rhs_jacobian_direction_transpose_action_numba(
        h, delta_a, delta_h, y3, y3_tangent, k3_bar
    )
    state_bar_tangent += delta_y3_bar
    k2_bar += 0.5 * half_dt * y3_bar
    delta_k2_bar += 0.5 * half_dt * delta_y3_bar

    y2_bar = jacobian2.T @ k2_bar
    delta_y2_bar = jacobian2.T @ delta_k2_bar + _rhs_jacobian_direction_transpose_action_numba(
        h, delta_a, delta_h, y2, y2_tangent, k2_bar
    )
    state_bar_tangent += delta_y2_bar
    k1_bar += 0.5 * half_dt * y2_bar
    delta_k1_bar += 0.5 * half_dt * delta_y2_bar

    state_bar_tangent += jacobian1.T @ delta_k1_bar + _rhs_jacobian_direction_transpose_action_numba(
        h, delta_a, delta_h, state, state_tangent, k1_bar
    )
    return state_bar_tangent


@njit(cache=True)
def _lagged_midpoint_reverse_step_numba(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, previous_state: np.ndarray, next_state: np.ndarray, dt: float, p0: np.ndarray, pq: np.ndarray, pm: np.ndarray, adjoint_next: np.ndarray, accumulate_parameters: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictor, _, _, _ = _rk4_half_predictor_numba(a, h, b, c, previous_state, dt, p0, pq, pm)
    linear_operator = a + _bilinear_action_numba(h, predictor)
    r = previous_state.shape[0]
    half_dt = 0.5 * dt
    system = np.eye(r, dtype=np.float64) - half_dt * linear_operator
    alpha = np.linalg.solve(system.T, adjoint_next)

    operator_bar = half_dt * (np.outer(alpha, previous_state) + np.outer(alpha, next_state))
    state_bar = alpha + half_dt * (linear_operator.T @ alpha)
    forcing_bar = dt * alpha

    a_grad = operator_bar.copy()
    h_grad = np.zeros_like(h)
    b_grad = np.zeros_like(b)
    c_grad = forcing_bar.copy()
    for row in range(r):
        for col in range(pm.shape[0]):
            b_grad[row, col] += forcing_bar[row] * pm[col]

    predictor_bar = _accumulate_bilinear_reverse_numba(h, predictor, operator_bar, h_grad)
    pred_state_bar, pred_a_grad, pred_h_grad, pred_b_grad, pred_c_grad = _rk4_half_reverse_numba(
        a, h, b, c, previous_state, dt, p0, pq, pm, predictor_bar, accumulate_parameters
    )
    state_bar += pred_state_bar
    if accumulate_parameters:
        a_grad += pred_a_grad
        h_grad += pred_h_grad
        b_grad += pred_b_grad
        c_grad += pred_c_grad
    return state_bar, a_grad, h_grad, b_grad, c_grad


@njit(cache=True)
def _lagged_midpoint_reverse_state_cached_numba(
    a: np.ndarray,
    h: np.ndarray,
    previous_state: np.ndarray,
    next_state: np.ndarray,
    dt: float,
    predictor: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    y4: np.ndarray,
    linear_operator: np.ndarray,
    system: np.ndarray,
    adjoint_next: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
) -> np.ndarray:
    half_dt = 0.5 * dt
    alpha = np.linalg.solve(system.T, adjoint_next)
    operator_bar = half_dt * (np.outer(alpha, previous_state) + np.outer(alpha, next_state))
    state_bar = alpha + half_dt * (linear_operator.T @ alpha)
    predictor_bar = _bilinear_reverse_state_numba(h, predictor, operator_bar)
    return state_bar + _rk4_half_reverse_state_cached_numba(
        previous_state, dt, predictor_bar, jacobian1, jacobian2, jacobian3, jacobian4
    )


@njit(cache=True)
def _lagged_midpoint_reverse_state_direction_cached_numba(
    h: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    previous_state: np.ndarray,
    next_state: np.ndarray,
    previous_state_tangent: np.ndarray,
    next_state_tangent: np.ndarray,
    dt: float,
    predictor: np.ndarray,
    predictor_tangent: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    y4: np.ndarray,
    y2_tangent: np.ndarray,
    y3_tangent: np.ndarray,
    y4_tangent: np.ndarray,
    linear_operator: np.ndarray,
    system: np.ndarray,
    adjoint_next: np.ndarray,
    adjoint_next_tangent: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
) -> np.ndarray:
    """Differentiate one cached lagged-midpoint reverse state action."""
    half_dt = 0.5 * dt
    linear_operator_tangent = (
        delta_a
        + _bilinear_action_numba(delta_h, predictor)
        + _bilinear_action_numba(h, predictor_tangent)
    )
    alpha = np.linalg.solve(system.T, adjoint_next)
    alpha_rhs_tangent = adjoint_next_tangent + half_dt * (linear_operator_tangent.T @ alpha)
    alpha_tangent = np.linalg.solve(system.T, alpha_rhs_tangent)

    state_sum = previous_state + next_state
    state_sum_tangent = previous_state_tangent + next_state_tangent
    operator_bar = half_dt * np.outer(alpha, state_sum)
    operator_bar_tangent = half_dt * (
        np.outer(alpha_tangent, state_sum) + np.outer(alpha, state_sum_tangent)
    )
    state_bar_tangent = alpha_tangent + half_dt * (
        linear_operator.T @ alpha_tangent + linear_operator_tangent.T @ alpha
    )
    predictor_bar = _bilinear_reverse_state_numba(h, predictor, operator_bar)
    predictor_bar_tangent = (
        _bilinear_reverse_state_numba(h, predictor, operator_bar_tangent)
        + _bilinear_reverse_state_numba(delta_h, predictor, operator_bar)
    )
    return state_bar_tangent + _rk4_half_reverse_state_direction_cached_numba(
        h, delta_a, delta_h, previous_state, previous_state_tangent, dt,
        predictor_bar, predictor_bar_tangent,
        y2, y3, y4, y2_tangent, y3_tangent, y4_tangent,
        jacobian1, jacobian2, jacobian3, jacobian4,
    )


@njit(cache=True)
def compute_lagged_midpoint_discrete_adjoint_presampled_kernel(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, states: np.ndarray, dt_history: np.ndarray, state_loss_gradients: np.ndarray, p0_values: np.ndarray, pq_values: np.ndarray, pm_values: np.ndarray) -> np.ndarray:
    adjoints = np.zeros_like(states)
    adjoints[-1, :] = state_loss_gradients[-1, :]
    for step in range(dt_history.shape[0] - 1, -1, -1):
        state_bar, _, _, _, _ = _lagged_midpoint_reverse_step_numba(
            a, h, b, c, states[step], states[step + 1], dt_history[step], p0_values[step], pq_values[step], pm_values[step], adjoints[step + 1], False
        )
        adjoints[step, :] = state_loss_gradients[step, :] + state_bar
    return adjoints


@njit(cache=True)
def compute_lagged_midpoint_discrete_adjoint_cached_presampled_kernel(
    a: np.ndarray,
    h: np.ndarray,
    states: np.ndarray,
    dt_history: np.ndarray,
    state_loss_gradients: np.ndarray,
    predictors: np.ndarray,
    stage2: np.ndarray,
    stage3: np.ndarray,
    stage4: np.ndarray,
    linear_operators: np.ndarray,
    system_matrices: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
) -> np.ndarray:
    adjoints = np.zeros_like(states)
    adjoints[-1, :] = state_loss_gradients[-1, :]
    for step in range(dt_history.shape[0] - 1, -1, -1):
        state_bar = _lagged_midpoint_reverse_state_cached_numba(
            a, h, states[step], states[step + 1], dt_history[step],
            predictors[step], stage2[step], stage3[step], stage4[step],
            linear_operators[step], system_matrices[step], adjoints[step + 1],
            jacobian1[step], jacobian2[step], jacobian3[step], jacobian4[step],
        )
        adjoints[step, :] = state_loss_gradients[step, :] + state_bar
    return adjoints


@njit(cache=True)
def compute_lagged_midpoint_incremental_discrete_adjoint_cached_presampled_kernel(
    h: np.ndarray,
    delta_a: np.ndarray,
    delta_h: np.ndarray,
    delta_b: np.ndarray,
    delta_c: np.ndarray,
    states: np.ndarray,
    tangent_states: np.ndarray,
    dt_history: np.ndarray,
    base_adjoints: np.ndarray,
    state_loss_gradient_direction: np.ndarray,
    predictors: np.ndarray,
    stage2: np.ndarray,
    stage3: np.ndarray,
    stage4: np.ndarray,
    linear_operators: np.ndarray,
    system_matrices: np.ndarray,
    jacobian1: np.ndarray,
    jacobian2: np.ndarray,
    jacobian3: np.ndarray,
    jacobian4: np.ndarray,
    feature1: np.ndarray,
    feature2: np.ndarray,
    feature3: np.ndarray,
    feature4: np.ndarray,
    p0_values: np.ndarray,
    pq_values: np.ndarray,
    pm_values: np.ndarray,
) -> np.ndarray:
    """Differentiate the cached lagged-midpoint discrete adjoint recursion."""
    incremental_adjoints = np.zeros_like(states)
    incremental_adjoints[-1, :] = state_loss_gradient_direction[-1, :]
    for step in range(dt_history.shape[0] - 1, -1, -1):
        predictor_tangent, y2_tangent, y3_tangent, y4_tangent = (
            _rk4_half_tangent_stages_from_cached_jacobians_numba(
                delta_a, delta_h, delta_b, delta_c,
                states[step], tangent_states[step], dt_history[step],
                p0_values[step], pq_values[step], pm_values[step],
                stage2[step], stage3[step], stage4[step],
                jacobian1[step], jacobian2[step], jacobian3[step], jacobian4[step],
                feature1[step], feature2[step], feature3[step], feature4[step],
            )
        )
        state_bar_tangent = _lagged_midpoint_reverse_state_direction_cached_numba(
            h, delta_a, delta_h,
            states[step], states[step + 1],
            tangent_states[step], tangent_states[step + 1],
            dt_history[step], predictors[step], predictor_tangent,
            stage2[step], stage3[step], stage4[step],
            y2_tangent, y3_tangent, y4_tangent,
            linear_operators[step], system_matrices[step],
            base_adjoints[step + 1], incremental_adjoints[step + 1],
            jacobian1[step], jacobian2[step], jacobian3[step], jacobian4[step],
        )
        incremental_adjoints[step, :] = state_loss_gradient_direction[step, :] + state_bar_tangent
    return incremental_adjoints


@njit(cache=True)
def accumulate_lagged_midpoint_parameter_gradients_presampled_kernel(a: np.ndarray, h: np.ndarray, b: np.ndarray, c: np.ndarray, states: np.ndarray, dt_history: np.ndarray, adjoints: np.ndarray, p0_values: np.ndarray, pq_values: np.ndarray, pm_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = states.shape[1]
    a_grad = np.zeros((r, r), dtype=np.float64)
    h_grad = np.zeros_like(h)
    b_grad = np.zeros_like(b)
    c_grad = np.zeros(r, dtype=np.float64)
    for step in range(dt_history.shape[0]):
        _, da, dh, db, dc = _lagged_midpoint_reverse_step_numba(
            a, h, b, c, states[step], states[step + 1], dt_history[step], p0_values[step], pq_values[step], pm_values[step], adjoints[step + 1], True
        )
        a_grad += da
        h_grad += dh
        b_grad += db
        c_grad += dc
    return a_grad, h_grad, b_grad, c_grad
