from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from goattm.core.parametrization import quadratic_features
from goattm.models.linear_dynamics import LinearDynamics
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.runtime import timed
from goattm.solvers.explicit_euler import (
    accumulate_explicit_euler_parameter_gradients,
    compute_explicit_euler_discrete_adjoint,
)
from goattm.solvers.implicit_midpoint import RolloutResult
from goattm.solvers.rk4 import (
    accumulate_rk4_parameter_gradients,
    compute_rk4_discrete_adjoint,
)
from goattm.solvers.time_integration import TimeIntegrator, rollout_to_final_time, rollout_to_observation_times, validate_time_integrator


DynamicsLike = LinearDynamics | QuadraticDynamics | StabilizedQuadraticDynamics


@dataclass(frozen=True)
class DecoderLossPartials:
    loss: float
    qoi_predictions: np.ndarray
    residuals: np.ndarray
    quadrature_weights: np.ndarray
    latent_state_gradients: np.ndarray
    v1_grad: np.ndarray
    v2_grad: np.ndarray
    v0_grad: np.ndarray


@dataclass(frozen=True)
class RolloutLossGradientResult:
    rollout: RolloutResult
    decoder_partials: DecoderLossPartials
    adjoints: np.ndarray
    dynamics_gradients: dict[str, np.ndarray]

    @property
    def loss(self) -> float:
        return self.decoder_partials.loss


@dataclass(frozen=True)
class ObservationAlignedRolloutLossGradientResult(RolloutLossGradientResult):
    observation_indices: np.ndarray


def trapezoidal_rule_weights_from_times(times: np.ndarray) -> np.ndarray:
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError(f"times must have shape (N,), N>=2, got {times.shape}")
    dt_history = np.diff(times)
    if np.any(dt_history <= 0.0):
        raise ValueError("times must be strictly increasing.")

    weights = np.zeros_like(times, dtype=np.float64)
    weights[0] = 0.5 * dt_history[0]
    weights[-1] = 0.5 * dt_history[-1]
    for i in range(1, times.shape[0] - 1):
        weights[i] = 0.5 * (dt_history[i - 1] + dt_history[i])
    return weights


def _validate_qoi_trajectory(states: np.ndarray, decoder: QuadraticDecoder, qoi_observations: np.ndarray) -> None:
    if states.ndim != 2 or states.shape[1] != decoder.latent_dimension:
        raise ValueError(
            f"states must have shape (N, {decoder.latent_dimension}), got {states.shape}"
        )
    if qoi_observations.ndim != 2 or qoi_observations.shape != (states.shape[0], decoder.output_dimension):
        raise ValueError(
            f"qoi_observations must have shape ({states.shape[0]}, {decoder.output_dimension}), "
            f"got {qoi_observations.shape}"
        )


def qoi_trajectory_loss_and_partials(
    states: np.ndarray,
    decoder: QuadraticDecoder,
    qoi_observations: np.ndarray,
    times: np.ndarray,
) -> DecoderLossPartials:
    _validate_qoi_trajectory(states, decoder, qoi_observations)
    weights = trapezoidal_rule_weights_from_times(times)

    qoi_predictions = np.zeros_like(qoi_observations, dtype=np.float64)
    residuals = np.zeros_like(qoi_observations, dtype=np.float64)
    latent_state_gradients = np.zeros_like(states, dtype=np.float64)
    v1_grad = np.zeros_like(decoder.v1, dtype=np.float64)
    v2_grad = np.zeros_like(decoder.v2, dtype=np.float64)
    v0_grad = np.zeros_like(decoder.v0, dtype=np.float64)
    loss = 0.0

    for n in range(states.shape[0]):
        u = states[n]
        q_pred = decoder.decode(u)
        residual = q_pred - qoi_observations[n]
        weighted_residual = weights[n] * residual
        zeta = quadratic_features(u)

        qoi_predictions[n] = q_pred
        residuals[n] = residual
        loss += 0.5 * weights[n] * float(np.dot(residual, residual))
        latent_state_gradients[n] = decoder.jacobian(u).T @ weighted_residual
        v1_grad += np.outer(weighted_residual, u)
        if decoder.form == "V1V2v":
            v2_grad += np.outer(weighted_residual, zeta)
        v0_grad += weighted_residual

    return DecoderLossPartials(
        loss=loss,
        qoi_predictions=qoi_predictions,
        residuals=residuals,
        quadrature_weights=weights,
        latent_state_gradients=latent_state_gradients,
        v1_grad=v1_grad,
        v2_grad=v2_grad,
        v0_grad=v0_grad,
    )


def qoi_trajectory_loss(
    states: np.ndarray,
    decoder: QuadraticDecoder,
    qoi_observations: np.ndarray,
    times: np.ndarray,
) -> float:
    return qoi_trajectory_loss_and_partials(states, decoder, qoi_observations, times).loss


@timed("goattm.losses.compute_midpoint_discrete_adjoint")
def compute_midpoint_discrete_adjoint(
    dynamics: DynamicsLike,
    states: np.ndarray,
    times: np.ndarray,
    dt_history: np.ndarray,
    state_loss_gradients: np.ndarray,
) -> np.ndarray:
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError(f"states must have shape (N, r), N>=2, got {states.shape}")
    if times.ndim != 1 or times.shape[0] != states.shape[0]:
        raise ValueError(f"times must have shape ({states.shape[0]},), got {times.shape}")
    if dt_history.ndim != 1 or dt_history.shape[0] != states.shape[0] - 1:
        raise ValueError(f"dt_history must have shape ({states.shape[0] - 1},), got {dt_history.shape}")
    if state_loss_gradients.shape != states.shape:
        raise ValueError(f"state_loss_gradients must have shape {states.shape}, got {state_loss_gradients.shape}")

    n_steps = dt_history.shape[0]
    r = states.shape[1]
    adjoints = np.zeros((n_steps + 1, r), dtype=np.float64)

    midpoint = 0.5 * (states[-2] + states[-1])
    terminal_matrix = np.eye(r) - 0.5 * dt_history[-1] * dynamics.rhs_jacobian(midpoint).T
    adjoints[-1] = np.linalg.solve(terminal_matrix, -state_loss_gradients[-1])

    for n in range(n_steps - 1, 0, -1):
        midpoint_prev = 0.5 * (states[n - 1] + states[n])
        midpoint_next = 0.5 * (states[n] + states[n + 1])
        lhs = np.eye(r) - 0.5 * dt_history[n - 1] * dynamics.rhs_jacobian(midpoint_prev).T
        rhs = (
            (np.eye(r) + 0.5 * dt_history[n] * dynamics.rhs_jacobian(midpoint_next).T) @ adjoints[n + 1]
            - state_loss_gradients[n]
        )
        adjoints[n] = np.linalg.solve(lhs, rhs)

    return adjoints


def _pullback_dynamics_gradients(
    dynamics: DynamicsLike,
    a_grad: np.ndarray,
    h_grad: np.ndarray,
    b_grad: np.ndarray | None,
    c_grad: np.ndarray,
) -> dict[str, np.ndarray]:
    gradients: dict[str, np.ndarray] = {"c": c_grad}
    if not isinstance(dynamics, LinearDynamics):
        gradients["mu_h"] = dynamics.pullback_h_gradient_to_mu_h(h_grad)
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        s_grad, w_grad = dynamics.pullback_a_gradient_to_stabilized_params(a_grad)
        gradients["s_params"] = s_grad
        gradients["w_params"] = w_grad
    else:
        gradients["a"] = a_grad
    if b_grad is not None:
        gradients["b"] = b_grad
    return gradients


def rollout_qoi_loss_and_gradients(
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    u0: np.ndarray,
    t_final: float,
    dt_initial: float,
    qoi_observations: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    dt_shrink: float = 0.8,
    dt_min: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> RolloutLossGradientResult:
    rollout = rollout_to_final_time(
        dynamics=dynamics,
        u0=u0,
        t_final=t_final,
        max_dt=dt_initial,
        input_function=input_function,
        time_integrator=time_integrator,
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
    return _rollout_result_to_loss_and_gradients(
        dynamics=dynamics,
        decoder=decoder,
        rollout=rollout,
        observation_indices=np.arange(rollout.states.shape[0], dtype=int),
        qoi_observations=qoi_observations,
        observation_times=rollout.times,
        input_function=input_function,
        time_integrator=time_integrator,
    )


def rollout_qoi_loss_and_gradients_from_observations(
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    u0: np.ndarray,
    observation_times: np.ndarray,
    max_dt: float,
    qoi_observations: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> ObservationAlignedRolloutLossGradientResult:
    rollout, observation_indices = rollout_to_observation_times(
        dynamics=dynamics,
        u0=u0,
        observation_times=observation_times,
        max_dt=max_dt,
        input_function=input_function,
        time_integrator=time_integrator,
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
    base = _rollout_result_to_loss_and_gradients(
        dynamics=dynamics,
        decoder=decoder,
        rollout=rollout,
        observation_indices=observation_indices,
        qoi_observations=qoi_observations,
        observation_times=observation_times,
        input_function=input_function,
        time_integrator=time_integrator,
    )
    return ObservationAlignedRolloutLossGradientResult(
        rollout=base.rollout,
        decoder_partials=base.decoder_partials,
        adjoints=base.adjoints,
        dynamics_gradients=base.dynamics_gradients,
        observation_indices=observation_indices,
    )


@timed("goattm.losses.rollout_qoi_loss_and_gradients_from_cached_observation_rollout")
def rollout_qoi_loss_and_gradients_from_cached_observation_rollout(
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    rollout: RolloutResult,
    observation_indices: np.ndarray,
    observation_times: np.ndarray,
    qoi_observations: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None = None,
    time_integrator: TimeIntegrator = "implicit_midpoint",
) -> ObservationAlignedRolloutLossGradientResult:
    base = _rollout_result_to_loss_and_gradients(
        dynamics=dynamics,
        decoder=decoder,
        rollout=rollout,
        observation_indices=observation_indices,
        qoi_observations=qoi_observations,
        observation_times=observation_times,
        input_function=input_function,
        time_integrator=time_integrator,
    )
    return ObservationAlignedRolloutLossGradientResult(
        rollout=base.rollout,
        decoder_partials=base.decoder_partials,
        adjoints=base.adjoints,
        dynamics_gradients=base.dynamics_gradients,
        observation_indices=observation_indices,
    )


def _rollout_result_to_loss_and_gradients(
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    rollout: RolloutResult,
    observation_indices: np.ndarray,
    qoi_observations: np.ndarray,
    observation_times: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None,
    time_integrator: TimeIntegrator,
) -> RolloutLossGradientResult:
    integrator = validate_time_integrator(time_integrator)
    if not rollout.success:
        raise RuntimeError("Forward rollout failed; loss gradients are undefined for an incomplete trajectory.")
    if observation_indices.ndim != 1 or observation_indices.shape[0] != observation_times.shape[0]:
        raise ValueError(
            f"observation_indices must have shape ({observation_times.shape[0]},), got {observation_indices.shape}"
        )

    observed_states = rollout.states[observation_indices]

    decoder_partials = qoi_trajectory_loss_and_partials(
        states=observed_states,
        decoder=decoder,
        qoi_observations=qoi_observations,
        times=observation_times,
    )
    full_state_loss_gradients = np.zeros_like(rollout.states, dtype=np.float64)
    full_state_loss_gradients[observation_indices] = decoder_partials.latent_state_gradients
    if integrator == "implicit_midpoint":
        adjoints = compute_midpoint_discrete_adjoint(
            dynamics=dynamics,
            states=rollout.states,
            times=rollout.times,
            dt_history=rollout.dt_history,
            state_loss_gradients=full_state_loss_gradients,
        )

        a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
        h_grad = np.zeros_like(dynamics.h_matrix, dtype=np.float64)
        c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
        b_grad = None if getattr(dynamics, "b", None) is None else np.zeros_like(dynamics.b, dtype=np.float64)

        for n in range(rollout.accepted_steps):
            dt = rollout.dt_history[n]
            midpoint_state = 0.5 * (rollout.states[n] + rollout.states[n + 1])
            midpoint_time = rollout.times[n] + 0.5 * dt
            lam = adjoints[n + 1]
            zeta = quadratic_features(midpoint_state)

            a_grad += -dt * np.outer(lam, midpoint_state)
            h_grad += -dt * np.outer(lam, zeta)
            c_grad += -dt * lam

            if b_grad is not None and input_function is not None:
                p_mid = np.asarray(input_function(midpoint_time), dtype=np.float64)
                b_grad += -dt * np.outer(lam, p_mid)
    elif integrator == "explicit_euler":
        adjoints = compute_explicit_euler_discrete_adjoint(
            dynamics=dynamics,
            states=rollout.states,
            times=rollout.times,
            dt_history=rollout.dt_history,
            state_loss_gradients=full_state_loss_gradients,
        )
        a_grad, h_grad, b_grad, c_grad = accumulate_explicit_euler_parameter_gradients(
            dynamics=dynamics,
            rollout=rollout,
            adjoints=adjoints,
            input_function=input_function,
        )
    else:
        adjoints = compute_rk4_discrete_adjoint(
            dynamics=dynamics,
            states=rollout.states,
            times=rollout.times,
            dt_history=rollout.dt_history,
            state_loss_gradients=full_state_loss_gradients,
            input_function=input_function,
        )
        a_grad, h_grad, b_grad, c_grad = accumulate_rk4_parameter_gradients(
            dynamics=dynamics,
            rollout=rollout,
            adjoints=adjoints,
            input_function=input_function,
        )

    dynamics_gradients = _pullback_dynamics_gradients(dynamics, a_grad, h_grad, b_grad, c_grad)

    return RolloutLossGradientResult(
        rollout=rollout,
        decoder_partials=decoder_partials,
        adjoints=adjoints,
        dynamics_gradients=dynamics_gradients,
    )
