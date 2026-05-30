from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension, quadratic_features  # noqa: E402
from goattm.losses.qoi_loss import rollout_qoi_loss_and_gradients_from_observations  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.quadratic_dynamics import QuadraticDynamics  # noqa: E402
from goattm.problems.reduced_qoi_best_response import _observation_state_loss_gradient_direction  # noqa: E402
from goattm.solvers import rollout_tangent_from_base_rollout, rollout_to_observation_times  # noqa: E402
from goattm.solvers.skew_lagged_midpoint import (  # noqa: E402
    accumulate_skew_lagged_midpoint_parameter_hessian_action_terms,
    compute_skew_lagged_midpoint_incremental_discrete_adjoint,
)


def input_function(t: float) -> np.ndarray:
    return np.array([np.sin(2.0 * np.pi * t), np.cos(1.3 * np.pi * t), 0.25 + 0.1 * t], dtype=np.float64)


def trajectory_taylor_test() -> None:
    print("=== trajectory-level Taylor test for skew_lagged_midpoint adjoint ===")
    rng = np.random.default_rng(20260528)
    r, dq, dp = 5, 4, 3
    skew_seed = rng.standard_normal((r, r))
    a = -0.12 * np.eye(r) + 0.02 * (skew_seed - skew_seed.T)
    mu_h = 0.025 * rng.standard_normal(mu_h_dimension(r))
    b = 0.035 * rng.standard_normal((r, dp))
    c = 0.01 * rng.standard_normal(r)
    u0 = 0.12 * rng.standard_normal(r)

    decoder = QuadraticDecoder(
        v1=0.2 * rng.standard_normal((dq, r)),
        v2=0.03 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
        v0=0.02 * rng.standard_normal(dq),
    )
    observation_times = np.linspace(0.0, 0.3, 31)
    base_dynamics = QuadraticDynamics(a=a, mu_h=mu_h, c=c, b=b)
    base_rollout, observation_indices = rollout_to_observation_times(
        base_dynamics,
        u0,
        observation_times,
        max_dt=1.0e-2,
        input_function=input_function,
        time_integrator="skew_lagged_midpoint",
    )
    if not base_rollout.success:
        raise RuntimeError("Base skew-lagged midpoint rollout failed.")

    clean_observations = np.vstack([decoder.decode(state) for state in base_rollout.states[observation_indices]])
    qoi_observations = clean_observations + 0.01 * np.std(clean_observations) * rng.standard_normal(
        clean_observations.shape
    )

    da = rng.standard_normal(a.shape)
    da /= np.linalg.norm(da)
    dmu_h = rng.standard_normal(mu_h.shape)
    dmu_h /= np.linalg.norm(dmu_h)
    db = rng.standard_normal(b.shape)
    db /= np.linalg.norm(db)
    dc = rng.standard_normal(c.shape)
    dc /= np.linalg.norm(dc)

    def make_dynamics(eps: float = 0.0) -> QuadraticDynamics:
        return QuadraticDynamics(a=a + eps * da, mu_h=mu_h + eps * dmu_h, b=b + eps * db, c=c + eps * dc)

    def evaluate(dynamics: QuadraticDynamics) -> tuple[float, dict[str, np.ndarray]]:
        result = rollout_qoi_loss_and_gradients_from_observations(
            dynamics=dynamics,
            decoder=decoder,
            u0=u0,
            observation_times=observation_times,
            max_dt=1.0e-2,
            qoi_observations=qoi_observations,
            input_function=input_function,
            time_integrator="skew_lagged_midpoint",
        )
        return result.loss, result.dynamics_gradients

    loss0, gradients = evaluate(base_dynamics)
    predicted = (
        np.sum(gradients["a"] * da)
        + np.sum(gradients["mu_h"] * dmu_h)
        + np.sum(gradients["b"] * db)
        + np.sum(gradients["c"] * dc)
    )
    print(f"base_loss {loss0:.16e}")
    print(f"adjoint_directional_derivative {predicted:.16e}")

    max_relative_error = 0.0
    for eps in [1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5]:
        loss_plus, _ = evaluate(make_dynamics(eps))
        loss_minus, _ = evaluate(make_dynamics(-eps))
        finite_difference = (loss_plus - loss_minus) / (2.0 * eps)
        absolute_error = abs(finite_difference - predicted)
        relative_error = absolute_error / max(1.0, abs(finite_difference), abs(predicted))
        max_relative_error = max(max_relative_error, relative_error)
        print(
            f"eps {eps:.1e} fd {finite_difference:.16e} "
            f"abs_err {absolute_error:.3e} rel_err {relative_error:.3e}"
        )

    if max_relative_error > 1.0e-8:
        raise AssertionError(f"Taylor test failed: max relative error {max_relative_error:.3e}")


def compare_against_rk4() -> None:
    print("\n=== 100-trajectory comparison: skew dt=1e-2 vs rk4 dt=1e-3 ===")
    rng = np.random.default_rng(20260529)
    r, dp = 8, 3
    skew_seed = rng.standard_normal((r, r))
    dynamics = QuadraticDynamics(
        a=-0.18 * np.eye(r) + 0.03 * (skew_seed - skew_seed.T),
        mu_h=0.015 * rng.standard_normal(mu_h_dimension(r)),
        b=0.025 * rng.standard_normal((r, dp)),
        c=0.005 * rng.standard_normal(r),
    )
    observation_times = np.linspace(0.0, 1.0, 101)
    rms_absolute_errors: list[float] = []
    trajectory_relative_errors: list[float] = []
    max_timepoint_relative_errors: list[float] = []

    for sample_index in range(100):
        u0 = 0.18 * rng.standard_normal(r)
        skew_rollout, skew_indices = rollout_to_observation_times(
            dynamics,
            u0,
            observation_times,
            max_dt=1.0e-2,
            input_function=input_function,
            time_integrator="skew_lagged_midpoint",
        )
        rk4_rollout, rk4_indices = rollout_to_observation_times(
            dynamics,
            u0,
            observation_times,
            max_dt=1.0e-3,
            input_function=input_function,
            time_integrator="rk4",
        )
        if not skew_rollout.success or not rk4_rollout.success:
            raise RuntimeError(
                f"Rollout failed for sample {sample_index}: "
                f"skew={skew_rollout.success}, rk4={rk4_rollout.success}"
            )

        skew_states = skew_rollout.states[skew_indices]
        rk4_states = rk4_rollout.states[rk4_indices]
        difference = skew_states - rk4_states
        rms_absolute_errors.append(float(np.linalg.norm(difference) / np.sqrt(difference.size)))
        trajectory_relative_errors.append(float(np.linalg.norm(difference) / max(1.0e-14, np.linalg.norm(rk4_states))))
        pointwise_relative_error = np.linalg.norm(difference, axis=1) / np.maximum(
            1.0e-14, np.linalg.norm(rk4_states, axis=1)
        )
        max_timepoint_relative_errors.append(float(np.max(pointwise_relative_error)))

    rms_absolute_errors_array = np.asarray(rms_absolute_errors)
    trajectory_relative_errors_array = np.asarray(trajectory_relative_errors)
    max_timepoint_relative_errors_array = np.asarray(max_timepoint_relative_errors)
    print("successful_pairs 100")
    print(
        "rms_abs_error "
        f"mean {rms_absolute_errors_array.mean():.6e} "
        f"median {np.median(rms_absolute_errors_array):.6e} "
        f"max {rms_absolute_errors_array.max():.6e}"
    )
    print(
        "trajectory_rel_error "
        f"mean {trajectory_relative_errors_array.mean():.6e} "
        f"median {np.median(trajectory_relative_errors_array):.6e} "
        f"max {trajectory_relative_errors_array.max():.6e}"
    )
    print(
        "max_timepoint_rel_error "
        f"mean {max_timepoint_relative_errors_array.mean():.6e} "
        f"median {np.median(max_timepoint_relative_errors_array):.6e} "
        f"max {max_timepoint_relative_errors_array.max():.6e}"
    )
    print("first_five_rel_errors", [f"{value:.6e}" for value in trajectory_relative_errors_array[:5]])

    if trajectory_relative_errors_array.max() > 5.0e-5:
        raise AssertionError(f"RK4 comparison failed: max relative error {trajectory_relative_errors_array.max():.3e}")


def hessian_action_taylor_test() -> None:
    print("\n=== Hessian-action Taylor test for skew_lagged_midpoint ===")
    rng = np.random.default_rng(20260530)
    r, dq, dp = 5, 4, 3
    skew_seed = rng.standard_normal((r, r))
    a = -0.10 * np.eye(r) + 0.018 * (skew_seed - skew_seed.T)
    mu_h = 0.020 * rng.standard_normal(mu_h_dimension(r))
    b = 0.025 * rng.standard_normal((r, dp))
    c = 0.008 * rng.standard_normal(r)
    u0 = 0.10 * rng.standard_normal(r)
    decoder = QuadraticDecoder(
        v1=0.20 * rng.standard_normal((dq, r)),
        v2=0.02 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
        v0=0.01 * rng.standard_normal(dq),
    )
    observation_times = np.linspace(0.0, 0.25, 26)

    def make_dynamics(eps: float = 0.0) -> QuadraticDynamics:
        return QuadraticDynamics(
            a=a + eps * da,
            mu_h=mu_h + eps * dmu_h,
            b=b + eps * db,
            c=c + eps * dc,
        )

    base_dynamics = QuadraticDynamics(a=a, mu_h=mu_h, b=b, c=c)
    base_result = rollout_qoi_loss_and_gradients_from_observations(
        base_dynamics,
        decoder,
        u0,
        observation_times,
        1.0e-2,
        np.zeros((observation_times.size, dq)),
        input_function=input_function,
        time_integrator="skew_lagged_midpoint",
    )
    clean_observations = np.vstack(
        [decoder.decode(state) for state in base_result.rollout.states[base_result.observation_indices]]
    )
    qoi_observations = clean_observations + 0.02 * np.std(clean_observations) * rng.standard_normal(
        clean_observations.shape
    )

    da = rng.standard_normal(a.shape)
    da /= np.linalg.norm(da)
    dmu_h = rng.standard_normal(mu_h.shape)
    dmu_h /= np.linalg.norm(dmu_h)
    db = rng.standard_normal(b.shape)
    db /= np.linalg.norm(db)
    dc = rng.standard_normal(c.shape)
    dc /= np.linalg.norm(dc)
    delta_h = QuadraticDynamics(a=a, mu_h=dmu_h, b=b, c=c).h_matrix - QuadraticDynamics(
        a=a,
        mu_h=np.zeros_like(mu_h),
        b=b,
        c=c,
    ).h_matrix

    def parameter_action(state: np.ndarray, time: float) -> np.ndarray:
        return da @ state + delta_h @ quadratic_features(state) + db @ input_function(time) + dc

    def evaluate_gradient(dynamics: QuadraticDynamics) -> tuple[np.ndarray, object]:
        result = rollout_qoi_loss_and_gradients_from_observations(
            dynamics,
            decoder,
            u0,
            observation_times,
            1.0e-2,
            qoi_observations,
            input_function=input_function,
            time_integrator="skew_lagged_midpoint",
        )
        gradients = result.dynamics_gradients
        return (
            np.concatenate(
                [
                    gradients["a"].ravel(),
                    gradients["mu_h"].ravel(),
                    gradients["b"].ravel(),
                    gradients["c"].ravel(),
                ]
            ),
            result,
        )

    _, base_result = evaluate_gradient(base_dynamics)
    tangent_states = rollout_tangent_from_base_rollout(
        dynamics=base_dynamics,
        base_rollout=base_result.rollout,
        parameter_action=parameter_action,
        input_function=input_function,
        time_integrator="skew_lagged_midpoint",
    )
    zero_decoder_direction = QuadraticDecoder(
        np.zeros_like(decoder.v1),
        np.zeros_like(decoder.v2),
        np.zeros_like(decoder.v0),
        form=decoder.form,
    )
    state_loss_gradient_direction = np.zeros_like(base_result.rollout.states)
    for local_idx, global_idx in enumerate(base_result.observation_indices):
        state_loss_gradient_direction[global_idx] = _observation_state_loss_gradient_direction(
            decoder=decoder,
            decoder_direction=zero_decoder_direction,
            state=base_result.rollout.states[global_idx],
            state_tangent=tangent_states[global_idx],
            residual=base_result.decoder_partials.residuals[local_idx],
            weight=float(base_result.decoder_partials.quadrature_weights[local_idx]),
        )
    adjoint_tangents = compute_skew_lagged_midpoint_incremental_discrete_adjoint(
        dynamics=base_dynamics,
        rollout=base_result.rollout,
        tangent_states=tangent_states,
        base_adjoints=base_result.adjoints,
        state_loss_gradient_direction=state_loss_gradient_direction,
        make_perturbed_dynamics=make_dynamics,
        input_function=input_function,
        finite_difference_epsilon=1.0e-6,
    )
    _, delta_a_grad, _, delta_h_grad, _, delta_b_grad, _, delta_c_grad = (
        accumulate_skew_lagged_midpoint_parameter_hessian_action_terms(
            dynamics=base_dynamics,
            rollout=base_result.rollout,
            tangent_states=tangent_states,
            adjoints=base_result.adjoints,
            adjoint_tangents=adjoint_tangents,
            make_perturbed_dynamics=make_dynamics,
            input_function=input_function,
            finite_difference_epsilon=1.0e-6,
        )
    )
    hessian_action = np.concatenate(
        [
            delta_a_grad.ravel(),
            base_dynamics.pullback_h_gradient_to_mu_h(delta_h_grad).ravel(),
            delta_b_grad.ravel(),
            delta_c_grad.ravel(),
        ]
    )
    print(f"hessian_action_norm {np.linalg.norm(hessian_action):.16e}")

    max_relative_error = 0.0
    for eps in [1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5]:
        gradient_plus, _ = evaluate_gradient(make_dynamics(eps))
        gradient_minus, _ = evaluate_gradient(make_dynamics(-eps))
        finite_difference = (gradient_plus - gradient_minus) / (2.0 * eps)
        absolute_error = float(np.linalg.norm(finite_difference - hessian_action))
        relative_error = absolute_error / max(1.0, float(np.linalg.norm(finite_difference)), float(np.linalg.norm(hessian_action)))
        max_relative_error = max(max_relative_error, relative_error)
        print(
            f"eps {eps:.1e} fd_norm {np.linalg.norm(finite_difference):.16e} "
            f"abs_err {absolute_error:.3e} rel_err {relative_error:.3e}"
        )

    if max_relative_error > 2.0e-5:
        raise AssertionError(f"Hessian-action Taylor test failed: max relative error {max_relative_error:.3e}")


def hessian_action_symmetry_test() -> None:
    print("\n=== Hessian-action symmetry test for skew_lagged_midpoint ===")
    rng = np.random.default_rng(20260531)
    r, dq, dp = 5, 4, 3
    skew_seed = rng.standard_normal((r, r))
    a = -0.10 * np.eye(r) + 0.018 * (skew_seed - skew_seed.T)
    mu_h = 0.020 * rng.standard_normal(mu_h_dimension(r))
    b = 0.025 * rng.standard_normal((r, dp))
    c = 0.008 * rng.standard_normal(r)
    u0 = 0.10 * rng.standard_normal(r)
    decoder = QuadraticDecoder(
        v1=0.20 * rng.standard_normal((dq, r)),
        v2=0.02 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
        v0=0.01 * rng.standard_normal(dq),
    )
    observation_times = np.linspace(0.0, 0.25, 26)
    base_dynamics = QuadraticDynamics(a=a, mu_h=mu_h, b=b, c=c)
    base_result0 = rollout_qoi_loss_and_gradients_from_observations(
        base_dynamics,
        decoder,
        u0,
        observation_times,
        1.0e-2,
        np.zeros((observation_times.size, dq)),
        input_function=input_function,
        time_integrator="skew_lagged_midpoint",
    )
    clean_observations = np.vstack(
        [decoder.decode(state) for state in base_result0.rollout.states[base_result0.observation_indices]]
    )
    qoi_observations = clean_observations + 0.02 * np.std(clean_observations) * rng.standard_normal(
        clean_observations.shape
    )
    base_result = rollout_qoi_loss_and_gradients_from_observations(
        base_dynamics,
        decoder,
        u0,
        observation_times,
        1.0e-2,
        qoi_observations,
        input_function=input_function,
        time_integrator="skew_lagged_midpoint",
    )
    zero_decoder_direction = QuadraticDecoder(
        np.zeros_like(decoder.v1),
        np.zeros_like(decoder.v2),
        np.zeros_like(decoder.v0),
        form=decoder.form,
    )

    a_size = a.size
    mu_size = mu_h.size
    b_size = b.size
    c_size = c.size

    def unpack_direction(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        offset = 0
        da = vector[offset : offset + a_size].reshape(a.shape)
        offset += a_size
        dmu_h = vector[offset : offset + mu_size].reshape(mu_h.shape)
        offset += mu_size
        db = vector[offset : offset + b_size].reshape(b.shape)
        offset += b_size
        dc = vector[offset : offset + c_size].reshape(c.shape)
        return da, dmu_h, db, dc

    def hessian_action(vector: np.ndarray) -> np.ndarray:
        da, dmu_h, db, dc = unpack_direction(vector)
        delta_h = QuadraticDynamics(a=a, mu_h=dmu_h, b=b, c=c).h_matrix - QuadraticDynamics(
            a=a,
            mu_h=np.zeros_like(mu_h),
            b=b,
            c=c,
        ).h_matrix

        def parameter_action(state: np.ndarray, time: float) -> np.ndarray:
            return da @ state + delta_h @ quadratic_features(state) + db @ input_function(time) + dc

        def make_perturbed_dynamics(eps: float) -> QuadraticDynamics:
            return QuadraticDynamics(a=a + eps * da, mu_h=mu_h + eps * dmu_h, b=b + eps * db, c=c + eps * dc)

        tangent_states = rollout_tangent_from_base_rollout(
            dynamics=base_dynamics,
            base_rollout=base_result.rollout,
            parameter_action=parameter_action,
            input_function=input_function,
            time_integrator="skew_lagged_midpoint",
        )
        state_loss_gradient_direction = np.zeros_like(base_result.rollout.states)
        for local_idx, global_idx in enumerate(base_result.observation_indices):
            state_loss_gradient_direction[global_idx] = _observation_state_loss_gradient_direction(
                decoder=decoder,
                decoder_direction=zero_decoder_direction,
                state=base_result.rollout.states[global_idx],
                state_tangent=tangent_states[global_idx],
                residual=base_result.decoder_partials.residuals[local_idx],
                weight=float(base_result.decoder_partials.quadrature_weights[local_idx]),
            )
        perturb_eps = 1.0e-6 / max(1.0, float(np.linalg.norm(vector)))
        adjoint_tangents = compute_skew_lagged_midpoint_incremental_discrete_adjoint(
            dynamics=base_dynamics,
            rollout=base_result.rollout,
            tangent_states=tangent_states,
            base_adjoints=base_result.adjoints,
            state_loss_gradient_direction=state_loss_gradient_direction,
            make_perturbed_dynamics=make_perturbed_dynamics,
            input_function=input_function,
            finite_difference_epsilon=perturb_eps,
        )
        _, delta_a_grad, _, delta_h_grad, _, delta_b_grad, _, delta_c_grad = (
            accumulate_skew_lagged_midpoint_parameter_hessian_action_terms(
                dynamics=base_dynamics,
                rollout=base_result.rollout,
                tangent_states=tangent_states,
                adjoints=base_result.adjoints,
                adjoint_tangents=adjoint_tangents,
                make_perturbed_dynamics=make_perturbed_dynamics,
                input_function=input_function,
                finite_difference_epsilon=perturb_eps,
            )
        )
        return np.concatenate(
            [
                delta_a_grad.ravel(),
                base_dynamics.pullback_h_gradient_to_mu_h(delta_h_grad).ravel(),
                delta_b_grad.ravel(),
                delta_c_grad.ravel(),
            ]
        )

    dimension = a_size + mu_size + b_size + c_size
    max_relative_error = 0.0
    for pair_idx in range(8):
        left = rng.standard_normal(dimension)
        left /= np.linalg.norm(left)
        right = rng.standard_normal(dimension)
        right /= np.linalg.norm(right)
        h_left = hessian_action(left)
        h_right = hessian_action(right)
        left_value = float(right @ h_left)
        right_value = float(left @ h_right)
        absolute_error = abs(left_value - right_value)
        relative_error = absolute_error / max(1.0, abs(left_value), abs(right_value))
        max_relative_error = max(max_relative_error, relative_error)
        print(
            f"pair {pair_idx:02d} wTHv {left_value:.16e} vTHw {right_value:.16e} "
            f"abs_err {absolute_error:.3e} rel_err {relative_error:.3e}"
        )

    print(f"max_relative_symmetry_error {max_relative_error:.3e}")
    if max_relative_error > 5.0e-8:
        raise AssertionError(f"Hessian-action symmetry test failed: max relative error {max_relative_error:.3e}")


def main() -> None:
    trajectory_taylor_test()
    compare_against_rk4()
    hessian_action_taylor_test()
    hessian_action_symmetry_test()


if __name__ == "__main__":
    main()
