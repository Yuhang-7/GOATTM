from __future__ import annotations

import json

import torch

from quadrode_gpu_goattm.decoders import QuadraticReadoutDecoder
from quadrode_gpu_goattm.dynamics import QuadraticDynamics
from quadrode_gpu_goattm.incremental import (
    add_scaled_parameters,
    decoder_feature_tangent,
    lagged_midpoint_rollout_incremental,
    random_direction_like,
    variable_projection_decoder_tangent,
)
from quadrode_gpu_goattm.linear import DissipativeSkewA
from quadrode_gpu_goattm.quadratic import EnergyTuckerTTQuadratic
from quadrode_gpu_goattm.source import LinearSource
from quadrode_gpu_goattm.steppers import DenseLaggedMidpointStepper
from quadrode_gpu_goattm.varpro import assemble_decoder_normal_terms, solve_decoder_normal_terms
from quadrode_gpu_goattm.decoders import decoder_loss_and_state_grad
from quadrode_gpu_goattm.incremental import exact_decoder_state_gradient_tangent, gauss_newton_state_cotangents


def relnorm(x: torch.Tensor, denom: torch.Tensor) -> float:
    return float((torch.linalg.norm(x) / (1.0 + torch.linalg.norm(denom))).detach().cpu())


def main() -> None:
    torch.manual_seed(20260705)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    batch = 40
    steps = 8
    r = 8
    inp = 4
    out = 3
    h = 0.04
    dynamics = QuadraticDynamics(
        DissipativeSkewA(r, skew_rank=3, damping_init=0.15, factor_scale=0.05, damping_shift=0.1),
        EnergyTuckerTTQuadratic(r, reduced_rank=5, tt_rank=4, scale=0.04, basis_trainable=True),
        LinearSource(r, inp, init_scale=0.05),
    ).to(device=device, dtype=dtype)
    decoder = QuadraticReadoutDecoder(r, out, include_quadratic=True, bias=True).to(device=device, dtype=dtype)
    stepper = DenseLaggedMidpointStepper(picard_iters=2)
    u0 = 0.1 * torch.randn(batch, r, device=device, dtype=dtype)
    p_mid = 0.1 * torch.randn(steps, batch, inp, device=device, dtype=dtype)
    qoi = 0.1 * torch.randn(steps + 1, batch, out, device=device, dtype=dtype)
    weights = torch.linspace(0.5, 1.5, steps + 1, device=device, dtype=dtype)[:, None].expand(steps + 1, batch)
    direction = random_direction_like(dynamics, seed=7, scale=0.05)

    with torch.no_grad():
        rollout = stepper.rollout_with_picard_history(dynamics, u0, h, steps, p_mid=p_mid)
        inc = lagged_midpoint_rollout_incremental(
            dynamics,
            u0,
            None,
            h,
            p_mid,
            direction,
            rollout=rollout,
        )
        normal = solve_decoder_normal_terms(
            decoder,
            assemble_decoder_normal_terms(decoder, rollout.states, qoi, weights=weights, chunk_size=64),
            ridge=1.0e-4,
        )
        d_features = decoder_feature_tangent(decoder, rollout.states, inc.states_dot, chunk_size=64)
        _, dcoeff, dnormal = variable_projection_decoder_tangent(
            decoder,
            normal,
            rollout.states,
            inc.states_dot,
            qoi,
            weights=weights,
            ridge=1.0e-4,
            chunk_size=64,
        )
        _, base_data_loss, base_state_grad = decoder_loss_and_state_grad(
            decoder,
            rollout.states,
            qoi,
            weights=weights,
            chunk_size=64,
            return_prediction=False,
            return_state_grad=True,
        )
        gn_state_cot, _ = gauss_newton_state_cotangents(
            decoder,
            normal,
            rollout.states,
            inc.states_dot,
            qoi,
            weights=weights,
            ridge=1.0e-4,
            chunk_size=64,
        )
        exact_state_cot_dot, _ = exact_decoder_state_gradient_tangent(
            decoder,
            normal,
            rollout.states,
            inc.states_dot,
            qoi,
            weights=weights,
            ridge=1.0e-4,
            chunk_size=64,
        )

    eps_values = torch.logspace(-7, -2, 6, device=device, dtype=dtype)
    state_errors = []
    feature_errors = []
    normal_errors = []
    coeff_errors = []
    state_grad_errors = []
    exact_state_grad_errors = []
    base_states = rollout.states.detach().clone()
    base_features = decoder.features(base_states.reshape(-1, r)).detach().clone()
    base_normal_matrix = normal.normal_matrix.detach().clone()
    base_coeff = normal.coefficients.detach().clone()
    for eps in eps_values:
        eps_float = float(eps.detach().cpu())
        add_scaled_parameters(dynamics, direction, eps_float)
        with torch.no_grad():
            pert_rollout = stepper.rollout_with_picard_history(dynamics, u0, h, steps, p_mid=p_mid)
            pert_features = decoder.features(pert_rollout.states.reshape(-1, r)).detach()
            pert_normal = solve_decoder_normal_terms(
                decoder,
                assemble_decoder_normal_terms(decoder, pert_rollout.states, qoi, weights=weights, chunk_size=64),
                ridge=1.0e-4,
            )
        add_scaled_parameters(dynamics, direction, -eps_float)
        state_res = pert_rollout.states - base_states - eps * inc.states_dot
        feature_res = pert_features - base_features - eps * d_features
        coeff_res = pert_normal.coefficients - base_coeff - eps * dcoeff
        normal_res = pert_normal.normal_matrix - base_normal_matrix - eps * dnormal
        _, _, pert_state_grad = decoder_loss_and_state_grad(
            decoder,
            pert_rollout.states,
            qoi,
            weights=weights,
            chunk_size=64,
            return_prediction=False,
            return_state_grad=True,
        )
        state_grad_res = pert_state_grad - base_state_grad - eps * gn_state_cot
        exact_state_grad_res = pert_state_grad - base_state_grad - eps * exact_state_cot_dot
        state_errors.append({"eps": eps_float, "error_over_eps": relnorm(state_res, eps * inc.states_dot) / eps_float})
        feature_errors.append({"eps": eps_float, "error_over_eps": relnorm(feature_res, eps * d_features) / eps_float})
        coeff_errors.append({"eps": eps_float, "error_over_eps": relnorm(coeff_res, eps * dcoeff) / eps_float})
        normal_errors.append({"eps": eps_float, "error_over_eps": relnorm(normal_res, eps * dnormal) / eps_float})
        state_grad_errors.append({"eps": eps_float, "error_over_eps": relnorm(state_grad_res, eps * gn_state_cot) / eps_float})
        exact_state_grad_errors.append({"eps": eps_float, "error_over_eps": relnorm(exact_state_grad_res, eps * exact_state_cot_dot) / eps_float})

    print(
        json.dumps(
            {
                "device": str(device),
                "state_taylor": state_errors,
                "feature_taylor": feature_errors,
                "normal_matrix_taylor": normal_errors,
                "decoder_coeff_taylor": coeff_errors,
                "state_gradient_taylor": state_grad_errors,
                "exact_state_gradient_taylor": exact_state_grad_errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
