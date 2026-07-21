from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from quadrode_gpu_goattm.data import ContinuousBatch
from quadrode_gpu_goattm.decoders import QuadraticReadoutDecoder
from quadrode_gpu_goattm.dynamics import QuadraticDynamics
from quadrode_gpu_goattm.incremental import (
    add_scaled_parameters,
    lagged_midpoint_rollout_incremental,
    projected_decoder_residual_tangent,
    random_direction_like,
)
from quadrode_gpu_goattm.linear import DissipativeSkewA
from quadrode_gpu_goattm.quadratic import EnergyTuckerTTQuadratic
from quadrode_gpu_goattm.reduced import trapezoidal_weights
from quadrode_gpu_goattm.source import LinearSource
from quadrode_gpu_goattm.steppers import DenseLaggedMidpointStepper
from quadrode_gpu_goattm.varpro import assemble_decoder_normal_terms, solve_decoder_normal_terms


def weighted_norm(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x.square().sum(dim=-1) * weights).sum())


def rel_error(numer: torch.Tensor, denom: torch.Tensor) -> float:
    return float((torch.linalg.norm(numer) / (1.0 + torch.linalg.norm(denom))).detach().cpu())


def main() -> None:
    torch.manual_seed(20260705)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    batch_size = 72
    steps = 12
    latent_dim = 10
    input_dim = 5
    output_dim = 4
    step_size = 0.025
    decoder_ridge = 1.0e-5

    dynamics = QuadraticDynamics(
        DissipativeSkewA(latent_dim, skew_rank=4, damping_init=0.12, factor_scale=0.04, damping_shift=0.08),
        EnergyTuckerTTQuadratic(latent_dim, reduced_rank=6, tt_rank=5, scale=0.03, basis_trainable=True),
        LinearSource(latent_dim, input_dim, init_scale=0.04),
    ).to(device=device, dtype=dtype)
    decoder = QuadraticReadoutDecoder(latent_dim, output_dim, include_quadratic=True, bias=True).to(device=device, dtype=dtype)
    stepper = DenseLaggedMidpointStepper(picard_iters=2)

    times = step_size * torch.arange(steps + 1, device=device, dtype=dtype)
    input_times = 0.5 * (times[:-1] + times[1:])
    p_mid = 0.1 * torch.randn(steps, batch_size, input_dim, device=device, dtype=dtype)
    u0 = torch.zeros(batch_size, latent_dim, device=device, dtype=dtype)

    with torch.no_grad():
        base_rollout = stepper.rollout_with_picard_history(dynamics, u0, step_size, steps, p_mid=p_mid)
        # Use a non-interpolatory target so the residual-weighted exact-Hessian
        # terms are visible but the ridged normal equation remains well posed.
        decoder.readout.weight.normal_(0.0, 0.05)
        decoder.readout.bias.zero_()
        qoi = decoder(base_rollout.states).detach() + 0.03 * torch.randn(
            steps + 1,
            batch_size,
            output_dim,
            device=device,
            dtype=dtype,
        )

    batch = ContinuousBatch(
        sample_ids=tuple(str(i) for i in range(batch_size)),
        observation_times=times,
        qoi=qoi,
        u0=None,
        input_times=input_times,
        input_values=p_mid,
    )
    weights = trapezoidal_weights(batch.observation_times)[:, None].expand(batch.qoi.shape[:-1])
    normal = solve_decoder_normal_terms(
        decoder,
        assemble_decoder_normal_terms(decoder, base_rollout.states, batch.qoi, weights=weights, chunk_size=256),
        ridge=decoder_ridge,
    )
    base_residual = (decoder(base_rollout.states) - batch.qoi).detach()

    eps_values = torch.logspace(-9, -2, 15, device=device, dtype=dtype)
    directions = [random_direction_like(dynamics, seed=500 + i, scale=1.0) for i in range(4)]
    summaries = []

    for i, direction in enumerate(directions):
        inc = lagged_midpoint_rollout_incremental(
            dynamics,
            u0,
            None,
            step_size,
            p_mid,
            direction,
            rollout=base_rollout,
        )
        residual_dot, coeff_dot, qform = projected_decoder_residual_tangent(
            decoder,
            normal,
            base_rollout.states,
            inc.states_dot,
            batch.qoi,
            weights=weights,
            ridge=decoder_ridge,
            chunk_size=256,
        )
        data_qform = weighted_norm(residual_dot, weights).square()
        ridge_qform = decoder_ridge * coeff_dot.square().sum()

        residual_curve = []
        coeff_curve = []
        exact_second_curve = []
        base_value = 0.5 * (base_residual.square().sum(dim=-1) * weights).sum() + 0.5 * decoder_ridge * normal.coefficients.square().sum()
        base_directional = ((base_residual * residual_dot).sum(dim=-1) * weights).sum() + decoder_ridge * (
            normal.coefficients * coeff_dot
        ).sum()

        for eps in eps_values:
            eps_float = float(eps.detach().cpu())
            add_scaled_parameters(dynamics, direction, eps_float)
            with torch.no_grad():
                pert_rollout = stepper.rollout_with_picard_history(dynamics, u0, step_size, steps, p_mid=p_mid)
                pert_normal = solve_decoder_normal_terms(
                    decoder,
                    assemble_decoder_normal_terms(decoder, pert_rollout.states, batch.qoi, weights=weights, chunk_size=256),
                    ridge=decoder_ridge,
                )
                pert_residual = decoder(pert_rollout.states) - batch.qoi
                pert_value = 0.5 * (pert_residual.square().sum(dim=-1) * weights).sum() + 0.5 * decoder_ridge * pert_normal.coefficients.square().sum()
            add_scaled_parameters(dynamics, direction, -eps_float)

            residual_fd = (pert_residual - base_residual) / eps
            coeff_fd = (pert_normal.coefficients - normal.coefficients) / eps
            residual_curve.append(
                {
                    "eps": eps_float,
                    "error_over_eps": float((weighted_norm(residual_fd - residual_dot, weights) / (1.0 + weighted_norm(residual_dot, weights)) / eps).detach().cpu()),
                    "absolute_error": float(weighted_norm(residual_fd - residual_dot, weights).detach().cpu()),
                }
            )
            coeff_curve.append(
                {
                    "eps": eps_float,
                    "error_over_eps": rel_error(coeff_fd - coeff_dot, coeff_dot) / eps_float,
                    "absolute_error": float(torch.linalg.norm(coeff_fd - coeff_dot).detach().cpu()),
                }
            )
            exact_second = 2.0 * (pert_value - base_value - eps * base_directional) / (eps * eps)
            exact_second_curve.append({"eps": eps_float, "value": float(exact_second.detach().cpu())})

        summaries.append(
            {
                "direction": i,
                "schur_gn_qform": float(qform.detach().cpu()),
                "data_qform": float(data_qform.detach().cpu()),
                "decoder_ridge_qform": float(ridge_qform.detach().cpu()),
                "residual_tangent": residual_curve,
                "coefficient_tangent": coeff_curve,
                "exact_reduced_second_difference": exact_second_curve,
            }
        )

    outdir = Path("outputs/incremental_hvp_validation")
    outdir.mkdir(parents=True, exist_ok=True)
    eps_cpu = [float(x.detach().cpu()) for x in eps_values]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for row in summaries:
        axes[0].loglog(eps_cpu, [v["error_over_eps"] for v in row["residual_tangent"]], linewidth=1.2)
        axes[1].loglog(eps_cpu, [v["error_over_eps"] for v in row["coefficient_tangent"]], linewidth=1.2)
        axes[2].semilogx(eps_cpu, [v["value"] for v in row["exact_reduced_second_difference"]], linewidth=1.2)
    axes[0].set_xlabel("eps")
    axes[0].set_ylabel("residual tangent error / eps")
    axes[0].set_title("Projected residual action")
    axes[1].set_xlabel("eps")
    axes[1].set_ylabel("decoder coefficient tangent error / eps")
    axes[1].set_title("Normal-equation action")
    axes[2].set_xlabel("eps")
    axes[2].set_ylabel("exact reduced second diff.")
    axes[2].set_title("Exact curvature diagnostic")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = outdir / "varpro_schur_components.png"
    fig.savefig(fig_path, dpi=180)

    summary = {
        "device": str(device),
        "decoder_ridge": decoder_ridge,
        "base_decoder_normal_relative_residual": normal.relative_residual,
        "directions": summaries,
        "figure": str(fig_path),
    }
    summary_path = outdir / "varpro_schur_components.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
