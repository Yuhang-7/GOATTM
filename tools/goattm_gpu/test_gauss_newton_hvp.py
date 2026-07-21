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
    flatten_direction,
    lagged_midpoint_gauss_newton_hvp,
    lagged_midpoint_schur_decoder_hvp,
    lagged_midpoint_rollout_incremental,
    projected_decoder_residual_tangent,
    random_direction_like,
)
from quadrode_gpu_goattm.linear import DissipativeSkewA
from quadrode_gpu_goattm.quadratic import EnergyTuckerTTQuadratic
from quadrode_gpu_goattm.reduced import ReducedObjective, trapezoidal_weights
from quadrode_gpu_goattm.source import LinearSource
from quadrode_gpu_goattm.steppers import DenseLaggedMidpointStepper
from quadrode_gpu_goattm.varpro import assemble_decoder_normal_terms, solve_decoder_normal_terms


def clone_direction(direction):
    return {k: v.detach().clone() for k, v in direction.items()}


def dot_direction(a, b, dynamics) -> torch.Tensor:
    total = None
    for name, param in dynamics.named_parameters():
        if not param.requires_grad:
            continue
        av = a.get(name, torch.zeros_like(param))
        bv = b.get(name, torch.zeros_like(param))
        piece = (av * bv).sum()
        total = piece if total is None else total + piece
    return next(dynamics.parameters()).new_zeros(()) if total is None else total


def grad_direction(result):
    return {k: v.detach().clone() for k, v in result.parameter_grads.items()}


def direction_norm(direction, dynamics) -> float:
    return float(torch.sqrt(dot_direction(direction, direction, dynamics)).detach().cpu())


def compute_base(dynamics, decoder, batch, objective):
    p_mid = batch.midpoint_inputs()
    u0 = batch.qoi.new_zeros(batch.batch_size, dynamics.latent_dim)
    rollout = objective.stepper.rollout_with_picard_history(
        dynamics,
        u0,
        batch.step_size,
        batch.steps,
        p_mid=p_mid,
    )
    weights = trapezoidal_weights(batch.observation_times)[:, None].expand(batch.qoi.shape[:-1])
    normal = solve_decoder_normal_terms(
        decoder,
        assemble_decoder_normal_terms(decoder, rollout.states, batch.qoi, weights=weights, chunk_size=256),
        ridge=objective.decoder_ridge,
    )
    value_grad = objective.value_and_grad(batch)
    return rollout, weights, normal, value_grad


def hvp(dynamics, decoder, batch, objective, rollout, weights, normal, direction):
    hv = lagged_midpoint_schur_decoder_hvp(
        dynamics,
        decoder,
        normal,
        batch,
        direction,
        rollout=rollout,
        weights=weights,
        picard_iters=objective.stepper.picard_iters,
        decoder_ridge=objective.decoder_ridge,
        linear_ridge=objective.linear_ridge,
        quadratic_ridge=objective.quadratic_ridge,
        source_ridge=objective.source_ridge,
        chunk_size=256,
    )
    return hv, None


def projected_data_qform(dynamics, decoder, batch, objective, rollout, weights, normal, direction):
    p_mid = batch.midpoint_inputs()
    u0 = batch.qoi.new_zeros(batch.batch_size, dynamics.latent_dim)
    inc = lagged_midpoint_rollout_incremental(
        dynamics,
        u0,
        None,
        batch.step_size,
        p_mid,
        direction,
        rollout=rollout,
    )
    _, _, qform = projected_decoder_residual_tangent(
        decoder,
        normal,
        rollout.states,
        inc.states_dot,
        batch.qoi,
        weights=weights,
        ridge=objective.decoder_ridge,
        chunk_size=256,
    )
    return qform


def main() -> None:
    torch.manual_seed(20260705)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    batch_size = 64
    steps = 10
    r = 10
    input_dim = 5
    output_dim = 4
    h = 0.03
    dynamics = QuadraticDynamics(
        DissipativeSkewA(r, skew_rank=4, damping_init=0.12, factor_scale=0.04, damping_shift=0.08),
        EnergyTuckerTTQuadratic(r, reduced_rank=6, tt_rank=5, scale=0.03, basis_trainable=True),
        LinearSource(r, input_dim, init_scale=0.04),
    ).to(device=device, dtype=dtype)
    decoder = QuadraticReadoutDecoder(r, output_dim, include_quadratic=True, bias=True).to(device=device, dtype=dtype)
    stepper = DenseLaggedMidpointStepper(picard_iters=2)
    times = h * torch.arange(steps + 1, device=device, dtype=dtype)
    input_times = 0.5 * (times[:-1] + times[1:])
    p_mid = 0.1 * torch.randn(steps, batch_size, input_dim, device=device, dtype=dtype)
    u0 = torch.zeros(batch_size, r, device=device, dtype=dtype)
    with torch.no_grad():
        rollout0 = stepper.rollout_with_picard_history(dynamics, u0, h, steps, p_mid=p_mid)
        decoder.readout.weight.normal_(0.0, 0.1)
        decoder.readout.bias.zero_()
        qoi = decoder(rollout0.states).detach()
    batch = ContinuousBatch(
        sample_ids=tuple(str(i) for i in range(batch_size)),
        observation_times=times,
        qoi=qoi,
        u0=None,
        input_times=input_times,
        input_values=p_mid,
    )
    objective = ReducedObjective(
        dynamics,
        decoder,
        stepper,
        decoder_ridge=0.0,
        dynamics_ridge=1.0e-5,
        normal_chunk_size=256,
        gradient_mode="lagged_adjoint",
    )
    rollout, weights, normal, base = compute_base(dynamics, decoder, batch, objective)
    g0 = grad_direction(base)
    base_grad_norm = direction_norm(g0, dynamics)
    base_residual_norm = float(torch.sqrt(2.0 * base.data_loss.detach()).cpu())

    directions = [random_direction_like(dynamics, seed=100 + i, scale=1.0) for i in range(8)]
    eps_values = torch.logspace(-7, -2, 11, device=device, dtype=dtype)
    taylor_curves = []
    for i, direction in enumerate(directions):
        hv, _ = hvp(dynamics, decoder, batch, objective, rollout, weights, normal, direction)
        data_qform = projected_data_qform(dynamics, decoder, batch, objective, rollout, weights, normal, direction)
        curve = []
        scalar_curve = []
        gtd0 = dot_direction(g0, direction, dynamics)
        vthv = dot_direction(direction, hv, dynamics)
        reg_v = vthv.new_zeros(())
        for name, param in dynamics.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("linear."):
                ridge = objective.linear_ridge
            elif name.startswith("quadratic."):
                ridge = objective.quadratic_ridge
            elif name.startswith("source."):
                ridge = objective.source_ridge
            else:
                ridge = objective.dynamics_ridge
            reg_v = reg_v + float(ridge) * direction.get(name, torch.zeros_like(param)).square().sum()
        for eps in eps_values:
            eps_float = float(eps.detach().cpu())
            add_scaled_parameters(dynamics, direction, eps_float)
            pert = objective.value_and_grad(batch)
            add_scaled_parameters(dynamics, direction, -eps_float)
            gd = grad_direction(pert)
            rem = {}
            for name, param in dynamics.named_parameters():
                if not param.requires_grad:
                    continue
                rem[name] = gd.get(name, torch.zeros_like(param)) - g0.get(name, torch.zeros_like(param)) - eps * hv.get(name, torch.zeros_like(param))
            denom = 1.0 + direction_norm(hv, dynamics)
            abs_err = torch.sqrt(dot_direction(rem, rem, dynamics))
            curve.append(
                {
                    "eps": eps_float,
                    "error_over_eps": float((abs_err / denom / eps).detach().cpu()),
                    "absolute_error": float(abs_err.detach().cpu()),
                }
            )
            scalar_second = 2.0 * (pert.loss.detach() - base.loss.detach() - eps * gtd0) / (eps * eps)
            scalar_curve.append(
                {
                    "eps": eps_float,
                    "second_difference": float(scalar_second.detach().cpu()),
                    "data_second_difference": float((scalar_second - reg_v).detach().cpu()),
                    "vTHv": float(vthv.detach().cpu()),
                    "projected_data_qform": float(data_qform.detach().cpu()),
                    "ridge_qform": float(reg_v.detach().cpu()),
                    "relative_gap": float((torch.abs(scalar_second - vthv) / (1.0 + torch.abs(scalar_second) + torch.abs(vthv))).detach().cpu()),
                }
            )
        taylor_curves.append({"direction": i, "values": curve, "scalar_values": scalar_curve, "hv_norm": direction_norm(hv, dynamics)})

    symmetry = []
    for i in range(4):
        u = directions[2 * i]
        v = directions[2 * i + 1]
        hu, _ = hvp(dynamics, decoder, batch, objective, rollout, weights, normal, u)
        hv_dir, _ = hvp(dynamics, decoder, batch, objective, rollout, weights, normal, v)
        lhs = dot_direction(u, hv_dir, dynamics)
        rhs = dot_direction(v, hu, dynamics)
        rel = torch.abs(lhs - rhs) / (1.0 + torch.abs(lhs) + torch.abs(rhs))
        symmetry.append({"pair": i, "uTHv": float(lhs.detach().cpu()), "vTHu": float(rhs.detach().cpu()), "relative_gap": float(rel.detach().cpu())})

    # Randomized eigenspectrum of the symmetric GN operator.
    probe_count = 24
    probes = [random_direction_like(dynamics, seed=300 + i, scale=1.0) for i in range(probe_count)]
    hv_cols = []
    for probe in probes:
        hp, _ = hvp(dynamics, decoder, batch, objective, rollout, weights, normal, probe)
        hv_cols.append(flatten_direction(hp, dynamics))
    y = torch.stack(hv_cols, dim=1)
    q, _ = torch.linalg.qr(y, mode="reduced")
    small = torch.empty(q.shape[1], q.shape[1], device=device, dtype=dtype)
    for j in range(q.shape[1]):
        qdir = {}
        offset = 0
        for name, param in dynamics.named_parameters():
            if not param.requires_grad:
                continue
            count = param.numel()
            qdir[name] = q[offset : offset + count, j].view_as(param)
            offset += count
        hq, _ = hvp(dynamics, decoder, batch, objective, rollout, weights, normal, qdir)
        hq_flat = flatten_direction(hq, dynamics)
        small[:, j] = q.T @ hq_flat
    small = 0.5 * (small + small.T)
    evals = torch.linalg.eigvalsh(small).flip(0).detach().cpu()

    outdir = Path("outputs/incremental_hvp_validation")
    outdir.mkdir(parents=True, exist_ok=True)
    eps_cpu = [float(x.detach().cpu()) for x in eps_values]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for curve in taylor_curves:
        axes[0].loglog(eps_cpu, [x["error_over_eps"] for x in curve["values"]], marker="o", linewidth=1)
    axes[0].set_xlabel("eps")
    axes[0].set_ylabel("gradient Taylor remainder / eps")
    axes[0].set_title("GN-HV Taylor Curves")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogy([x["pair"] for x in symmetry], [x["relative_gap"] for x in symmetry], marker="o")
    axes[1].set_xlabel("direction pair")
    axes[1].set_ylabel("relative symmetry gap")
    axes[1].set_title(r"$u^T Hv$ symmetry")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[2].semilogy(range(1, len(evals) + 1), torch.clamp(evals, min=1.0e-30).numpy(), marker="o")
    axes[2].set_xlabel("randomized Ritz index")
    axes[2].set_ylabel("eigenvalue")
    axes[2].set_title("Randomized GN Spectrum")
    axes[2].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = outdir / "gn_hvp_validation.png"
    fig.savefig(fig_path, dpi=180)
    summary = {
        "device": str(device),
        "parameter_dim": int(flatten_direction(directions[0], dynamics).numel()),
        "base_loss": float(base.loss.detach().cpu()),
        "base_data_loss": float(base.data_loss.detach().cpu()),
        "base_residual_norm": base_residual_norm,
        "base_grad_norm": base_grad_norm,
        "taylor": [
            {
                "direction": c["direction"],
                "hv_norm": c["hv_norm"],
                "eps": eps_cpu,
                "error_over_eps": [x["error_over_eps"] for x in c["values"]],
                "absolute_error": [x["absolute_error"] for x in c["values"]],
                "scalar_second_difference": c["scalar_values"],
            }
            for c in taylor_curves
        ],
        "symmetry": symmetry,
        "randomized_eigenvalues": [float(x) for x in evals],
        "figure": str(fig_path),
    }
    summary_path = outdir / "gn_hvp_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
