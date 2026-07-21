from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.func import functional_call

from quadrode_gpu_goattm.data import ContinuousBatch
from quadrode_gpu_goattm.decoders import QuadraticReadoutDecoder
from quadrode_gpu_goattm.dynamics import QuadraticDynamics
from quadrode_gpu_goattm.incremental import (
    flatten_direction,
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


class ReducedLossModule(nn.Module):
    def __init__(
        self,
        dynamics: QuadraticDynamics,
        decoder: QuadraticReadoutDecoder,
        stepper: DenseLaggedMidpointStepper,
        *,
        decoder_ridge: float,
        dynamics_ridge: float,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.decoder = decoder
        self.stepper = stepper
        self.decoder_ridge = float(decoder_ridge)
        self.dynamics_ridge = float(dynamics_ridge)

    def _differentiable_normal_solve(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading = int(states.shape[:-1].numel())
        flat_states = states.reshape(leading, states.shape[-1])
        flat_targets = targets.reshape(leading, targets.shape[-1])
        features = self.decoder.features(flat_states)
        if self.decoder.readout.bias is not None:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            features = torch.cat((features, ones), dim=1)
        flat_weights = weights.reshape(leading).to(device=states.device, dtype=states.dtype)
        weighted_features = features * flat_weights[:, None]
        normal = features.T @ weighted_features
        normal = normal + self.decoder_ridge * torch.eye(normal.shape[0], device=normal.device, dtype=normal.dtype)
        rhs = flat_targets.T @ weighted_features
        coeff = torch.linalg.solve(normal, rhs.T).T
        return features, coeff

    def forward(self, batch: ContinuousBatch) -> torch.Tensor:
        p_mid = batch.midpoint_inputs()
        u0 = batch.qoi.new_zeros(batch.batch_size, self.dynamics.latent_dim) if batch.u0 is None else batch.u0
        states = self.stepper.rollout(self.dynamics, u0, batch.step_size, batch.steps, p_mid=p_mid)
        weights = trapezoidal_weights(batch.observation_times)[:, None].expand(batch.qoi.shape[:-1])
        features, coeff = self._differentiable_normal_solve(states, batch.qoi, weights)
        prediction = features @ coeff.T
        residual = prediction.reshape_as(batch.qoi) - batch.qoi
        loss = 0.5 * (residual.square().sum(dim=-1) * weights).sum()
        loss = loss + 0.5 * self.decoder_ridge * coeff.square().sum()
        if self.dynamics_ridge > 0.0:
            for param in self.dynamics.parameters():
                if param.requires_grad:
                    loss = loss + 0.5 * self.dynamics_ridge * param.square().sum()
        return loss


def dot_tuples(xs: tuple[torch.Tensor, ...], ys: tuple[torch.Tensor, ...]) -> torch.Tensor:
    total = None
    for x, y in zip(xs, ys):
        piece = (x * y).sum()
        total = piece if total is None else total + piece
    if total is None:
        raise ValueError("empty tuple")
    return total


def tuple_norm(xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.sqrt(dot_tuples(xs, xs))


def main() -> None:
    torch.manual_seed(20260705)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    batch_size = 24
    steps = 5
    latent_dim = 6
    input_dim = 4
    output_dim = 3
    step_size = 0.035
    decoder_ridge = 1.0e-4
    dynamics_ridge = 1.0e-5

    dynamics = QuadraticDynamics(
        DissipativeSkewA(latent_dim, skew_rank=3, damping_init=0.10, factor_scale=0.04, damping_shift=0.03),
        EnergyTuckerTTQuadratic(latent_dim, reduced_rank=4, tt_rank=3, scale=0.025, basis_trainable=True),
        LinearSource(latent_dim, input_dim, init_scale=0.03),
    ).to(device=device, dtype=dtype)
    decoder = QuadraticReadoutDecoder(latent_dim, output_dim, include_quadratic=True, bias=True).to(device=device, dtype=dtype)
    stepper = DenseLaggedMidpointStepper(picard_iters=2)
    model = ReducedLossModule(dynamics, decoder, stepper, decoder_ridge=decoder_ridge, dynamics_ridge=dynamics_ridge)

    times = step_size * torch.arange(steps + 1, device=device, dtype=dtype)
    input_times = 0.5 * (times[:-1] + times[1:])
    p_mid = 0.1 * torch.randn(steps, batch_size, input_dim, device=device, dtype=dtype)
    u0 = torch.zeros(batch_size, latent_dim, device=device, dtype=dtype)
    with torch.no_grad():
        rollout = stepper.rollout(dynamics, u0, step_size, steps, p_mid=p_mid)
        decoder.readout.weight.normal_(0.0, 0.04)
        decoder.readout.bias.zero_()
        qoi = decoder(rollout).detach() + 0.02 * torch.randn(steps + 1, batch_size, output_dim, device=device, dtype=dtype)
    batch = ContinuousBatch(
        sample_ids=tuple(str(i) for i in range(batch_size)),
        observation_times=times,
        qoi=qoi,
        u0=None,
        input_times=input_times,
        input_values=p_mid,
    )

    params = OrderedDict(
        (f"dynamics.{name}", param.detach().clone().requires_grad_(True))
        for name, param in dynamics.named_parameters()
        if param.requires_grad
    )
    names = tuple(params.keys())
    values = tuple(params.values())

    def loss_from_values(*param_values: torch.Tensor) -> torch.Tensor:
        return functional_call(model, OrderedDict(zip(names, param_values)), (batch,))

    value = loss_from_values(*values)
    grads = torch.autograd.grad(value, values, create_graph=True)
    directions = []
    for i in range(6):
        direction_dict = random_direction_like(dynamics, seed=900 + i, scale=1.0)
        by_name = {f"dynamics.{name}": direction_dict[name].to(device=device, dtype=dtype) for name, _ in dynamics.named_parameters()}
        directions.append(tuple(by_name[name] for name in names))

    hvps = []
    symmetry = []
    taylor = []
    eps_values = torch.logspace(-7, -2, 11, device=device, dtype=dtype)
    for i, direction in enumerate(directions):
        directional_grad = dot_tuples(grads, direction)
        hvp_values = torch.autograd.grad(directional_grad, values, retain_graph=True)
        hvps.append(hvp_values)
        vthv = dot_tuples(direction, hvp_values)
        grad_dot = dot_tuples(grads, direction)
        curve = []
        for eps in eps_values:
            perturbed = tuple(param + eps * d for param, d in zip(values, direction))
            pert_value = loss_from_values(*perturbed)
            second = 2.0 * (pert_value - value - eps * grad_dot) / (eps * eps)
            curve.append(
                {
                    "eps": float(eps.detach().cpu()),
                    "second_difference": float(second.detach().cpu()),
                    "vTHv": float(vthv.detach().cpu()),
                    "relative_gap": float((torch.abs(second - vthv) / (1.0 + torch.abs(second) + torch.abs(vthv))).detach().cpu()),
                }
            )
        taylor.append({"direction": i, "values": curve, "hvp_norm": float(tuple_norm(hvp_values).detach().cpu())})

    for i in range(3):
        u = directions[2 * i]
        v = directions[2 * i + 1]
        hu = hvps[2 * i]
        hv = hvps[2 * i + 1]
        lhs = dot_tuples(u, hv)
        rhs = dot_tuples(v, hu)
        symmetry.append(
            {
                "pair": i,
                "uTHv": float(lhs.detach().cpu()),
                "vTHu": float(rhs.detach().cpu()),
                "relative_gap": float((torch.abs(lhs - rhs) / (1.0 + torch.abs(lhs) + torch.abs(rhs))).detach().cpu()),
            }
        )

    # Compare the exact reduced curvature with the positive Schur/GN decoder component.
    with torch.no_grad():
        history = stepper.rollout_with_picard_history(dynamics, u0, step_size, steps, p_mid=p_mid)
        weights = trapezoidal_weights(batch.observation_times)[:, None].expand(batch.qoi.shape[:-1])
        normal = solve_decoder_normal_terms(
            decoder,
            assemble_decoder_normal_terms(decoder, history.states, batch.qoi, weights=weights, chunk_size=128),
            ridge=decoder_ridge,
        )
        gn_components = []
        for i in range(3):
            direction_dict = random_direction_like(dynamics, seed=900 + i, scale=1.0)
            inc = lagged_midpoint_rollout_incremental(dynamics, u0, None, step_size, p_mid, direction_dict, rollout=history)
            _, _, qform = projected_decoder_residual_tangent(
                decoder,
                normal,
                history.states,
                inc.states_dot,
                batch.qoi,
                weights=weights,
                ridge=decoder_ridge,
                chunk_size=128,
            )
            gn_components.append(float(qform.detach().cpu()))

    outdir = Path("outputs/incremental_hvp_validation")
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    eps_cpu = [float(e.detach().cpu()) for e in eps_values]
    for row in taylor:
        axes[0].loglog(eps_cpu, [x["relative_gap"] for x in row["values"]], linewidth=1.2)
    axes[0].set_xlabel("eps")
    axes[0].set_ylabel("second-difference vs exact HVP gap")
    axes[0].set_title("Exact reduced HVP reference")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogy([x["pair"] for x in symmetry], [x["relative_gap"] for x in symmetry], marker="o")
    axes[1].set_xlabel("pair")
    axes[1].set_ylabel("symmetry gap")
    axes[1].set_title(r"$u^T Hv=v^T Hu$")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = outdir / "exact_reduced_hessian_reference.png"
    fig.savefig(fig_path, dpi=180)

    summary = {
        "device": str(device),
        "base_loss": float(value.detach().cpu()),
        "gradient_norm": float(tuple_norm(grads).detach().cpu()),
        "taylor": taylor,
        "symmetry": symmetry,
        "gn_decoder_qforms_first3": gn_components,
        "figure": str(fig_path),
    }
    summary_path = outdir / "exact_reduced_hessian_reference.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
