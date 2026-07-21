from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call


DEFAULT_GOATTM_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm")
GOATTM_ROOT = Path(os.environ.get("QUAD_GOATTM_ROOT", str(DEFAULT_GOATTM_ROOT))).expanduser()
sys.path.insert(0, str(GOATTM_ROOT))

from quadrode_gpu_goattm.reduced import trapezoidal_weights  # noqa: E402
from quadrode_gpu_goattm.steppers import DenseLaggedMidpointStepper  # noqa: E402
from tools.train_cascadia_packed import (  # noqa: E402
    MaskedCrossQuadraticReadoutDecoder,
    apply_pod_initializer,
    batch_from_payload,
    load_packed_payload,
    make_dynamics,
)


class DifferentiableVarProLoss(nn.Module):
    def __init__(
        self,
        dynamics: nn.Module,
        decoder: nn.Module,
        stepper: DenseLaggedMidpointStepper,
        *,
        decoder_ridge: float,
        dynamics_ridge: float,
        normal_chunk_size: int,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.decoder = decoder
        self.stepper = stepper
        self.decoder_ridge = float(decoder_ridge)
        self.dynamics_ridge = float(dynamics_ridge)
        self.normal_chunk_size = int(normal_chunk_size)

    def _features(self, states: torch.Tensor) -> torch.Tensor:
        flat_states = states.reshape(-1, states.shape[-1])
        features = self.decoder.features(flat_states)
        if self.decoder.readout.bias is not None:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            features = torch.cat((features, ones), dim=1)
        return features

    def _normal_solve(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(states)
        flat_targets = targets.reshape(-1, targets.shape[-1])
        flat_weights = weights.reshape(-1).to(device=states.device, dtype=states.dtype)
        weighted_features = features * flat_weights[:, None]
        normal = features.T @ weighted_features
        eye = torch.eye(normal.shape[0], device=normal.device, dtype=normal.dtype)
        normal = normal + self.decoder_ridge * eye
        rhs = flat_targets.T @ weighted_features
        coeff = torch.linalg.solve(normal, rhs.T).T
        return features, coeff

    def forward(self, batch) -> torch.Tensor:
        p_mid = batch.midpoint_inputs()
        u0 = batch.qoi.new_zeros(batch.batch_size, self.dynamics.latent_dim) if batch.u0 is None else batch.u0
        states = self.stepper.rollout(self.dynamics, u0, batch.step_size, batch.steps, p_mid=p_mid)
        weights = trapezoidal_weights(batch.observation_times)[:, None].expand(batch.qoi.shape[:-1])
        features, coeff = self._normal_solve(states, batch.qoi, weights)
        prediction = features @ coeff.T
        residual = prediction.reshape_as(batch.qoi) - batch.qoi
        data_loss = 0.5 * (residual.square().sum(dim=-1) * weights).sum()
        decoder_reg = 0.5 * self.decoder_ridge * coeff.square().sum()
        dynamics_reg = states.new_zeros(())
        if self.dynamics_ridge > 0.0:
            for param in self.dynamics.parameters():
                if param.requires_grad:
                    dynamics_reg = dynamics_reg + 0.5 * self.dynamics_ridge * param.square().sum()
        return data_loss + decoder_reg + dynamics_reg


def tuple_dot(xs: tuple[torch.Tensor, ...], ys: tuple[torch.Tensor, ...]) -> torch.Tensor:
    total = None
    for x, y in zip(xs, ys):
        piece = (x * y).sum()
        total = piece if total is None else total + piece
    if total is None:
        raise ValueError("empty tuple")
    return total


def tuple_norm(xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.sqrt(tuple_dot(xs, xs))


def tuple_add(xs: tuple[torch.Tensor, ...], ys: tuple[torch.Tensor, ...], alpha: float = 1.0) -> tuple[torch.Tensor, ...]:
    return tuple(x + float(alpha) * y for x, y in zip(xs, ys))


def tuple_scale(xs: tuple[torch.Tensor, ...], alpha: float) -> tuple[torch.Tensor, ...]:
    return tuple(float(alpha) * x for x in xs)


def tuple_sub(xs: tuple[torch.Tensor, ...], ys: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(x - y for x, y in zip(xs, ys))


def tuple_zeros_like(xs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros_like(x) for x in xs)


def values_to_json_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu())


def truncated_cg(
    hvp,
    grad: tuple[torch.Tensor, ...],
    *,
    damping: float,
    max_iter: int,
    rel_tol: float,
) -> tuple[tuple[torch.Tensor, ...], list[dict]]:
    x = tuple_zeros_like(grad)
    residual = tuple_scale(grad, -1.0)
    direction = residual
    residual_norm0 = tuple_norm(residual)
    residual_sq = tuple_dot(residual, residual)
    records: list[dict] = []
    for k in range(int(max_iter)):
        h_direction_raw = hvp(direction)
        h_direction = tuple_add(h_direction_raw, direction, alpha=float(damping))
        p_hp = tuple_dot(direction, h_direction)
        p_hp_value = values_to_json_float(p_hp)
        if p_hp_value <= 0.0:
            records.append(
                {
                    "iter": k,
                    "event": "negative_curvature",
                    "pHp": p_hp_value,
                    "residual_norm": values_to_json_float(torch.sqrt(residual_sq)),
                }
            )
            return (direction if k == 0 else x), records
        alpha = residual_sq / p_hp
        x = tuple_add(x, direction, alpha=values_to_json_float(alpha))
        residual_next = tuple_sub(residual, tuple_scale(h_direction, values_to_json_float(alpha)))
        residual_sq_next = tuple_dot(residual_next, residual_next)
        rel_res = torch.sqrt(residual_sq_next) / (residual_norm0 + residual_norm0.new_tensor(1.0e-30))
        records.append(
            {
                "iter": k,
                "event": "cg_step",
                "alpha": values_to_json_float(alpha),
                "pHp": p_hp_value,
                "residual_norm": values_to_json_float(torch.sqrt(residual_sq_next)),
                "relative_residual": values_to_json_float(rel_res),
            }
        )
        if values_to_json_float(rel_res) <= float(rel_tol):
            return x, records
        beta = residual_sq_next / residual_sq
        direction = tuple_add(residual_next, direction, alpha=values_to_json_float(beta))
        residual = residual_next
        residual_sq = residual_sq_next
    return x, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe exact reduced Newton-CG directions on a Cascadia checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-sample-limit", type=int, default=128)
    parser.add_argument("--cg-max-iter", type=int, default=8)
    parser.add_argument("--cg-damping", type=float, default=1.0e-3)
    parser.add_argument("--cg-rel-tol", type=float, default=1.0e-2)
    parser.add_argument("--armijo-initial-alpha", type=float, default=1.0)
    parser.add_argument("--armijo-shrink", type=float, default=0.5)
    parser.add_argument("--armijo-max-trials", type=int, default=18)
    parser.add_argument("--armijo-c1", type=float, default=1.0e-4)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--save-direction", action="store_true")
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = dict(checkpoint.get("metadata", {}))
    train_packed = Path(metadata["train_packed"])
    initializer = Path(metadata["initializer"]["path"])
    latent_dim = int(metadata["latent_dim"])
    linear_a = metadata["linear_a"]
    a_rank = int(metadata["linear_a_rank"])
    quadratic = metadata["quadratic"]
    h_reduced_rank = int(metadata["h_reduced_rank"])
    h_tt_rank = int(metadata["h_tt_rank"])
    decoder_cross_terms = int(metadata["decoder_cross_terms"])
    decoder_mask_seed = int(metadata["decoder_mask_seed"])
    decoder_ridge = float(metadata["optimizer"].get("decoder_ridge", 1.0e-5))
    dynamics_ridge = float(metadata["optimizer"].get("dynamics_ridge", 1.0e-7))
    picard_iters = int(metadata["optimizer"].get("picard_iters", 2))
    normal_chunk_size = int(metadata["optimizer"].get("normal_chunk_size", 4096))
    time_mode = metadata.get("time_mode", "normalized")

    payload, _ = load_packed_payload(train_packed)
    if int(args.train_sample_limit) > 0:
        payload = dict(payload)
        payload["sample_ids"] = list(payload["sample_ids"][: int(args.train_sample_limit)])
        payload["qoi"] = payload["qoi"][:, : int(args.train_sample_limit)]
        payload["input_values"] = payload["input_values"][:, : int(args.train_sample_limit)]
    batch = batch_from_payload(payload, device, time_mode=time_mode)

    input_dim = int(batch.input_values.shape[-1])
    output_dim = int(batch.qoi.shape[-1])
    dynamics = make_dynamics(
        latent_dim,
        input_dim,
        0.01,
        device,
        linear_a=linear_a,
        a_rank=a_rank,
        damping_init=0.1,
        damping_shift=0.0,
        quadratic=quadratic,
        h_reduced_rank=h_reduced_rank,
        h_tt_rank=h_tt_rank,
    )
    decoder = MaskedCrossQuadraticReadoutDecoder(
        latent_dim,
        output_dim,
        cross_terms=decoder_cross_terms,
        mask_seed=decoder_mask_seed,
        bias=True,
    ).double().to(device)
    apply_pod_initializer(dynamics, decoder, initializer)
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    stepper = DenseLaggedMidpointStepper(picard_iters=picard_iters)
    model = DifferentiableVarProLoss(
        dynamics,
        decoder,
        stepper,
        decoder_ridge=decoder_ridge,
        dynamics_ridge=dynamics_ridge,
        normal_chunk_size=normal_chunk_size,
    )
    params = OrderedDict(
        (f"dynamics.{name}", param.detach().clone().requires_grad_(True))
        for name, param in dynamics.named_parameters()
        if param.requires_grad
    )
    names = tuple(params.keys())
    values = tuple(params.values())

    def loss_from_values(vals: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return functional_call(model, OrderedDict(zip(names, vals)), (batch,))

    t0 = time.perf_counter()
    loss = loss_from_values(values)
    grads = torch.autograd.grad(loss, values, create_graph=True)
    grad_norm = tuple_norm(grads)

    def exact_hvp(direction: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        directional_grad = tuple_dot(grads, direction)
        return torch.autograd.grad(directional_grad, values, retain_graph=True)

    newton_direction, cg_records = truncated_cg(
        exact_hvp,
        grads,
        damping=float(args.cg_damping),
        max_iter=int(args.cg_max_iter),
        rel_tol=float(args.cg_rel_tol),
    )
    neggrad_direction = tuple_scale(grads, -1.0)

    def line_search(direction: tuple[torch.Tensor, ...], label: str) -> dict:
        gtd = tuple_dot(grads, direction)
        alpha = float(args.armijo_initial_alpha)
        records = []
        best = {"alpha": 0.0, "loss": values_to_json_float(loss), "accepted": False}
        with torch.no_grad():
            for trial in range(1, int(args.armijo_max_trials) + 1):
                trial_values = tuple_add(values, direction, alpha=alpha)
                trial_loss_tensor = loss_from_values(trial_values)
                trial_loss = values_to_json_float(trial_loss_tensor)
                threshold = values_to_json_float(loss + float(args.armijo_c1) * alpha * gtd)
                accepted = bool(trial_loss <= threshold)
                record = {
                    "trial": trial,
                    "alpha": alpha,
                    "loss": trial_loss,
                    "threshold": threshold,
                    "accepted": accepted,
                }
                records.append(record)
                if trial_loss < best["loss"]:
                    best = {"alpha": alpha, "loss": trial_loss, "accepted": accepted}
                if accepted:
                    break
                alpha *= float(args.armijo_shrink)
        return {
            "label": label,
            "direction_norm": values_to_json_float(tuple_norm(direction)),
            "directional_derivative": values_to_json_float(gtd),
            "best": best,
            "records": records,
        }

    newton_line = line_search(newton_direction, "damped_exact_newton_cg")
    grad_line = line_search(neggrad_direction, "negative_gradient")
    elapsed = time.perf_counter() - t0
    summary = {
        "checkpoint": str(checkpoint_path),
        "train_sample_limit": int(args.train_sample_limit),
        "device": str(device),
        "base_loss": values_to_json_float(loss),
        "grad_norm": values_to_json_float(grad_norm),
        "cg_damping": float(args.cg_damping),
        "cg_records": cg_records,
        "newton_line_search": newton_line,
        "negative_gradient_line_search": grad_line,
        "elapsed_seconds": elapsed,
        "peak_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "config": {
            "latent_dim": latent_dim,
            "a_rank": a_rank,
            "h_reduced_rank": h_reduced_rank,
            "h_tt_rank": h_tt_rank,
            "decoder_cross_terms": decoder_cross_terms,
            "steps": int(batch.steps),
            "batch_size": int(batch.batch_size),
            "picard_iters": picard_iters,
        },
    }
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_best_checkpoint:
        best_alpha = float(newton_line["best"]["alpha"])
        best_values = tuple_add(values, newton_direction, alpha=best_alpha)
        params_by_name = dict(dynamics.named_parameters())
        with torch.no_grad():
            for full_name, value in zip(names, best_values):
                param_name = full_name.removeprefix("dynamics.")
                params_by_name[param_name].copy_(value)
        candidate = {
            "metadata": {
                **metadata,
                "second_order_probe": {
                    "source_checkpoint": str(checkpoint_path),
                    "train_sample_limit": int(args.train_sample_limit),
                    "cg_damping": float(args.cg_damping),
                    "cg_max_iter": int(args.cg_max_iter),
                    "accepted_alpha": best_alpha,
                    "base_loss_on_probe": values_to_json_float(loss),
                    "best_loss_on_probe": float(newton_line["best"]["loss"]),
                },
            },
            "dynamics_state_dict": dynamics.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        }
        torch.save(candidate, outdir / "checkpoint_newton_best.pt")
        summary["saved_best_checkpoint"] = str(outdir / "checkpoint_newton_best.pt")
    if args.save_direction:
        torch.save(
            {
                "checkpoint": str(checkpoint_path),
                "names": names,
                "values": tuple(v.detach().cpu() for v in values),
                "newton_direction": tuple(v.detach().cpu() for v in newton_direction),
                "negative_gradient_direction": tuple(v.detach().cpu() for v in neggrad_direction),
                "summary": summary,
            },
            outdir / "newton_direction.pt",
        )
        summary["saved_direction"] = str(outdir / "newton_direction.pt")
    (outdir / "newton_cg_probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
