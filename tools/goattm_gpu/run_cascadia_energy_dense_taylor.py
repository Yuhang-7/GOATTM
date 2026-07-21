from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_GOATTM_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm")
GOATTM_ROOT = Path(os.environ.get("QUAD_GOATTM_ROOT", str(DEFAULT_GOATTM_ROOT))).expanduser()
sys.path.insert(0, str(GOATTM_ROOT))

from quadrode_gpu_goattm import (  # noqa: E402
    ContinuousBatch,
    ContinuousDataset,
    DenseLaggedMidpointStepper,
    DenseLinearA,
    EnergyDenseQuadratic,
    LinearSource,
    QuadraticDynamics,
    QuadraticReadoutDecoder,
    ReducedObjective,
    linear_interpolate,
    load_manifest_npz,
)


@dataclass(frozen=True)
class ChannelStats:
    mean: torch.Tensor
    scale: torch.Tensor
    target_max_abs: float

    def to_json(self) -> dict[str, object]:
        return {
            "target_max_abs": float(self.target_max_abs),
            "mean": self.mean.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
        }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def channel_stats(values: torch.Tensor, target_max_abs: float, eps: float = 1.0e-12) -> ChannelStats:
    flat = values.reshape(-1, values.shape[-1])
    mean = flat.mean(dim=0)
    centered_max_abs = (flat - mean).abs().amax(dim=0)
    scale = centered_max_abs / float(target_max_abs)
    scale = torch.where(centered_max_abs < float(eps), torch.ones_like(scale), scale)
    return ChannelStats(mean=mean, scale=scale, target_max_abs=float(target_max_abs))


def normalize(values: torch.Tensor, stats: ChannelStats) -> torch.Tensor:
    return (values - stats.mean) / stats.scale


def select_cascadia_qoi(batch: ContinuousBatch) -> torch.Tensor:
    indices = torch.arange(0, 150, 3, device=batch.qoi.device)
    if indices.numel() != 50:
        raise RuntimeError(f"expected 50 selected QoIs, got {indices.numel()}")
    return batch.qoi.index_select(-1, indices)


def build_normalized_input_on_observation_grid(
    observation_times: torch.Tensor,
    active_input_times: torch.Tensor,
    active_input_values: torch.Tensor,
    stats: ChannelStats,
) -> torch.Tensor:
    active_norm = normalize(active_input_values, stats)
    zero_time = observation_times.new_zeros(1)
    zero_value = active_norm.new_zeros((1, active_norm.shape[1], active_norm.shape[2]))
    interp_times = torch.cat((zero_time, active_input_times), dim=0)
    interp_values = torch.cat((zero_value, active_norm), dim=0)

    full = active_norm.new_zeros(
        (observation_times.numel(), active_norm.shape[1], active_norm.shape[2])
    )
    active_mask = observation_times <= active_input_times[-1]
    full[active_mask] = linear_interpolate(
        interp_times,
        interp_values,
        observation_times[active_mask],
        check_bounds=True,
    )
    return full


def load_cascadia_batch(
    manifest_path: Path,
    train_count: int,
    *,
    device: torch.device,
    target_max_abs: float,
    normalize_time: bool,
) -> tuple[ContinuousBatch, dict[str, object]]:
    manifest = load_manifest_npz(manifest_path)
    count = min(int(train_count), len(manifest))
    dataset = ContinuousDataset(manifest, device=device, dtype=torch.float64)
    raw = dataset.batch(indices=range(count))
    if raw.input_times is None or raw.input_values is None:
        raise ValueError("Cascadia batch must contain input_times and input_values")
    if raw.qoi.shape[-1] != 150:
        raise ValueError(f"expected raw qoi dimension 150, got {raw.qoi.shape[-1]}")
    if raw.input_values.shape[-1] != 150:
        raise ValueError(f"expected input dimension 150, got {raw.input_values.shape[-1]}")

    qoi = select_cascadia_qoi(raw)
    qoi_stats = channel_stats(qoi, target_max_abs)
    qoi_norm = normalize(qoi, qoi_stats)

    input_stats = channel_stats(raw.input_values, target_max_abs)
    input_full = build_normalized_input_on_observation_grid(
        raw.observation_times,
        raw.input_times,
        raw.input_values,
        input_stats,
    )
    active_input_norm = normalize(raw.input_values, input_stats)
    post_source_mask = raw.observation_times > raw.input_times[-1]

    observation_times = raw.observation_times
    input_times = raw.observation_times.clone()
    physical_step_size = float((observation_times[1] - observation_times[0]).detach().cpu())
    if normalize_time:
        t0 = observation_times[0]
        t1 = observation_times[-1]
        time_scale = t1 - t0
        observation_times = (observation_times - t0) / time_scale
        input_times = (input_times - t0) / time_scale
    batch = ContinuousBatch(
        sample_ids=raw.sample_ids,
        observation_times=observation_times,
        qoi=qoi_norm,
        u0=None,
        input_times=input_times,
        input_values=input_full,
    )
    metadata = {
        "raw_qoi_shape": list(raw.qoi.shape),
        "selected_qoi_shape": list(qoi_norm.shape),
        "raw_input_shape": list(raw.input_values.shape),
        "streamed_input_shape": list(input_full.shape),
        "qoi_indices": list(range(0, 150, 3)),
        "physical_step_size_seconds": physical_step_size,
        "steps": int(batch.steps),
        "normalized_time": bool(normalize_time),
        "selected_qoi_normalized_max_abs": float(qoi_norm.abs().amax().detach().cpu()),
        "active_input_normalized_max_abs": float(active_input_norm.abs().amax().detach().cpu()),
        "post_source_input_max_abs": float(input_full[post_source_mask].abs().amax().detach().cpu()),
        "qoi_normalization": qoi_stats.to_json(),
        "input_normalization": {
            **input_stats.to_json(),
            "computed_on": "active_input_times_10_to_180_seconds_only",
            "post_180_seconds_value": "exact_zero_in_normalized_coordinates",
        },
    }
    return batch, metadata


def make_dynamics(latent_dim: int, input_dim: int, init_scale: float, device: torch.device) -> QuadraticDynamics:
    dynamics = QuadraticDynamics(
        DenseLinearA(latent_dim, init_scale=init_scale),
        EnergyDenseQuadratic(latent_dim, scale=init_scale),
        LinearSource(latent_dim, input_dim, init_scale=init_scale),
    )
    return dynamics.double().to(device)


def clone_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def restore_params(module: torch.nn.Module, values: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in values:
                param.copy_(values[name])


def random_unit_direction(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=next(module.parameters()).device)
    gen.manual_seed(int(seed))
    direction: dict[str, torch.Tensor] = {}
    norm2 = None
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        value = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=gen)
        direction[name] = value
        piece = value.square().sum()
        norm2 = piece if norm2 is None else norm2 + piece
    if norm2 is None:
        raise ValueError("no trainable parameters")
    scale = torch.sqrt(norm2)
    return {name: value / scale for name, value in direction.items()}


def add_direction(module: torch.nn.Module, direction: dict[str, torch.Tensor], alpha: float) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in direction:
                param.add_(direction[name], alpha=float(alpha))


def direction_dot_grad(module: torch.nn.Module, direction: dict[str, torch.Tensor]) -> torch.Tensor:
    out = None
    for name, param in module.named_parameters():
        if name not in direction:
            continue
        piece = (
            torch.zeros((), device=param.device, dtype=param.dtype)
            if param.grad is None
            else (param.grad * direction[name]).sum()
        )
        out = piece if out is None else out + piece
    if out is None:
        raise ValueError("empty direction")
    return out


def write_plot(points: list[dict[str, float]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eps = [p["eps"] for p in points]
        err = [p["directional_error"] for p in points]
        rem = [p["first_remainder"] for p in points]
        plt.figure(figsize=(7.5, 4.8))
        plt.loglog(eps, err, "-o", label="directional error / eps")
        plt.loglog(eps, rem, "-s", label="first-order remainder")
        plt.gca().invert_xaxis()
        plt.xlabel("epsilon")
        plt.ylabel("Taylor error")
        plt.title("Cascadia r=60 dense A energy-dense H Taylor test")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
    except Exception as exc:
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascadia Taylor test: r=60, dense A, energy-dense H.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=60)
    parser.add_argument("--picard-iters", type=int, default=2)
    parser.add_argument("--decoder-ridge", type=float, default=1.0e-5)
    parser.add_argument("--dynamics-ridge", type=float, default=1.0e-7)
    parser.add_argument("--normal-chunk-size", type=int, default=4096)
    parser.add_argument("--normalization-target-max-abs", type=float, default=0.9)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--eps-min", type=float, default=1.0e-8)
    parser.add_argument("--eps-max", type=float, default=1.0e-2)
    parser.add_argument("--eps-count", type=int, default=8)
    parser.add_argument("--no-normalize-time", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed))

    batch, data_metadata = load_cascadia_batch(
        Path(args.manifest),
        args.train_count,
        device=device,
        target_max_abs=args.normalization_target_max_abs,
        normalize_time=not args.no_normalize_time,
    )
    input_dim = int(batch.input_values.shape[-1]) if batch.input_values is not None else 0
    output_dim = int(batch.qoi.shape[-1])
    dynamics = make_dynamics(args.latent_dim, input_dim, args.init_scale, device)
    decoder = QuadraticReadoutDecoder(
        args.latent_dim,
        output_dim,
        include_quadratic=True,
        bias=True,
    ).double().to(device)
    objective = ReducedObjective(
        dynamics,
        decoder,
        DenseLaggedMidpointStepper(picard_iters=args.picard_iters),
        decoder_ridge=args.decoder_ridge,
        dynamics_ridge=args.dynamics_ridge,
        normal_chunk_size=args.normal_chunk_size,
        gradient_mode="lagged_adjoint",
    )

    base = clone_params(objective.dynamics)
    direction = random_unit_direction(objective.dynamics, args.seed + 17)
    sync(device)
    start = time.perf_counter()
    result = objective.value_and_grad(batch)
    sync(device)
    gradient_seconds = time.perf_counter() - start
    base_loss = float(result.loss.detach().cpu())
    grad_dot = float(direction_dot_grad(objective.dynamics, direction).detach().cpu())

    eps_values = torch.logspace(
        math.log10(args.eps_max),
        math.log10(args.eps_min),
        args.eps_count,
        dtype=torch.float64,
    ).tolist()
    points: list[dict[str, float]] = []
    for eps in eps_values:
        restore_params(objective.dynamics, base)
        add_direction(objective.dynamics, direction, float(eps))
        sync(device)
        trial_start = time.perf_counter()
        trial = objective.evaluate(batch)
        sync(device)
        trial_seconds = time.perf_counter() - trial_start
        trial_loss = float(trial.loss.detach().cpu())
        first = abs(trial_loss - base_loss - float(eps) * grad_dot)
        points.append(
            {
                "eps": float(eps),
                "trial_loss": trial_loss,
                "first_remainder": first,
                "directional_error": first / float(eps),
                "eval_seconds": trial_seconds,
            }
        )
    restore_params(objective.dynamics, base)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(Path(args.manifest)),
        "goattm_root": str(GOATTM_ROOT),
        "device": str(device),
        "torch_version": torch.__version__,
        "train_count": int(args.train_count),
        "latent_dim": int(args.latent_dim),
        "linear_a": "dense",
        "quadratic": "energy_dense",
        "picard_iters": int(args.picard_iters),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "steps": int(batch.steps),
        "dt": float(batch.step_size),
        "base_loss": base_loss,
        "data_loss": float(result.data_loss.detach().cpu()),
        "dynamics_regularization_loss": float(result.dynamics_regularization_loss.detach().cpu()),
        "decoder_regularization_loss": float(result.decoder_regularization_loss.detach().cpu()),
        "grad_dot": grad_dot,
        "gradient_seconds": gradient_seconds,
        "normal_relative_residual": float(result.normal_solve.relative_residual),
        "peak_memory_mib": (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "data_streaming": data_metadata,
        "points": points,
    }
    (outdir / "taylor.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_plot(points, outdir / "taylor.png")

    summary_keys = [
        "device",
        "train_count",
        "latent_dim",
        "linear_a",
        "quadratic",
        "input_dim",
        "output_dim",
        "steps",
        "dt",
        "base_loss",
        "gradient_seconds",
        "normal_relative_residual",
        "peak_memory_mib",
    ]
    print(json.dumps({key: payload[key] for key in summary_keys}, indent=2))
    print("wrote", outdir / "taylor.json")
    print("wrote", outdir / "taylor.png")


if __name__ == "__main__":
    main()
