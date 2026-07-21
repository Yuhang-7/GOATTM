from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


DEFAULT_GOATTM_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm")
GOATTM_ROOT = Path(os.environ.get("QUAD_GOATTM_ROOT", str(DEFAULT_GOATTM_ROOT))).expanduser()
sys.path.insert(0, str(GOATTM_ROOT))

from quadrode_gpu_goattm import (  # noqa: E402
    ContinuousBatch,
    ContinuousDataset,
    DenseLaggedMidpointStepper,
    DenseLinearA,
    DistributedReducedObjective,
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
            "mean_shape": list(self.mean.shape),
            "scale_shape": list(self.scale.shape),
            "scale_min": float(self.scale.min().detach().cpu()),
            "scale_max": float(self.scale.max().detach().cpu()),
        }


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_reduce_max(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distributed_channel_stats(values: torch.Tensor, target_max_abs: float, eps: float = 1.0e-12) -> ChannelStats:
    flat = values.reshape(-1, values.shape[-1])
    total = flat.sum(dim=0)
    count = torch.tensor([flat.shape[0]], device=values.device, dtype=values.dtype)
    all_reduce_sum(total)
    all_reduce_sum(count)
    mean = total / count.clamp_min(1.0)
    centered_max_abs = (flat - mean).abs().amax(dim=0)
    all_reduce_max(centered_max_abs)
    scale = centered_max_abs / float(target_max_abs)
    scale = torch.where(centered_max_abs < float(eps), torch.ones_like(scale), scale)
    return ChannelStats(mean=mean, scale=scale, target_max_abs=float(target_max_abs))


def normalize(values: torch.Tensor, stats: ChannelStats) -> torch.Tensor:
    return (values - stats.mean) / stats.scale


def select_cascadia_qoi(batch: ContinuousBatch) -> torch.Tensor:
    indices = torch.arange(0, 150, 3, device=batch.qoi.device)
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
    full = active_norm.new_zeros((observation_times.numel(), active_norm.shape[1], active_norm.shape[2]))
    active_mask = observation_times <= active_input_times[-1]
    full[active_mask] = linear_interpolate(
        interp_times,
        interp_values,
        observation_times[active_mask],
        check_bounds=True,
    )
    return full


def load_local_cascadia_batch(
    manifest_path: Path,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    target_max_abs: float,
    normalize_time: bool,
) -> tuple[ContinuousBatch, dict[str, object]]:
    manifest = load_manifest_npz(manifest_path)
    total_count = len(manifest)
    indices = list(range(total_count))[rank::world_size]
    if not indices:
        raise ValueError(f"rank {rank} received no samples from {total_count}")
    dataset = ContinuousDataset(manifest, device=device, dtype=torch.float64)
    raw = dataset.batch(indices=indices)
    if raw.input_times is None or raw.input_values is None:
        raise ValueError("Cascadia batch must contain input_times and input_values")

    qoi = select_cascadia_qoi(raw)
    qoi_stats = distributed_channel_stats(qoi, target_max_abs)
    qoi_norm = normalize(qoi, qoi_stats)

    input_stats = distributed_channel_stats(raw.input_values, target_max_abs)
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
        scale = t1 - t0
        observation_times = (observation_times - t0) / scale
        input_times = (input_times - t0) / scale

    batch = ContinuousBatch(
        sample_ids=raw.sample_ids,
        observation_times=observation_times,
        qoi=qoi_norm,
        u0=None,
        input_times=input_times,
        input_values=input_full,
    )
    local_count_tensor = torch.tensor([batch.batch_size], device=device, dtype=torch.float64)
    total_count_tensor = local_count_tensor.clone()
    all_reduce_sum(total_count_tensor)
    metadata = {
        "manifest_count": total_count,
        "global_count": int(total_count_tensor.item()),
        "local_count": int(batch.batch_size),
        "raw_qoi_shape_local": list(raw.qoi.shape),
        "selected_qoi_shape_local": list(qoi_norm.shape),
        "raw_input_shape_local": list(raw.input_values.shape),
        "streamed_input_shape_local": list(input_full.shape),
        "physical_step_size_seconds": physical_step_size,
        "steps": int(batch.steps),
        "dt": float(batch.step_size),
        "selected_qoi_normalized_max_abs": float(qoi_norm.abs().amax().detach().cpu()),
        "active_input_normalized_max_abs": float(active_input_norm.abs().amax().detach().cpu()),
        "post_source_input_max_abs": float(input_full[post_source_mask].abs().amax().detach().cpu()),
        "qoi_normalization": qoi_stats.to_json(),
        "input_normalization": input_stats.to_json(),
    }
    return batch, metadata


def make_dynamics(latent_dim: int, input_dim: int, init_scale: float, device: torch.device) -> QuadraticDynamics:
    return QuadraticDynamics(
        DenseLinearA(latent_dim, init_scale=init_scale),
        EnergyDenseQuadratic(latent_dim, scale=init_scale),
        LinearSource(latent_dim, input_dim, init_scale=init_scale),
    ).double().to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one Cascadia reduced-objective closure.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-dim", type=int, default=60)
    parser.add_argument("--picard-iters", type=int, default=2)
    parser.add_argument("--closures", type=int, default=1)
    parser.add_argument("--decoder-ridge", type=float, default=1.0e-5)
    parser.add_argument("--dynamics-ridge", type=float, default=1.0e-7)
    parser.add_argument("--normal-chunk-size", type=int, default=4096)
    parser.add_argument("--normalization-target-max-abs", type=float, default=0.9)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--no-normalize-time", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    rank, world_size, local_rank = setup_distributed()
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(int(args.seed))

        load_start = time.perf_counter()
        batch, data_metadata = load_local_cascadia_batch(
            Path(args.manifest),
            device=device,
            rank=rank,
            world_size=world_size,
            target_max_abs=args.normalization_target_max_abs,
            normalize_time=not args.no_normalize_time,
        )
        sync(device)
        load_seconds = time.perf_counter() - load_start

        input_dim = int(batch.input_values.shape[-1]) if batch.input_values is not None else 0
        output_dim = int(batch.qoi.shape[-1])
        dynamics = make_dynamics(args.latent_dim, input_dim, args.init_scale, device)
        decoder = QuadraticReadoutDecoder(
            args.latent_dim,
            output_dim,
            include_quadratic=True,
            bias=True,
        ).double().to(device)
        base_objective = ReducedObjective(
            dynamics,
            decoder,
            DenseLaggedMidpointStepper(picard_iters=args.picard_iters),
            decoder_ridge=args.decoder_ridge,
            dynamics_ridge=args.dynamics_ridge,
            normal_chunk_size=args.normal_chunk_size,
            gradient_mode="lagged_adjoint",
        )
        objective = DistributedReducedObjective(base_objective) if world_size > 1 else base_objective

        closure_records = []
        for closure_index in range(int(args.closures)):
            sync(device)
            start = time.perf_counter()
            result = objective.value_and_grad(batch)
            sync(device)
            seconds = time.perf_counter() - start
            closure_records.append(
                {
                    "closure_index": closure_index,
                    "closure_seconds": seconds,
                    "loss": float(result.loss.detach().cpu()),
                    "data_loss": float(result.data_loss.detach().cpu()),
                    "decoder_regularization_loss": float(result.decoder_regularization_loss.detach().cpu()),
                    "dynamics_regularization_loss": float(result.dynamics_regularization_loss.detach().cpu()),
                    "normal_relative_residual": float(result.normal_solve.relative_residual),
                }
            )
            del result

        peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        peak_tensor = torch.tensor([peak_memory_mib], device=device, dtype=torch.float64)
        all_reduce_max(peak_tensor)
        payload = {
            "manifest": str(Path(args.manifest)),
            "goattm_root": str(GOATTM_ROOT),
            "device": str(device),
            "world_size": int(world_size),
            "rank": int(rank),
            "local_rank": int(local_rank),
            "torch_version": torch.__version__,
            "latent_dim": int(args.latent_dim),
            "linear_a": "dense",
            "quadratic": "energy_dense",
            "picard_iters": int(args.picard_iters),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "steps": int(batch.steps),
            "dt": float(batch.step_size),
            "load_seconds": load_seconds,
            "closures": closure_records,
            "local_peak_memory_mib": peak_memory_mib,
            "global_peak_memory_mib": float(peak_tensor.item()),
            "data_streaming": data_metadata,
        }
        outdir = Path(args.output_dir)
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "closure_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps({
                "world_size": world_size,
                "global_count": data_metadata["global_count"],
                "local_count_rank0": data_metadata["local_count"],
                "steps": payload["steps"],
                "dt": payload["dt"],
                "closure_seconds": [r["closure_seconds"] for r in closure_records],
                "global_peak_memory_mib": payload["global_peak_memory_mib"],
                "normal_relative_residual": closure_records[-1]["normal_relative_residual"],
            }, indent=2))
            print("wrote", outdir / "closure_benchmark.json")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
