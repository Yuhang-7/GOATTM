from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


DEFAULT_GOATTM_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm")
GOATTM_ROOT = Path(os.environ.get("QUAD_GOATTM_ROOT", str(DEFAULT_GOATTM_ROOT))).expanduser()
sys.path.insert(0, str(GOATTM_ROOT))

from quadrode_gpu_goattm import (  # noqa: E402
    ContinuousBatch,
    DenseLaggedMidpointStepper,
    DissipativeSkewA,
    DistributedReducedObjective,
    EnergyTuckerTTQuadratic,
    LinearSource,
    QuadraticDynamics,
    QuadraticReadoutDecoder,
    ReducedObjective,
    SubstepRungeKutta4Stepper,
    objective_backend_metadata,
)


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_max(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    return list(range(rank, total, world_size))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_packed_batch(path: Path, device: torch.device, *, rank: int, world_size: int, sample_count: int | None) -> tuple[ContinuousBatch, dict]:
    payload = torch.load(path, map_location="cpu")
    total = int(payload["qoi"].shape[1])
    if sample_count is not None:
        total = min(total, int(sample_count))
    indices = shard_indices(total, rank, world_size)
    qoi = payload["qoi"][:, indices]
    inp = payload["input_values"][:, indices]
    sample_ids = [payload["sample_ids"][i] for i in indices]
    observation_times = payload["observation_times"]
    return ContinuousBatch(
        sample_ids=tuple(str(x) for x in sample_ids),
        observation_times=observation_times.to(device=device, dtype=torch.float64),
        qoi=qoi.to(device=device, dtype=torch.float64),
        u0=None,
        input_times=observation_times.to(device=device, dtype=torch.float64),
        input_values=inp.to(device=device, dtype=torch.float64),
    ), dict(payload.get("metadata", {})) | {
        "packed_path": str(path),
        "global_selected_samples": total,
        "local_samples": len(indices),
    }


def make_objective(args: argparse.Namespace, batch: ContinuousBatch, device: torch.device) -> ReducedObjective:
    linear = DissipativeSkewA(
        int(args.latent_dim),
        skew_rank=int(args.a_rank),
        damping_init=float(args.a_damping_init),
        factor_scale=float(args.init_scale),
        damping_shift=float(args.a_damping_shift),
    )
    quadratic = EnergyTuckerTTQuadratic(
        int(args.latent_dim),
        reduced_rank=int(args.h_rank),
        tt_rank=int(args.tt_rank),
        scale=float(args.init_scale),
    )
    dynamics = QuadraticDynamics(
        linear,
        quadratic,
        LinearSource(int(args.latent_dim), int(batch.input_values.shape[-1]), init_scale=float(args.init_scale)),
    ).double().to(device)
    decoder = QuadraticReadoutDecoder(
        int(args.latent_dim),
        int(batch.qoi.shape[-1]),
        include_quadratic=True,
        bias=True,
    ).double().to(device)
    if args.stepper == "lagged":
        stepper = DenseLaggedMidpointStepper(picard_iters=int(args.picard_iters))
        gradient_mode = "lagged_adjoint"
    elif args.stepper == "rk4_substep":
        stepper = SubstepRungeKutta4Stepper(substeps=int(args.substeps))
        gradient_mode = "rk4_adjoint"
    else:
        raise ValueError(f"unknown stepper {args.stepper}")
    return ReducedObjective(
        dynamics,
        decoder,
        stepper,
        decoder_ridge=float(args.decoder_ridge),
        dynamics_ridge=float(args.dynamics_ridge),
        normal_chunk_size=int(args.normal_chunk_size),
        gradient_mode=gradient_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Cascadia lagged midpoint vs RK4 substep closures.")
    parser.add_argument("--train-packed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stepper", choices=("lagged", "rk4_substep"), required=True)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--a-rank", type=int, default=30)
    parser.add_argument("--h-rank", type=int, default=30)
    parser.add_argument("--tt-rank", type=int, default=30)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--picard-iters", type=int, default=2)
    parser.add_argument("--closures", type=int, default=1)
    parser.add_argument("--decoder-ridge", type=float, default=1.0e-5)
    parser.add_argument("--dynamics-ridge", type=float, default=1.0e-7)
    parser.add_argument("--normal-chunk-size", type=int, default=4096)
    parser.add_argument("--init-scale", type=float, default=1.0e-2)
    parser.add_argument("--a-damping-init", type=float, default=0.1)
    parser.add_argument("--a-damping-shift", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    rank, world_size, local_rank = setup_distributed()
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
        torch.manual_seed(int(args.seed))
        load_start = time.perf_counter()
        batch, data_meta = load_packed_batch(
            Path(args.train_packed),
            device,
            rank=rank,
            world_size=world_size,
            sample_count=args.sample_count,
        )
        objective = make_objective(args, batch, device)
        train_objective = DistributedReducedObjective(objective) if world_size > 1 else objective
        sync(device)
        load_seconds = time.perf_counter() - load_start
        records = []
        for k in range(int(args.closures)):
            sync(device)
            t0 = time.perf_counter()
            result = train_objective.value_and_grad(batch)
            sync(device)
            records.append(
                {
                    "closure_index": k,
                    "closure_seconds": time.perf_counter() - t0,
                    "loss": float(result.loss.detach().cpu()),
                    "data_loss": float(result.data_loss.detach().cpu()),
                    "decoder_regularization_loss": float(result.decoder_regularization_loss.detach().cpu()),
                    "dynamics_regularization_loss": float(result.dynamics_regularization_loss.detach().cpu()),
                    "normal_relative_residual": float(result.normal_solve.relative_residual),
                }
            )
        peak = torch.tensor(
            [torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0],
            device=device,
            dtype=torch.float64,
        )
        all_reduce_max(peak)
        payload = {
            "args": vars(args),
            "world_size": int(world_size),
            "rank": int(rank),
            "local_rank": int(local_rank),
            "device": str(device),
            "load_seconds": load_seconds,
            "data": data_meta,
            "backend": objective_backend_metadata(train_objective),
            "steps_observed": int(batch.steps),
            "dt_observed_normalized": float(batch.step_size),
            "dt_observed_physical_seconds": 5.0,
            "dt_latent_physical_seconds": 5.0 / float(args.substeps) if args.stepper == "rk4_substep" else 5.0,
            "closures": records,
            "global_peak_memory_mib": float(peak.item()),
        }
        outdir = Path(args.output_dir)
        if rank == 0:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "stepper_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps({
                "stepper": args.stepper,
                "world_size": world_size,
                "global_samples": data_meta["global_selected_samples"],
                "local_samples_rank0": data_meta["local_samples"],
                "dt_latent_physical_seconds": payload["dt_latent_physical_seconds"],
                "closure_seconds": [r["closure_seconds"] for r in records],
                "peak_memory_mib": payload["global_peak_memory_mib"],
                "backend": payload["backend"],
                "normal_relative_residual": records[-1]["normal_relative_residual"],
                "output": str(outdir / "stepper_benchmark.json"),
            }, indent=2))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
