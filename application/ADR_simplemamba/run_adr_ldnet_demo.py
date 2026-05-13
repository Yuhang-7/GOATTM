from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.data import load_npz_qoi_sample, load_npz_sample_manifest  # noqa: E402
from goattm.ldnet import (  # noqa: E402
    AlternatingLDNetConfig,
    LDNetBatch,
    LDNetModel,
    SelectiveSSMDecoder,
    SelectiveSSMDecoderConfig,
    TorchQuadraticLatentDynamics,
    alternating_train_ldnet,
)
from goattm.ldnet.mpi import TorchMPIContext  # noqa: E402
from goattm.ldnet.trainer import evaluate_mse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ADR r=10 LDNet/simple-Mamba smoke training job.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--latent-rank", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--max-time-steps", type=int, default=121)
    parser.add_argument("--ssm-state-dim", type=int, default=32)
    parser.add_argument("--outer-cycles", type=int, default=5)
    parser.add_argument("--dynamics-steps", type=int, default=5)
    parser.add_argument("--decoder-steps", type=int, default=10)
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260513)
    return parser.parse_args()


def selected_sample_paths(manifest_path: Path, train_samples: int, test_samples: int) -> tuple[list[Path], list[Path]]:
    manifest = load_npz_sample_manifest(manifest_path)
    total = train_samples + test_samples
    if len(manifest) < total:
        raise ValueError(f"Manifest has {len(manifest)} samples, but requested {total}.")
    paths = list(manifest.absolute_paths())
    return paths[:train_samples], paths[train_samples:total]


def qoi_stats(paths: list[Path], max_time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    qois = []
    for path in paths:
        sample = load_npz_qoi_sample(path)
        qois.append(np.asarray(sample.qoi_observations[:max_time_steps], dtype=np.float64))
    stacked = np.concatenate(qois, axis=0)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def make_batch(
    paths: list[Path],
    latent_rank: int,
    max_time_steps: int,
    qoi_mean: np.ndarray,
    qoi_std: np.ndarray,
    context: TorchMPIContext,
    dtype: torch.dtype,
) -> LDNetBatch:
    local_paths = [path for idx, path in enumerate(paths) if idx % context.size == context.rank]
    if not local_paths:
        first = load_npz_qoi_sample(paths[0])
        steps = min(max_time_steps, first.observation_times.shape[0])
        qoi_dim = first.qoi_observations.shape[1]
        input_dim = 0 if first.input_values is None else first.input_values.shape[1]
        return LDNetBatch(
            times=torch.as_tensor(first.observation_times[:steps], dtype=dtype),
            u0=torch.zeros((0, latent_rank), dtype=dtype),
            qoi=torch.zeros((0, steps, qoi_dim), dtype=dtype),
            inputs=torch.zeros((0, steps, input_dim), dtype=dtype),
        )

    qois = []
    inputs = []
    times = None
    for path in local_paths:
        sample = load_npz_qoi_sample(path)
        steps = min(max_time_steps, sample.observation_times.shape[0])
        sample_times = np.asarray(sample.observation_times[:steps], dtype=np.float64)
        if times is None:
            times = sample_times
        elif not np.allclose(times, sample_times):
            raise ValueError(f"Sample {path} has a different observation time grid.")
        qoi = np.asarray(sample.qoi_observations[:steps], dtype=np.float64)
        qois.append((qoi - qoi_mean[None, :]) / qoi_std[None, :])
        if sample.input_values is None:
            inputs.append(np.zeros((steps, 0), dtype=np.float64))
        else:
            inputs.append(np.asarray(sample.input_values[:steps], dtype=np.float64))
    assert times is not None
    return LDNetBatch(
        times=torch.as_tensor(times, dtype=dtype),
        u0=torch.zeros((len(local_paths), latent_rank), dtype=dtype),
        qoi=torch.as_tensor(np.stack(qois, axis=0), dtype=dtype),
        inputs=torch.as_tensor(np.stack(inputs, axis=0), dtype=dtype),
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    context = TorchMPIContext.from_mpi4py()
    train_paths, test_paths = selected_sample_paths(args.manifest_path, args.train_samples, args.test_samples)
    qoi_mean, qoi_std = qoi_stats(train_paths, args.max_time_steps)
    train_batch = make_batch(train_paths, args.latent_rank, args.max_time_steps, qoi_mean, qoi_std, context, dtype)
    test_batch = make_batch(test_paths, args.latent_rank, args.max_time_steps, qoi_mean, qoi_std, context, dtype)
    input_dim = 0 if train_batch.inputs is None else train_batch.inputs.shape[-1]
    qoi_dim = train_batch.qoi.shape[-1]
    dynamics = TorchQuadraticLatentDynamics(args.latent_rank, input_dim=input_dim, dtype=dtype, h_scale=1e-4)
    decoder = SelectiveSSMDecoder(
        SelectiveSSMDecoderConfig(
            latent_dim=args.latent_rank,
            qoi_dim=qoi_dim,
            state_dim=args.ssm_state_dim,
            dtype=dtype,
        )
    )
    model = LDNetModel(dynamics, decoder)
    config = AlternatingLDNetConfig(
        outer_cycles=args.outer_cycles,
        dynamics_steps_per_cycle=args.dynamics_steps,
        decoder_steps_per_cycle=args.decoder_steps,
        dynamics_learning_rate=args.dynamics_lr,
        decoder_learning_rate=args.decoder_lr,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = alternating_train_ldnet(model, train_batch, config, test_batch=test_batch, context=context)
    train_mse = evaluate_mse(model, train_batch, context)
    test_mse = evaluate_mse(model, test_batch, context)
    if context.is_root:
        summary = {
            "latent_rank": args.latent_rank,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "max_time_steps": args.max_time_steps,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "history_records": len(history.records),
            "qoi_mean": qoi_mean.tolist(),
            "qoi_std": qoi_std.tolist(),
        }
        summary_path = args.output_dir / "adr_ldnet_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    context.barrier()


if __name__ == "__main__":
    main()
