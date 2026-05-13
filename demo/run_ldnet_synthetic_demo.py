from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic LDNet/simple-Mamba alternating-training demo.")
    parser.add_argument("--outer-cycles", type=int, default=3)
    parser.add_argument("--dynamics-steps", type=int, default=4)
    parser.add_argument("--decoder-steps", type=int, default=6)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--test-samples", type=int, default=12)
    parser.add_argument("--time-steps", type=int, default=41)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--qoi-dim", type=int, default=4)
    parser.add_argument("--ssm-state-dim", type=int, default=20)
    parser.add_argument("--dynamics-lr", type=float, default=2e-3)
    parser.add_argument("--decoder-lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_synthetic_batch(
    n_samples: int,
    n_steps: int,
    latent_dim: int,
    qoi_dim: int,
    dtype: torch.dtype,
    seed: int,
    sample_offset: int = 0,
) -> LDNetBatch:
    generator = torch.Generator().manual_seed(seed)
    times = torch.linspace(0.0, 1.0, n_steps, dtype=dtype)
    u0 = 0.4 * torch.randn(n_samples, latent_dim, generator=generator, dtype=dtype)
    phases = torch.linspace(0.0, 1.0, qoi_dim, dtype=dtype)
    trajectories = []
    for idx in range(n_samples):
        local_phase = float(idx + sample_offset) * 0.07
        base = torch.stack(
            [
                torch.sin((j + 1) * times * 2.0 + local_phase) + 0.2 * torch.cos((j + 2) * times)
                for j in range(latent_dim)
            ],
            dim=-1,
        )
        latent = base + u0[idx][None, :]
        qoi = torch.stack(
            [
                torch.tanh(latent[:, j % latent_dim] + 0.25 * latent[:, (j + 1) % latent_dim] ** 2)
                + 0.1 * torch.sin(3.0 * times + phases[j])
                for j in range(qoi_dim)
            ],
            dim=-1,
        )
        trajectories.append(qoi)
    return LDNetBatch(times=times, u0=u0, qoi=torch.stack(trajectories, dim=0))


def shard_batch(batch: LDNetBatch, context: TorchMPIContext) -> LDNetBatch:
    indices = torch.arange(batch.u0.shape[0])
    keep = indices[indices % context.size == context.rank]
    return LDNetBatch(times=batch.times, u0=batch.u0[keep], qoi=batch.qoi[keep], inputs=None)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    context = TorchMPIContext.from_mpi4py()
    latent_dim = args.latent_dim
    qoi_dim = args.qoi_dim
    train_batch = shard_batch(make_synthetic_batch(args.train_samples, args.time_steps, latent_dim, qoi_dim, dtype, seed=7), context)
    test_batch = shard_batch(
        make_synthetic_batch(args.test_samples, args.time_steps, latent_dim, qoi_dim, dtype, seed=11, sample_offset=100),
        context,
    )

    dynamics = TorchQuadraticLatentDynamics(latent_dim, dtype=dtype, h_scale=1e-4)
    decoder = SelectiveSSMDecoder(
        SelectiveSSMDecoderConfig(latent_dim=latent_dim, qoi_dim=qoi_dim, state_dim=args.ssm_state_dim, dtype=dtype)
    )
    model = LDNetModel(dynamics, decoder)
    config = AlternatingLDNetConfig(
        outer_cycles=args.outer_cycles,
        dynamics_steps_per_cycle=args.dynamics_steps,
        decoder_steps_per_cycle=args.decoder_steps,
        dynamics_learning_rate=args.dynamics_lr,
        decoder_learning_rate=args.decoder_lr,
        output_dir=REPO_ROOT / "demo" / "ldnet_synthetic_output",
    )
    alternating_train_ldnet(model, train_batch, config, test_batch=test_batch, context=context)


if __name__ == "__main__":
    main()
