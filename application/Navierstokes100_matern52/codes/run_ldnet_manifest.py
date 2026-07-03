from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.latent_dynamics import (  # noqa: E402
    LDNetConfig,
    LDNetModel,
    LDNetTrainConfig,
    make_train_test_tensors,
    train_ldnet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LDNet baseline from a GOATTM NPZ QoI manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ns100_matern52")
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--ntest", type=int, required=True)
    parser.add_argument("--latent-rank", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=("tanh", "gelu", "relu", "silu"), default="tanh")
    parser.add_argument("--no-time-feature", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--target-max-abs", type=float, default=0.9)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--test-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--stop-test-rel-below", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-stamp", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    train, test, normalization = make_train_test_tensors(
        args.manifest_path,
        ntrain=args.ntrain,
        ntest=args.ntest,
        normalize=not args.no_normalize,
        target_max_abs=args.target_max_abs,
        torch_dtype=torch_dtype,
    )
    model_config = LDNetConfig(
        latent_dim=args.latent_rank,
        input_dim=train.input_dimension,
        qoi_dim=train.qoi_dimension,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        activation=args.activation,
        include_time=not args.no_time_feature,
        residual_scale=args.residual_scale,
    )
    run_stamp = args.run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.output_dir
        / f"{args.dataset_name}_ldnet_r{args.latent_rank}_ntrain{args.ntrain}_ntest{args.ntest}_{run_stamp}"
    )
    train_config = LDNetTrainConfig(
        output_dir=run_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        test_every=args.test_every,
        checkpoint_every=args.checkpoint_every,
        stop_test_rel_below=args.stop_test_rel_below,
        seed=args.seed,
        device=args.device,
    )
    model = LDNetModel(model_config)
    result = train_ldnet(model, train, test, train_config, normalization=normalization)
    print(json.dumps({
        "run_dir": str(result.output_dir),
        "summary_path": str(result.summary_path),
        "metrics_path": str(result.metrics_path),
        "best_checkpoint_path": str(result.best_checkpoint_path),
        "best_epoch": result.best_epoch,
        "best_test_relative_error": result.best_test_relative_error,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
