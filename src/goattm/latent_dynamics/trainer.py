from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .data import LatentDynamicsDataset, LatentDynamicsNormalization
from .model import LDNetConfig, LDNetModel


@dataclass(frozen=True)
class LDNetTrainConfig:
    output_dir: str | Path
    max_epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    test_every: int = 10
    checkpoint_every: int = 100
    stop_test_rel_below: float | None = None
    seed: int = 20260603
    device: str = "cpu"


@dataclass(frozen=True)
class LDNetTrainResult:
    output_dir: Path
    metrics_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    summary_path: Path
    best_epoch: int
    best_test_relative_error: float


def relative_error_torch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.clamp(torch.sum(target**2), min=torch.finfo(target.dtype).tiny)
    return torch.sqrt(numerator / denominator)


@torch.no_grad()
def evaluate_ldnet(model: LDNetModel, dataset: LatentDynamicsDataset) -> dict[str, float]:
    model.eval()
    prediction = model(dataset.times, dataset.inputs, dataset.sample_count)
    mse = F.mse_loss(prediction, dataset.qoi, reduction="mean")
    rel = relative_error_torch(prediction, dataset.qoi)
    max_abs_state = torch.max(torch.abs(model.rollout(dataset.times, dataset.inputs, dataset.sample_count)))
    return {
        "mse": float(mse.detach().cpu()),
        "relative_error": float(rel.detach().cpu()),
        "max_abs_state": float(max_abs_state.detach().cpu()),
    }


def train_ldnet(
    model: LDNetModel,
    train: LatentDynamicsDataset,
    test: LatentDynamicsDataset,
    config: LDNetTrainConfig,
    normalization: LatentDynamicsNormalization | None = None,
) -> LDNetTrainResult:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    train = train.to(device)
    test = test.to(device)

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    if normalization is not None:
        normalization.to_npz(output_dir / "normalization_stats.npz")

    config_payload = {
        "model": model.config.to_dict(),
        "train": asdict(config) | {"output_dir": str(output_dir)},
        "data": {
            "ntrain": train.sample_count,
            "ntest": test.sample_count,
            "time_count": train.time_count,
            "qoi_dimension": train.qoi_dimension,
            "input_dimension": train.input_dimension,
            "time_first": float(train.times[0].detach().cpu()),
            "time_last": float(train.times[-1].detach().cpu()),
            "normalization": normalization is not None,
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    best_epoch = -1
    best_test_rel = float("inf")
    start = perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(config.max_epochs + 1):
            model.train()
            permutation = torch.randperm(train.sample_count, generator=generator, device=device)
            train_loss_sum = 0.0
            train_count = 0
            for start_index in range(0, train.sample_count, config.batch_size):
                batch_indices = permutation[start_index : start_index + config.batch_size]
                batch_inputs = None if train.inputs is None else train.inputs[batch_indices]
                batch_qoi = train.qoi[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(train.times, batch_inputs, int(batch_indices.numel()))
                loss = F.mse_loss(prediction, batch_qoi, reduction="mean")
                loss.backward()
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                train_loss_sum += float(loss.detach().cpu()) * int(batch_indices.numel())
                train_count += int(batch_indices.numel())

            should_eval = epoch % config.test_every == 0 or epoch == config.max_epochs
            if should_eval:
                train_metrics = evaluate_ldnet(model, train)
                test_metrics = evaluate_ldnet(model, test)
                is_best = test_metrics["relative_error"] < best_test_rel
                if is_best:
                    best_epoch = epoch
                    best_test_rel = test_metrics["relative_error"]
                record = {
                    "epoch": epoch,
                    "elapsed_seconds": perf_counter() - start,
                    "train_epoch_mse": train_loss_sum / max(train_count, 1),
                    "train_mse": train_metrics["mse"],
                    "train_relative_error": train_metrics["relative_error"],
                    "test_mse": test_metrics["mse"],
                    "test_relative_error": test_metrics["relative_error"],
                    "train_max_abs_state": train_metrics["max_abs_state"],
                    "test_max_abs_state": test_metrics["max_abs_state"],
                    "best_epoch": best_epoch,
                    "best_test_relative_error": best_test_rel,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(
                    f"[epoch {epoch:05d}] train_rel={record['train_relative_error']:.6e} "
                    f"test_rel={record['test_relative_error']:.6e} best={best_test_rel:.6e}",
                    flush=True,
                )
                _save_checkpoint(latest_checkpoint_path, model, epoch, record)
                if is_best:
                    _save_checkpoint(best_checkpoint_path, model, epoch, record)
                if config.stop_test_rel_below is not None and best_test_rel < config.stop_test_rel_below:
                    break
            elif epoch % config.checkpoint_every == 0:
                _save_checkpoint(latest_checkpoint_path, model, epoch, {"epoch": epoch})

    summary = {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_test_relative_error": best_test_rel,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return LDNetTrainResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        latest_checkpoint_path=latest_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        summary_path=summary_path,
        best_epoch=best_epoch,
        best_test_relative_error=best_test_rel,
    )


def _save_checkpoint(path: Path, model: LDNetModel, epoch: int, metrics: dict[str, float | int]) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )
