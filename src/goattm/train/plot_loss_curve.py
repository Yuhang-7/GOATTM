from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_metrics_jsonl(metrics_path: str | Path) -> list[dict[str, object]]:
    metrics_path = Path(metrics_path)
    records: list[dict[str, object]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    if not records:
        raise ValueError(f"No metric records found in {metrics_path}.")
    return records


def plot_loss_curve(metrics_path: str | Path, output_path: str | Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = Path(metrics_path)
    if output_path is None:
        output_path = metrics_path.with_name("loss_curve.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_metrics_jsonl(metrics_path)
    iterations = np.asarray([int(record["iteration"]) for record in records], dtype=int)
    train_objective = np.asarray([float(record["train_objective"]) for record in records], dtype=float)
    train_data_loss = np.asarray([float(record["train_data_loss"]) for record in records], dtype=float)
    train_decoder_reg = np.asarray([float(record["train_decoder_regularization_loss"]) for record in records], dtype=float)
    gradient_norm = np.asarray([float(record["gradient_norm"]) for record in records], dtype=float)
    step_norm = np.asarray([float(record["step_norm"]) for record in records], dtype=float)
    test_data_loss = np.asarray(
        [
            np.nan if record["test_data_loss"] is None else float(record["test_data_loss"])
            for record in records
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)

    axes[0].plot(iterations, train_objective, marker="o", linewidth=2.0, label="train objective")
    axes[0].plot(iterations, train_data_loss, marker="s", linewidth=1.8, label="train data loss")
    axes[0].plot(iterations, train_decoder_reg, marker="^", linewidth=1.8, label="decoder regularization")
    if not np.all(np.isnan(test_data_loss)):
        axes[0].plot(iterations, test_data_loss, marker="d", linewidth=1.8, label="test data loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("GOATTM Training Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(iterations, gradient_norm, marker="o", linewidth=2.0, label="gradient norm")
    axes[1].plot(iterations, step_norm, marker="s", linewidth=1.8, label="step norm")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for axis, values in ((axes[0], train_objective), (axes[1], gradient_norm)):
        positive_mask = np.isfinite(values) & (values > 0.0)
        if np.any(positive_mask):
            axis.set_yscale("log")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot GOATTM training loss curves from metrics.jsonl.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.jsonl.")
    parser.add_argument("--output", default=None, help="Optional output PNG path.")
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    output_path = plot_loss_curve(metrics_path=args.metrics, output_path=args.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
