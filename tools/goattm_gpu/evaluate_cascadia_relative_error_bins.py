from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch


DEFAULT_TOOLS_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm/tools")
TOOLS_ROOT = Path(os.environ.get("QUAD_GOATTM_TOOLS", str(DEFAULT_TOOLS_ROOT))).expanduser()
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from plot_cascadia_qoi_uniform_samples import (  # noqa: E402
    build_model,
    nested_get,
    per_sample_relative_errors,
)
from train_cascadia_packed import batch_from_payload, load_packed_payload  # noqa: E402


def bin_for_index(index: int, total: int, bins: int) -> int:
    return min(int(bins) - 1, int(math.floor(int(index) * int(bins) / int(total))))


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values, torch.tensor(float(q), dtype=values.dtype)))


def summarize_bins(rows: list[dict], bins: int) -> list[dict]:
    out = []
    for b in range(int(bins)):
        members = [row for row in rows if int(row["bin"]) == b]
        raw = torch.tensor([float(row["raw_relative_error"]) for row in members], dtype=torch.float64)
        norm = torch.tensor([float(row["normalized_relative_error"]) for row in members], dtype=torch.float64)
        first = int(members[0]["sample_index"]) if members else -1
        last = int(members[-1]["sample_index"]) if members else -1
        out.append(
            {
                "bin": b,
                "count": len(members),
                "sample_index_start": first,
                "sample_index_end": last,
                "raw_mean": float(raw.mean()) if raw.numel() else float("nan"),
                "raw_median": quantile(raw, 0.5),
                "raw_q75": quantile(raw, 0.75),
                "raw_q90": quantile(raw, 0.9),
                "raw_q95": quantile(raw, 0.95),
                "raw_max": float(raw.max()) if raw.numel() else float("nan"),
                "normalized_mean": float(norm.mean()) if norm.numel() else float("nan"),
                "normalized_median": quantile(norm, 0.5),
                "normalized_q75": quantile(norm, 0.75),
                "normalized_q90": quantile(norm, 0.9),
                "normalized_q95": quantile(norm, 0.95),
                "normalized_max": float(norm.max()) if norm.numel() else float("nan"),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_split(
    split_name: str,
    payload: dict,
    objective,
    device: torch.device,
    *,
    time_mode: str,
    bins: int,
    chunk_samples: int,
    write_sample_errors: bool,
    outdir: Path,
) -> tuple[list[dict], list[dict]]:
    total = int(payload["qoi"].shape[1])
    sample_rows: list[dict] = []
    for start in range(0, total, int(chunk_samples)):
        end = min(start + int(chunk_samples), total)
        indices = list(range(start, end))
        batch = batch_from_payload(payload, device, sample_indices=indices, time_mode=time_mode)
        with torch.no_grad():
            rollout = objective.rollout(batch)
            pred_norm = objective.decoder(rollout.states)
            raw_err, norm_err = per_sample_relative_errors(
                pred_norm,
                batch.qoi,
                batch.observation_times,
                payload["qoi_stats"],
            )
        for local_pos, sample_index in enumerate(indices):
            sample_rows.append(
                {
                    "split": split_name,
                    "sample_index": int(sample_index),
                    "sample_id": str(payload["sample_ids"][sample_index]),
                    "bin": bin_for_index(sample_index, total, int(bins)),
                    "raw_relative_error": float(raw_err[local_pos].detach().cpu()),
                    "normalized_relative_error": float(norm_err[local_pos].detach().cpu()),
                }
            )
        del batch, rollout, pred_norm, raw_err, norm_err
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bin_rows = summarize_bins(sample_rows, int(bins))
    for row in bin_rows:
        row["split"] = split_name
    if write_sample_errors:
        write_csv(
            outdir / f"{split_name}_sample_relative_errors.csv",
            sample_rows,
            ["split", "sample_index", "sample_id", "bin", "raw_relative_error", "normalized_relative_error"],
        )
    return sample_rows, bin_rows


def plot_bin_summary(outdir: Path, train_bins: list[dict], test_bins: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [int(row["bin"]) for row in train_bins]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), sharex=True)
    panels = [
        ("raw", "raw relative error"),
        ("normalized", "normalized relative error"),
    ]
    for col, (prefix, ylabel) in enumerate(panels):
        ax = axes[0][col]
        ax.plot(bins, [row[f"{prefix}_mean"] for row in train_bins], label="train mean", linewidth=1.2)
        ax.plot(bins, [row[f"{prefix}_median"] for row in train_bins], label="train median", linewidth=1.2)
        ax.plot(bins, [row[f"{prefix}_q90"] for row in train_bins], label="train q90", linewidth=1.2)
        ax.set_title(f"Train {ylabel} by packed-index bin")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1][col]
        ax.plot(bins, [row[f"{prefix}_mean"] for row in test_bins], label="test mean", linewidth=1.2)
        ax.plot(bins, [row[f"{prefix}_median"] for row in test_bins], label="test median", linewidth=1.2)
        ax.plot(bins, [row[f"{prefix}_q90"] for row in test_bins], label="test q90", linewidth=1.2)
        ax.set_title(f"Test {ylabel} by packed-index bin")
        ax.set_xlabel("bin")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = outdir / "relative_error_bin_summary.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute full train/test relative-error statistics by packed-index bin.")
    parser.add_argument("--train-packed", required=True)
    parser.add_argument("--test-packed", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--chunk-samples", type=int, default=128)
    parser.add_argument("--write-sample-errors", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_payload, _ = load_packed_payload(Path(args.train_packed))
    test_payload, _ = load_packed_payload(Path(args.test_packed))
    output_dim = int(train_payload["qoi"].shape[-1])
    metadata, objective = build_model(checkpoint, output_dim, device)
    time_mode = str(nested_get(metadata, "time_mode", "normalized"))

    train_samples, train_bins = evaluate_split(
        "train",
        train_payload,
        objective,
        device,
        time_mode=time_mode,
        bins=int(args.bins),
        chunk_samples=int(args.chunk_samples),
        write_sample_errors=bool(args.write_sample_errors),
        outdir=outdir,
    )
    test_samples, test_bins = evaluate_split(
        "test",
        test_payload,
        objective,
        device,
        time_mode=time_mode,
        bins=int(args.bins),
        chunk_samples=int(args.chunk_samples),
        write_sample_errors=bool(args.write_sample_errors),
        outdir=outdir,
    )
    bin_rows = train_bins + test_bins
    fieldnames = [
        "split",
        "bin",
        "count",
        "sample_index_start",
        "sample_index_end",
        "raw_mean",
        "raw_median",
        "raw_q75",
        "raw_q90",
        "raw_q95",
        "raw_max",
        "normalized_mean",
        "normalized_median",
        "normalized_q75",
        "normalized_q90",
        "normalized_q95",
        "normalized_max",
    ]
    write_csv(outdir / "relative_error_bins.csv", bin_rows, fieldnames)
    plot_path = plot_bin_summary(outdir, train_bins, test_bins)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_optimizer_step": int(checkpoint.get("optimizer_step", -1)),
        "device": str(device),
        "time_mode": time_mode,
        "bins": int(args.bins),
        "chunk_samples": int(args.chunk_samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "bin_csv": str(outdir / "relative_error_bins.csv"),
        "plot": str(plot_path),
        "train_bins": train_bins,
        "test_bins": test_bins,
    }
    (outdir / "relative_error_bins.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
