from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_cascadia_packed import (  # noqa: E402
    MaskedCrossQuadraticReadoutDecoder,
    batch_from_payload,
    load_packed_payload,
    make_dynamics,
)
from quadrode_gpu_goattm import DenseLaggedMidpointStepper, QuadraticReadoutDecoder, ReducedObjective  # noqa: E402


def make_decoder(metadata: dict, output_dim: int, device: torch.device) -> torch.nn.Module:
    latent_dim = int(metadata["latent_dim"])
    mode = metadata.get("decoder_quadratic_mode", "full")
    if mode == "masked_cross":
        decoder = MaskedCrossQuadraticReadoutDecoder(
            latent_dim,
            output_dim,
            cross_terms=int(metadata.get("decoder_cross_terms", 0)),
            mask_seed=int(metadata.get("decoder_mask_seed", 20260705)),
            bias=True,
        )
    else:
        decoder = QuadraticReadoutDecoder(
            latent_dim,
            output_dim,
            include_quadratic=mode == "full",
            bias=True,
        )
    return decoder.double().to(device)


def raw_values(norm: torch.Tensor, qoi_stats: dict) -> torch.Tensor:
    scale = qoi_stats["scale"].to(device=norm.device, dtype=norm.dtype)
    mean = qoi_stats["mean"].to(device=norm.device, dtype=norm.dtype)
    return norm * scale + mean


def sample_raw_relative_errors(pred_norm: torch.Tensor, target_norm: torch.Tensor, qoi_stats: dict) -> torch.Tensor:
    pred_raw = raw_values(pred_norm, qoi_stats)
    target_raw = raw_values(target_norm, qoi_stats)
    residual = pred_raw - target_raw
    num = residual.square().sum(dim=(0, 2))
    den = target_raw.square().sum(dim=(0, 2)).clamp_min(1.0e-300)
    return torch.sqrt(num / den)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot true/predicted Cascadia QoI trajectories.")
    parser.add_argument("--test-packed", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scan-count", type=int, default=128)
    parser.add_argument("--qoi-indices", default="0,10,20,30,40,49")
    parser.add_argument("--picard-iters", type=int, default=2)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = dict(checkpoint["metadata"])
    payload, packed_metadata = load_packed_payload(Path(args.test_packed))
    output_dim = int(payload["qoi"].shape[-1])
    qoi_stats = payload["qoi_stats"]

    dynamics = make_dynamics(
        int(metadata["latent_dim"]),
        int(metadata["input_dim"]),
        0.01,
        device,
        linear_a=metadata.get("linear_a", "dense"),
        a_rank=int(metadata.get("linear_a_rank", metadata.get("a_rank", 20))),
        damping_init=0.1,
        damping_shift=0.0,
        quadratic=metadata.get("quadratic", "energy_dense"),
        h_reduced_rank=int(metadata.get("h_reduced_rank", 50)),
        h_tt_rank=int(metadata.get("h_tt_rank", 16)),
    )
    decoder = make_decoder(metadata, output_dim, device)
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    objective = ReducedObjective(
        dynamics,
        decoder,
        DenseLaggedMidpointStepper(picard_iters=int(args.picard_iters)),
        normal_chunk_size=4096,
        gradient_mode="lagged_adjoint",
    )

    scan_count = min(int(args.scan_count), int(payload["qoi"].shape[1]))
    scan_batch = batch_from_payload(payload, device, sample_indices=list(range(scan_count)))
    with torch.no_grad():
        scan_rollout = objective.rollout(scan_batch)
        scan_pred = decoder(scan_rollout.states)
    errs = sample_raw_relative_errors(scan_pred, scan_batch.qoi, qoi_stats).detach().cpu()
    order = torch.argsort(errs)
    chosen_scan = [
        int(order[0]),
        int(order[len(order) // 2]),
        int(order[-1]),
    ]
    chosen_labels = ["best", "median", "worst"]
    chosen_global = chosen_scan

    plot_batch = batch_from_payload(payload, device, sample_indices=chosen_global)
    with torch.no_grad():
        rollout = objective.rollout(plot_batch)
        pred_norm = decoder(rollout.states)
    pred_raw = raw_values(pred_norm, qoi_stats).detach().cpu()
    target_raw = raw_values(plot_batch.qoi, qoi_stats).detach().cpu()

    qoi_indices = [int(x) for x in args.qoi_indices.split(",") if x.strip()]
    original_qoi_indices = [3 * idx for idx in qoi_indices]
    physical_dt = float(packed_metadata.get("physical_step_size_seconds", 5.0))
    t_sec = torch.arange(target_raw.shape[0], dtype=torch.float64) * physical_dt

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(chosen_global),
        len(qoi_indices),
        figsize=(3.0 * len(qoi_indices), 2.25 * len(chosen_global)),
        sharex=True,
        squeeze=False,
    )
    for row, (sample_idx, label) in enumerate(zip(chosen_global, chosen_labels)):
        sample_id = str(payload["sample_ids"][sample_idx])
        for col, qidx in enumerate(qoi_indices):
            ax = axes[row][col]
            ax.plot(t_sec, target_raw[:, row, qidx], color="black", linewidth=1.25, label="true")
            ax.plot(t_sec, pred_raw[:, row, qidx], color="#d95f02", linewidth=1.05, linestyle="--", label="pred")
            if row == 0:
                ax.set_title(f"QoI {qidx} (orig {3*qidx})", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{label}\nidx {sample_idx}\nerr {errs[sample_idx]:.3f}", fontsize=8)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=8)
    axes[0][0].legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle("Cascadia test QoI: true vs learned prediction (raw scale)", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    panel_path = outdir / "qoi_true_vs_pred_best_median_worst.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # One easier-to-read plot for the median sample only.
    median_row = 1
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), sharex=True, squeeze=False)
    for ax, qidx in zip(axes.ravel(), qoi_indices):
        ax.plot(t_sec, target_raw[:, median_row, qidx], color="black", linewidth=1.5, label="true")
        ax.plot(t_sec, pred_raw[:, median_row, qidx], color="#d95f02", linewidth=1.25, linestyle="--", label="pred")
        ax.set_title(f"QoI {qidx} (orig {3*qidx})", fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=9)
    axes[0][0].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(
        f"Median test sample idx {chosen_global[median_row]}: raw QoI relerr {errs[chosen_global[median_row]]:.3f}",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    median_path = outdir / "qoi_true_vs_pred_median_sample.png"
    fig.savefig(median_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "checkpoint": str(args.checkpoint),
        "test_packed": str(args.test_packed),
        "device": str(device),
        "scan_count": scan_count,
        "chosen_samples": [
            {
                "label": label,
                "test_index": int(idx),
                "sample_id": str(payload["sample_ids"][idx]),
                "raw_relative_error": float(errs[idx]),
            }
            for label, idx in zip(chosen_labels, chosen_global)
        ],
        "qoi_indices": qoi_indices,
        "original_qoi_indices": original_qoi_indices,
        "panel_plot": str(panel_path),
        "median_plot": str(median_path),
    }
    (outdir / "qoi_plot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
