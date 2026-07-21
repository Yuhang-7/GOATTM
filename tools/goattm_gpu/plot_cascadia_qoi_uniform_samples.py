from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import torch


DEFAULT_TOOLS_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm/tools")
TOOLS_ROOT = Path(os.environ.get("QUAD_GOATTM_TOOLS", str(DEFAULT_TOOLS_ROOT))).expanduser()
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from train_cascadia_packed import (  # noqa: E402
    MaskedCrossQuadraticReadoutDecoder,
    batch_from_payload,
    load_packed_payload,
    make_dynamics,
)
from quadrode_gpu_goattm import (  # noqa: E402
    DenseLaggedMidpointStepper,
    QuadraticReadoutDecoder,
    ReducedObjective,
    RungeKutta4Stepper,
    SubstepRungeKutta4Stepper,
    trapezoidal_weights,
)


_PLOT_STATE = {}


def nested_get(metadata: dict, key: str, default=None):
    if key in metadata:
        return metadata[key]
    optimizer = metadata.get("optimizer", {})
    if isinstance(optimizer, dict) and key in optimizer:
        return optimizer[key]
    return default


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


def make_stepper(metadata: dict):
    stepper_name = str(nested_get(metadata, "stepper", "lagged"))
    if stepper_name == "lagged":
        return DenseLaggedMidpointStepper(picard_iters=int(nested_get(metadata, "picard_iters", 2))), "lagged_adjoint"
    if stepper_name == "rk4":
        return RungeKutta4Stepper(), "rk4_adjoint"
    if stepper_name == "rk4_substep":
        return SubstepRungeKutta4Stepper(substeps=int(nested_get(metadata, "rk4_substeps", 1) or 1)), "rk4_adjoint"
    raise ValueError(f"unknown stepper {stepper_name!r}")


def build_model(checkpoint: dict, output_dim: int, device: torch.device) -> tuple[dict, ReducedObjective]:
    metadata = dict(checkpoint["metadata"])
    dynamics = make_dynamics(
        int(metadata["latent_dim"]),
        int(metadata["input_dim"]),
        0.01,
        device,
        linear_a=metadata.get("linear_a", "dense"),
        a_rank=int(metadata.get("linear_a_rank", metadata.get("a_rank", 20)) or 20),
        damping_init=0.1,
        damping_shift=float(nested_get(metadata, "a_damping_shift", 1.0)),
        quadratic=metadata.get("quadratic", "energy_dense"),
        h_reduced_rank=int(metadata.get("h_reduced_rank", 50) or 50),
        h_tt_rank=int(metadata.get("h_tt_rank", 16) or 16),
    )
    decoder = make_decoder(metadata, output_dim, device)
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    dynamics.eval()
    decoder.eval()
    stepper, gradient_mode = make_stepper(metadata)
    objective = ReducedObjective(
        dynamics,
        decoder,
        stepper,
        decoder_ridge=float(nested_get(metadata, "decoder_ridge", 0.0)),
        dynamics_ridge=float(nested_get(metadata, "dynamics_ridge", 0.0)),
        normal_chunk_size=int(nested_get(metadata, "normal_chunk_size", 4096)),
        gradient_mode=gradient_mode,
    )
    return metadata, objective


def raw_values(norm: torch.Tensor, qoi_stats: dict) -> torch.Tensor:
    scale = qoi_stats["scale"].to(device=norm.device, dtype=norm.dtype)
    mean = qoi_stats["mean"].to(device=norm.device, dtype=norm.dtype)
    return norm * scale + mean


def per_sample_relative_errors(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    observation_times: torch.Tensor,
    qoi_stats: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = trapezoidal_weights(observation_times).to(device=target_norm.device, dtype=target_norm.dtype)
    residual_norm = pred_norm - target_norm
    norm_num = (residual_norm.square().sum(dim=-1) * weights[:, None]).sum(dim=0)
    norm_den = (target_norm.square().sum(dim=-1) * weights[:, None]).sum(dim=0).clamp_min(1.0e-300)
    scale = qoi_stats["scale"].to(device=target_norm.device, dtype=target_norm.dtype)
    mean = qoi_stats["mean"].to(device=target_norm.device, dtype=target_norm.dtype)
    residual_raw = residual_norm * scale
    target_raw = target_norm * scale + mean
    raw_num = (residual_raw.square().sum(dim=-1) * weights[:, None]).sum(dim=0)
    raw_den = (target_raw.square().sum(dim=-1) * weights[:, None]).sum(dim=0).clamp_min(1.0e-300)
    return torch.sqrt(raw_num / raw_den), torch.sqrt(norm_num / norm_den)


def uniform_bin_indices(total: int, bins: int) -> list[dict]:
    rows = []
    for bin_idx in range(int(bins)):
        start = int(math.floor(bin_idx * total / bins))
        end = int(math.floor((bin_idx + 1) * total / bins))
        if end <= start:
            end = min(start + 1, total)
        index = start + max(0, (end - start - 1) // 2)
        rows.append({"bin": bin_idx, "bin_start": start, "bin_end": end, "index": int(index)})
    return rows


def uniform_sample_indices(total: int, count: int) -> list[dict]:
    rows = []
    count = min(int(count), int(total))
    for sample_rank in range(count):
        start = int(math.floor(sample_rank * total / count))
        end = int(math.floor((sample_rank + 1) * total / count))
        if end <= start:
            end = min(start + 1, total)
        index = start + max(0, (end - start - 1) // 2)
        rows.append({"sample_rank": sample_rank, "bin": sample_rank, "bin_start": start, "bin_end": end, "index": int(index)})
    return rows


def uniform_qoi_indices(output_dim: int, qoi_count: int) -> list[int]:
    count = min(int(qoi_count), int(output_dim))
    if count >= output_dim:
        return list(range(output_dim))
    if count == 1:
        return [0]
    return [int(round(i * (output_dim - 1) / (count - 1))) for i in range(count)]


def evaluate_split(payload: dict, objective: ReducedObjective, indices: list[int], device: torch.device, time_mode: str):
    batch = batch_from_payload(payload, device, sample_indices=indices, time_mode=time_mode)
    with torch.no_grad():
        rollout = objective.rollout(batch)
        pred_norm = objective.decoder(rollout.states)
        raw_err, norm_err = per_sample_relative_errors(pred_norm, batch.qoi, batch.observation_times, payload["qoi_stats"])
    pred_raw = raw_values(pred_norm, payload["qoi_stats"]).detach().cpu()
    target_raw = raw_values(batch.qoi, payload["qoi_stats"]).detach().cpu()
    return batch, pred_raw, target_raw, raw_err.detach().cpu(), norm_err.detach().cpu()


def _init_plot_worker(state: dict) -> None:
    global _PLOT_STATE
    _PLOT_STATE = state


def _plot_one_sample(task: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = _PLOT_STATE
    split_name = str(state["split_name"])
    pred_raw = state["pred_raw"]
    target_raw = state["target_raw"]
    qoi_indices = state["qoi_indices"]
    time_seconds = state["time_seconds"]
    sample_ids = state["sample_ids"]
    split_dir = Path(state["split_dir"])
    row = task["row"]
    local_pos = int(task["local_pos"])
    raw_value = float(task["raw_relative_error"])
    norm_value = float(task["normalized_relative_error"])
    sample_index = int(row["index"])
    sample_id = str(sample_ids[sample_index])

    cols = int(state["cols"])
    rows_count = int(state["rows_count"])
    y_margin = float(state["y_margin"])
    fig, axes = plt.subplots(rows_count, cols, figsize=(3.25 * cols, 2.15 * rows_count), sharex=True, squeeze=False)
    true_scale = float(abs(target_raw[:, local_pos, qoi_indices]).max())
    if true_scale <= 0.0:
        true_scale = float(abs(pred_raw[:, local_pos, qoi_indices]).max())
    y_limit = max(true_scale * y_margin, 1.0e-12)
    for ax, qidx in zip(axes.ravel(), qoi_indices):
        ax.plot(time_seconds, target_raw[:, local_pos, qidx], color="black", linewidth=1.0, label="true")
        ax.plot(time_seconds, pred_raw[:, local_pos, qidx], color="#d95f02", linewidth=0.95, linestyle="--", label="pred")
        ax.set_ylim(-y_limit, y_limit)
        ax.set_title(f"QoI {qidx} / orig {3 * qidx}", fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(qoi_indices) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{split_name} bin {row['bin']:02d} | index {sample_index} | id {sample_id} | "
        f"raw rel {raw_value:.4f} | norm rel {norm_value:.4f}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    png_path = split_dir / f"{split_name}_sample{row['sample_rank']:03d}_idx{sample_index:05d}_{len(qoi_indices)}qoi_unified_y.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "split": split_name,
        "bin": int(row["bin"]),
        "sample_rank": int(row["sample_rank"]),
        "bin_start": int(row["bin_start"]),
        "bin_end": int(row["bin_end"]),
        "sample_index": sample_index,
        "sample_id": sample_id,
        "raw_relative_error": raw_value,
        "normalized_relative_error": norm_value,
        "true_y_limit_abs": y_limit,
        "png": str(png_path),
    }


def plot_split(
    *,
    split_name: str,
    outdir: Path,
    payload: dict,
    selected_rows: list[dict],
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    raw_err: torch.Tensor,
    norm_err: torch.Tensor,
    qoi_indices: list[int],
    physical_dt: float,
    plot_workers: int,
    write_pdf: bool,
    y_margin: float,
) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    split_dir = outdir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{split_name}_{len(selected_rows)}_uniform_samples_{len(qoi_indices)}qoi_unified_y.pdf"
    time_seconds = torch.arange(target_raw.shape[0], dtype=torch.float64).numpy() * float(physical_dt)
    cols = 5
    rows_count = int(math.ceil(len(qoi_indices) / cols))
    manifest: list[dict] = []
    tasks = [
        {
            "row": row,
            "local_pos": local_pos,
            "raw_relative_error": float(raw_err[local_pos]),
            "normalized_relative_error": float(norm_err[local_pos]),
        }
        for local_pos, row in enumerate(selected_rows)
    ]
    if int(plot_workers) > 1:
        state = {
            "split_name": split_name,
            "split_dir": str(split_dir),
            "pred_raw": pred_raw.numpy(),
            "target_raw": target_raw.numpy(),
            "qoi_indices": qoi_indices,
            "time_seconds": time_seconds,
            "sample_ids": list(payload["sample_ids"]),
            "cols": cols,
            "rows_count": rows_count,
            "y_margin": float(y_margin),
        }
        ctx = get_context("fork")
        with ctx.Pool(processes=int(plot_workers), initializer=_init_plot_worker, initargs=(state,)) as pool:
            manifest = list(pool.imap_unordered(_plot_one_sample, tasks))
        manifest.sort(key=lambda item: int(item["bin"]))
        if not write_pdf:
            return manifest

    with PdfPages(pdf_path) as pdf:
        for local_pos, row in enumerate(selected_rows):
            sample_index = int(row["index"])
            sample_id = str(payload["sample_ids"][sample_index])
            fig, axes = plt.subplots(rows_count, cols, figsize=(3.25 * cols, 2.15 * rows_count), sharex=True, squeeze=False)
            true_scale = float(target_raw[:, local_pos, qoi_indices].abs().max())
            if true_scale <= 0.0:
                true_scale = float(pred_raw[:, local_pos, qoi_indices].abs().max())
            y_limit = max(true_scale * float(y_margin), 1.0e-12)
            for ax, qidx in zip(axes.ravel(), qoi_indices):
                ax.plot(time_seconds, target_raw[:, local_pos, qidx], color="black", linewidth=1.0, label="true")
                ax.plot(time_seconds, pred_raw[:, local_pos, qidx], color="#d95f02", linewidth=0.95, linestyle="--", label="pred")
                ax.set_ylim(-y_limit, y_limit)
                ax.set_title(f"QoI {qidx} / orig {3 * qidx}", fontsize=8)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                ax.tick_params(labelsize=7)
            for ax in axes.ravel()[len(qoi_indices) :]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel("time (s)", fontsize=8)
            axes[0][0].legend(frameon=False, fontsize=7)
            fig.suptitle(
                f"{split_name} bin {row['bin']:02d} | index {sample_index} | id {sample_id} | "
                f"raw rel {float(raw_err[local_pos]):.4f} | norm rel {float(norm_err[local_pos]):.4f}",
                fontsize=12,
                y=0.995,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.965])
            png_path = split_dir / f"{split_name}_sample{row['sample_rank']:03d}_idx{sample_index:05d}_{len(qoi_indices)}qoi_unified_y.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            manifest.append(
                {
                    "split": split_name,
                    "bin": int(row["bin"]),
                    "sample_rank": int(row["sample_rank"]),
                    "bin_start": int(row["bin_start"]),
                    "bin_end": int(row["bin_end"]),
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "raw_relative_error": float(raw_err[local_pos]),
                    "normalized_relative_error": float(norm_err[local_pos]),
                    "true_y_limit_abs": y_limit,
                    "png": str(png_path),
                }
            )
    return manifest


def write_latex_document(outdir: Path, train_manifest: list[dict], test_manifest: list[dict], title: str) -> Path:
    def rel(path: str) -> str:
        return Path(path).relative_to(outdir).as_posix()

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.35in,landscape]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\usepackage{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        rf"\section*{{{title}}}",
        r"Each page uses one shared symmetric y-axis range for all QoI panels in that case. The y-limit is based on the true QoI amplitude for the selected QoIs, so tiny-amplitude QoIs are not visually over-weighted by autoscaling.",
    ]
    for split_name, manifest in [("Train", train_manifest), ("Test", test_manifest)]:
        lines.append(rf"\section*{{{split_name} samples}}")
        for row in sorted(manifest, key=lambda item: int(item["sample_rank"])):
            caption = (
                rf"{split_name} sample {int(row['sample_rank']):03d}, packed index {int(row['sample_index'])}, "
                rf"id {row['sample_id']}, raw rel. err. {float(row['raw_relative_error']):.4f}, "
                rf"norm rel. err. {float(row['normalized_relative_error']):.4f}, "
                rf"$|y|_{{\max}}$ {float(row['true_y_limit_abs']):.3g}."
            )
            lines.extend(
                [
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.98\textwidth,height=0.86\textheight,keepaspectratio]{{{rel(row['png'])}}}",
                    rf"\caption*{{{caption}}}",
                    r"\end{figure}",
                    r"\clearpage",
                ]
            )
    lines.append(r"\end{document}")
    path = outdir / "qoi_uniform_samples_report.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot_error_summary(outdir: Path, train_manifest: list[dict], test_manifest: list[dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_raw = [row["raw_relative_error"] for row in train_manifest]
    test_raw = [row["raw_relative_error"] for row in test_manifest]
    train_norm = [row["normalized_relative_error"] for row in train_manifest]
    test_norm = [row["normalized_relative_error"] for row in test_manifest]
    bins = [row["bin"] for row in train_manifest]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), sharex=True)
    axes[0].plot(bins, train_raw, marker="o", linewidth=1.2, label="train raw")
    axes[0].plot(bins, test_raw, marker="o", linewidth=1.2, label="test raw")
    axes[0].set_title("Raw relative error for selected samples")
    axes[0].set_xlabel("uniform bin")
    axes[0].set_ylabel("relative error")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(bins, train_norm, marker="o", linewidth=1.2, label="train normalized")
    axes[1].plot(bins, test_norm, marker="o", linewidth=1.2, label="test normalized")
    axes[1].set_title("Normalized relative error for selected samples")
    axes[1].set_xlabel("uniform bin")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = outdir / "uniform_40_sample_error_summary.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 40 uniform train/test Cascadia QoI prediction samples.")
    parser.add_argument("--train-packed", required=True)
    parser.add_argument("--test-packed", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--samples-per-split", type=int, default=40)
    parser.add_argument("--qois-per-sample", type=int, default=20)
    parser.add_argument("--plot-workers", type=int, default=1)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--no-latex", action="store_true")
    parser.add_argument("--y-margin", type=float, default=1.08)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_payload, train_meta = load_packed_payload(Path(args.train_packed))
    test_payload, test_meta = load_packed_payload(Path(args.test_packed))
    output_dim = int(train_payload["qoi"].shape[-1])
    metadata, objective = build_model(checkpoint, output_dim, device)
    time_mode = str(nested_get(metadata, "time_mode", "normalized"))
    physical_dt = float(train_meta.get("physical_step_size_seconds", test_meta.get("physical_step_size_seconds", 5.0)))

    train_rows = uniform_sample_indices(int(train_payload["qoi"].shape[1]), int(args.samples_per_split))
    test_rows = uniform_sample_indices(int(test_payload["qoi"].shape[1]), int(args.samples_per_split))
    qoi_indices = uniform_qoi_indices(output_dim, int(args.qois_per_sample))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_batch, train_pred, train_true, train_raw, train_norm = evaluate_split(
        train_payload,
        objective,
        [row["index"] for row in train_rows],
        device,
        time_mode,
    )
    del train_batch
    test_batch, test_pred, test_true, test_raw, test_norm = evaluate_split(
        test_payload,
        objective,
        [row["index"] for row in test_rows],
        device,
        time_mode,
    )
    del test_batch

    train_manifest = plot_split(
        split_name="train",
        outdir=outdir,
        payload=train_payload,
        selected_rows=train_rows,
        pred_raw=train_pred,
        target_raw=train_true,
        raw_err=train_raw,
        norm_err=train_norm,
        qoi_indices=qoi_indices,
        physical_dt=physical_dt,
        plot_workers=int(args.plot_workers),
        write_pdf=not bool(args.no_pdf),
        y_margin=float(args.y_margin),
    )
    test_manifest = plot_split(
        split_name="test",
        outdir=outdir,
        payload=test_payload,
        selected_rows=test_rows,
        pred_raw=test_pred,
        target_raw=test_true,
        raw_err=test_raw,
        norm_err=test_norm,
        qoi_indices=qoi_indices,
        physical_dt=physical_dt,
        plot_workers=int(args.plot_workers),
        write_pdf=not bool(args.no_pdf),
        y_margin=float(args.y_margin),
    )
    summary_plot = plot_error_summary(outdir, train_manifest, test_manifest)
    latex_path = None
    if not bool(args.no_latex):
        latex_path = write_latex_document(
            outdir,
            train_manifest,
            test_manifest,
            title=f"Cascadia QoI true vs predicted, checkpoint step {int(checkpoint.get('optimizer_step', -1))}",
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_optimizer_step": int(checkpoint.get("optimizer_step", -1)),
        "device": str(device),
        "train_packed": str(args.train_packed),
        "test_packed": str(args.test_packed),
        "time_mode": time_mode,
        "stepper": str(nested_get(metadata, "stepper", "unknown")),
        "samples_per_split": int(args.samples_per_split),
        "y_axis": "shared symmetric y-limit per case, based on selected true QoI max abs times y_margin",
        "y_margin": float(args.y_margin),
        "qoi_indices": qoi_indices,
        "original_qoi_indices": [3 * idx for idx in qoi_indices],
        "train_pdf": None if bool(args.no_pdf) else str(outdir / f"train_{len(train_manifest)}_uniform_samples_{len(qoi_indices)}qoi_unified_y.pdf"),
        "test_pdf": None if bool(args.no_pdf) else str(outdir / f"test_{len(test_manifest)}_uniform_samples_{len(qoi_indices)}qoi_unified_y.pdf"),
        "latex_file": None if latex_path is None else str(latex_path),
        "summary_plot": str(summary_plot),
        "train_selected": train_manifest,
        "test_selected": test_manifest,
        "train_raw_relative_error_mean": float(torch.tensor([r["raw_relative_error"] for r in train_manifest]).mean()),
        "test_raw_relative_error_mean": float(torch.tensor([r["raw_relative_error"] for r in test_manifest]).mean()),
        "train_normalized_relative_error_mean": float(torch.tensor([r["normalized_relative_error"] for r in train_manifest]).mean()),
        "test_normalized_relative_error_mean": float(torch.tensor([r["normalized_relative_error"] for r in test_manifest]).mean()),
    }
    (outdir / "uniform_qoi_plot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
