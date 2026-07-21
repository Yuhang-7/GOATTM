from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_cascadia_packed import (  # noqa: E402
    MaskedCrossQuadraticReadoutDecoder,
    batch_from_payload,
    load_packed_payload,
    make_dynamics,
)
from quadrode_gpu_goattm import DenseLaggedMidpointStepper, QuadraticReadoutDecoder, ReducedObjective, trapezoidal_weights  # noqa: E402


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    return list(range(int(rank), int(total), int(world_size)))


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


def per_sample_relative_errors(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    observation_times: torch.Tensor,
    qoi_stats: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = trapezoidal_weights(observation_times).to(device=target_norm.device, dtype=target_norm.dtype)
    residual_norm = prediction_norm - target_norm
    norm_num = (residual_norm.square().sum(dim=-1) * weights[:, None]).sum(dim=0)
    norm_den = (target_norm.square().sum(dim=-1) * weights[:, None]).sum(dim=0).clamp_min(1.0e-300)
    scale = qoi_stats["scale"].to(device=target_norm.device, dtype=target_norm.dtype)
    mean = qoi_stats["mean"].to(device=target_norm.device, dtype=target_norm.dtype)
    residual_raw = residual_norm * scale
    target_raw = target_norm * scale + mean
    raw_num = (residual_raw.square().sum(dim=-1) * weights[:, None]).sum(dim=0)
    raw_den = (target_raw.square().sum(dim=-1) * weights[:, None]).sum(dim=0).clamp_min(1.0e-300)
    return torch.sqrt(raw_num / raw_den), torch.sqrt(norm_num / norm_den)


def build_model(checkpoint: dict, output_dim: int, device: torch.device, damping_shift: float, picard_iters: int, normal_chunk_size: int):
    metadata = dict(checkpoint["metadata"])
    dynamics = make_dynamics(
        int(metadata["latent_dim"]),
        int(metadata["input_dim"]),
        0.01,
        device,
        linear_a=metadata.get("linear_a", "dense"),
        a_rank=int(metadata.get("linear_a_rank", metadata.get("a_rank", 20)) or 20),
        damping_init=0.1,
        damping_shift=float(damping_shift),
        quadratic=metadata.get("quadratic", "energy_dense"),
        h_reduced_rank=int(metadata.get("h_reduced_rank", 50) or 50),
        h_tt_rank=int(metadata.get("h_tt_rank", 16) or 16),
    )
    decoder = make_decoder(metadata, output_dim, device)
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    dynamics.eval()
    decoder.eval()
    objective = ReducedObjective(
        dynamics,
        decoder,
        DenseLaggedMidpointStepper(picard_iters=int(picard_iters)),
        normal_chunk_size=int(normal_chunk_size),
        gradient_mode="lagged_adjoint",
    )
    return metadata, objective, decoder


def write_error_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_index", "sample_id", "raw_relative_error", "normalized_relative_error"])
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(outdir: Path, raw_errors: torch.Tensor, norm_errors: torch.Tensor) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = raw_errors.cpu().numpy()
    norm = norm_errors.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].hist(raw, bins=60, color="#4c78a8", alpha=0.82)
    axes[0].axvline(raw.mean(), color="#d95f02", linewidth=1.4, label=f"mean {raw.mean():.4f}")
    axes[0].axvline(float(torch.median(raw_errors)), color="black", linewidth=1.2, linestyle="--", label=f"median {float(torch.median(raw_errors)):.4f}")
    axes[0].set_title("Raw QoI relative error per test sample")
    axes[0].set_xlabel("relative error")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False, fontsize=8)
    sorted_raw = torch.sort(raw_errors).values.cpu().numpy()
    sorted_norm = torch.sort(norm_errors).values.cpu().numpy()
    y = (torch.arange(raw_errors.numel(), dtype=torch.float64).numpy() + 1.0) / float(raw_errors.numel())
    axes[1].plot(sorted_raw, y, color="#4c78a8", linewidth=1.4, label="raw")
    axes[1].plot(sorted_norm, y, color="#59a14f", linewidth=1.4, label="normalized")
    axes[1].set_title("Empirical CDF")
    axes[1].set_xlabel("relative error")
    axes[1].set_ylabel("fraction")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = outdir / "test_error_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selected_samples(
    outdir: Path,
    payload: dict,
    objective: ReducedObjective,
    decoder: torch.nn.Module,
    selected_indices: list[int],
    qoi_choices: list[list[int]],
    error_by_index: dict[int, dict],
    *,
    device: torch.device,
    seed: int,
    write_pngs: bool,
    pdf_name: str = "selected_50_samples_true_vs_pred.pdf",
    plot_orders: list[int] | None = None,
    png_prefix: str = "",
) -> tuple[Path, list[dict]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    samples_dir = outdir / "selected_sample_pngs"
    if write_pngs:
        samples_dir.mkdir(parents=True, exist_ok=True)
    qoi_stats = payload["qoi_stats"]
    physical_dt = float(payload.get("metadata", {}).get("physical_step_size_seconds", 5.0))
    time_seconds = torch.arange(payload["qoi"].shape[0], dtype=torch.float64).numpy() * physical_dt
    batch = batch_from_payload(payload, device, sample_indices=selected_indices)
    with torch.no_grad():
        rollout = objective.rollout(batch)
        pred_norm = decoder(rollout.states)
    pred_raw = raw_values(pred_norm, qoi_stats).detach().cpu()
    target_raw = raw_values(batch.qoi, qoi_stats).detach().cpu()

    if plot_orders is None:
        plot_orders = list(range(len(selected_indices)))
    pdf_path = outdir / pdf_name
    manifest: list[dict] = []
    with PdfPages(pdf_path) as pdf:
        for row, sample_idx in enumerate(selected_indices):
            plot_order = int(plot_orders[row])
            sample_id = str(payload["sample_ids"][sample_idx])
            raw_err = float(error_by_index[sample_idx]["raw_relative_error"])
            norm_err = float(error_by_index[sample_idx]["normalized_relative_error"])
            fig, axes = plt.subplots(4, 5, figsize=(15.5, 10.0), sharex=True, squeeze=False)
            for ax, qidx in zip(axes.ravel(), qoi_choices[row]):
                ax.plot(time_seconds, target_raw[:, row, qidx], color="black", linewidth=1.1, label="true")
                ax.plot(time_seconds, pred_raw[:, row, qidx], color="#d95f02", linewidth=1.0, linestyle="--", label="pred")
                ax.set_title(f"QoI {qidx} (orig {3 * qidx})", fontsize=8)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                ax.tick_params(labelsize=7)
            for ax in axes[-1]:
                ax.set_xlabel("time (s)", fontsize=8)
            axes[0][0].legend(loc="best", fontsize=7, frameon=False)
            fig.suptitle(
                f"Test sample {sample_idx} | id {sample_id} | raw relerr {raw_err:.4f} | normalized relerr {norm_err:.4f}",
                fontsize=12,
                y=0.995,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.965])
            png_path = samples_dir / f"{png_prefix}sample_{plot_order:02d}_testidx_{sample_idx}_true_vs_pred.png"
            if write_pngs:
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            manifest.append(
                {
                    "plot_index": row,
                    "global_plot_index": plot_order,
                    "test_index": int(sample_idx),
                    "sample_id": sample_id,
                    "raw_relative_error": raw_err,
                    "normalized_relative_error": norm_err,
                    "qoi_indices": [int(x) for x in qoi_choices[row]],
                    "original_qoi_indices": [int(3 * x) for x in qoi_choices[row]],
                    "png": str(png_path) if write_pngs else None,
                    "seed": int(seed),
                }
            )
    return pdf_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Cascadia test error distribution and plot selected QoIs.")
    parser.add_argument("--test-packed", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-samples", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--qois-per-sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--picard-iters", type=int, default=2)
    parser.add_argument("--normal-chunk-size", type=int, default=4096)
    parser.add_argument("--a-damping-shift", type=float, default=1.0)
    parser.add_argument("--write-pngs", action="store_true")
    parser.add_argument("--parallel-plot", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    rank, world_size, _local_rank, device = setup_distributed()
    outdir = Path(args.output_dir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    payload, packed_metadata = load_packed_payload(Path(args.test_packed))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata, objective, decoder = build_model(
        checkpoint,
        int(payload["qoi"].shape[-1]),
        device,
        float(args.a_damping_shift),
        int(args.picard_iters),
        int(args.normal_chunk_size),
    )
    total = int(payload["qoi"].shape[1])
    local_indices = shard_indices(total, rank, world_size)
    local_rows: list[dict] = []
    for start in range(0, len(local_indices), int(args.chunk_samples)):
        indices = local_indices[start : start + int(args.chunk_samples)]
        batch = batch_from_payload(payload, device, sample_indices=indices)
        with torch.no_grad():
            rollout = objective.rollout(batch)
            pred = decoder(rollout.states)
            raw_err, norm_err = per_sample_relative_errors(pred, batch.qoi, batch.observation_times, payload["qoi_stats"])
        for idx, raw_value, norm_value in zip(indices, raw_err.detach().cpu(), norm_err.detach().cpu()):
            local_rows.append(
                {
                    "test_index": int(idx),
                    "sample_id": str(payload["sample_ids"][idx]),
                    "raw_relative_error": float(raw_value),
                    "normalized_relative_error": float(norm_value),
                }
            )
        del batch, rollout, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rank_path = outdir / f"errors_rank{rank:02d}.json"
    rank_path.write_text(json.dumps(local_rows), encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    rows: list[dict] = []
    for rr in range(world_size):
        rows.extend(json.loads((outdir / f"errors_rank{rr:02d}.json").read_text(encoding="utf-8")))
    rows.sort(key=lambda item: int(item["test_index"]))
    raw_errors = torch.tensor([float(row["raw_relative_error"]) for row in rows], dtype=torch.float64)
    norm_errors = torch.tensor([float(row["normalized_relative_error"]) for row in rows], dtype=torch.float64)
    error_by_index = {int(row["test_index"]): row for row in rows}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    selected = torch.randperm(total, generator=generator)[: int(args.sample_count)].tolist()
    qoi_choices = []
    output_dim = int(payload["qoi"].shape[-1])
    qoi_count = min(int(args.qois_per_sample), output_dim)
    for sample_idx in selected:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(args.seed) + 1009 * int(sample_idx))
        qoi_choices.append(torch.randperm(output_dim, generator=g)[:qoi_count].tolist())

    if rank == 0:
        write_error_csv(outdir / "test_sample_errors.csv", rows)
        dist_plot = plot_distribution(outdir, raw_errors, norm_errors)
    else:
        dist_plot = outdir / "test_error_distribution.png"

    if args.parallel_plot:
        local_positions = [idx for idx in range(len(selected)) if idx % world_size == rank]
        local_selected = [selected[idx] for idx in local_positions]
        local_qois = [qoi_choices[idx] for idx in local_positions]
        pdf_path, selected_manifest = plot_selected_samples(
            outdir,
            payload,
            objective,
            decoder,
            local_selected,
            local_qois,
            error_by_index,
            device=device,
            seed=int(args.seed),
            write_pngs=bool(args.write_pngs),
            pdf_name=f"selected_samples_rank{rank:02d}.pdf",
            plot_orders=local_positions,
            png_prefix=f"rank{rank:02d}_",
        )
        (outdir / f"selected_samples_rank{rank:02d}.json").write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "pdf": str(pdf_path),
                    "selected_samples": selected_manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif rank == 0:
        pdf_path, selected_manifest = plot_selected_samples(
            outdir,
            payload,
            objective,
            decoder,
            selected,
            qoi_choices,
            error_by_index,
            device=device,
            seed=int(args.seed),
            write_pngs=bool(args.write_pngs),
        )
        (outdir / "selected_samples_rank00.json").write_text(
            json.dumps({"rank": 0, "pdf": str(pdf_path), "selected_samples": selected_manifest}, indent=2),
            encoding="utf-8",
        )

    if args.parallel_plot and rank == 0:
        for rr in range(world_size):
            marker = outdir / f"selected_samples_rank{rr:02d}.json"
            while not marker.exists():
                time.sleep(1.0)

    if rank == 0:
        rank_manifests = [
            json.loads((outdir / f"selected_samples_rank{rr:02d}.json").read_text(encoding="utf-8"))
            for rr in range(world_size if args.parallel_plot else 1)
        ]
        selected_manifest = []
        pdf_paths = []
        for rank_manifest in rank_manifests:
            pdf_paths.append(rank_manifest["pdf"])
            selected_manifest.extend(rank_manifest["selected_samples"])
        selected_manifest.sort(key=lambda item: int(item.get("global_plot_index", item["plot_index"])))

        quantile_levels = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64)
        summary = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_optimizer_step": int(checkpoint.get("optimizer_step", -1)),
            "test_packed": str(args.test_packed),
            "packed_metadata": packed_metadata,
            "model_metadata": metadata,
            "world_size": int(world_size),
            "sample_count": total,
            "selected_sample_count": len(selected),
            "qois_per_sample": qoi_count,
            "seed": int(args.seed),
            "raw_relative_error": {
                "mean": float(raw_errors.mean()),
                "std": float(raw_errors.std(unbiased=False)),
                "quantiles": {str(float(q)): float(v) for q, v in zip(quantile_levels, torch.quantile(raw_errors, quantile_levels))},
            },
            "normalized_relative_error": {
                "mean": float(norm_errors.mean()),
                "std": float(norm_errors.std(unbiased=False)),
                "quantiles": {str(float(q)): float(v) for q, v in zip(quantile_levels, torch.quantile(norm_errors, quantile_levels))},
            },
            "distribution_plot": str(dist_plot),
            "selected_pdf": pdf_paths[0] if len(pdf_paths) == 1 else None,
            "selected_pdfs": pdf_paths,
            "selected_samples": selected_manifest,
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "selected_samples.json").write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
