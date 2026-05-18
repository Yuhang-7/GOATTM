from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-goattm-coastal-gauge"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


REPO = Path("/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT")
sys.path.insert(0, str(REPO / "src"))

from goattm.data import load_npz_qoi_sample  # noqa: E402
from goattm.train.plot_qoi_predictions import (  # noqa: E402
    PlotConfig,
    denormalize_qoi,
    evaluate_sample_predictions,
    load_run_artifacts,
)


RUN_DIR = Path(
    "/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/outputs/"
    "production_1344_v1v_init_v1v2_train_lbfgs/"
    "20260513_141424_swe_skewcp_n1344_lbfgs_v1vinit_v1v2_r50_R100_r50_R100/"
    "swe_skewcp_smoke_r50_R100_20260513_163947/runs/"
    "swe_skewcp_lbfgs_r50_R100_ntrain1344_ntest384_20260513_164044_e3d910ce"
)
OUT_DIR = RUN_DIR / "diagnostic_plots_current" / "coastal_gauge_30_train_30_test"
N_PER_SPLIT = 30
PANELS_PER_PAGE = 10


def latest_iteration_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints").glob("iter_*.npz"))
    if candidates:
        return candidates[-1]
    return run_dir / "checkpoints" / "latest.npz"


def choose_coastal_gauge(sample_path: Path) -> tuple[int, np.ndarray]:
    with np.load(sample_path, allow_pickle=True) as data:
        sensors = np.asarray(data["sensor_locations"], dtype=np.float64)
    if sensors.ndim != 2 or sensors.shape[1] < 2:
        raise RuntimeError(f"Unexpected sensor_locations shape: {sensors.shape}")
    gauge_index = int(np.argmax(sensors[:, 0]))
    return gauge_index, sensors


def evaluate_cases(split: str, manifest, artifacts, gauge_index: int, count: int) -> list[dict]:
    rows: list[dict] = []
    for sample_path in list(manifest.absolute_paths())[:count]:
        sample = load_npz_qoi_sample(sample_path)
        targets_norm, preds_norm = evaluate_sample_predictions(
            sample=sample,
            dynamics=artifacts.dynamics,
            decoder=artifacts.decoder,
            time_integrator=artifacts.time_integrator,
            max_dt=artifacts.max_dt,
        )
        targets = denormalize_qoi(targets_norm, artifacts.normalization_stats, True)
        preds = denormalize_qoi(preds_norm, artifacts.normalization_stats, True)
        truth = np.asarray(targets[:, gauge_index], dtype=np.float64)
        pred = np.asarray(preds[:, gauge_index], dtype=np.float64)
        rel_l2 = float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-14))
        rows.append(
            {
                "split": split,
                "sample_id": str(sample.sample_id),
                "sample_path": str(sample_path),
                "times": np.asarray(sample.observation_times, dtype=np.float64),
                "truth": truth,
                "prediction": pred,
                "relative_l2": rel_l2,
            }
        )
    return rows


def plot_split(rows: list[dict], split: str, gauge_label: str, checkpoint_label: str) -> dict:
    gauge_slug = gauge_label.lower().replace(" ", "")
    pdf_path = OUT_DIR / f"swe_skewcp_r50_R100_{split}_first30_{gauge_slug}_truth_vs_prediction.pdf"
    png_paths: list[str] = []
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(rows), PANELS_PER_PAGE):
            page_rows = rows[page_start : page_start + PANELS_PER_PAGE]
            fig, axes = plt.subplots(5, 2, figsize=(11.0, 13.0), sharex=False)
            axes_flat = axes.ravel()
            for ax in axes_flat[len(page_rows) :]:
                ax.axis("off")
            for local_idx, (ax, row) in enumerate(zip(axes_flat, page_rows), start=page_start + 1):
                ax.plot(row["times"], row["truth"], color="black", linewidth=2.1, label="Truth")
                ax.plot(row["times"], row["prediction"], color="#d62728", linewidth=2.0, linestyle="--", label="Prediction")
                ax.axhline(0.0, color="0.3", linewidth=0.7, linestyle=":", alpha=0.7)
                ax.set_title(f"{split.capitalize()} Case {local_idx:02d}: {row['sample_id']}", fontsize=11, pad=5)
                ax.grid(True, alpha=0.25, linewidth=0.7)
                ax.tick_params(labelsize=9)
                ax.text(
                    0.98,
                    0.92,
                    f"rel={row['relative_l2']:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
                )
                if local_idx == page_start + 1:
                    ax.legend(loc="upper left", frameon=False, fontsize=9)
            for ax in axes[:, 0]:
                ax.set_ylabel(f"{gauge_label} QoI", fontsize=10)
            for ax in axes[-1, :]:
                ax.set_xlabel("time", fontsize=10)
            page_num = page_start // PANELS_PER_PAGE + 1
            fig.suptitle(
                f"SWE skewCP r=50, R=100: {split.capitalize()} {gauge_label} Truth vs Prediction "
                f"({checkpoint_label}, page {page_num})",
                fontsize=14,
                y=0.995,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
            pdf.savefig(fig)
            png_path = OUT_DIR / (
                f"swe_skewcp_r50_R100_{split}_first30_{gauge_slug}"
                f"_truth_vs_prediction_page{page_num}.png"
            )
            fig.savefig(png_path, dpi=220)
            png_paths.append(str(png_path))
            plt.close(fig)
    return {"pdf": str(pdf_path), "png_pages": png_paths}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = latest_iteration_checkpoint(RUN_DIR)
    checkpoint_label = checkpoint_path.stem
    config = PlotConfig(
        run_dir=RUN_DIR.resolve(),
        checkpoint_mode="latest",
        checkpoint_path=checkpoint_path.resolve(),
        train_manifest_path=None,
        test_manifest_path=None,
        output_root=None,
        max_pages_per_split=1,
        use_raw_qoi_scale=True,
    )
    artifacts = load_run_artifacts(config)
    if artifacts.test_manifest is None:
        raise RuntimeError("Run artifacts do not include a test manifest")

    first_train = next(iter(artifacts.train_manifest.absolute_paths()))
    gauge_index, sensors = choose_coastal_gauge(Path(first_train))
    gauge_label = f"Gauge {gauge_index + 1}"

    train_rows = evaluate_cases("train", artifacts.train_manifest, artifacts, gauge_index, N_PER_SPLIT)
    test_rows = evaluate_cases("test", artifacts.test_manifest, artifacts, gauge_index, N_PER_SPLIT)
    train_plot = plot_split(train_rows, "train", gauge_label, checkpoint_label)
    test_plot = plot_split(test_rows, "test", gauge_label, checkpoint_label)

    cache_path = OUT_DIR / f"swe_skewcp_r50_R100_{gauge_label.lower().replace(' ', '')}_first30_train_test_qoi_data.npz"
    np.savez(
        cache_path,
        gauge_index_zero_based=gauge_index,
        gauge_label=gauge_label,
        sensor_locations=sensors,
        train_sample_id=np.array([row["sample_id"] for row in train_rows]),
        train_times=np.array([row["times"] for row in train_rows], dtype=object),
        train_truth=np.array([row["truth"] for row in train_rows], dtype=object),
        train_prediction=np.array([row["prediction"] for row in train_rows], dtype=object),
        train_relative_l2=np.array([row["relative_l2"] for row in train_rows]),
        test_sample_id=np.array([row["sample_id"] for row in test_rows]),
        test_times=np.array([row["times"] for row in test_rows], dtype=object),
        test_truth=np.array([row["truth"] for row in test_rows], dtype=object),
        test_prediction=np.array([row["prediction"] for row in test_rows], dtype=object),
        test_relative_l2=np.array([row["relative_l2"] for row in test_rows]),
        checkpoint_path=str(checkpoint_path),
    )

    summary = {
        "run_dir": str(RUN_DIR),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(OUT_DIR),
        "gauge_index_zero_based": gauge_index,
        "gauge_label": gauge_label,
        "gauge_location": sensors[gauge_index].tolist(),
        "train_plot": train_plot,
        "test_plot": test_plot,
        "cache_path": str(cache_path),
        "train_relative_l2_mean": float(np.mean([row["relative_l2"] for row in train_rows])),
        "train_relative_l2_median": float(np.median([row["relative_l2"] for row in train_rows])),
        "test_relative_l2_mean": float(np.mean([row["relative_l2"] for row in test_rows])),
        "test_relative_l2_median": float(np.median([row["relative_l2"] for row in test_rows])),
        "train_cases": [{"sample_id": row["sample_id"], "relative_l2": row["relative_l2"]} for row in train_rows],
        "test_cases": [{"sample_id": row["sample_id"], "relative_l2": row["relative_l2"]} for row in test_rows],
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
