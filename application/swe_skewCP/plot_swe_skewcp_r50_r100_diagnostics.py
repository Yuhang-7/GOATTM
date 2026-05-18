from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-goattm-skewcp-r50"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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
OUT_DIR = RUN_DIR / "diagnostic_plots_current"
GAUGE_INDEX = 10
GAUGE_LABEL = f"Gauge {GAUGE_INDEX + 1}"
FIXED_TRAIN_SAMPLE_ID = "sample_000137"


@dataclass(frozen=True)
class SampleScore:
    split: str
    sample_id: str
    sample_path: str
    amplitude: float
    signed_peak: float
    extrema_count: int
    total_variation: float
    complexity_score: float


def moving_average(values: np.ndarray, width: int = 9) -> np.ndarray:
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values, kernel, mode="same")


def score_curve(values: np.ndarray) -> tuple[float, float, int, float, float]:
    y = np.asarray(values, dtype=np.float64)
    y_smooth = moving_average(y, width=11)
    amplitude = float(np.max(np.abs(y_smooth)))
    signed_peak = float(y_smooth[np.argmax(np.abs(y_smooth))]) if y_smooth.size else 0.0
    if amplitude <= 1e-14:
        return 0.0, 0.0, 0, 0.0, 0.0
    dy = np.diff(y_smooth)
    threshold = 0.015 * amplitude
    signs = np.sign(dy)
    signs[np.abs(dy) < threshold] = 0.0
    nz = signs[signs != 0.0]
    extrema_count = int(np.sum(nz[1:] * nz[:-1] < 0.0)) if nz.size > 1 else 0
    total_variation = float(np.sum(np.abs(np.diff(y_smooth))) / amplitude)
    complexity_score = float(extrema_count + 0.25 * total_variation + 0.20 * math.log1p(amplitude))
    return amplitude, signed_peak, extrema_count, total_variation, complexity_score


def score_manifest(split: str, manifest, artifacts) -> list[SampleScore]:
    scores: list[SampleScore] = []
    for sample_path in manifest.absolute_paths():
        sample = load_npz_qoi_sample(sample_path)
        qoi = denormalize_qoi(sample.qoi_observations, artifacts.normalization_stats, True)
        curve = qoi[:, GAUGE_INDEX]
        amplitude, signed_peak, extrema_count, total_variation, complexity_score = score_curve(curve)
        scores.append(
            SampleScore(
                split=split,
                sample_id=str(sample.sample_id),
                sample_path=str(sample_path),
                amplitude=amplitude,
                signed_peak=signed_peak,
                extrema_count=extrema_count,
                total_variation=total_variation,
                complexity_score=complexity_score,
            )
        )
    return scores


def choose_samples(train_scores: list[SampleScore], test_scores: list[SampleScore]) -> list[SampleScore]:
    train_by_id = {item.sample_id: item for item in train_scores}
    selected: list[SampleScore] = []
    if FIXED_TRAIN_SAMPLE_ID in train_by_id:
        selected.append(train_by_id[FIXED_TRAIN_SAMPLE_ID])

    train_complex = sorted(
        [item for item in train_scores if item.amplitude >= 0.30 and item.sample_id != FIXED_TRAIN_SAMPLE_ID],
        key=lambda item: item.complexity_score,
        reverse=True,
    )
    train_simple = sorted(
        [item for item in train_scores if item.amplitude >= 0.15 and item.sample_id not in {s.sample_id for s in selected}],
        key=lambda item: (item.complexity_score, -item.amplitude),
    )
    for item in train_complex:
        if item.sample_id not in {s.sample_id for s in selected}:
            selected.append(item)
        if len(selected) == 2:
            break
    for item in train_simple:
        if item.sample_id not in {s.sample_id for s in selected}:
            selected.append(item)
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise RuntimeError("Could not select four train samples")

    test_complex = sorted(
        [item for item in test_scores if item.amplitude >= 0.30],
        key=lambda item: item.complexity_score,
        reverse=True,
    )
    if not test_complex:
        test_complex = sorted(test_scores, key=lambda item: item.complexity_score, reverse=True)
    test_simple = sorted(
        [item for item in test_scores if item.amplitude >= 0.15 and item.sample_id != test_complex[0].sample_id],
        key=lambda item: (item.complexity_score, -item.amplitude),
    )
    if not test_simple:
        test_simple = sorted(
            [item for item in test_scores if item.sample_id != test_complex[0].sample_id],
            key=lambda item: (item.complexity_score, -item.amplitude),
        )
    selected.extend([test_complex[0], test_simple[0]])
    return selected


def latest_iteration_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints").glob("iter_*.npz"))
    if not candidates:
        return run_dir / "checkpoints" / "latest.npz"
    return candidates[-1]


def evaluate_selected(selected: list[SampleScore], artifacts):
    rows = []
    for item in selected:
        sample = load_npz_qoi_sample(Path(item.sample_path))
        targets_norm, preds_norm = evaluate_sample_predictions(
            sample=sample,
            dynamics=artifacts.dynamics,
            decoder=artifacts.decoder,
            time_integrator=artifacts.time_integrator,
            max_dt=artifacts.max_dt,
        )
        targets = denormalize_qoi(targets_norm, artifacts.normalization_stats, True)
        preds = denormalize_qoi(preds_norm, artifacts.normalization_stats, True)
        truth = targets[:, GAUGE_INDEX]
        pred = preds[:, GAUGE_INDEX]
        rel_l2 = float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-14))
        rows.append(
            {
                "score": item,
                "times": np.asarray(sample.observation_times, dtype=np.float64),
                "truth": truth,
                "prediction": pred,
                "relative_l2_gauge13": rel_l2,
            }
        )
    return rows


def plot_qoi_rows(rows, output_prefix: Path, checkpoint_label: str) -> None:
    plot_titles = ["Train Case A", "Train Case B", "Train Case C", "Train Case D", "Test Case A", "Test Case B"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.2), sharex=True)
    axes_flat = axes.ravel()
    for panel_index, (ax, row, title) in enumerate(zip(axes_flat, rows, plot_titles)):
        ax.plot(row["times"], row["truth"], color="black", linewidth=2.5, label="Truth")
        ax.plot(row["times"], row["prediction"], color="#d62728", linewidth=2.4, linestyle="--", label="Prediction")
        ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":", alpha=0.65)
        ax.set_title(title, fontsize=13, pad=7)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.tick_params(labelsize=11)
        if panel_index == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{GAUGE_LABEL} QoI", fontsize=12)
    for ax in axes[-1, :]:
        ax.set_xlabel("time", fontsize=12)
    fig.suptitle(f"SWE skewCP r=50, R=100: {GAUGE_LABEL} Truth vs Prediction ({checkpoint_label})", fontsize=15, y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def load_metrics(run_dir: Path) -> list[dict[str, float]]:
    records = []
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def arr(records: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in records], dtype=np.float64)


def plot_loss_and_regularization(records: list[dict[str, float]], output_prefix: Path) -> None:
    it = arr(records, "iteration")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2))

    axes[0].plot(it, arr(records, "train_relative_error"), linewidth=2.1, label="train rel error")
    axes[0].plot(it, arr(records, "test_relative_error"), linewidth=2.1, label="test rel error")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("relative error")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=10)

    axes[1].semilogy(it, arr(records, "train_data_loss"), linewidth=2.1, label="train data")
    axes[1].semilogy(it, arr(records, "test_data_loss"), linewidth=2.1, label="test data")
    axes[1].semilogy(it, arr(records, "train_objective"), linewidth=1.8, linestyle="--", label="train objective")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=10)

    axes[2].semilogy(it, arr(records, "train_decoder_regularization_loss"), linewidth=2.1, label="decoder reg")
    axes[2].semilogy(it, arr(records, "train_dynamics_regularization_loss"), linewidth=2.1, label="dynamics reg")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("regularization contribution")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, fontsize=10)

    fig.suptitle("SWE skewCP r=50, R=100: loss and regularization decay", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_operator_norms(records: list[dict[str, float]], output_prefix: Path) -> None:
    it = arr(records, "iteration")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for key, label in [
        ("dynamics_a_fro_norm", "|A|"),
        ("dynamics_h_fro_norm", "|H|"),
        ("dynamics_b_fro_norm", "|B|"),
        ("dynamics_c_l2_norm", "|c|"),
        ("dynamics_skew_cp_l2_norm", "skewCP param norm"),
        ("decoder_parameter_norm", "decoder param norm"),
    ]:
        values = arr(records, key)
        if np.all(~np.isfinite(values)):
            continue
        ax.semilogy(it, values, linewidth=2.0, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("norm")
    ax.set_title("SWE skewCP r=50, R=100: parameter norms")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


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
    train_scores = score_manifest("train", artifacts.train_manifest, artifacts)
    if artifacts.test_manifest is None:
        raise RuntimeError("Run artifacts do not include a test manifest")
    test_scores = score_manifest("test", artifacts.test_manifest, artifacts)
    selected = choose_samples(train_scores, test_scores)
    rows = evaluate_selected(selected, artifacts)

    qoi_prefix = OUT_DIR / f"swe_skewcp_r50_R100_{checkpoint_label}_gauge{GAUGE_INDEX + 1}_truth_vs_prediction_2x3"
    plot_qoi_rows(rows, qoi_prefix, checkpoint_label=checkpoint_label)

    records = load_metrics(RUN_DIR)
    loss_prefix = OUT_DIR / "swe_skewcp_r50_R100_loss_regularization_decay"
    plot_loss_and_regularization(records, loss_prefix)
    norm_prefix = OUT_DIR / "swe_skewcp_r50_R100_parameter_norm_decay"
    plot_operator_norms(records, norm_prefix)

    cache_path = OUT_DIR / f"swe_skewcp_r50_R100_{checkpoint_label}_gauge{GAUGE_INDEX + 1}_truth_vs_prediction_data.npz"
    np.savez(
        cache_path,
        times=np.vstack([row["times"] for row in rows]),
        truth=np.vstack([row["truth"] for row in rows]),
        prediction=np.vstack([row["prediction"] for row in rows]),
        split=np.array([row["score"].split for row in rows]),
        sample_id=np.array([row["score"].sample_id for row in rows]),
        sample_path=np.array([row["score"].sample_path for row in rows]),
        relative_l2_gauge13=np.array([row["relative_l2_gauge13"] for row in rows]),
        complexity_score=np.array([row["score"].complexity_score for row in rows]),
        extrema_count=np.array([row["score"].extrema_count for row in rows]),
        total_variation=np.array([row["score"].total_variation for row in rows]),
        checkpoint_path=str(checkpoint_path),
        gauge_index_zero_based=GAUGE_INDEX,
    )

    summary = {
        "run_dir": str(RUN_DIR),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "output_dir": str(OUT_DIR),
        "qoi_pdf": str(qoi_prefix.with_suffix(".pdf")),
        "loss_regularization_pdf": str(loss_prefix.with_suffix(".pdf")),
        "parameter_norm_pdf": str(norm_prefix.with_suffix(".pdf")),
        "cache_path": str(cache_path),
        "latest_metric": records[-1],
        "cases": [
            {
                **asdict(row["score"]),
                "relative_l2_gauge13": row["relative_l2_gauge13"],
            }
            for row in rows
        ],
    }
    summary_path = OUT_DIR / "plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
