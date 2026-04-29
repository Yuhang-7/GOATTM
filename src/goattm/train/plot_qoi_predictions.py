from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-goattm"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from goattm.data import NpzQoiSample, NpzSampleManifest, load_npz_qoi_sample, load_npz_sample_manifest
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.preprocess.normalization import DatasetNormalizationStats
from goattm.solvers import rollout_to_observation_times, validate_time_integrator


DynamicsLike = QuadraticDynamics | StabilizedQuadraticDynamics


@dataclass(frozen=True)
class PlotConfig:
    run_dir: Path
    checkpoint_mode: str
    checkpoint_path: Path | None
    train_manifest_path: Path | None
    test_manifest_path: Path | None
    output_root: Path | None
    max_pages_per_split: int
    use_raw_qoi_scale: bool


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    train_manifest: NpzSampleManifest
    test_manifest: NpzSampleManifest | None
    decoder: QuadraticDecoder
    dynamics: DynamicsLike
    normalization_stats: DatasetNormalizationStats | None
    time_integrator: str
    max_dt: float
    checkpoint_path: Path


def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(
        description="Plot true vs predicted QoI curves from a GOATTM training run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing config.json, preprocess.json, and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=("latest", "best"),
        default="latest",
        help="Which checkpoint inside checkpoints/ to use by default.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Overrides --checkpoint-mode.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Optional explicit train manifest path. Needed if preprocess.json does not record one.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=None,
        help="Optional explicit test manifest path. Overrides any inferred test manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to <run-dir>/qoi_prediction_plots/<checkpoint-mode-or-custom>.",
    )
    parser.add_argument(
        "--max-pages-per-split",
        type=int,
        default=100,
        help="Maximum number of sample-pages in each of the train/test PDFs.",
    )
    parser.add_argument(
        "--normalized-qoi",
        action="store_true",
        help="Plot normalized QoIs instead of mapping predictions back to original QoI scale.",
    )
    args = parser.parse_args()

    max_pages = int(args.max_pages_per_split)
    if max_pages <= 0:
        raise ValueError(f"--max-pages-per-split must be positive, got {max_pages}")

    return PlotConfig(
        run_dir=args.run_dir.resolve(),
        checkpoint_mode=str(args.checkpoint_mode),
        checkpoint_path=None if args.checkpoint_path is None else args.checkpoint_path.resolve(),
        train_manifest_path=None if args.train_manifest is None else args.train_manifest.resolve(),
        test_manifest_path=None if args.test_manifest is None else args.test_manifest.resolve(),
        output_root=None if args.output_root is None else args.output_root.resolve(),
        max_pages_per_split=max_pages,
        use_raw_qoi_scale=not bool(args.normalized_qoi),
    )


def load_run_artifacts(config: PlotConfig) -> RunArtifacts:
    run_dir = config.run_dir
    config_path = run_dir / "config.json"
    preprocess_path = run_dir / "preprocess.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        run_config = json.load(fh)

    preprocess = {"applied": False}
    if preprocess_path.exists():
        with preprocess_path.open("r", encoding="utf-8") as fh:
            preprocess = json.load(fh)

    checkpoint_path = resolve_checkpoint_path(run_dir, config)
    decoder, dynamics = load_checkpoint_models(checkpoint_path)
    train_manifest, test_manifest = resolve_manifests(config, preprocess)
    normalization_stats = resolve_normalization_stats(preprocess)
    if config.use_raw_qoi_scale and normalization_stats is None:
        warnings.warn(
            "Normalization stats are unavailable, so plots will remain on normalized/current manifest scale.",
            stacklevel=2,
        )

    return RunArtifacts(
        run_dir=run_dir,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        decoder=decoder,
        dynamics=dynamics,
        normalization_stats=normalization_stats,
        time_integrator=validate_time_integrator(str(run_config["time_integrator"])),
        max_dt=float(run_config["solver"]["max_dt"]),
        checkpoint_path=checkpoint_path,
    )


def resolve_checkpoint_path(run_dir: Path, config: PlotConfig) -> Path:
    if config.checkpoint_path is not None:
        return config.checkpoint_path
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"{config.checkpoint_mode}.npz"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_manifests(
    config: PlotConfig,
    preprocess: dict[str, object],
) -> tuple[NpzSampleManifest, NpzSampleManifest | None]:
    train_manifest_path = config.train_manifest_path
    test_manifest_path = config.test_manifest_path

    if train_manifest_path is None:
        train_manifest_value = preprocess.get("latent_train_manifest_path")
        if train_manifest_value is None:
            train_manifest_value = preprocess.get("normalized_train_manifest_path")
        if train_manifest_value is None:
            raise ValueError(
                "Could not infer the train manifest from preprocess.json. Pass --train-manifest explicitly."
            )
        train_manifest_path = Path(str(train_manifest_value))

    if test_manifest_path is None:
        test_manifest_value = preprocess.get("latent_test_manifest_path")
        if test_manifest_value is None:
            test_manifest_value = preprocess.get("normalized_test_manifest_path")
        if test_manifest_value is not None:
            test_manifest_path = Path(str(test_manifest_value))

    train_manifest = load_npz_sample_manifest(train_manifest_path)
    test_manifest = None if test_manifest_path is None else load_npz_sample_manifest(test_manifest_path)
    return train_manifest, test_manifest


def resolve_normalization_stats(preprocess: dict[str, object]) -> DatasetNormalizationStats | None:
    stats_path_value = preprocess.get("normalization_stats_path")
    if stats_path_value is None:
        return None
    stats_path = Path(str(stats_path_value))
    if not stats_path.exists():
        return None
    return load_normalization_stats(stats_path)


def load_normalization_stats(path: Path) -> DatasetNormalizationStats:
    with np.load(path, allow_pickle=True) as data:
        input_mean = _optional_float_array(data, "input_mean")
        input_std = _optional_float_array(data, "input_std")
        scale_mode = str(data["scale_mode"].item()) if "scale_mode" in data.files else "zscore"
        target_max_abs = float(np.asarray(data["target_max_abs"], dtype=np.float64)) if "target_max_abs" in data.files else 1.0
        qoi_centered_max_abs = _optional_float_array(data, "qoi_centered_max_abs")
        input_centered_max_abs = _optional_float_array(data, "input_centered_max_abs")
        return DatasetNormalizationStats(
            qoi_mean=np.asarray(data["qoi_mean"], dtype=np.float64),
            qoi_std=np.asarray(data["qoi_std"], dtype=np.float64),
            input_mean=input_mean,
            input_std=input_std,
            epsilon=float(np.asarray(data["epsilon"], dtype=np.float64)),
            scale_mode=scale_mode,
            target_max_abs=target_max_abs,
            qoi_centered_max_abs=qoi_centered_max_abs,
            input_centered_max_abs=input_centered_max_abs,
        )


def _optional_float_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    raw = data[key]
    if getattr(raw, "shape", None) == () and raw.dtype == object and raw.item() is None:
        return None
    return np.asarray(raw, dtype=np.float64)


def load_checkpoint_models(path: Path) -> tuple[QuadraticDecoder, DynamicsLike]:
    with np.load(path, allow_pickle=True) as data:
        decoder = QuadraticDecoder(
            v1=np.asarray(data["decoder_v1"], dtype=np.float64),
            v2=np.asarray(data["decoder_v2"], dtype=np.float64),
            v0=np.asarray(data["decoder_v0"], dtype=np.float64),
        )
        dynamics_type = str(data["dynamics_type"].item())
        mu_h = np.asarray(data["mu_h"], dtype=np.float64)
        c = np.asarray(data["c_vector"], dtype=np.float64)
        b = None if "b_matrix" not in data.files else np.asarray(data["b_matrix"], dtype=np.float64)
        if dynamics_type == "stabilized":
            dynamics = StabilizedQuadraticDynamics(
                s_params=np.asarray(data["s_params"], dtype=np.float64),
                w_params=np.asarray(data["w_params"], dtype=np.float64),
                mu_h=mu_h,
                c=c,
                b=b,
            )
        elif dynamics_type == "general":
            dynamics = QuadraticDynamics(
                a=np.asarray(data["a_matrix"], dtype=np.float64),
                mu_h=mu_h,
                c=c,
                b=b,
            )
        else:
            raise ValueError(f"Unsupported dynamics_type '{dynamics_type}' in checkpoint {path}")
    return decoder, dynamics


def denormalize_qoi(values: np.ndarray, stats: DatasetNormalizationStats | None, use_raw_qoi_scale: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not use_raw_qoi_scale or stats is None:
        return array
    return array * stats.qoi_std[None, :] + stats.qoi_mean[None, :]


def evaluate_sample_predictions(
    sample: NpzQoiSample,
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    time_integrator: str,
    max_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    explicit_dynamics = dynamics.explicit_dynamics if isinstance(dynamics, StabilizedQuadraticDynamics) else dynamics
    rollout, observation_indices = rollout_to_observation_times(
        dynamics=explicit_dynamics,
        u0=np.asarray(sample.u0, dtype=np.float64),
        observation_times=np.asarray(sample.observation_times, dtype=np.float64),
        max_dt=max_dt,
        input_function=sample.build_input_function(),
        time_integrator=time_integrator,
    )
    latent_states = np.asarray(rollout.states[observation_indices], dtype=np.float64)
    predictions = np.vstack([decoder.decode(state) for state in latent_states])
    targets = np.asarray(sample.qoi_observations, dtype=np.float64)
    return targets, predictions


def qoi_grid(output_dimension: int) -> tuple[int, int]:
    if output_dimension <= 4:
        cols = 2
    elif output_dimension <= 9:
        cols = 3
    else:
        cols = 4
    rows = int(math.ceil(output_dimension / cols))
    return rows, cols


def render_case_page(
    sample: NpzQoiSample,
    targets: np.ndarray,
    predictions: np.ndarray,
    split_name: str,
    pdf: PdfPages,
    page_index: int,
) -> None:
    output_dimension = targets.shape[1]
    rows, cols = qoi_grid(output_dimension)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.5 * rows), squeeze=False, sharex=True)
    axes_flat = axes.ravel()
    times = np.asarray(sample.observation_times, dtype=np.float64)
    residual = predictions - targets
    relative_l2 = float(np.linalg.norm(residual) / max(np.linalg.norm(targets), 1e-14))

    for qoi_idx in range(output_dimension):
        ax = axes_flat[qoi_idx]
        ax.plot(times, targets[:, qoi_idx], color="black", linewidth=1.8, label="true")
        ax.plot(times, predictions[:, qoi_idx], color="tab:red", linewidth=1.5, linestyle="--", label="pred")
        ax.set_title(f"QoI {qoi_idx}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if qoi_idx % cols == 0:
            ax.set_ylabel("QoI")
        if qoi_idx >= (rows - 1) * cols:
            ax.set_xlabel("time")
        if qoi_idx == 0:
            ax.legend(loc="best", fontsize=8)

    for extra_ax in axes_flat[output_dimension:]:
        extra_ax.axis("off")

    fig.suptitle(
        f"{split_name.upper()} sample {page_index + 1}: {sample.sample_id} | "
        f"relative L2 error = {relative_l2:.3e}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def write_split_pdf(
    split_name: str,
    manifest: NpzSampleManifest | None,
    artifacts: RunArtifacts,
    config: PlotConfig,
    output_root: Path,
) -> dict[str, object] | None:
    if manifest is None or len(manifest.sample_ids) == 0:
        return None

    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = split_dir / f"{split_name}_qoi_curves.pdf"

    plotted_ids: list[str] = []
    total_available = len(manifest.sample_ids)
    plotted_count = min(total_available, config.max_pages_per_split)

    with PdfPages(pdf_path) as pdf:
        for page_idx, sample_path in enumerate(manifest.absolute_paths()[:plotted_count]):
            sample = load_npz_qoi_sample(sample_path)
            targets_norm, predictions_norm = evaluate_sample_predictions(
                sample=sample,
                dynamics=artifacts.dynamics,
                decoder=artifacts.decoder,
                time_integrator=artifacts.time_integrator,
                max_dt=artifacts.max_dt,
            )
            targets = denormalize_qoi(targets_norm, artifacts.normalization_stats, config.use_raw_qoi_scale)
            predictions = denormalize_qoi(predictions_norm, artifacts.normalization_stats, config.use_raw_qoi_scale)
            render_case_page(
                sample=sample,
                targets=targets,
                predictions=predictions,
                split_name=split_name,
                pdf=pdf,
                page_index=page_idx,
            )
            plotted_ids.append(sample.sample_id)

    summary = {
        "split": split_name,
        "pdf_path": str(pdf_path),
        "total_available_samples": total_available,
        "plotted_samples": plotted_count,
        "truncated": total_available > plotted_count,
        "plotted_sample_ids": plotted_ids,
    }
    summary_path = split_dir / "plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def default_output_root(config: PlotConfig, checkpoint_path: Path) -> Path:
    if config.output_root is not None:
        return config.output_root
    checkpoint_label = checkpoint_path.stem
    return config.run_dir / "qoi_prediction_plots" / checkpoint_label


def plot_qoi_predictions_from_run_dir(config: PlotConfig) -> dict[str, object]:
    artifacts = load_run_artifacts(config)
    output_root = default_output_root(config, artifacts.checkpoint_path)
    output_root.mkdir(parents=True, exist_ok=True)

    train_summary = write_split_pdf("train", artifacts.train_manifest, artifacts, config, output_root)
    test_summary = write_split_pdf("test", artifacts.test_manifest, artifacts, config, output_root)

    combined_summary = {
        "run_dir": str(artifacts.run_dir),
        "checkpoint_path": str(artifacts.checkpoint_path),
        "output_root": str(output_root),
        "checkpoint_mode": config.checkpoint_mode,
        "max_pages_per_split": config.max_pages_per_split,
        "qoi_scale": "raw" if config.use_raw_qoi_scale else "normalized",
        "time_integrator": artifacts.time_integrator,
        "max_dt": artifacts.max_dt,
        "train": train_summary,
        "test": test_summary,
    }
    combined_summary_path = output_root / "plotting_summary.json"
    combined_summary_path.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return combined_summary


def plot_qoi_prediction_reports(
    run_dir: str | Path,
    checkpoint_mode: str = "latest",
    checkpoint_path: str | Path | None = None,
    train_manifest_path: str | Path | None = None,
    test_manifest_path: str | Path | None = None,
    output_root: str | Path | None = None,
    max_pages_per_split: int = 100,
    use_raw_qoi_scale: bool = True,
) -> dict[str, object]:
    config = PlotConfig(
        run_dir=Path(run_dir).resolve(),
        checkpoint_mode=str(checkpoint_mode),
        checkpoint_path=None if checkpoint_path is None else Path(checkpoint_path).resolve(),
        train_manifest_path=None if train_manifest_path is None else Path(train_manifest_path).resolve(),
        test_manifest_path=None if test_manifest_path is None else Path(test_manifest_path).resolve(),
        output_root=None if output_root is None else Path(output_root).resolve(),
        max_pages_per_split=int(max_pages_per_split),
        use_raw_qoi_scale=bool(use_raw_qoi_scale),
    )
    if config.max_pages_per_split <= 0:
        raise ValueError(f"max_pages_per_split must be positive, got {config.max_pages_per_split}")
    if config.checkpoint_mode not in {"latest", "best"}:
        raise ValueError(
            f"checkpoint_mode must be 'latest' or 'best' unless checkpoint_path is provided, got '{config.checkpoint_mode}'"
        )
    return plot_qoi_predictions_from_run_dir(config)


def main() -> None:
    summary = plot_qoi_predictions_from_run_dir(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
