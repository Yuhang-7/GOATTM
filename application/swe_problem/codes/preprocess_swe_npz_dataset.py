from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.data import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest  # noqa: E402


INPUT_ROOT_ENV_VAR = "SWE_ORIGINAL_DATA_ROOT"
CURRENT_ORIGINAL_DATA_ROOT = Path("/storage/yuhang/swedata/originaldata/swe_data_2026_510510")
DEFAULT_OUTPUT_ROOT = THIS_FILE.parents[1] / "data" / "processed_data"
DEFAULT_QOI_STRIDE = 5
INPUT_MODE_UPLIFT_PARAMETERS = "uplift_parameters"
INPUT_MODE_UPLIFT_TIME_DERIVATIVE = "uplift_time_derivative"
INPUT_MODE_CHOICES = (INPUT_MODE_UPLIFT_PARAMETERS, INPUT_MODE_UPLIFT_TIME_DERIVATIVE)


def default_input_root() -> Path:
    env_value = os.environ.get(INPUT_ROOT_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return CURRENT_ORIGINAL_DATA_ROOT


@dataclass(frozen=True)
class ConvertedSample:
    sample: NpzQoiSample
    sample_path: Path
    observation_count: int
    output_dimension: int
    input_dimension: int
    input_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SWE application NPZ files into GOATTM NpzQoiSample format."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Directory containing original per-sample subdirectories. "
            f"Defaults to ${INPUT_ROOT_ENV_VAR}, or {CURRENT_ORIGINAL_DATA_ROOT} if the environment variable is unset."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where converted samples and manifest will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of samples to process, for smoke testing.",
    )
    parser.add_argument(
        "--qoi-stride",
        type=int,
        default=DEFAULT_QOI_STRIDE,
        help=(
            "Keep every qoi_stride-th physical QoI time. The default keeps "
            "5, 10, ..., 1500 and prepends a zero QoI at t=0."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODE_CHOICES,
        default=INPUT_MODE_UPLIFT_PARAMETERS,
        help=(
            "Select the latent-dynamics input p(t). "
            f"'{INPUT_MODE_UPLIFT_PARAMETERS}' uses the original constant "
            "Gaussian uplift parameters [xi, yi, Ti, sigma_i, Hi]. "
            f"'{INPUT_MODE_UPLIFT_TIME_DERIVATIVE}' uses the same packet descriptors but replaces Hi "
            "with d/dt[Hi * phi(t; Ti)] evaluated at each observation time."
        ),
    )
    return parser.parse_args()


def discover_original_samples(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    sample_paths = sorted(input_root.glob("sample_*/*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No .npz samples found under {input_root}")
    return sample_paths


def _load_required_vector(npz_data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz_data.files:
        raise ValueError(f"Missing required field '{key}'")
    value = np.asarray(npz_data[key], dtype=np.float64)
    if value.ndim != 1:
        raise ValueError(f"Field '{key}' must be rank-1, got shape {value.shape}")
    return value


def _load_required_matrix(npz_data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz_data.files:
        raise ValueError(f"Missing required field '{key}'")
    value = np.asarray(npz_data[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"Field '{key}' must be rank-2, got shape {value.shape}")
    return value


def _mollifier_time_derivative(times: np.ndarray, ti: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    ti = np.asarray(ti, dtype=np.float64)
    if np.any(ti <= 0.0):
        raise ValueError("All Ti values must be positive to evaluate the uplift time derivative.")
    tau = times[:, None] / ti[None, :]
    active = (tau > 0.0) & (tau < 1.0)
    clipped_tau = np.clip(tau, 0.0, 1.0)
    derivative = 30.0 * clipped_tau**2 * (clipped_tau - 1.0) ** 2 / ti[None, :]
    return np.where(active, derivative, 0.0)


def build_input_values(
    observation_times: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    ti: np.ndarray,
    sigma_i: np.ndarray,
    hi: np.ndarray,
    input_mode: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if input_mode == INPUT_MODE_UPLIFT_PARAMETERS:
        constant_input = np.concatenate([xi, yi, ti, sigma_i, hi], axis=0)
        input_values = np.repeat(constant_input[None, :], observation_times.shape[0], axis=0)
        feature_names = (
            tuple(f"xi_{idx}" for idx in range(xi.shape[0]))
            + tuple(f"yi_{idx}" for idx in range(yi.shape[0]))
            + tuple(f"Ti_{idx}" for idx in range(ti.shape[0]))
            + tuple(f"sigma_i_{idx}" for idx in range(sigma_i.shape[0]))
            + tuple(f"Hi_{idx}" for idx in range(hi.shape[0]))
        )
        return input_values, feature_names
    if input_mode == INPUT_MODE_UPLIFT_TIME_DERIVATIVE:
        dhi_dt = hi[None, :] * _mollifier_time_derivative(observation_times, ti)
        packet_prefix = np.repeat(
            np.concatenate([xi, yi, ti, sigma_i], axis=0)[None, :],
            observation_times.shape[0],
            axis=0,
        )
        input_values = np.concatenate([packet_prefix, dhi_dt], axis=1)
        feature_names = (
            tuple(f"xi_{idx}" for idx in range(xi.shape[0]))
            + tuple(f"yi_{idx}" for idx in range(yi.shape[0]))
            + tuple(f"Ti_{idx}" for idx in range(ti.shape[0]))
            + tuple(f"sigma_i_{idx}" for idx in range(sigma_i.shape[0]))
            + tuple(f"dHi_dt_{idx}" for idx in range(hi.shape[0]))
        )
        return input_values, feature_names
    raise ValueError(f"Unknown input_mode '{input_mode}'. Expected one of {INPUT_MODE_CHOICES}.")


def convert_original_sample(
    original_path: Path,
    output_samples_root: Path,
    qoi_stride: int,
    input_mode: str,
) -> ConvertedSample:
    if qoi_stride <= 0:
        raise ValueError(f"qoi_stride must be positive, got {qoi_stride}")
    sample_id = original_path.stem
    with np.load(original_path, allow_pickle=True) as data:
        sensor_locations = _load_required_matrix(data, "sensor_locations")
        xi = _load_required_vector(data, "xi")
        yi = _load_required_vector(data, "yi")
        ti = _load_required_vector(data, "Ti")
        sigma_i = _load_required_vector(data, "sigma_i")
        hi = _load_required_vector(data, "Hi")
        qoi_times = _load_required_vector(data, "qoi_times")
        qoi_values = _load_required_matrix(data, "qoi_values")

    if qoi_times.shape[0] < 2:
        raise ValueError(f"{original_path} must contain at least two observation times.")
    if not np.all(np.diff(qoi_times) > 0.0):
        raise ValueError(f"{original_path} has non-increasing qoi_times.")
    if qoi_values.shape[1] != qoi_times.shape[0]:
        raise ValueError(
            f"{original_path} has qoi_values shape {qoi_values.shape}, "
            f"expected second axis to match qoi_times length {qoi_times.shape[0]}."
        )
    qoi_time_indices = np.flatnonzero(np.isclose(np.remainder(qoi_times, float(qoi_stride)), 0.0))
    if qoi_time_indices.size == 0:
        raise ValueError(f"{original_path} has no qoi_times divisible by qoi_stride={qoi_stride}.")

    source_count = xi.shape[0]
    if not all(vector.shape[0] == source_count for vector in (yi, ti, sigma_i, hi)):
        raise ValueError(f"{original_path} has inconsistent source parameter lengths.")

    selected_qoi_times = qoi_times[qoi_time_indices]
    selected_qoi_values = qoi_values[:, qoi_time_indices].T
    observation_times = np.concatenate([[0.0], selected_qoi_times], axis=0)
    qoi_observations = np.vstack(
        [
            np.zeros((1, qoi_values.shape[0]), dtype=np.float64),
            np.asarray(selected_qoi_values, dtype=np.float64),
        ]
    )
    input_values, input_feature_names = build_input_values(
        observation_times=observation_times,
        xi=xi,
        yi=yi,
        ti=ti,
        sigma_i=sigma_i,
        hi=hi,
        input_mode=input_mode,
    )
    sample = NpzQoiSample(
        sample_id=sample_id,
        observation_times=observation_times,
        u0=qoi_observations[0].copy(),
        qoi_observations=qoi_observations,
        input_times=observation_times.copy(),
        input_values=input_values,
        metadata={
            "dataset_kind": "swe_sensor_qoi",
            "input_mode": input_mode,
            "input_feature_names": np.asarray(input_feature_names),
            "original_npz_path": str(original_path),
            "original_time_offset": 0.0,
            "qoi_stride": int(qoi_stride),
            "prepended_zero_initial_qoi": 1,
            "kept_original_qoi_time_first": float(selected_qoi_times[0]),
            "kept_original_qoi_time_last": float(selected_qoi_times[-1]),
            "sensor_locations": sensor_locations,
            "xi": xi,
            "yi": yi,
            "Ti": ti,
            "sigma_i": sigma_i,
            "Hi": hi,
            "source_count": int(source_count),
        },
    )
    output_path = output_samples_root / f"{sample_id}.npz"
    return ConvertedSample(
        sample=sample,
        sample_path=output_path,
        observation_count=observation_times.shape[0],
        output_dimension=qoi_observations.shape[1],
        input_dimension=input_values.shape[1],
        input_mode=input_mode,
    )


def write_summary(
    output_root: Path,
    input_root: Path,
    manifest: NpzSampleManifest,
    converted_samples: list[ConvertedSample],
    qoi_stride: int,
    input_mode: str,
) -> None:
    first = converted_samples[0]
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "sample_count": len(converted_samples),
        "manifest_path": str(output_root / "manifest.npz"),
        "samples_root": str(output_root / "samples"),
        "observation_count_per_sample": int(first.observation_count),
        "qoi_output_dimension": int(first.output_dimension),
        "input_parameter_dimension": int(first.input_dimension),
        "input_mode": input_mode,
        "qoi_stride": int(qoi_stride),
        "prepended_zero_initial_qoi": True,
        "kept_original_qoi_time_first": first.sample.metadata["kept_original_qoi_time_first"],
        "kept_original_qoi_time_last": first.sample.metadata["kept_original_qoi_time_last"],
        "sample_id_first": manifest.sample_ids[0],
        "sample_id_last": manifest.sample_ids[-1],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = default_input_root() if args.input_root is None else args.input_root.expanduser()
    if args.qoi_stride <= 0:
        raise ValueError(f"--qoi-stride must be positive, got {args.qoi_stride}")
    sample_paths = discover_original_samples(input_root)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be positive, got {args.limit}")
        sample_paths = sample_paths[: args.limit]

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_samples_root = output_root / "samples"
    output_samples_root.mkdir(parents=True, exist_ok=True)

    converted_samples: list[ConvertedSample] = []
    manifest_sample_paths: list[Path] = []
    manifest_sample_ids: list[str] = []
    for original_path in sample_paths:
        converted = convert_original_sample(
            original_path.resolve(),
            output_samples_root,
            qoi_stride=args.qoi_stride,
            input_mode=args.input_mode,
        )
        save_npz_qoi_sample(converted.sample_path, converted.sample)
        converted_samples.append(converted)
        manifest_sample_paths.append(Path("samples") / converted.sample_path.name)
        manifest_sample_ids.append(converted.sample.sample_id)

    manifest = NpzSampleManifest(
        root_dir=output_root,
        sample_paths=tuple(manifest_sample_paths),
        sample_ids=tuple(manifest_sample_ids),
    )
    save_npz_sample_manifest(output_root / "manifest.npz", manifest)
    write_summary(
        output_root,
        input_root.resolve(),
        manifest,
        converted_samples,
        qoi_stride=args.qoi_stride,
        input_mode=args.input_mode,
    )

    print(
        f"Converted {len(converted_samples)} samples from {input_root} to {output_root}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
