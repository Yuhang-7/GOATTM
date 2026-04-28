from __future__ import annotations

import argparse
import json
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


DEFAULT_INPUT_ROOT = THIS_FILE.parents[1] / "data" / "original_data"
DEFAULT_OUTPUT_ROOT = THIS_FILE.parents[1] / "data" / "processed_data"


@dataclass(frozen=True)
class ConvertedSample:
    sample: NpzQoiSample
    sample_path: Path
    observation_count: int
    output_dimension: int
    input_dimension: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SWE application NPZ files into GOATTM NpzQoiSample format."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing original per-sample subdirectories.",
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


def convert_original_sample(original_path: Path, output_samples_root: Path) -> ConvertedSample:
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

    source_count = xi.shape[0]
    if not all(vector.shape[0] == source_count for vector in (yi, ti, sigma_i, hi)):
        raise ValueError(f"{original_path} has inconsistent source parameter lengths.")

    observation_times = qoi_times - float(qoi_times[0])
    qoi_observations = np.asarray(qoi_values.T, dtype=np.float64)
    constant_input = np.concatenate([xi, yi, ti, sigma_i, hi], axis=0)
    input_values = np.repeat(constant_input[None, :], observation_times.shape[0], axis=0)
    sample = NpzQoiSample(
        sample_id=sample_id,
        observation_times=observation_times,
        u0=qoi_observations[0].copy(),
        qoi_observations=qoi_observations,
        input_times=observation_times.copy(),
        input_values=input_values,
        metadata={
            "dataset_kind": "swe_sensor_qoi",
            "original_npz_path": str(original_path),
            "original_time_offset": float(qoi_times[0]),
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
    )


def write_summary(
    output_root: Path,
    input_root: Path,
    manifest: NpzSampleManifest,
    converted_samples: list[ConvertedSample],
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
        "sample_id_first": manifest.sample_ids[0],
        "sample_id_last": manifest.sample_ids[-1],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_paths = discover_original_samples(args.input_root)
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
        converted = convert_original_sample(original_path.resolve(), output_samples_root)
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
    write_summary(output_root, args.input_root.resolve(), manifest, converted_samples)

    print(
        f"Converted {len(converted_samples)} samples from {args.input_root} to {output_root}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
