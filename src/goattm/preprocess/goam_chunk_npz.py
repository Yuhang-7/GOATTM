from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest


TIME_MODE_PHYSICAL = "physical"
TIME_MODE_UNIT = "unit"
TIME_MODE_CHOICES = (TIME_MODE_PHYSICAL, TIME_MODE_UNIT)


@dataclass(frozen=True)
class GoamChunkNpzDefaults:
    train_root: Path
    test_root: Path | None
    output_root: Path
    dataset_kind: str
    sample_prefix: str
    qoi_stride: int = 1


@dataclass(frozen=True)
class ConvertedSample:
    sample: NpzQoiSample
    sample_path: Path
    source_path: Path
    source_local_index: int
    source_sample_index: int
    split_name: str


def parse_args(defaults: GoamChunkNpzDefaults, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--train-root",
        type=Path,
        default=defaults.train_root,
        help="GOAM tmpdirectory containing numeric training .npz chunk files.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=defaults.test_root,
        help=(
            "GOAM tmpdirectory containing test samples. If omitted, the converter uses "
            "test.npz under --train-root when that file exists."
        ),
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Only convert training samples, even if a test root or train-root/test.npz exists.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=defaults.output_root,
        help="Directory where samples, manifest.npz, train_manifest.npz, and test_manifest.npz are written.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Number of training samples to convert. Defaults to all samples found in --train-root.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Number of test samples to convert. Defaults to all samples found in the test source.",
    )
    parser.add_argument(
        "--qoi-stride",
        type=int,
        default=defaults.qoi_stride,
        help="Keep every qoi_stride-th time point from Tlist/QoI_list/bc_datas_list.",
    )
    parser.add_argument(
        "--time-mode",
        choices=TIME_MODE_CHOICES,
        default=TIME_MODE_PHYSICAL,
        help="Keep physical GOAM times or divide times by the final selected time.",
    )
    parser.add_argument(
        "--dataset-kind",
        default=defaults.dataset_kind,
        help="Metadata label stored in each converted sample.",
    )
    parser.add_argument(
        "--sample-prefix",
        default=defaults.sample_prefix,
        help="Prefix used for generated sample ids.",
    )
    return parser.parse_args()


def run_cli(defaults: GoamChunkNpzDefaults, description: str) -> None:
    args = parse_args(defaults, description)
    convert_goam_chunk_npz_dataset(
        train_root=args.train_root,
        test_root=None if args.no_test else args.test_root,
        output_root=args.output_root,
        dataset_kind=args.dataset_kind,
        sample_prefix=args.sample_prefix,
        train_count=args.train_count,
        test_count=args.test_count,
        qoi_stride=args.qoi_stride,
        time_mode=args.time_mode,
        allow_train_root_test_file=not args.no_test,
    )


def convert_goam_chunk_npz_dataset(
    train_root: Path,
    test_root: Path | None,
    output_root: Path,
    dataset_kind: str,
    sample_prefix: str,
    train_count: int | None = None,
    test_count: int | None = None,
    qoi_stride: int = 1,
    time_mode: str = TIME_MODE_PHYSICAL,
    allow_train_root_test_file: bool = True,
) -> NpzSampleManifest:
    train_root = train_root.expanduser().resolve()
    if test_root is not None:
        test_root = test_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if qoi_stride <= 0:
        raise ValueError(f"qoi_stride must be positive, got {qoi_stride}")
    if time_mode not in TIME_MODE_CHOICES:
        raise ValueError(f"time_mode must be one of {TIME_MODE_CHOICES}, got {time_mode!r}")
    if train_count is not None and train_count < 0:
        raise ValueError(f"train_count must be nonnegative, got {train_count}")
    if test_count is not None and test_count < 0:
        raise ValueError(f"test_count must be nonnegative, got {test_count}")

    output_samples_root = output_root / "samples"
    output_samples_root.mkdir(parents=True, exist_ok=True)

    train_samples = convert_split(
        root=train_root,
        split_name="train",
        output_samples_root=output_samples_root,
        dataset_kind=dataset_kind,
        sample_prefix=sample_prefix,
        max_count=train_count,
        qoi_stride=qoi_stride,
        time_mode=time_mode,
        prefer_test_file=False,
    )
    if test_root is None and allow_train_root_test_file and (train_root / "test.npz").exists():
        test_root = train_root
    test_samples: list[ConvertedSample] = []
    if test_root is not None:
        test_samples = convert_split(
            root=test_root,
            split_name="test",
            output_samples_root=output_samples_root,
            dataset_kind=dataset_kind,
            sample_prefix=sample_prefix,
            max_count=test_count,
            qoi_stride=qoi_stride,
            time_mode=time_mode,
            prefer_test_file=True,
        )

    if not train_samples:
        raise ValueError(f"No training samples were converted from {train_root}")
    all_samples = train_samples + test_samples
    manifest = make_manifest(output_root, all_samples)
    train_manifest = make_manifest(output_root, train_samples)
    test_manifest = make_manifest(output_root, test_samples)
    save_npz_sample_manifest(output_root / "manifest.npz", manifest)
    save_npz_sample_manifest(output_root / "train_manifest.npz", train_manifest)
    save_npz_sample_manifest(output_root / "test_manifest.npz", test_manifest)
    write_summary(
        output_root=output_root,
        train_root=train_root,
        test_root=test_root,
        dataset_kind=dataset_kind,
        sample_prefix=sample_prefix,
        train_samples=train_samples,
        test_samples=test_samples,
        qoi_stride=qoi_stride,
        time_mode=time_mode,
    )
    print(
        f"Converted {len(train_samples)} train and {len(test_samples)} test samples to {output_root}.",
        flush=True,
    )
    return manifest


def convert_split(
    root: Path,
    split_name: str,
    output_samples_root: Path,
    dataset_kind: str,
    sample_prefix: str,
    max_count: int | None,
    qoi_stride: int,
    time_mode: str,
    prefer_test_file: bool,
) -> list[ConvertedSample]:
    source_paths = discover_source_paths(root, prefer_test_file=prefer_test_file)
    converted_samples: list[ConvertedSample] = []
    split_sample_counter = 0
    for source_path in source_paths:
        with np.load(source_path, allow_pickle=True) as data:
            converted_from_file = convert_source_file(
                data=data,
                source_path=source_path,
                split_name=split_name,
                output_samples_root=output_samples_root,
                dataset_kind=dataset_kind,
                sample_prefix=sample_prefix,
                split_sample_start=split_sample_counter,
                max_remaining=None if max_count is None else max_count - split_sample_counter,
                qoi_stride=qoi_stride,
                time_mode=time_mode,
            )
        for converted in converted_from_file:
            save_npz_qoi_sample(converted.sample_path, converted.sample)
        converted_samples.extend(converted_from_file)
        split_sample_counter += len(converted_from_file)
        if max_count is not None and split_sample_counter >= max_count:
            break
    return converted_samples


def discover_source_paths(root: Path, prefer_test_file: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"GOAM source root does not exist: {root}")
    test_file = root / "test.npz"
    if prefer_test_file and test_file.exists():
        return [test_file]
    source_paths = sorted(
        (path for path in root.glob("*.npz") if is_numeric_npz(path)),
        key=lambda path: int(path.stem),
    )
    if not source_paths and test_file.exists():
        return [test_file]
    if not source_paths:
        raise FileNotFoundError(f"No numeric .npz chunk files found under {root}")
    return source_paths


def is_numeric_npz(path: Path) -> bool:
    return path.suffix == ".npz" and re.fullmatch(r"\d+", path.stem) is not None


def convert_source_file(
    data: np.lib.npyio.NpzFile,
    source_path: Path,
    split_name: str,
    output_samples_root: Path,
    dataset_kind: str,
    sample_prefix: str,
    split_sample_start: int,
    max_remaining: int | None,
    qoi_stride: int,
    time_mode: str,
) -> list[ConvertedSample]:
    qoi_values = load_goam_tensor(data, "QoI_list", source_path)
    input_values = load_goam_tensor(data, "bc_datas_list", source_path)
    tlist = load_goam_time_grid(data, source_path)
    if qoi_values.shape[1] != tlist.shape[0]:
        raise ValueError(f"{source_path}: QoI_list time axis {qoi_values.shape[1]} does not match Tlist {tlist.shape[0]}")
    if input_values.shape[1] != tlist.shape[0]:
        raise ValueError(
            f"{source_path}: bc_datas_list time axis {input_values.shape[1]} does not match Tlist {tlist.shape[0]}"
        )
    if qoi_values.shape[2] != input_values.shape[2]:
        raise ValueError(
            f"{source_path}: QoI_list sample count {qoi_values.shape[2]} does not match "
            f"bc_datas_list sample count {input_values.shape[2]}"
        )

    time_indices = np.arange(0, tlist.shape[0], qoi_stride, dtype=int)
    if time_indices[-1] != tlist.shape[0] - 1:
        time_indices = np.concatenate([time_indices, np.asarray([tlist.shape[0] - 1], dtype=int)])
    observation_times = np.asarray(tlist[time_indices], dtype=np.float64)
    if abs(float(observation_times[0])) > 1e-12:
        observation_times = observation_times - float(observation_times[0])
    if time_mode == TIME_MODE_UNIT:
        final_time = float(observation_times[-1])
        if final_time <= 0.0:
            raise ValueError(f"{source_path}: cannot rescale nonpositive final time {final_time}")
        observation_times = observation_times / final_time

    source_indices = source_index_array(data, qoi_values.shape[2])
    local_limit = qoi_values.shape[2] if max_remaining is None else min(qoi_values.shape[2], max_remaining)
    converted_samples: list[ConvertedSample] = []
    for local_index in range(local_limit):
        split_index = split_sample_start + local_index
        source_sample_index = int(source_indices[local_index])
        qoi = np.asarray(qoi_values[:, time_indices, local_index], dtype=np.float64).T
        inputs = np.asarray(input_values[:, time_indices, local_index], dtype=np.float64).T
        sample_id = f"{sample_prefix}_{split_name}_{split_index:06d}"
        sample = NpzQoiSample(
            sample_id=sample_id,
            observation_times=observation_times.copy(),
            u0=qoi[0].copy(),
            qoi_observations=qoi,
            input_times=observation_times.copy(),
            input_values=inputs,
            metadata={
                "dataset_kind": dataset_kind,
                "source_npz_path": str(source_path),
                "source_split": split_name,
                "source_local_index": int(local_index),
                "source_sample_index": source_sample_index,
                "source_qoi_layout": "QoI_list[dq, nt, nsample]",
                "source_input_layout": "bc_datas_list[dp, nt, nsample]",
                "qoi_stride": int(qoi_stride),
                "time_mode": time_mode,
                "qoi_feature_names": np.asarray([f"qoi_{idx}" for idx in range(qoi.shape[1])]),
                "input_feature_names": np.asarray([f"input_{idx}" for idx in range(inputs.shape[1])]),
                "qoi_dt_available": int("QoI_dt_list" in data.files),
            },
        )
        converted_samples.append(
            ConvertedSample(
                sample=sample,
                sample_path=output_samples_root / f"{sample_id}.npz",
                source_path=source_path,
                source_local_index=int(local_index),
                source_sample_index=source_sample_index,
                split_name=split_name,
            )
        )
    return converted_samples


def load_goam_tensor(data: np.lib.npyio.NpzFile, key: str, source_path: Path) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"{source_path}: missing required field {key!r}")
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 3:
        raise ValueError(f"{source_path}: {key} must have shape (d, nt, nsample), got {value.shape}")
    return value


def load_goam_time_grid(data: np.lib.npyio.NpzFile, source_path: Path) -> np.ndarray:
    if "Tlist" not in data.files:
        raise ValueError(f"{source_path}: missing required field 'Tlist'")
    value = np.asarray(data["Tlist"], dtype=np.float64)
    if value.ndim != 1 or value.shape[0] < 2:
        raise ValueError(f"{source_path}: Tlist must have shape (nt,), nt >= 2, got {value.shape}")
    if np.any(np.diff(value) <= 0.0):
        raise ValueError(f"{source_path}: Tlist must be strictly increasing")
    return value


def source_index_array(data: np.lib.npyio.NpzFile, sample_count: int) -> np.ndarray:
    if "indexlist" not in data.files:
        return np.arange(sample_count, dtype=int)
    indexlist = np.asarray(data["indexlist"], dtype=int)
    if indexlist.ndim != 1 or indexlist.shape[0] != sample_count:
        raise ValueError(f"indexlist must have shape ({sample_count},), got {indexlist.shape}")
    return indexlist


def make_manifest(output_root: Path, samples: list[ConvertedSample]) -> NpzSampleManifest:
    return NpzSampleManifest(
        root_dir=output_root,
        sample_paths=tuple(Path("samples") / sample.sample_path.name for sample in samples),
        sample_ids=tuple(sample.sample.sample_id for sample in samples),
    )


def write_summary(
    output_root: Path,
    train_root: Path,
    test_root: Path | None,
    dataset_kind: str,
    sample_prefix: str,
    train_samples: list[ConvertedSample],
    test_samples: list[ConvertedSample],
    qoi_stride: int,
    time_mode: str,
) -> None:
    first = train_samples[0]
    summary = {
        "dataset_kind": dataset_kind,
        "sample_prefix": sample_prefix,
        "train_root": str(train_root),
        "test_root": None if test_root is None else str(test_root),
        "output_root": str(output_root),
        "manifest_path": str(output_root / "manifest.npz"),
        "train_manifest_path": str(output_root / "train_manifest.npz"),
        "test_manifest_path": str(output_root / "test_manifest.npz"),
        "samples_root": str(output_root / "samples"),
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "total_sample_count": len(train_samples) + len(test_samples),
        "observation_count_per_sample": int(first.sample.observation_times.shape[0]),
        "qoi_output_dimension": int(first.sample.output_dimension),
        "input_dimension": int(first.sample.input_dimension),
        "qoi_stride": int(qoi_stride),
        "time_mode": time_mode,
        "time_first": float(first.sample.observation_times[0]),
        "time_last": float(first.sample.observation_times[-1]),
        "sample_id_first": first.sample.sample_id,
        "sample_id_last": (test_samples or train_samples)[-1].sample.sample_id,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
