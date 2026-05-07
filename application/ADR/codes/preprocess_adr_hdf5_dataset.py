from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


THIS_FILE = Path(__file__).resolve()
ADR_PROBLEM_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.data import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest  # noqa: E402


INPUT_FILE_ENV_VAR = "ADR_GOAM_HDF5"
INPUT_ROOT_ENV_VAR = "ADR_GOAM_DATA_ROOT"
CURRENT_GOAM_ADR_ROOT = Path("/storage/yuhang/Myresearch/GOAM_clean/Example/ADR/dataset/ADR_quadp")
DEFAULT_DATASET_NAME = "ADR_quadp_trainsize=256_testsize=200.hdf5"
DEFAULT_OUTPUT_ROOT = ADR_PROBLEM_ROOT / "data" / "processed_data"
TIME_MODE_PHYSICAL = "physical"
TIME_MODE_UNIT = "unit"
TIME_MODE_CHOICES = (TIME_MODE_PHYSICAL, TIME_MODE_UNIT)


@dataclass(frozen=True)
class ConvertedSample:
    sample: NpzQoiSample
    sample_path: Path
    source_index: int
    observation_count: int
    output_dimension: int
    input_dimension: int


def default_input_file() -> Path:
    env_file = os.environ.get(INPUT_FILE_ENV_VAR)
    if env_file:
        return Path(env_file).expanduser()
    env_root = os.environ.get(INPUT_ROOT_ENV_VAR)
    root = Path(env_root).expanduser() if env_root else CURRENT_GOAM_ADR_ROOT
    return root / DEFAULT_DATASET_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GOAM ADR HDF5 files into GOATTM NpzQoiSample format."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "GOAM ADR HDF5 file. Defaults to $ADR_GOAM_HDF5, or "
            f"$ADR_GOAM_DATA_ROOT/{DEFAULT_DATASET_NAME}, or {CURRENT_GOAM_ADR_ROOT / DEFAULT_DATASET_NAME}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where converted samples and manifest will be written.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample cap for smoke tests.")
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="First source sample index to convert from the HDF5 file.",
    )
    parser.add_argument(
        "--qoi-stride",
        type=int,
        default=2,
        help="Keep every qoi_stride-th time point. Default keeps indices 0,2,4,...,1000 for ADR_quadp.",
    )
    parser.add_argument(
        "--time-mode",
        choices=TIME_MODE_CHOICES,
        default=TIME_MODE_PHYSICAL,
        help="Keep physical GOAM times or divide each sample by its final time.",
    )
    parser.add_argument(
        "--normalize-qoi-from-file",
        action="store_true",
        help="Divide QoI_list by Qs_normalize_constant if present. Off by default to match GOAM ADR runs.",
    )
    parser.add_argument(
        "--normalize-input-from-file",
        action="store_true",
        help="Divide bc_datas_list by Ps_normalize_constant if present. Off by default to match GOAM ADR runs.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Number of leading samples treated as training samples. Inferred from filename when omitted.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Number of trailing samples treated as test samples. Inferred from filename when omitted.",
    )
    return parser.parse_args()


def _infer_count(pattern: str, filename: str) -> int | None:
    match = re.search(pattern, filename)
    if match is None:
        return None
    return int(match.group(1))


def infer_split_counts(input_file: Path, total_count: int, train_count: int | None, test_count: int | None) -> tuple[int, int]:
    inferred_train = _infer_count(r"trainsize=(\d+)", input_file.name)
    inferred_test = _infer_count(r"testsize=(\d+)", input_file.name)
    if train_count is None:
        train_count = inferred_train
    if test_count is None:
        test_count = inferred_test
    if train_count is None and test_count is None:
        train_count = total_count
        test_count = 0
    elif train_count is None:
        train_count = total_count - int(test_count)
    elif test_count is None:
        test_count = total_count - int(train_count)
    train_count = int(train_count)
    test_count = int(test_count)
    if train_count < 0 or test_count < 0:
        raise ValueError(f"train_count and test_count must be nonnegative, got {train_count}, {test_count}")
    if train_count + test_count != total_count:
        raise ValueError(
            f"train_count + test_count must equal total samples, got {train_count}+{test_count}!={total_count}"
        )
    return train_count, test_count


def _normalization_vector(file: h5py.File, key: str, expected_dim: int, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones(expected_dim, dtype=np.float64)
    if key not in file:
        raise ValueError(f"Requested normalization from missing HDF5 field {key!r}.")
    value = np.asarray(file[key], dtype=np.float64)
    if value.shape != (expected_dim,):
        raise ValueError(f"{key} must have shape ({expected_dim},), got {value.shape}")
    if np.any(value == 0.0):
        raise ValueError(f"{key} contains zero entries.")
    return value


def convert_sample(
    file: h5py.File,
    source_index: int,
    time_indices: np.ndarray,
    output_samples_root: Path,
    dataset_stem: str,
    observation_times: np.ndarray,
    qoi_scale: np.ndarray,
    input_scale: np.ndarray,
    train_count: int,
) -> ConvertedSample:
    qoi = np.asarray(file["QoI_list"][:, time_indices, source_index], dtype=np.float64).T / qoi_scale[None, :]
    inputs = np.asarray(file["bc_datas_list"][:, time_indices, source_index], dtype=np.float64).T / input_scale[None, :]
    split_name = "train" if source_index < train_count else "test"
    sample_id = f"{dataset_stem}_{split_name}_{source_index:06d}"
    sample = NpzQoiSample(
        sample_id=sample_id,
        observation_times=observation_times.copy(),
        u0=qoi[0].copy(),
        qoi_observations=qoi,
        input_times=observation_times.copy(),
        input_values=inputs,
        metadata={
            "dataset_kind": "adr_quadp_goam_hdf5",
            "source_hdf5_path": str(Path(file.filename).resolve()),
            "source_sample_index": int(source_index),
            "source_split": split_name,
            "qoi_source_layout": "QoI_list[dq, nt, nsample]",
            "input_source_layout": "bc_datas_list[dp, nt, nsample]",
            "qoi_feature_names": np.asarray(
                [f"density_{idx + 1}" for idx in range(5)]
                + [f"flux_{idx + 1}" for idx in range(5)]
                + ["energy"]
            ),
            "input_feature_names": np.asarray(["boundary_signal_g", "boundary_signal_g_squared"][: inputs.shape[1]]),
            "goam_train_count": int(train_count),
            "goam_test_count": int(file["param_num"][()] - train_count),
            "input_channel_semantics": "bc_datas_list[0]=g(t), bc_datas_list[1]=g(t)^2 for ADR_quadp",
        },
    )
    output_path = output_samples_root / f"{sample_id}.npz"
    return ConvertedSample(
        sample=sample,
        sample_path=output_path,
        source_index=source_index,
        observation_count=observation_times.shape[0],
        output_dimension=qoi.shape[1],
        input_dimension=inputs.shape[1],
    )


def write_summary(
    output_root: Path,
    input_file: Path,
    manifest: NpzSampleManifest,
    converted_samples: list[ConvertedSample],
    train_count: int,
    test_count: int,
    source_total_count: int,
    qoi_stride: int,
    time_mode: str,
    normalize_qoi: bool,
    normalize_input: bool,
) -> None:
    first = converted_samples[0]
    source_indices = [sample.source_index for sample in converted_samples]
    summary = {
        "input_file": str(input_file),
        "output_root": str(output_root),
        "manifest_path": str(output_root / "manifest.npz"),
        "samples_root": str(output_root / "samples"),
        "converted_sample_count": len(converted_samples),
        "source_total_count": int(source_total_count),
        "goam_train_count": int(train_count),
        "goam_test_count": int(test_count),
        "converted_source_index_first": int(source_indices[0]),
        "converted_source_index_last": int(source_indices[-1]),
        "sample_id_first": manifest.sample_ids[0],
        "sample_id_last": manifest.sample_ids[-1],
        "observation_count_per_sample": int(first.observation_count),
        "qoi_output_dimension": int(first.output_dimension),
        "input_dimension": int(first.input_dimension),
        "qoi_stride": int(qoi_stride),
        "time_mode": time_mode,
        "time_first": float(first.sample.observation_times[0]),
        "time_last": float(first.sample.observation_times[-1]),
        "normalize_qoi_from_file": bool(normalize_qoi),
        "normalize_input_from_file": bool(normalize_input),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = default_input_file() if args.input_file is None else args.input_file.expanduser()
    input_file = input_file.resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input HDF5 file does not exist: {input_file}")
    if args.sample_start < 0:
        raise ValueError(f"sample_start must be nonnegative, got {args.sample_start}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {args.limit}")
    if args.qoi_stride <= 0:
        raise ValueError(f"qoi_stride must be positive, got {args.qoi_stride}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_samples_root = output_root / "samples"
    output_samples_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, "r") as file:
        required = ("QoI_list", "bc_datas_list", "Tlist", "param_num")
        missing = [key for key in required if key not in file]
        if missing:
            raise ValueError(f"{input_file} is missing required fields: {missing}")
        qoi_data = file["QoI_list"]
        input_data = file["bc_datas_list"]
        if qoi_data.ndim != 3 or input_data.ndim != 3:
            raise ValueError("QoI_list and bc_datas_list must both be rank-3 arrays.")
        dq, nt, qoi_total = qoi_data.shape
        dp, input_nt, input_total = input_data.shape
        if input_nt != nt or input_total != qoi_total:
            raise ValueError(
                "QoI_list and bc_datas_list disagree: "
                f"QoI shape={qoi_data.shape}, input shape={input_data.shape}"
            )
        source_total_count = int(np.asarray(file["param_num"]).item())
        if source_total_count != qoi_total:
            raise ValueError(f"param_num={source_total_count} but QoI_list has {qoi_total} samples")
        train_count, test_count = infer_split_counts(input_file, source_total_count, args.train_count, args.test_count)

        raw_times = np.asarray(file["Tlist"], dtype=np.float64)
        if raw_times.shape != (nt,):
            raise ValueError(f"Tlist must have shape ({nt},), got {raw_times.shape}")
        time_indices = np.arange(0, nt, args.qoi_stride, dtype=int)
        if time_indices[-1] != nt - 1:
            time_indices = np.concatenate([time_indices, np.asarray([nt - 1], dtype=int)])
        observation_times = raw_times[time_indices].astype(np.float64, copy=True)
        if args.time_mode == TIME_MODE_UNIT:
            final_time = float(observation_times[-1])
            if final_time <= 0.0:
                raise ValueError(f"Cannot normalize nonpositive final time {final_time}")
            observation_times /= final_time
        if abs(float(observation_times[0])) > 1e-14:
            observation_times -= float(observation_times[0])
        if np.any(np.diff(observation_times) <= 0.0):
            raise ValueError("Selected observation times are not strictly increasing.")

        qoi_scale = _normalization_vector(file, "Qs_normalize_constant", dq, args.normalize_qoi_from_file)
        input_scale = _normalization_vector(file, "Ps_normalize_constant", dp, args.normalize_input_from_file)
        sample_stop = source_total_count if args.limit is None else min(source_total_count, args.sample_start + args.limit)
        if args.sample_start >= source_total_count:
            raise ValueError(f"sample_start={args.sample_start} is outside {source_total_count} source samples")
        dataset_stem = input_file.stem

        converted_samples: list[ConvertedSample] = []
        manifest_sample_paths: list[Path] = []
        manifest_sample_ids: list[str] = []
        for source_index in range(args.sample_start, sample_stop):
            converted = convert_sample(
                file=file,
                source_index=source_index,
                time_indices=time_indices,
                output_samples_root=output_samples_root,
                dataset_stem=dataset_stem,
                observation_times=observation_times,
                qoi_scale=qoi_scale,
                input_scale=input_scale,
                train_count=train_count,
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
        output_root=output_root,
        input_file=input_file,
        manifest=manifest,
        converted_samples=converted_samples,
        train_count=train_count,
        test_count=test_count,
        source_total_count=source_total_count,
        qoi_stride=args.qoi_stride,
        time_mode=args.time_mode,
        normalize_qoi=args.normalize_qoi_from_file,
        normalize_input=args.normalize_input_from_file,
    )
    print(
        f"Converted {len(converted_samples)} ADR samples from {input_file} to {output_root} "
        f"with dq={converted_samples[0].output_dimension}, dp={converted_samples[0].input_dimension}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
