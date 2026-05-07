from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("QoI_list", "bc_datas_list", "Tlist")
OPTIONAL_KEYS = ("QoI_dt_list", "param_index", "param_num")
DTYPE_BY_HDF5_NAME = {
    "H5T_IEEE_F64LE": np.dtype("<f8"),
    "H5T_IEEE_F32LE": np.dtype("<f4"),
    "H5T_STD_I64LE": np.dtype("<i8"),
    "H5T_STD_I32LE": np.dtype("<i4"),
    "H5T_STD_U64LE": np.dtype("<u8"),
    "H5T_STD_U32LE": np.dtype("<u4"),
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dtype: np.dtype
    shape: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GOAM HDF5 file into chunked GOAM-style NPZ files without requiring h5py."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Source GOAM HDF5 file.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for 0.npz, test.npz, summary.json.")
    parser.add_argument("--train-count", type=int, required=True, help="Number of samples stored in numeric train chunks.")
    parser.add_argument("--test-count", type=int, required=True, help="Number of samples stored in test.npz.")
    parser.add_argument("--sample-start", type=int, default=0, help="First source sample index to export.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Train samples per numeric chunk. Defaults to train-count.")
    parser.add_argument(
        "--required-keys",
        default=",".join(REQUIRED_KEYS),
        help="Comma-separated HDF5 datasets that must exist and be exported.",
    )
    parser.add_argument(
        "--optional-keys",
        default=",".join(OPTIONAL_KEYS),
        help="Comma-separated HDF5 datasets exported when present.",
    )
    parser.add_argument("--h5dump-bin", default=None, help="Path to h5dump. Defaults to h5dump on PATH.")
    parser.add_argument(
        "--engine",
        choices=("auto", "h5py", "h5dump"),
        default="auto",
        help="HDF5 read engine. auto tries h5py first, then h5dump.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_keys = parse_key_list(args.required_keys)
    optional_keys = parse_key_list(args.optional_keys)
    convert_hdf5_to_goam_npz(
        input_file=args.input_file,
        output_root=args.output_root,
        train_count=args.train_count,
        test_count=args.test_count,
        sample_start=args.sample_start,
        chunk_size=args.chunk_size,
        required_keys=required_keys,
        optional_keys=optional_keys,
        h5dump_bin=args.h5dump_bin,
        engine=args.engine,
    )


def parse_key_list(value: str) -> tuple[str, ...]:
    return tuple(key.strip() for key in value.split(",") if key.strip())


def convert_hdf5_to_goam_npz(
    input_file: Path,
    output_root: Path,
    train_count: int,
    test_count: int,
    sample_start: int = 0,
    chunk_size: int | None = None,
    required_keys: tuple[str, ...] = REQUIRED_KEYS,
    optional_keys: tuple[str, ...] = OPTIONAL_KEYS,
    h5dump_bin: str | None = None,
    engine: str = "auto",
) -> None:
    input_file = input_file.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input HDF5 file does not exist: {input_file}")
    if train_count < 0 or test_count < 0:
        raise ValueError("train_count and test_count must be nonnegative")
    if train_count + test_count <= 0:
        raise ValueError("train_count + test_count must be positive")
    if sample_start < 0:
        raise ValueError("sample_start must be nonnegative")
    if chunk_size is None:
        chunk_size = max(train_count, 1)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    reader = Hdf5Reader(input_file=input_file, h5dump_bin=h5dump_bin, engine=engine)
    available = reader.available_keys()
    missing = [key for key in required_keys if key not in available]
    if missing:
        raise ValueError(f"Missing required HDF5 datasets in {input_file}: {missing}")
    keys = tuple(required_keys) + tuple(key for key in optional_keys if key in available and key not in required_keys)

    arrays = {key: reader.read_dataset(key) for key in keys}
    sample_count = infer_sample_count(arrays)
    sample_end = sample_start + train_count + test_count
    if sample_end > sample_count:
        raise ValueError(
            f"Requested samples [{sample_start}, {sample_end}) exceed source sample count {sample_count}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    train_indices = np.arange(sample_start, sample_start + train_count, dtype=np.int64)
    test_indices = np.arange(sample_start + train_count, sample_end, dtype=np.int64)
    for chunk_id, start in enumerate(range(0, train_count, chunk_size)):
        stop = min(start + chunk_size, train_count)
        chunk_indices = train_indices[start:stop]
        chunk_arrays = slice_arrays_for_indices(arrays, chunk_indices)
        chunk_arrays["indexlist"] = chunk_indices.copy()
        np.savez_compressed(output_root / f"{chunk_id}.npz", **chunk_arrays)
    if test_count > 0:
        test_arrays = slice_arrays_for_indices(arrays, test_indices)
        test_arrays["indexlist"] = test_indices.copy()
        np.savez_compressed(output_root / "test.npz", **test_arrays)

    summary = {
        "input_file": str(input_file),
        "output_root": str(output_root),
        "engine": reader.engine_name,
        "exported_keys": list(keys),
        "source_sample_count": int(sample_count),
        "sample_start": int(sample_start),
        "train_count": int(train_count),
        "test_count": int(test_count),
        "chunk_size": int(chunk_size),
        "train_indices_first": None if train_indices.size == 0 else int(train_indices[0]),
        "train_indices_last": None if train_indices.size == 0 else int(train_indices[-1]),
        "test_indices_first": None if test_indices.size == 0 else int(test_indices[0]),
        "test_indices_last": None if test_indices.size == 0 else int(test_indices[-1]),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(f"Converted {input_file} to {output_root} with {train_count} train and {test_count} test samples.")


def infer_sample_count(arrays: dict[str, np.ndarray]) -> int:
    qoi = arrays["QoI_list"]
    inputs = arrays["bc_datas_list"]
    if qoi.ndim != 3 or inputs.ndim != 3:
        raise ValueError("QoI_list and bc_datas_list must have shape (d, nt, nsample).")
    if qoi.shape[2] != inputs.shape[2]:
        raise ValueError(f"QoI_list sample count {qoi.shape[2]} != bc_datas_list sample count {inputs.shape[2]}")
    return int(qoi.shape[2])


def slice_arrays_for_indices(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    source_count = infer_sample_count(arrays)
    for key, value in arrays.items():
        if value.ndim >= 1 and value.shape[-1] == source_count and key != "Tlist":
            result[key] = np.asarray(value[..., indices], order="C")
        else:
            result[key] = np.asarray(value, order="C")
    return result


class Hdf5Reader:
    def __init__(self, input_file: Path, h5dump_bin: str | None, engine: str) -> None:
        self.input_file = input_file
        self.h5dump_bin = h5dump_bin or "h5dump"
        self._h5py = None
        if engine in {"auto", "h5py"}:
            try:
                import h5py  # type: ignore
            except Exception:
                if engine == "h5py":
                    raise
            else:
                self._h5py = h5py
                self.engine_name = "h5py"
                return
        if engine in {"auto", "h5dump"}:
            if shutil.which(self.h5dump_bin) is None:
                raise RuntimeError(
                    "h5py is unavailable and h5dump was not found on PATH. "
                    "Load an HDF5 module or set --h5dump-bin."
                )
            self.engine_name = "h5dump"
            return
        raise ValueError(f"Unknown engine {engine!r}")

    def available_keys(self) -> set[str]:
        if self._h5py is not None:
            with self._h5py.File(self.input_file, "r") as handle:
                return set(handle.keys())
        text = subprocess.check_output([self.h5dump_bin, "-H", str(self.input_file)], text=True)
        return set(re.findall(r'DATASET "([^"]+)"', text))

    def read_dataset(self, key: str) -> np.ndarray:
        if self._h5py is not None:
            with self._h5py.File(self.input_file, "r") as handle:
                return np.asarray(handle[key])
        spec = self.dataset_spec(key)
        with tempfile.TemporaryDirectory(prefix="goattm_h5dump_") as tmpdir:
            raw_path = Path(tmpdir) / f"{key}.bin"
            subprocess.check_call(
                [self.h5dump_bin, "-d", f"/{key}", "-b", "LE", "-o", str(raw_path), str(self.input_file)],
                stdout=subprocess.DEVNULL,
            )
            data = np.fromfile(raw_path, dtype=spec.dtype)
        expected_size = int(np.prod(spec.shape, dtype=np.int64)) if spec.shape else 1
        if data.size != expected_size:
            raise ValueError(f"{key}: expected {expected_size} values from h5dump, got {data.size}")
        if not spec.shape:
            return data.reshape(())
        return data.reshape(spec.shape)

    def dataset_spec(self, key: str) -> DatasetSpec:
        text = subprocess.check_output([self.h5dump_bin, "-H", "-d", f"/{key}", str(self.input_file)], text=True)
        dtype_match = re.search(r"DATATYPE\s+([^\n]+)", text)
        if dtype_match is None:
            raise ValueError(f"Could not parse datatype for {key}")
        hdf5_dtype = dtype_match.group(1).strip()
        if hdf5_dtype not in DTYPE_BY_HDF5_NAME:
            raise ValueError(f"Unsupported datatype for {key}: {hdf5_dtype}")
        scalar_match = re.search(r"DATASPACE\s+SCALAR", text)
        if scalar_match is not None:
            return DatasetSpec(key=key, dtype=DTYPE_BY_HDF5_NAME[hdf5_dtype], shape=())
        shape_match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s*\(\s*([^\)]*?)\s*\)", text)
        if shape_match is None:
            raise ValueError(f"Could not parse shape for {key}")
        shape = tuple(int(part.strip()) for part in shape_match.group(1).split(",") if part.strip())
        return DatasetSpec(key=key, dtype=DTYPE_BY_HDF5_NAME[hdf5_dtype], shape=shape)


if __name__ == "__main__":
    main()
