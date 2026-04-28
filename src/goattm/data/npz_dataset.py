from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class NpzQoiSample:
    sample_id: str
    observation_times: np.ndarray
    u0: np.ndarray
    qoi_observations: np.ndarray
    input_times: np.ndarray | None = None
    input_values: np.ndarray | None = None
    metadata: dict[str, np.ndarray | str | float | int] | None = None

    def __post_init__(self) -> None:
        if self.observation_times.ndim != 1 or self.observation_times.shape[0] < 2:
            raise ValueError(
                f"observation_times must have shape (N,), N>=2, got {self.observation_times.shape}"
            )
        if abs(float(self.observation_times[0])) > 1e-14:
            raise ValueError("observation_times must start at 0.0.")
        if np.any(np.diff(self.observation_times) <= 0.0):
            raise ValueError("observation_times must be strictly increasing.")
        if self.u0.ndim != 1:
            raise ValueError(f"u0 must be rank-1, got shape {self.u0.shape}")
        if self.qoi_observations.ndim != 2 or self.qoi_observations.shape[0] != self.observation_times.shape[0]:
            raise ValueError(
                f"qoi_observations must have shape ({self.observation_times.shape[0]}, dq), "
                f"got {self.qoi_observations.shape}"
            )
        if self.input_times is None and self.input_values is not None:
            raise ValueError("input_times must be provided when input_values is present.")
        if self.input_times is not None and self.input_values is None:
            raise ValueError("input_values must be provided when input_times is present.")
        if self.input_times is not None:
            if self.input_times.ndim != 1:
                raise ValueError(f"input_times must be rank-1, got shape {self.input_times.shape}")
            if np.any(np.diff(self.input_times) <= 0.0):
                raise ValueError("input_times must be strictly increasing.")
            if self.input_values is None or self.input_values.ndim != 2:
                raise ValueError("input_values must have shape (M, dp).")
            if self.input_values.shape[0] != self.input_times.shape[0]:
                raise ValueError(
                    f"input_values must have shape ({self.input_times.shape[0]}, dp), got {self.input_values.shape}"
                )

    @property
    def latent_dimension(self) -> int:
        return self.u0.shape[0]

    @property
    def output_dimension(self) -> int:
        return self.qoi_observations.shape[1]

    @property
    def input_dimension(self) -> int:
        return 0 if self.input_values is None else self.input_values.shape[1]

    def build_input_function(self) -> Callable[[float], np.ndarray] | None:
        if self.input_times is None or self.input_values is None:
            return None
        return build_cubic_spline_input_function(self.input_times, self.input_values)


@dataclass(frozen=True)
class NpzSampleManifest:
    root_dir: Path
    sample_paths: tuple[Path, ...]
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.sample_paths) != len(self.sample_ids):
            raise ValueError("sample_paths and sample_ids must have the same length.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique within a manifest.")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def absolute_paths(self) -> tuple[Path, ...]:
        return tuple(path if path.is_absolute() else (self.root_dir / path) for path in self.sample_paths)

    def subset_by_indices(self, indices: list[int] | tuple[int, ...] | np.ndarray) -> "NpzSampleManifest":
        index_list = [int(index) for index in indices]
        if any(index < 0 or index >= len(self.sample_ids) for index in index_list):
            raise ValueError("subset indices must lie within the manifest range.")
        return NpzSampleManifest(
            root_dir=self.root_dir,
            sample_paths=tuple(self.sample_paths[index] for index in index_list),
            sample_ids=tuple(self.sample_ids[index] for index in index_list),
        )

    def subset_by_ids(self, sample_ids: list[str] | tuple[str, ...] | np.ndarray) -> "NpzSampleManifest":
        id_to_index = {sample_id: idx for idx, sample_id in enumerate(self.sample_ids)}
        selected_indices: list[int] = []
        for sample_id in sample_ids:
            sample_id_str = str(sample_id)
            if sample_id_str not in id_to_index:
                raise ValueError(f"Unknown sample_id '{sample_id_str}' requested in manifest subset.")
            selected_indices.append(id_to_index[sample_id_str])
        return self.subset_by_indices(selected_indices)

    def entries_for_rank(self, rank: int, size: int) -> tuple[tuple[str, Path], ...]:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if rank < 0 or rank >= size:
            raise ValueError(f"rank must satisfy 0 <= rank < size, got rank={rank}, size={size}")
        selected: list[tuple[str, Path]] = []
        for idx, (sample_id, path) in enumerate(zip(self.sample_ids, self.absolute_paths())):
            if idx % size == rank:
                selected.append((sample_id, path))
        return tuple(selected)


@dataclass(frozen=True)
class NpzTrainTestSplit:
    train_manifest: NpzSampleManifest
    test_manifest: NpzSampleManifest
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    sample_seed: int | None = None


def make_npz_train_test_split(
    manifest: str | Path | NpzSampleManifest,
    train_sample_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
    test_sample_ids: list[str] | tuple[str, ...] | np.ndarray | None = None,
    sample_seed: int | None = None,
    train_fraction: float = 0.8,
    shuffle: bool = True,
) -> NpzTrainTestSplit:
    if isinstance(manifest, (str, Path)):
        manifest = load_npz_sample_manifest(manifest)
    if len(manifest) < 2:
        raise ValueError("Need at least two samples to create a train/test split.")

    if train_sample_ids is not None or test_sample_ids is not None:
        if sample_seed is not None:
            raise ValueError("sample_seed cannot be provided together with an explicit split.")
        return _make_explicit_train_test_split(
            manifest=manifest,
            train_sample_ids=train_sample_ids,
            test_sample_ids=test_sample_ids,
        )

    if sample_seed is None:
        raise ValueError("Either explicit train/test sample ids or sample_seed must be provided.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must lie in (0, 1), got {train_fraction}")

    indices = np.arange(len(manifest), dtype=int)
    if shuffle:
        rng = np.random.default_rng(sample_seed)
        rng.shuffle(indices)
    train_count = int(np.floor(train_fraction * len(manifest)))
    train_count = min(max(train_count, 1), len(manifest) - 1)
    train_indices = np.sort(indices[:train_count]).tolist()
    test_indices = np.sort(indices[train_count:]).tolist()
    return NpzTrainTestSplit(
        train_manifest=manifest.subset_by_indices(train_indices),
        test_manifest=manifest.subset_by_indices(test_indices),
        train_indices=tuple(train_indices),
        test_indices=tuple(test_indices),
        sample_seed=int(sample_seed),
    )


def _make_explicit_train_test_split(
    manifest: NpzSampleManifest,
    train_sample_ids: list[str] | tuple[str, ...] | np.ndarray | None,
    test_sample_ids: list[str] | tuple[str, ...] | np.ndarray | None,
) -> NpzTrainTestSplit:
    all_ids = tuple(str(sample_id) for sample_id in manifest.sample_ids)
    all_id_set = set(all_ids)
    train_ids = None if train_sample_ids is None else tuple(str(sample_id) for sample_id in train_sample_ids)
    test_ids = None if test_sample_ids is None else tuple(str(sample_id) for sample_id in test_sample_ids)

    if train_ids is None and test_ids is None:
        raise ValueError("At least one of train_sample_ids or test_sample_ids must be provided for an explicit split.")
    if train_ids is None:
        missing = set(test_ids) - all_id_set
        if missing:
            raise ValueError(f"Unknown test sample ids: {sorted(missing)}")
        train_ids = tuple(sample_id for sample_id in all_ids if sample_id not in set(test_ids))
    if test_ids is None:
        missing = set(train_ids) - all_id_set
        if missing:
            raise ValueError(f"Unknown train sample ids: {sorted(missing)}")
        test_ids = tuple(sample_id for sample_id in all_ids if sample_id not in set(train_ids))

    train_set = set(train_ids)
    test_set = set(test_ids)
    if len(train_set) != len(train_ids):
        raise ValueError("train_sample_ids contains duplicates.")
    if len(test_set) != len(test_ids):
        raise ValueError("test_sample_ids contains duplicates.")
    if train_set & test_set:
        raise ValueError("Explicit train/test split must be disjoint.")
    if train_set | test_set != all_id_set:
        missing = sorted(all_id_set - (train_set | test_set))
        extra = sorted((train_set | test_set) - all_id_set)
        details = []
        if missing:
            details.append(f"missing ids {missing}")
        if extra:
            details.append(f"unknown ids {extra}")
        raise ValueError("Explicit train/test split must cover the manifest exactly: " + ", ".join(details))

    id_to_index = {sample_id: idx for idx, sample_id in enumerate(all_ids)}
    train_indices = tuple(id_to_index[sample_id] for sample_id in train_ids)
    test_indices = tuple(id_to_index[sample_id] for sample_id in test_ids)
    return NpzTrainTestSplit(
        train_manifest=manifest.subset_by_indices(train_indices),
        test_manifest=manifest.subset_by_indices(test_indices),
        train_indices=train_indices,
        test_indices=test_indices,
        sample_seed=None,
    )


def build_piecewise_linear_input_function(times: np.ndarray, values: np.ndarray) -> Callable[[float], np.ndarray]:
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError(f"times must have shape (N,), N>=2, got {times.shape}")
    if values.ndim != 2 or values.shape[0] != times.shape[0]:
        raise ValueError(f"values must have shape ({times.shape[0]}, dp), got {values.shape}")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")

    grid = times.astype(np.float64, copy=True)
    table = values.astype(np.float64, copy=True)

    def input_function(t: float) -> np.ndarray:
        return np.asarray([np.interp(t, grid, table[:, j]) for j in range(table.shape[1])], dtype=np.float64)

    return input_function


def build_cubic_spline_input_function(times: np.ndarray, values: np.ndarray) -> Callable[[float], np.ndarray]:
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError(f"times must have shape (N,), N>=2, got {times.shape}")
    if values.ndim != 2 or values.shape[0] != times.shape[0]:
        raise ValueError(f"values must have shape ({times.shape[0]}, dp), got {values.shape}")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")

    grid = times.astype(np.float64, copy=True)
    table = values.astype(np.float64, copy=True)
    spline = CubicSpline(grid, table, axis=0, bc_type="natural", extrapolate=True)

    def input_function(t: float) -> np.ndarray:
        return np.asarray(spline(float(t)), dtype=np.float64)

    return input_function


def _read_scalar_string(npz_data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz_data.files:
        return default
    raw = npz_data[key]
    if getattr(raw, "shape", None) == ():
        return str(raw.item())
    if np.isscalar(raw):
        return str(raw)
    if raw.size == 1:
        return str(np.asarray(raw).reshape(-1)[0])
    raise ValueError(f"{key} must be scalar-like, got shape {raw.shape}")


def load_npz_qoi_sample(path: str | Path) -> NpzQoiSample:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        if "observation_times" not in data.files:
            raise ValueError(f"{path} is missing required field 'observation_times'.")
        if "u0" not in data.files:
            raise ValueError(f"{path} is missing required field 'u0'.")
        if "qoi_observations" not in data.files:
            raise ValueError(f"{path} is missing required field 'qoi_observations'.")

        observation_times = np.asarray(data["observation_times"], dtype=np.float64)
        u0 = np.asarray(data["u0"], dtype=np.float64)
        qoi_observations = np.asarray(data["qoi_observations"], dtype=np.float64)
        input_times = None if "input_times" not in data.files else np.asarray(data["input_times"], dtype=np.float64)
        input_values = None if "input_values" not in data.files else np.asarray(data["input_values"], dtype=np.float64)
        sample_id = _read_scalar_string(data, "sample_id", path.stem)

        metadata: dict[str, np.ndarray | str | float | int] = {}
        for key in data.files:
            if key in {"observation_times", "u0", "qoi_observations", "input_times", "input_values", "sample_id"}:
                continue
            raw = data[key]
            metadata[key] = raw.item() if getattr(raw, "shape", None) == () else raw.copy()

    return NpzQoiSample(
        sample_id=sample_id,
        observation_times=observation_times,
        u0=u0,
        qoi_observations=qoi_observations,
        input_times=input_times,
        input_values=input_values,
        metadata=metadata or None,
    )


def load_npz_sample_manifest(path: str | Path) -> NpzSampleManifest:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        if "sample_paths" not in data.files:
            raise ValueError(f"{path} is missing required field 'sample_paths'.")
        sample_paths = tuple(Path(str(item)) for item in data["sample_paths"].tolist())
        if "sample_ids" in data.files:
            sample_ids = tuple(str(item) for item in data["sample_ids"].tolist())
        else:
            sample_ids = tuple(sample_path.stem for sample_path in sample_paths)

    return NpzSampleManifest(root_dir=path.parent, sample_paths=sample_paths, sample_ids=sample_ids)


def save_npz_qoi_sample(path: str | Path, sample: NpzQoiSample) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray | str | float | int] = {
        "sample_id": np.asarray(sample.sample_id),
        "observation_times": np.asarray(sample.observation_times, dtype=np.float64),
        "u0": np.asarray(sample.u0, dtype=np.float64),
        "qoi_observations": np.asarray(sample.qoi_observations, dtype=np.float64),
    }
    if sample.input_times is not None:
        payload["input_times"] = np.asarray(sample.input_times, dtype=np.float64)
    if sample.input_values is not None:
        payload["input_values"] = np.asarray(sample.input_values, dtype=np.float64)
    if sample.metadata is not None:
        for key, value in sample.metadata.items():
            payload[key] = value
    np.savez(path, **payload)


def save_npz_sample_manifest(path: str | Path, manifest: NpzSampleManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root_dir = path.parent
    sample_paths = []
    for sample_path in manifest.absolute_paths():
        try:
            relative_path = sample_path.relative_to(root_dir)
        except ValueError:
            relative_path = sample_path
        sample_paths.append(str(relative_path))
    np.savez(
        path,
        sample_paths=np.asarray(sample_paths, dtype=object),
        sample_ids=np.asarray(manifest.sample_ids, dtype=object),
    )
