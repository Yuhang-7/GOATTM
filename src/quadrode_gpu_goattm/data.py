from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def _as_1d_time(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(values.shape)}")
    if values.numel() < 2:
        raise ValueError(f"{name} must contain at least two time points")
    if not bool(torch.all(values[1:] > values[:-1]).detach().cpu()):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def linear_interpolate(
    time_grid: torch.Tensor,
    values: torch.Tensor,
    query_times: torch.Tensor,
    *,
    check_bounds: bool = True,
) -> torch.Tensor:
    """Piecewise-linear interpolation along the first axis.

    ``values`` has shape ``[T, ...]``. ``query_times`` may have any shape; the
    output has shape ``query_times.shape + values.shape[1:]``.
    """

    time_grid = _as_1d_time(time_grid, "time_grid")
    if values.shape[0] != time_grid.numel():
        raise ValueError("values first dimension must match time_grid length")
    query = query_times.to(device=time_grid.device, dtype=time_grid.dtype)
    if check_bounds:
        below = bool((query < time_grid[0]).any().detach().cpu())
        above = bool((query > time_grid[-1]).any().detach().cpu())
        if below or above:
            raise ValueError("query_times are outside the interpolation time_grid")
    query_flat = query.reshape(-1)
    right = torch.searchsorted(time_grid, query_flat, right=False)
    right = right.clamp(1, time_grid.numel() - 1)
    left = right - 1
    t0 = time_grid[left]
    t1 = time_grid[right]
    weight = ((query_flat - t0) / (t1 - t0)).to(dtype=values.dtype)
    left_values = values.index_select(0, left.to(device=values.device))
    right_values = values.index_select(0, right.to(device=values.device))
    while weight.ndim < left_values.ndim:
        weight = weight.unsqueeze(-1)
    left_values.mul_(1.0 - weight)
    right_values.mul_(weight)
    out = left_values.add_(right_values)
    return out.reshape(*query.shape, *values.shape[1:])


@dataclass(frozen=True)
class ContinuousSample:
    sample_id: str
    observation_times: torch.Tensor
    qoi: torch.Tensor
    u0: torch.Tensor | None = None
    input_times: torch.Tensor | None = None
    input_values: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _as_1d_time(self.observation_times, "observation_times")
        if self.qoi.shape[0] != self.observation_times.numel():
            raise ValueError("qoi first dimension must match observation_times")
        if (self.input_times is None) != (self.input_values is None):
            raise ValueError("input_times and input_values must be provided together")
        if self.input_times is not None and self.input_values is not None:
            _as_1d_time(self.input_times, "input_times")
            if self.input_values.shape[0] != self.input_times.numel():
                raise ValueError("input_values first dimension must match input_times")

    @property
    def output_dim(self) -> int:
        return int(self.qoi.shape[-1])

    @property
    def input_dim(self) -> int:
        if self.input_values is None:
            return 0
        return int(self.input_values.shape[-1])

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ContinuousSample":
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ContinuousSample(
            sample_id=self.sample_id,
            observation_times=self.observation_times.to(**kwargs),
            qoi=self.qoi.to(**kwargs),
            u0=None if self.u0 is None else self.u0.to(**kwargs),
            input_times=None if self.input_times is None else self.input_times.to(**kwargs),
            input_values=None if self.input_values is None else self.input_values.to(**kwargs),
            metadata=None if self.metadata is None else dict(self.metadata),
        )


@dataclass(frozen=True)
class ContinuousBatch:
    sample_ids: tuple[str, ...]
    observation_times: torch.Tensor
    qoi: torch.Tensor
    u0: torch.Tensor | None = None
    input_times: torch.Tensor | None = None
    input_values: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    @property
    def steps(self) -> int:
        return int(self.observation_times.numel() - 1)

    @property
    def step_size(self) -> float:
        dt = self.observation_times[1:] - self.observation_times[:-1]
        if not torch.allclose(dt, dt[0].expand_as(dt)):
            raise ValueError("step_size is only defined for a uniform observation grid")
        return float(dt[0].detach().cpu())

    def midpoint_times(self) -> torch.Tensor:
        return 0.5 * (self.observation_times[:-1] + self.observation_times[1:])

    def midpoint_inputs(self, *, check_bounds: bool = True) -> torch.Tensor | None:
        if self.input_times is None or self.input_values is None:
            return None
        midpoint_times = self.midpoint_times()
        input_times = self.input_times.to(device=midpoint_times.device, dtype=midpoint_times.dtype)
        if input_times.shape == midpoint_times.shape and torch.allclose(input_times, midpoint_times):
            return self.input_values
        if input_times.numel() == 2 * midpoint_times.numel() + 1:
            midpoint_grid = input_times[1::2]
            if torch.allclose(midpoint_grid, midpoint_times):
                return self.input_values[1::2]
        return linear_interpolate(self.input_times, self.input_values, midpoint_times, check_bounds=check_bounds)

    def substep_midpoint_times(self, substeps: int) -> torch.Tensor:
        substeps = int(substeps)
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        left = self.observation_times[:-1]
        right = self.observation_times[1:]
        dt = right - left
        offsets = (torch.arange(substeps, device=left.device, dtype=left.dtype) + 0.5) / float(substeps)
        return left[:, None] + dt[:, None] * offsets[None, :]

    def substep_midpoint_inputs(self, substeps: int, *, check_bounds: bool = True) -> torch.Tensor | None:
        if self.input_times is None or self.input_values is None:
            return None
        substeps = int(substeps)
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        if substeps == 1:
            values = self.midpoint_inputs(check_bounds=check_bounds)
            return None if values is None else values[:, None]
        query = self.substep_midpoint_times(substeps)
        input_times = self.input_times.to(device=query.device, dtype=query.dtype)
        if input_times.shape == self.midpoint_times().shape and self.input_values.shape[0] == self.steps:
            return self.input_values[:, None].expand(-1, substeps, -1, -1)
        values = linear_interpolate(self.input_times, self.input_values, query.reshape(-1), check_bounds=check_bounds)
        return values.reshape(self.steps, substeps, *self.input_values.shape[1:])

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ContinuousBatch":
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ContinuousBatch(
            sample_ids=self.sample_ids,
            observation_times=self.observation_times.to(**kwargs),
            qoi=self.qoi.to(**kwargs),
            u0=None if self.u0 is None else self.u0.to(**kwargs),
            input_times=None if self.input_times is None else self.input_times.to(**kwargs),
            input_values=None if self.input_values is None else self.input_values.to(**kwargs),
        )


def slice_batch(batch: ContinuousBatch, start: int, end: int) -> ContinuousBatch:
    start = int(start)
    end = int(end)
    if start < 0 or end < start or end > batch.batch_size:
        raise ValueError("invalid batch slice")
    sample_slice = slice(start, end)
    return ContinuousBatch(
        sample_ids=batch.sample_ids[sample_slice],
        observation_times=batch.observation_times,
        qoi=batch.qoi[:, sample_slice],
        u0=None if batch.u0 is None else batch.u0[sample_slice],
        input_times=batch.input_times,
        input_values=None if batch.input_values is None else batch.input_values[:, sample_slice],
    )


@dataclass(frozen=True)
class SampleManifest:
    sample_ids: tuple[str, ...]
    sample_paths: tuple[Path, ...]
    root: Path | None = None

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(self.sample_paths):
            raise ValueError("sample_ids and sample_paths must have the same length")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def absolute_paths(self) -> tuple[Path, ...]:
        if self.root is None:
            return tuple(path.expanduser().resolve() for path in self.sample_paths)
        return tuple((self.root / path).expanduser().resolve() for path in self.sample_paths)

    def subset(self, indices: Iterable[int]) -> "SampleManifest":
        idx = tuple(int(i) for i in indices)
        return SampleManifest(
            sample_ids=tuple(self.sample_ids[i] for i in idx),
            sample_paths=tuple(self.sample_paths[i] for i in idx),
            root=self.root,
        )


def split_manifest(
    manifest: SampleManifest,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[SampleManifest, SampleManifest]:
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(len(manifest), generator=generator).tolist()
    train_count = max(1, min(len(manifest) - 1, int(round(len(manifest) * float(train_fraction)))))
    return manifest.subset(perm[:train_count]), manifest.subset(perm[train_count:])


class ContinuousDataset:
    """Manifest-backed continuous sample loader."""

    def __init__(
        self,
        manifest: SampleManifest,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.manifest = manifest
        self.paths = manifest.absolute_paths()
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.manifest)

    def load_sample(self, index: int) -> ContinuousSample:
        return load_sample_npz(self.paths[int(index)], device=self.device, dtype=self.dtype)

    def load_indices(self, indices: Iterable[int]) -> ContinuousBatch:
        return batch_samples([self.load_sample(i) for i in indices])

    def batch(
        self,
        *,
        batch_size: int | None = None,
        indices: Iterable[int] | None = None,
    ) -> ContinuousBatch:
        if indices is None:
            if batch_size is None:
                indices = range(len(self))
            else:
                indices = range(min(int(batch_size), len(self)))
        return self.load_indices(indices)

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Iterable[ContinuousBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = list(range(len(self)))
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]
        for start in range(0, len(indices), int(batch_size)):
            chunk = indices[start : start + int(batch_size)]
            if drop_last and len(chunk) < int(batch_size):
                continue
            yield self.load_indices(chunk)


def batch_samples(samples: list[ContinuousSample]) -> ContinuousBatch:
    if not samples:
        raise ValueError("samples must be nonempty")
    base_t = samples[0].observation_times
    for sample in samples[1:]:
        torch.testing.assert_close(sample.observation_times, base_t)
    qoi = torch.stack([sample.qoi for sample in samples], dim=1)
    u0 = None
    if samples[0].u0 is not None:
        if any(sample.u0 is None for sample in samples):
            raise ValueError("either all samples or no samples must define u0")
        u0 = torch.stack([sample.u0 for sample in samples if sample.u0 is not None], dim=0)
    input_times = None
    input_values = None
    if samples[0].input_times is not None:
        if any(sample.input_times is None or sample.input_values is None for sample in samples):
            raise ValueError("either all samples or no samples must define inputs")
        input_times = samples[0].input_times
        for sample in samples[1:]:
            torch.testing.assert_close(sample.input_times, input_times)
        input_values = torch.stack([sample.input_values for sample in samples if sample.input_values is not None], dim=1)
    return ContinuousBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        observation_times=base_t,
        qoi=qoi,
        u0=u0,
        input_times=input_times,
        input_values=input_values,
    )


def save_sample_npz(path: str | Path, sample: ContinuousSample) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "sample_id": np.array(sample.sample_id),
        "observation_times": sample.observation_times.detach().cpu().numpy(),
        "qoi": sample.qoi.detach().cpu().numpy(),
        "metadata_json": np.array(json.dumps(sample.metadata or {}, ensure_ascii=True)),
    }
    if sample.u0 is not None:
        payload["u0"] = sample.u0.detach().cpu().numpy()
    if sample.input_times is not None and sample.input_values is not None:
        payload["input_times"] = sample.input_times.detach().cpu().numpy()
        payload["input_values"] = sample.input_values.detach().cpu().numpy()
    np.savez(path, **payload)


def load_sample_npz(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> ContinuousSample:
    data = np.load(path, allow_pickle=True)
    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(data[name], device=device, dtype=dtype)
    qoi_key = "qoi" if "qoi" in data else "qoi_observations"
    if qoi_key not in data:
        raise KeyError(f"{path} does not contain 'qoi' or 'qoi_observations'")
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"].item()))
    return ContinuousSample(
        sample_id=str(data["sample_id"].item()),
        observation_times=tensor("observation_times"),
        qoi=tensor(qoi_key),
        u0=tensor("u0") if "u0" in data else None,
        input_times=tensor("input_times") if "input_times" in data else None,
        input_values=tensor("input_values") if "input_values" in data else None,
        metadata=metadata,
    )


def save_manifest_npz(path: str | Path, manifest: SampleManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sample_ids=np.asarray(manifest.sample_ids, dtype=object),
        sample_paths=np.asarray([str(p) for p in manifest.sample_paths], dtype=object),
        root=np.array("" if manifest.root is None else str(manifest.root)),
    )


def load_manifest_npz(path: str | Path) -> SampleManifest:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    root_str = str(data["root"].item()) if "root" in data else ""
    sample_paths = tuple(Path(str(x)) for x in data["sample_paths"].tolist())
    root = None if root_str == "" else Path(root_str)
    if root is None and any(not sample_path.is_absolute() for sample_path in sample_paths):
        root = path.expanduser().resolve().parent
    return SampleManifest(
        sample_ids=tuple(str(x) for x in data["sample_ids"].tolist()),
        sample_paths=sample_paths,
        root=root,
    )
