from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from goattm.data import NpzSampleManifest, load_npz_qoi_sample, load_npz_sample_manifest


@dataclass(frozen=True)
class LatentDynamicsNormalization:
    qoi_mean: np.ndarray
    qoi_scale: np.ndarray
    input_mean: np.ndarray | None
    input_scale: np.ndarray | None
    target_max_abs: float = 0.9
    epsilon: float = 1e-12

    @classmethod
    def from_training_arrays(
        cls,
        qoi: np.ndarray,
        inputs: np.ndarray | None,
        target_max_abs: float = 0.9,
        epsilon: float = 1e-12,
    ) -> "LatentDynamicsNormalization":
        if qoi.ndim != 3:
            raise ValueError(f"qoi must have shape (N,T,dq), got {qoi.shape}")
        qoi_mean = np.mean(qoi.reshape(-1, qoi.shape[-1]), axis=0)
        qoi_max_abs = np.max(np.abs(qoi.reshape(-1, qoi.shape[-1]) - qoi_mean[None, :]), axis=0)
        qoi_scale = qoi_max_abs / float(target_max_abs)
        qoi_scale[qoi_max_abs < epsilon] = 1.0

        input_mean = None
        input_scale = None
        if inputs is not None:
            if inputs.ndim != 3:
                raise ValueError(f"inputs must have shape (N,T,dp), got {inputs.shape}")
            input_mean = np.mean(inputs.reshape(-1, inputs.shape[-1]), axis=0)
            input_max_abs = np.max(
                np.abs(inputs.reshape(-1, inputs.shape[-1]) - input_mean[None, :]),
                axis=0,
            )
            input_scale = input_max_abs / float(target_max_abs)
            input_scale[input_max_abs < epsilon] = 1.0

        return cls(
            qoi_mean=qoi_mean.astype(np.float64),
            qoi_scale=qoi_scale.astype(np.float64),
            input_mean=None if input_mean is None else input_mean.astype(np.float64),
            input_scale=None if input_scale is None else input_scale.astype(np.float64),
            target_max_abs=float(target_max_abs),
            epsilon=float(epsilon),
        )

    def normalize_qoi(self, qoi: np.ndarray) -> np.ndarray:
        return (np.asarray(qoi, dtype=np.float64) - self.qoi_mean[None, None, :]) / self.qoi_scale[None, None, :]

    def unnormalize_qoi(self, qoi: np.ndarray) -> np.ndarray:
        return np.asarray(qoi, dtype=np.float64) * self.qoi_scale[None, None, :] + self.qoi_mean[None, None, :]

    def normalize_inputs(self, inputs: np.ndarray | None) -> np.ndarray | None:
        if inputs is None:
            return None
        if self.input_mean is None or self.input_scale is None:
            raise ValueError("Input normalization stats are unavailable.")
        return (np.asarray(inputs, dtype=np.float64) - self.input_mean[None, None, :]) / self.input_scale[None, None, :]

    def to_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray | float] = {
            "qoi_mean": self.qoi_mean,
            "qoi_scale": self.qoi_scale,
            "target_max_abs": float(self.target_max_abs),
            "epsilon": float(self.epsilon),
        }
        if self.input_mean is not None:
            payload["input_mean"] = self.input_mean
        if self.input_scale is not None:
            payload["input_scale"] = self.input_scale
        np.savez(path, **payload)

    @classmethod
    def from_npz(cls, path: str | Path) -> "LatentDynamicsNormalization":
        with np.load(path) as data:
            return cls(
                qoi_mean=np.asarray(data["qoi_mean"], dtype=np.float64),
                qoi_scale=np.asarray(data["qoi_scale"], dtype=np.float64),
                input_mean=None if "input_mean" not in data.files else np.asarray(data["input_mean"], dtype=np.float64),
                input_scale=None if "input_scale" not in data.files else np.asarray(data["input_scale"], dtype=np.float64),
                target_max_abs=float(data["target_max_abs"]) if "target_max_abs" in data.files else 0.9,
                epsilon=float(data["epsilon"]) if "epsilon" in data.files else 1e-12,
            )


@dataclass(frozen=True)
class LatentDynamicsDataset:
    times: torch.Tensor
    qoi: torch.Tensor
    inputs: torch.Tensor | None
    sample_ids: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return int(self.qoi.shape[0])

    @property
    def time_count(self) -> int:
        return int(self.qoi.shape[1])

    @property
    def qoi_dimension(self) -> int:
        return int(self.qoi.shape[2])

    @property
    def input_dimension(self) -> int:
        return 0 if self.inputs is None else int(self.inputs.shape[2])

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "LatentDynamicsDataset":
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return LatentDynamicsDataset(
            times=self.times.to(**kwargs),
            qoi=self.qoi.to(**kwargs),
            inputs=None if self.inputs is None else self.inputs.to(**kwargs),
            sample_ids=self.sample_ids,
        )


def load_manifest_tensors(
    manifest: str | Path | NpzSampleManifest,
    sample_count: int | None = None,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...]]:
    if isinstance(manifest, (str, Path)):
        manifest = load_npz_sample_manifest(manifest)
    if sample_count is None:
        sample_count = len(manifest)
    if sample_count <= 0 or sample_count > len(manifest):
        raise ValueError(f"sample_count must be in [1,{len(manifest)}], got {sample_count}")

    selected = manifest.subset_by_indices(tuple(range(sample_count)))
    paths = selected.absolute_paths()
    first = load_npz_qoi_sample(paths[0])
    times = np.asarray(first.observation_times, dtype=dtype)
    qoi = np.empty((sample_count, times.shape[0], first.output_dimension), dtype=dtype)
    inputs = None
    if first.input_values is not None:
        inputs = np.empty((sample_count, times.shape[0], first.input_dimension), dtype=dtype)

    for idx, path in enumerate(paths):
        sample = load_npz_qoi_sample(path)
        if sample.observation_times.shape != times.shape or not np.allclose(sample.observation_times, times):
            raise ValueError("LDNet runner currently expects all samples to share observation_times.")
        qoi[idx] = np.asarray(sample.qoi_observations, dtype=dtype)
        if inputs is not None:
            if sample.input_times is None or sample.input_values is None:
                raise ValueError("Some samples are missing input data.")
            inputs[idx] = _interpolate_inputs_to_times(
                np.asarray(sample.input_times, dtype=dtype),
                np.asarray(sample.input_values, dtype=dtype),
                times,
            )
    return times, qoi, inputs, selected.sample_ids


def make_train_test_tensors(
    manifest: str | Path | NpzSampleManifest,
    ntrain: int,
    ntest: int,
    normalize: bool = True,
    target_max_abs: float = 0.9,
    torch_dtype: torch.dtype = torch.float32,
) -> tuple[LatentDynamicsDataset, LatentDynamicsDataset, LatentDynamicsNormalization | None]:
    times, qoi, inputs, sample_ids = load_manifest_tensors(manifest, sample_count=ntrain + ntest)
    train_qoi = qoi[:ntrain]
    test_qoi = qoi[ntrain : ntrain + ntest]
    train_inputs = None if inputs is None else inputs[:ntrain]
    test_inputs = None if inputs is None else inputs[ntrain : ntrain + ntest]

    stats = None
    if normalize:
        stats = LatentDynamicsNormalization.from_training_arrays(
            train_qoi,
            train_inputs,
            target_max_abs=target_max_abs,
        )
        train_qoi = stats.normalize_qoi(train_qoi)
        test_qoi = stats.normalize_qoi(test_qoi)
        train_inputs = stats.normalize_inputs(train_inputs)
        test_inputs = stats.normalize_inputs(test_inputs)

    train = LatentDynamicsDataset(
        times=torch.as_tensor(times, dtype=torch_dtype),
        qoi=torch.as_tensor(train_qoi, dtype=torch_dtype),
        inputs=None if train_inputs is None else torch.as_tensor(train_inputs, dtype=torch_dtype),
        sample_ids=sample_ids[:ntrain],
    )
    test = LatentDynamicsDataset(
        times=torch.as_tensor(times, dtype=torch_dtype),
        qoi=torch.as_tensor(test_qoi, dtype=torch_dtype),
        inputs=None if test_inputs is None else torch.as_tensor(test_inputs, dtype=torch_dtype),
        sample_ids=sample_ids[ntrain : ntrain + ntest],
    )
    return train, test, stats


def _interpolate_inputs_to_times(input_times: np.ndarray, input_values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if input_values.ndim != 2:
        raise ValueError(f"input_values must have shape (M,dp), got {input_values.shape}")
    out = np.empty((times.shape[0], input_values.shape[1]), dtype=input_values.dtype)
    for j in range(input_values.shape[1]):
        out[:, j] = np.interp(times, input_times, input_values[:, j])
    return out
