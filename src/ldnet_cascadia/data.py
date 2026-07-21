from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


def _sample_ids(payload: dict[str, Any], count: int) -> tuple[str, ...]:
    ids = payload.get("sample_ids")
    if ids is None:
        return tuple(str(i) for i in range(count))
    return tuple(str(x) for x in ids[:count])


def _take_samples(value: torch.Tensor, indices: Sequence[int] | None) -> torch.Tensor:
    if indices is None:
        return value
    index = torch.as_tensor(indices, dtype=torch.long)
    return value.index_select(1, index)


def _observation_times(payload: dict[str, Any], *, time_mode: str) -> torch.Tensor:
    times = payload.get("observation_times")
    if times is None:
        times = torch.arange(payload["qoi"].shape[0], dtype=torch.float64)
    else:
        times = torch.as_tensor(times, dtype=torch.float64)
    if time_mode == "step_index":
        return torch.arange(times.numel(), dtype=torch.float64)
    if time_mode != "normalized":
        raise ValueError(f"unknown time_mode {time_mode!r}")
    return times


def _metadata_text(metadata: dict[str, Any]) -> str:
    try:
        return json.dumps(metadata, sort_keys=True).lower()
    except TypeError:
        return str(metadata).lower()


def assert_displacement_payload(metadata: dict[str, Any]) -> None:
    """Raise when metadata strongly indicates raw increments rather than displacement."""

    text = _metadata_text(metadata)
    has_increment = "increment" in text
    has_displacement = "displacement" in text or "cumsum" in text or "accum" in text
    if has_increment and not has_displacement:
        raise ValueError(
            "packed payload metadata looks like raw increment data; expected seafloor displacement"
        )


@dataclass(frozen=True)
class CascadiaPackedBatch:
    sample_ids: tuple[str, ...]
    observation_times: torch.Tensor
    qoi: torch.Tensor
    input_values: torch.Tensor
    metadata: dict[str, Any]

    @property
    def batch_size(self) -> int:
        return int(self.qoi.shape[1])

    @property
    def steps(self) -> int:
        return int(self.observation_times.numel() - 1)

    @property
    def input_dim(self) -> int:
        return int(self.input_values.shape[-1])

    @property
    def output_dim(self) -> int:
        return int(self.qoi.shape[-1])

    @property
    def step_size(self) -> float:
        dt = self.observation_times[1:] - self.observation_times[:-1]
        if not torch.allclose(dt, dt[0].expand_as(dt)):
            raise ValueError("observation_times are not uniformly spaced")
        return float(dt[0].detach().cpu())

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "CascadiaPackedBatch":
        kwargs: dict[str, Any] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return CascadiaPackedBatch(
            sample_ids=self.sample_ids,
            observation_times=self.observation_times.to(**kwargs),
            qoi=self.qoi.to(**kwargs),
            input_values=self.input_values.to(**kwargs),
            metadata=dict(self.metadata),
        )

    def midpoint_inputs(self) -> torch.Tensor:
        if self.input_values.shape[0] == self.steps:
            return self.input_values
        if self.input_values.shape[0] == self.steps + 1:
            return 0.5 * (self.input_values[:-1] + self.input_values[1:])
        raise ValueError(
            "input_values first dimension must be steps or steps + 1; "
            f"got {self.input_values.shape[0]} for {self.steps} steps"
        )

    def input_context(self, mode: str = "final") -> torch.Tensor:
        if mode == "initial":
            return self.input_values[0]
        if mode == "final":
            return self.input_values[-1]
        if mode == "mean":
            return self.input_values.mean(dim=0)
        if mode == "maxabs":
            idx = self.input_values.abs().argmax(dim=0, keepdim=True)
            return self.input_values.gather(0, idx).squeeze(0)
        if mode == "flatten":
            return self.input_values.permute(1, 0, 2).reshape(self.batch_size, -1)
        raise ValueError(f"unknown input context mode {mode!r}")


def load_packed_payload(path: str | Path, *, require_displacement: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    if require_displacement:
        assert_displacement_payload(metadata)
    return payload, metadata


def batch_from_payload(
    payload: dict[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
    sample_indices: Sequence[int] | None = None,
    sample_limit: int | None = None,
    time_mode: str = "step_index",
    max_time_steps: int | None = None,
) -> CascadiaPackedBatch:
    if sample_indices is None and sample_limit is not None and int(sample_limit) > 0:
        sample_indices = list(range(min(int(sample_limit), int(payload["qoi"].shape[1]))))
    qoi = _take_samples(payload["qoi"], sample_indices)
    inputs = _take_samples(payload["input_values"], sample_indices)
    if max_time_steps is not None and int(max_time_steps) > 0:
        keep = int(max_time_steps) + 1
        qoi = qoi[:keep]
        if inputs.shape[0] >= keep:
            inputs = inputs[:keep]
        else:
            inputs = inputs[: int(max_time_steps)]
    sample_count = int(qoi.shape[1])
    if sample_indices is None:
        sample_ids = _sample_ids(payload, sample_count)
    else:
        ids = payload.get("sample_ids")
        sample_ids = tuple(str(ids[i]) for i in sample_indices) if ids is not None else tuple(str(i) for i in sample_indices)
    times = _observation_times(payload, time_mode=time_mode)[: qoi.shape[0]]
    metadata = dict(payload.get("metadata", {}))
    return CascadiaPackedBatch(
        sample_ids=sample_ids,
        observation_times=times.to(device=device, dtype=dtype),
        qoi=qoi.to(device=device, dtype=dtype),
        input_values=inputs.to(device=device, dtype=dtype),
        metadata=metadata,
    )


def load_packed_batch(
    path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
    sample_limit: int | None = None,
    time_mode: str = "step_index",
    max_time_steps: int | None = None,
    require_displacement: bool = True,
) -> CascadiaPackedBatch:
    payload, _ = load_packed_payload(path, require_displacement=require_displacement)
    return batch_from_payload(
        payload,
        device=device,
        dtype=dtype,
        sample_limit=sample_limit,
        time_mode=time_mode,
        max_time_steps=max_time_steps,
    )
