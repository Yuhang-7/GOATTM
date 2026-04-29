from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.npz_dataset import (
    NpzQoiSample,
    NpzSampleManifest,
    load_npz_qoi_sample,
    load_npz_sample_manifest,
    save_npz_qoi_sample,
    save_npz_sample_manifest,
)
from ..runtime.distributed import DistributedContext


@dataclass(frozen=True)
class DatasetNormalizationStats:
    qoi_mean: np.ndarray
    qoi_std: np.ndarray
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None
    epsilon: float = 1e-12
    scale_mode: str = "max_abs"
    target_max_abs: float = 0.9
    qoi_centered_max_abs: np.ndarray | None = None
    input_centered_max_abs: np.ndarray | None = None


@dataclass(frozen=True)
class NormalizedDatasetArtifacts:
    output_dir: Path
    stats_path: Path
    train_manifest_path: Path
    test_manifest_path: Path | None
    train_manifest: NpzSampleManifest
    test_manifest: NpzSampleManifest | None
    stats: DatasetNormalizationStats


def compute_training_normalization_stats(
    manifest: str | Path | NpzSampleManifest,
    context: DistributedContext | None = None,
    epsilon: float = 1e-12,
    target_max_abs: float = 0.9,
) -> DatasetNormalizationStats:
    if isinstance(manifest, (str, Path)):
        manifest = load_npz_sample_manifest(manifest)
    if len(manifest) == 0:
        raise ValueError("Training manifest must contain at least one sample.")
    if target_max_abs <= 0.0:
        raise ValueError(f"target_max_abs must be positive, got {target_max_abs}")

    context = DistributedContext() if context is None else context
    reference_sample = load_npz_qoi_sample(manifest.absolute_paths()[0])
    dq = reference_sample.output_dimension
    dp = reference_sample.input_dimension
    has_inputs = reference_sample.input_values is not None

    local_qoi_sum = np.zeros(dq, dtype=np.float64)
    local_qoi_count = 0

    local_input_sum = None if not has_inputs else np.zeros(dp, dtype=np.float64)
    local_input_count = 0

    for _, path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(path)
        qoi = np.asarray(sample.qoi_observations, dtype=np.float64)
        local_qoi_sum += np.sum(qoi, axis=0)
        local_qoi_count += int(qoi.shape[0])

        if has_inputs:
            if sample.input_values is None:
                raise ValueError("Input normalization requested but a sample is missing input_values.")
            inputs = np.asarray(sample.input_values, dtype=np.float64)
            local_input_sum += np.sum(inputs, axis=0)
            local_input_count += int(inputs.shape[0])

    qoi_sum = context.allreduce_array_sum(local_qoi_sum)
    qoi_count = context.allreduce_int_sum(local_qoi_count)
    if qoi_count <= 0:
        raise ValueError("Training manifest does not contain any QoI observations.")
    qoi_mean = qoi_sum / float(qoi_count)

    input_mean = None
    if has_inputs:
        assert local_input_sum is not None
        input_sum = context.allreduce_array_sum(local_input_sum)
        input_count = context.allreduce_int_sum(local_input_count)
        if input_count <= 0:
            raise ValueError("Training manifest declares inputs but no input samples were found.")
        input_mean = input_sum / float(input_count)

    local_qoi_centered_max_abs = np.zeros(dq, dtype=np.float64)
    local_input_centered_max_abs = None if not has_inputs else np.zeros(dp, dtype=np.float64)
    for _, path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(path)
        qoi = np.asarray(sample.qoi_observations, dtype=np.float64)
        local_qoi_centered_max_abs = np.maximum(local_qoi_centered_max_abs, np.max(np.abs(qoi - qoi_mean[None, :]), axis=0))

        if has_inputs:
            if sample.input_values is None or input_mean is None:
                raise ValueError("Input normalization requested but a sample is missing input_values.")
            assert local_input_centered_max_abs is not None
            inputs = np.asarray(sample.input_values, dtype=np.float64)
            local_input_centered_max_abs = np.maximum(
                local_input_centered_max_abs,
                np.max(np.abs(inputs - input_mean[None, :]), axis=0),
            )

    qoi_centered_max_abs = context.allreduce_array_max(local_qoi_centered_max_abs)
    qoi_std = qoi_centered_max_abs / float(target_max_abs)
    qoi_std[qoi_centered_max_abs < epsilon] = 1.0

    input_std = None
    input_centered_max_abs = None
    if has_inputs:
        assert local_input_centered_max_abs is not None
        input_centered_max_abs = context.allreduce_array_max(local_input_centered_max_abs)
        input_std = input_centered_max_abs / float(target_max_abs)
        input_std[input_centered_max_abs < epsilon] = 1.0

    return DatasetNormalizationStats(
        qoi_mean=qoi_mean,
        qoi_std=qoi_std,
        input_mean=input_mean,
        input_std=input_std,
        epsilon=float(epsilon),
        scale_mode="max_abs",
        target_max_abs=float(target_max_abs),
        qoi_centered_max_abs=qoi_centered_max_abs,
        input_centered_max_abs=input_centered_max_abs,
    )


def normalize_npz_qoi_sample(
    sample: NpzQoiSample,
    stats: DatasetNormalizationStats,
) -> NpzQoiSample:
    normalized_qoi = (np.asarray(sample.qoi_observations, dtype=np.float64) - stats.qoi_mean[None, :]) / stats.qoi_std[None, :]
    normalized_inputs = None
    if sample.input_values is not None:
        if stats.input_mean is None or stats.input_std is None:
            raise ValueError("Input values are present, but input normalization stats are unavailable.")
        normalized_inputs = (np.asarray(sample.input_values, dtype=np.float64) - stats.input_mean[None, :]) / stats.input_std[None, :]
    return NpzQoiSample(
        sample_id=sample.sample_id,
        observation_times=np.asarray(sample.observation_times, dtype=np.float64),
        u0=np.asarray(sample.u0, dtype=np.float64),
        qoi_observations=normalized_qoi,
        input_times=None if sample.input_times is None else np.asarray(sample.input_times, dtype=np.float64),
        input_values=normalized_inputs,
        metadata=sample.metadata,
    )


def materialize_normalized_train_test_split(
    train_manifest: str | Path | NpzSampleManifest,
    output_dir: str | Path,
    test_manifest: str | Path | NpzSampleManifest | None = None,
    context: DistributedContext | None = None,
    epsilon: float = 1e-12,
    target_max_abs: float = 0.9,
) -> NormalizedDatasetArtifacts:
    if isinstance(train_manifest, (str, Path)):
        train_manifest = load_npz_sample_manifest(train_manifest)
    if test_manifest is not None and isinstance(test_manifest, (str, Path)):
        test_manifest = load_npz_sample_manifest(test_manifest)

    context = DistributedContext() if context is None else context
    output_root = Path(output_dir)
    samples_root = output_root / "samples"
    train_root = samples_root / "train"
    test_root = samples_root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    if test_manifest is not None:
        test_root.mkdir(parents=True, exist_ok=True)

    stats = compute_training_normalization_stats(
        train_manifest,
        context=context,
        epsilon=epsilon,
        target_max_abs=target_max_abs,
    )
    stats_path = output_root / "normalization_stats.npz"
    train_manifest_path = output_root / "train_manifest.npz"
    test_manifest_path = None if test_manifest is None else output_root / "test_manifest.npz"

    train_sample_paths = _materialize_normalized_manifest_partition(
        manifest=train_manifest,
        destination_root=train_root,
        split_name="train",
        stats=stats,
        context=context,
    )
    test_sample_paths = None
    if test_manifest is not None:
        test_sample_paths = _materialize_normalized_manifest_partition(
            manifest=test_manifest,
            destination_root=test_root,
            split_name="test",
            stats=stats,
            context=context,
        )

    context.barrier()
    normalized_train_manifest = NpzSampleManifest(
        root_dir=output_root,
        sample_paths=tuple(path.relative_to(output_root) for path in train_sample_paths),
        sample_ids=train_manifest.sample_ids,
    )
    normalized_test_manifest = None
    if test_manifest is not None:
        assert test_sample_paths is not None
        normalized_test_manifest = NpzSampleManifest(
            root_dir=output_root,
            sample_paths=tuple(path.relative_to(output_root) for path in test_sample_paths),
            sample_ids=test_manifest.sample_ids,
        )

    if context.rank == 0:
        np.savez(
            stats_path,
            qoi_mean=np.asarray(stats.qoi_mean, dtype=np.float64),
            qoi_std=np.asarray(stats.qoi_std, dtype=np.float64),
            input_mean=None if stats.input_mean is None else np.asarray(stats.input_mean, dtype=np.float64),
            input_std=None if stats.input_std is None else np.asarray(stats.input_std, dtype=np.float64),
            epsilon=np.asarray(stats.epsilon, dtype=np.float64),
            scale_mode=np.asarray(stats.scale_mode),
            target_max_abs=np.asarray(stats.target_max_abs, dtype=np.float64),
            qoi_centered_max_abs=None
            if stats.qoi_centered_max_abs is None
            else np.asarray(stats.qoi_centered_max_abs, dtype=np.float64),
            input_centered_max_abs=None
            if stats.input_centered_max_abs is None
            else np.asarray(stats.input_centered_max_abs, dtype=np.float64),
        )
        save_npz_sample_manifest(train_manifest_path, normalized_train_manifest)
        if normalized_test_manifest is not None and test_manifest_path is not None:
            save_npz_sample_manifest(test_manifest_path, normalized_test_manifest)

    context.barrier()
    return NormalizedDatasetArtifacts(
        output_dir=output_root,
        stats_path=stats_path,
        train_manifest_path=train_manifest_path,
        test_manifest_path=test_manifest_path,
        train_manifest=normalized_train_manifest,
        test_manifest=normalized_test_manifest,
        stats=stats,
    )


def _materialize_normalized_manifest_partition(
    manifest: NpzSampleManifest,
    destination_root: Path,
    split_name: str,
    stats: DatasetNormalizationStats,
    context: DistributedContext,
) -> tuple[Path, ...]:
    normalized_paths = []
    absolute_paths = manifest.absolute_paths()
    destination_root.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(absolute_paths):
        output_path = destination_root / f"{idx:06d}_{manifest.sample_ids[idx]}.npz"
        normalized_paths.append(output_path)

    for local_index, (sample_id, path) in enumerate(manifest.entries_for_rank(context.rank, context.size)):
        global_index = manifest.sample_ids.index(sample_id)
        sample = load_npz_qoi_sample(path)
        normalized = normalize_npz_qoi_sample(sample, stats)
        metadata = dict(normalized.metadata or {})
        metadata["split"] = split_name
        save_npz_qoi_sample(
            normalized_paths[global_index],
            NpzQoiSample(
                sample_id=normalized.sample_id,
                observation_times=normalized.observation_times,
                u0=normalized.u0,
                qoi_observations=normalized.qoi_observations,
                input_times=normalized.input_times,
                input_values=normalized.input_values,
                metadata=metadata,
            ),
        )
    return tuple(normalized_paths)
