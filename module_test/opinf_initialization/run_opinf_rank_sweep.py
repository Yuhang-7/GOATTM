from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.data import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest
from goattm.preprocess import (
    OpInfInitializationRegularization,
    OpInfLatentEmbeddingConfig,
    initialize_reduced_model_via_opinf,
)
from goattm.runtime import DistributedContext


OUTPUT_DIR = ROOT / "module_test" / "output_plots" / "opinf_rank_sweep"


def _build_context() -> DistributedContext:
    mpi_markers = [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS",
    ]
    if any(name in os.environ for name in mpi_markers):
        from mpi4py import MPI  # type: ignore

        return DistributedContext.from_comm(MPI.COMM_WORLD)
    return DistributedContext()


def _build_dataset(
    root: Path,
    sample_count: int,
    output_dimension: int,
    rng: np.random.Generator,
) -> tuple[Path, dict[str, object]]:
    observation_times = np.arange(0.0, 1.0 + 1e-12, 0.1, dtype=float)
    input_dimension = 1
    ab_pairs = rng.uniform(-2.0, 2.0, size=(output_dimension, 2))

    sample_paths: list[Path] = []
    sample_ids: list[str] = []
    all_qoi_values: list[np.ndarray] = []
    for sample_idx in range(sample_count):
        p_values = rng.uniform(-2.0, 2.0, size=(observation_times.shape[0], input_dimension))
        p_column = p_values[:, 0]
        qoi_columns = []
        for a_value, b_value in ab_pairs:
            qoi_columns.append(np.exp(a_value * p_column) + b_value * np.square(p_column - a_value))
        qoi_observations = np.column_stack(qoi_columns)
        sample = NpzQoiSample(
            sample_id=f"sample-{sample_idx:03d}",
            observation_times=observation_times,
            u0=qoi_observations[0].copy(),
            qoi_observations=qoi_observations,
            input_times=observation_times,
            input_values=p_values,
            metadata={"dataset_kind": "raw_qoi_exp_plus_quadratic"},
        )
        sample_path = root / f"sample_{sample_idx:03d}.npz"
        save_npz_qoi_sample(sample_path, sample)
        sample_paths.append(sample_path)
        sample_ids.append(sample.sample_id)
        all_qoi_values.append(qoi_observations)

    manifest = NpzSampleManifest(
        root_dir=root,
        sample_paths=tuple(Path(path.name) for path in sample_paths),
        sample_ids=tuple(sample_ids),
    )
    manifest_path = root / "manifest.npz"
    save_npz_sample_manifest(manifest_path, manifest)

    qoi_stack = np.concatenate(all_qoi_values, axis=0)
    metadata = {
        "sample_count": sample_count,
        "output_dimension": output_dimension,
        "ab_pairs": ab_pairs.tolist(),
        "qoi_abs_mean": float(np.mean(np.abs(qoi_stack))),
        "qoi_abs_max": float(np.max(np.abs(qoi_stack))),
        "qoi_l2_mean_per_observation": float(np.mean(np.linalg.norm(qoi_stack, axis=1))),
    }
    return manifest_path, metadata


def _norm_summary(rank: int, result) -> dict[str, object]:
    dynamics = result.dynamics
    return {
        "rank": rank,
        "regression_relative_residual": float(result.regression_relative_residual),
        "a_fro_norm": float(np.linalg.norm(dynamics.a, ord="fro")),
        "a_l2_norm": float(np.linalg.norm(dynamics.a, ord=2)),
        "h_fro_norm": float(np.linalg.norm(dynamics.h_matrix, ord="fro")),
        "h_l2_norm": float(np.linalg.norm(dynamics.h_matrix, ord=2)),
        "b_fro_norm": 0.0 if dynamics.b is None else float(np.linalg.norm(dynamics.b, ord="fro")),
        "b_l2_norm": 0.0 if dynamics.b is None else float(np.linalg.norm(dynamics.b, ord=2)),
        "c_l2_norm": float(np.linalg.norm(dynamics.c)),
        "s_param_l2_norm": float(np.linalg.norm(dynamics.s_params)),
        "w_param_l2_norm": float(np.linalg.norm(dynamics.w_params)),
        "mu_h_l2_norm": float(np.linalg.norm(dynamics.mu_h)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = _build_context()
    rng = np.random.default_rng(20260428)
    ranks = [6, 8, 10, 12, 14]
    output_dimension = 10
    sample_count = 10

    if context.rank == 0:
        run_root = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        manifest_path, dataset_metadata = _build_dataset(
            root=run_root / "raw_dataset",
            sample_count=sample_count,
            output_dimension=output_dimension,
            rng=rng,
        )
    else:
        run_root = None
        manifest_path = None
        dataset_metadata = None

    run_root = Path(context.bcast_object(str(run_root) if run_root is not None else None, root=0))
    manifest_path = Path(context.bcast_object(str(manifest_path) if manifest_path is not None else None, root=0))
    dataset_metadata = dict(context.bcast_object(dataset_metadata, root=0))

    from goattm.data import load_npz_sample_manifest

    manifest = load_npz_sample_manifest(manifest_path)
    results = []
    for rank in ranks:
        init_result = initialize_reduced_model_via_opinf(
            train_manifest=manifest,
            output_dir=run_root / f"opinf_rank_{rank}",
            rank=rank,
            context=context,
            apply_normalization=True,
            time_rescale_to_unit_interval=True,
            regularization=OpInfInitializationRegularization(
                coeff_w=1.0e-6,
                coeff_h=1.0e-6,
                coeff_b=1.0e-6,
                coeff_c=1.0e-7,
            ),
            latent_embedding=OpInfLatentEmbeddingConfig(
                mode="qoi_augmentation",
                augmentation_seed=12345,
                augmentation_scale=0.1,
            ),
        )
        results.append(_norm_summary(rank, init_result))

    if context.rank == 0:
        summary = {
            "dataset_metadata": dataset_metadata,
            "latent_embedding_mode": "qoi_augmentation",
            "augmentation_seed": 12345,
            "augmentation_scale": 0.1,
            "regularization": {
                "coeff_w": 1.0e-6,
                "coeff_h": 1.0e-6,
                "coeff_b": 1.0e-6,
                "coeff_c": 1.0e-7,
            },
            "results": results,
        }
        summary_path = run_root / "opinf_rank_sweep_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
