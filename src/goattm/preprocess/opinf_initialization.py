from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from ..core.parametrization import (
    compressed_quadratic_dimension,
    njit,
    mu_h_dimension,
    s_params_to_matrix,
    skew_symmetric_dimension,
    upper_triangular_dimension,
    w_params_to_matrix,
)
from ..data.npz_dataset import (
    NpzQoiSample,
    NpzSampleManifest,
    load_npz_qoi_sample,
    load_npz_sample_manifest,
    save_npz_qoi_sample,
    save_npz_sample_manifest,
)
from ..models.linear_dynamics import LinearDynamics
from ..models.quadratic_decoder import QuadraticDecoder
from ..models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from ..problems.decoder_normal_equation import DecoderTikhonovRegularization
from ..runtime.distributed import DistributedContext
from ..solvers.time_integration import TimeIntegrator, rollout_to_observation_times, validate_time_integrator
from .constrained_least_squares import build_energy_preserving_compressed_h_basis
from .normalization import DatasetNormalizationStats, NormalizedDatasetArtifacts, materialize_normalized_train_test_split


_DYNAMICS_ASSEMBLY_CHUNK_SIZE = 64


@dataclass(frozen=True)
class OpInfInitializationRegularization:
    coeff_w: float = 1e-8
    coeff_h: float = 1e-8
    coeff_b: float = 1e-8
    coeff_c: float = 1e-8

    def scaled(self, factor: float) -> "OpInfInitializationRegularization":
        return OpInfInitializationRegularization(
            coeff_w=float(factor) * self.coeff_w,
            coeff_h=float(factor) * self.coeff_h,
            coeff_b=float(factor) * self.coeff_b,
            coeff_c=float(factor) * self.coeff_c,
        )


@dataclass(frozen=True)
class OpInfLatentEmbeddingConfig:
    mode: str = "pod_projection"
    augmentation_seed: int = 12345
    augmentation_scale: float = 0.1


@dataclass(frozen=True)
class OpInfInitializationResult:
    output_dir: Path
    normalized_artifacts: NormalizedDatasetArtifacts
    latent_train_manifest: NpzSampleManifest
    latent_test_manifest: NpzSampleManifest | None
    latent_train_manifest_path: Path
    latent_test_manifest_path: Path | None
    decoder_basis: np.ndarray
    dynamics: LinearDynamics | StabilizedQuadraticDynamics
    decoder: QuadraticDecoder
    time_scale: float
    time_rescaled_to_unit_interval: bool
    regression_relative_residual: float
    validation_success: bool
    validation_attempt_count: int
    validation_log_path: Path
    summary_path: Path

    def as_preprocess_record(self) -> dict[str, object]:
        return {
            "applied": True,
            "pipeline": "opinf_initialization",
            "normalized_output_dir": str(self.normalized_artifacts.output_dir),
            "normalization_stats_path": str(self.normalized_artifacts.stats_path),
            "normalized_train_manifest_path": str(self.normalized_artifacts.train_manifest_path),
            "normalized_test_manifest_path": None
            if self.normalized_artifacts.test_manifest_path is None
            else str(self.normalized_artifacts.test_manifest_path),
            "latent_train_manifest_path": str(self.latent_train_manifest_path),
            "latent_test_manifest_path": None if self.latent_test_manifest_path is None else str(self.latent_test_manifest_path),
            "time_scale": float(self.time_scale),
            "time_rescaled_to_unit_interval": bool(self.time_rescaled_to_unit_interval),
            "normalization_scale_mode": self.normalized_artifacts.stats.scale_mode,
            "normalization_target_max_abs": float(self.normalized_artifacts.stats.target_max_abs),
            "regression_relative_residual": float(self.regression_relative_residual),
            "validation_success": bool(self.validation_success),
            "validation_attempt_count": int(self.validation_attempt_count),
            "validation_log_path": str(self.validation_log_path),
            "summary_path": str(self.summary_path),
        }


def initialize_reduced_model_via_opinf(
    train_manifest: str | Path | NpzSampleManifest,
    output_dir: str | Path,
    rank: int,
    test_manifest: str | Path | NpzSampleManifest | None = None,
    context: DistributedContext | None = None,
    apply_normalization: bool = True,
    normalization_target_max_abs: float = 0.9,
    time_rescale_to_unit_interval: bool = True,
    max_dt: float = 0.04,
    regularization: OpInfInitializationRegularization | None = None,
    decoder_regularization: DecoderTikhonovRegularization | None = None,
    latent_embedding: OpInfLatentEmbeddingConfig | None = None,
    validation_time_integrator: TimeIntegrator = "rk4",
    max_regularization_retries: int = 5,
    regularization_growth: float = 10.0,
    dynamic_form: str = "AHBc",
    decoder_form: str = "V1V2v",
) -> OpInfInitializationResult:
    context = DistributedContext.from_comm() if context is None else context
    regularization = OpInfInitializationRegularization() if regularization is None else regularization
    latent_embedding = OpInfLatentEmbeddingConfig() if latent_embedding is None else latent_embedding
    validation_time_integrator = validate_time_integrator(validation_time_integrator)
    if dynamic_form not in {"ABc", "AHBc"}:
        raise ValueError(f"dynamic_form must be 'ABc' or 'AHBc', got {dynamic_form!r}")
    if decoder_form not in {"V1v", "V1V2v"}:
        raise ValueError(f"decoder_form must be 'V1v' or 'V1V2v', got {decoder_form!r}")
    _ = decoder_regularization
    if max_regularization_retries < 0:
        raise ValueError(f"max_regularization_retries must be nonnegative, got {max_regularization_retries}")
    if regularization_growth <= 1.0:
        raise ValueError(f"regularization_growth must be greater than 1.0, got {regularization_growth}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if apply_normalization:
        normalized_artifacts = materialize_normalized_train_test_split(
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            output_dir=output_root / "normalized",
            context=context,
            target_max_abs=normalization_target_max_abs,
        )
        normalized_train_manifest = normalized_artifacts.train_manifest
        normalized_test_manifest = normalized_artifacts.test_manifest
    else:
        if isinstance(train_manifest, (str, Path)):
            train_manifest = load_npz_sample_manifest(train_manifest)
        if test_manifest is not None and isinstance(test_manifest, (str, Path)):
            test_manifest = load_npz_sample_manifest(test_manifest)
        normalized_train_manifest = train_manifest
        normalized_test_manifest = test_manifest
        normalized_artifacts = NormalizedDatasetArtifacts(
            output_dir=output_root,
            stats_path=output_root / "normalization_stats.npz",
            train_manifest_path=output_root / "train_manifest.npz",
            test_manifest_path=None if test_manifest is None else output_root / "test_manifest.npz",
            train_manifest=normalized_train_manifest,
            test_manifest=normalized_test_manifest,
            stats=DatasetNormalizationStats(
                qoi_mean=np.zeros(load_npz_qoi_sample(normalized_train_manifest.absolute_paths()[0]).output_dimension, dtype=np.float64),
                qoi_std=np.ones(load_npz_qoi_sample(normalized_train_manifest.absolute_paths()[0]).output_dimension, dtype=np.float64),
                input_mean=None if _infer_input_dimension(normalized_train_manifest) == 0 else np.zeros(_infer_input_dimension(normalized_train_manifest), dtype=np.float64),
                input_std=None if _infer_input_dimension(normalized_train_manifest) == 0 else np.ones(_infer_input_dimension(normalized_train_manifest), dtype=np.float64),
                scale_mode="identity",
                target_max_abs=1.0,
            ),
        )

    basis = _compute_latent_embedding_matrix(
        normalized_train_manifest,
        rank=rank,
        context=context,
        config=latent_embedding,
    )
    time_scale = _infer_shared_time_scale(normalized_train_manifest) if time_rescale_to_unit_interval else 1.0
    latent_dir = output_root / "latent_dataset"
    latent_train_manifest = _materialize_latent_manifest(
        normalized_train_manifest,
        destination_root=latent_dir / "train",
        basis=basis,
        time_scale=time_scale,
        context=context,
    )
    latent_test_manifest = None
    if normalized_test_manifest is not None:
        latent_test_manifest = _materialize_latent_manifest(
            normalized_test_manifest,
            destination_root=latent_dir / "test",
            basis=basis,
            time_scale=time_scale,
            context=context,
        )

    latent_train_manifest_path = output_root / "latent_train_manifest.npz"
    latent_test_manifest_path = None if latent_test_manifest is None else output_root / "latent_test_manifest.npz"
    if context.rank == 0:
        save_npz_sample_manifest(latent_train_manifest_path, latent_train_manifest)
        if latent_test_manifest is not None and latent_test_manifest_path is not None:
            save_npz_sample_manifest(latent_test_manifest_path, latent_test_manifest)
    context.barrier()

    s_params = _identity_s_params(rank)
    validation_log_path = output_root / "opinf_initialization_log.jsonl"
    debug_log_path = output_root / "opinf_initialization_debug.log"
    if context.rank == 0:
        validation_log_path.write_text("", encoding="utf-8")
        debug_log_path.write_text("", encoding="utf-8")
    context.barrier()
    _write_opinf_debug(
        context,
        debug_log_path,
        "initialization_start",
        rank=rank,
        train_sample_count=len(latent_train_manifest),
        test_sample_count=0 if latent_test_manifest is None else len(latent_test_manifest),
    )

    dynamics = None
    regression_relative_residual = float("nan")
    validation_records: list[dict[str, object]] = []
    validation_success = False
    current_regularization = regularization
    validation_manifests: tuple[tuple[str, NpzSampleManifest], ...]
    if latent_test_manifest is None:
        validation_manifests = (("train", latent_train_manifest),)
    else:
        validation_manifests = (("train", latent_train_manifest), ("test", latent_test_manifest))
    for attempt in range(max_regularization_retries + 1):
        if dynamic_form == "ABc":
            dynamics, regression_relative_residual = _fit_linear_dynamics_from_latent_dataset(
                manifest=latent_train_manifest,
                rank=rank,
                regularization=current_regularization,
                context=context,
                debug_log_path=debug_log_path,
                attempt=attempt,
            )
        else:
            dynamics, regression_relative_residual = _fit_stabilized_dynamics_from_latent_dataset(
                manifest=latent_train_manifest,
                rank=rank,
                regularization=current_regularization,
                context=context,
                s_params=s_params,
                debug_log_path=debug_log_path,
                attempt=attempt,
            )
        validation_start = time.perf_counter()
        _write_opinf_debug(context, debug_log_path, "validation_start", attempt=attempt)
        validation_record = _validate_opinf_forward_rollouts(
            manifests=validation_manifests,
            dynamics=dynamics,
            context=context,
            max_dt=max_dt,
            time_integrator=validation_time_integrator,
        )
        _write_opinf_debug(
            context,
            debug_log_path,
            "validation_done",
            attempt=attempt,
            elapsed_seconds=time.perf_counter() - validation_start,
            success=bool(validation_record["success"]),
            checked_sample_count=int(validation_record["global_checked_sample_count"]),
            failure_count=int(validation_record["global_failure_count"]),
        )
        validation_record.update(
            {
                "attempt": attempt,
                "regularization": _regularization_to_record(current_regularization),
                "regression_relative_residual": float(regression_relative_residual),
                "a_fro_norm": float(np.linalg.norm(dynamics.a)),
                "h_fro_norm": float(np.linalg.norm(dynamics.h_matrix)),
                "b_fro_norm": 0.0 if dynamics.b is None else float(np.linalg.norm(dynamics.b)),
                "c_l2_norm": float(np.linalg.norm(dynamics.c)),
            }
        )
        validation_records.append(validation_record)
        if context.rank == 0:
            with validation_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(validation_record, ensure_ascii=True) + "\n")
        if bool(validation_record["success"]):
            validation_success = True
            break
        current_regularization = current_regularization.scaled(regularization_growth)
    context.barrier()

    if dynamics is None:
        raise RuntimeError("OpInf initialization did not produce a dynamics model.")
    if not validation_success:
        raise RuntimeError(
            f"OpInf initialization failed forward validation after {max_regularization_retries + 1} attempts. "
            f"See {validation_log_path}."
        )
    decoder = QuadraticDecoder(
        v1=basis.copy(),
        v2=np.zeros((basis.shape[0], compressed_quadratic_dimension(rank)), dtype=np.float64),
        v0=np.zeros(basis.shape[0], dtype=np.float64),
        form=decoder_form,
    )

    summary_path = output_root / "opinf_initialization_summary.json"
    if context.rank == 0:
        summary = {
            "rank": rank,
            "time_scale": float(time_scale),
            "time_rescaled_to_unit_interval": bool(time_rescale_to_unit_interval),
            "apply_normalization": bool(apply_normalization),
            "normalization_target_max_abs": float(normalization_target_max_abs),
            "latent_embedding_mode": latent_embedding.mode,
            "augmentation_seed": int(latent_embedding.augmentation_seed),
            "augmentation_scale": float(latent_embedding.augmentation_scale),
            "decoder_initialization": "pod_projection_only",
            "dynamic_form": dynamic_form,
            "decoder_form": decoder_form,
            "dynamics_regression": "midpoint finite difference: du=(u[n+1]-u[n])/dt, rhs at (u[n]+u[n+1])/2",
            "default_s_params": "identity",
            "regression_relative_residual": float(regression_relative_residual),
            "validation_time_integrator": validation_time_integrator,
            "validation_max_dt": float(max_dt),
            "validation_success": bool(validation_success),
            "validation_attempt_count": len(validation_records),
            "validation_log_path": str(validation_log_path),
            "debug_log_path": str(debug_log_path),
            "final_regularization": _regularization_to_record(current_regularization),
            "train_sample_count": len(latent_train_manifest),
            "test_sample_count": 0 if latent_test_manifest is None else len(latent_test_manifest),
            "latent_train_manifest_path": str(latent_train_manifest_path),
            "latent_test_manifest_path": None if latent_test_manifest_path is None else str(latent_test_manifest_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    context.barrier()
    return OpInfInitializationResult(
        output_dir=output_root,
        normalized_artifacts=normalized_artifacts,
        latent_train_manifest=latent_train_manifest,
        latent_test_manifest=latent_test_manifest,
        latent_train_manifest_path=latent_train_manifest_path,
        latent_test_manifest_path=latent_test_manifest_path,
        decoder_basis=basis,
        dynamics=dynamics,
        decoder=decoder,
        time_scale=float(time_scale),
        time_rescaled_to_unit_interval=bool(time_rescale_to_unit_interval),
        regression_relative_residual=float(regression_relative_residual),
        validation_success=bool(validation_success),
        validation_attempt_count=len(validation_records),
        validation_log_path=validation_log_path,
        summary_path=summary_path,
    )


def _write_opinf_debug(
    context: DistributedContext,
    log_path: Path | None,
    event: str,
    **fields: object,
) -> None:
    if context.rank != 0 or log_path is None:
        return
    record = {"event": event, **fields}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[opinf] {event}" + (f" {field_text}" if field_text else ""), flush=True)


def _compute_latent_embedding_matrix(
    manifest: NpzSampleManifest,
    rank: int,
    context: DistributedContext,
    config: OpInfLatentEmbeddingConfig,
) -> np.ndarray:
    reference = load_npz_qoi_sample(manifest.absolute_paths()[0])
    dq = reference.output_dimension
    if config.mode == "qoi_augmentation":
        return _build_qoi_augmentation_embedding(
            output_dimension=dq,
            rank=rank,
            seed=config.augmentation_seed,
            scale=config.augmentation_scale,
        )
    if rank > dq:
        raise ValueError(
            f"POD projection mode requires rank <= output dimension. Got rank={rank}, dq={dq}. "
            "Use latent_embedding.mode='qoi_augmentation' for rank > dq."
        )
    local_gram = np.zeros((dq, dq), dtype=np.float64)
    for _, sample_path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(sample_path)
        snapshots = np.asarray(sample.qoi_observations, dtype=np.float64).T
        local_gram += snapshots @ snapshots.T
    gram = context.allreduce_array_sum(local_gram)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:rank]].copy()
    signs = np.where(basis[0, :] < 0.0, -1.0, 1.0)
    basis *= signs[None, :]
    return basis


def _build_qoi_augmentation_embedding(
    output_dimension: int,
    rank: int,
    seed: int,
    scale: float,
) -> np.ndarray:
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    basis = np.zeros((output_dimension, rank), dtype=np.float64)
    direct_dim = min(rank, output_dimension)
    basis[:, :direct_dim] = np.eye(output_dimension, direct_dim, dtype=np.float64)
    if rank <= output_dimension:
        return basis
    rng = np.random.default_rng(seed)
    extra_dim = rank - output_dimension
    extra = np.zeros((output_dimension, extra_dim), dtype=np.float64)
    for col in range(extra_dim):
        row = col % output_dimension
        extra[row, col] = scale * rng.choice((-1.0, 1.0))
    basis[:, output_dimension:] = extra
    return basis


def _infer_shared_time_scale(manifest: NpzSampleManifest, tolerance: float = 1e-12) -> float:
    end_times = []
    for path in manifest.absolute_paths():
        sample = load_npz_qoi_sample(path)
        end_times.append(float(sample.observation_times[-1]))
    reference = end_times[0]
    for end_time in end_times[1:]:
        if abs(end_time - reference) > tolerance:
            raise ValueError("All samples must share a common final time to rescale to [0, 1].")
    if reference <= 0.0:
        raise ValueError(f"Final time must be positive, got {reference}")
    return reference


def _materialize_latent_manifest(
    manifest: NpzSampleManifest,
    destination_root: Path,
    basis: np.ndarray,
    time_scale: float,
    context: DistributedContext,
) -> NpzSampleManifest:
    destination_root.mkdir(parents=True, exist_ok=True)
    absolute_paths = manifest.absolute_paths()
    output_paths = tuple(destination_root / f"{idx:06d}_{manifest.sample_ids[idx]}.npz" for idx in range(len(manifest)))
    for sample_id, sample_path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(sample_path)
        global_index = manifest.sample_ids.index(sample_id)
        qoi = np.asarray(sample.qoi_observations, dtype=np.float64)
        latent_states = qoi @ basis
        observation_times = np.asarray(sample.observation_times, dtype=np.float64) / float(time_scale)
        input_times = None if sample.input_times is None else np.asarray(sample.input_times, dtype=np.float64) / float(time_scale)
        metadata = dict(sample.metadata or {})
        metadata["time_rescaled_to_unit_interval"] = bool(abs(time_scale - 1.0) > 1e-14)
        metadata["time_scale"] = float(time_scale)
        metadata["raw_sample_id"] = sample.sample_id
        metadata["latent_trajectory"] = latent_states
        processed_sample = NpzQoiSample(
            sample_id=sample.sample_id,
            observation_times=observation_times,
            u0=latent_states[0].copy(),
            qoi_observations=qoi,
            input_times=input_times,
            input_values=None if sample.input_values is None else np.asarray(sample.input_values, dtype=np.float64),
            metadata=metadata,
        )
        save_npz_qoi_sample(output_paths[global_index], processed_sample)
    context.barrier()
    return NpzSampleManifest(
        root_dir=destination_root,
        sample_paths=tuple(Path(path.name) for path in output_paths),
        sample_ids=manifest.sample_ids,
    )


def _fit_stabilized_dynamics_from_latent_dataset(
    manifest: NpzSampleManifest,
    rank: int,
    regularization: OpInfInitializationRegularization,
    context: DistributedContext,
    s_params: np.ndarray,
    debug_log_path: Path | None = None,
    attempt: int = 0,
) -> tuple[StabilizedQuadraticDynamics, float]:
    fit_start = time.perf_counter()
    _write_opinf_debug(context, debug_log_path, "fit_start", attempt=attempt, rank=rank)
    nw = skew_symmetric_dimension(rank)
    nh = mu_h_dimension(rank)
    basis_start = time.perf_counter()
    h_basis_tensor = _compressed_h_basis_tensor(build_energy_preserving_compressed_h_basis(rank), rank)
    w_basis = _build_skew_symmetric_basis(rank)
    _write_opinf_debug(
        context,
        debug_log_path,
        "basis_done",
        attempt=attempt,
        elapsed_seconds=time.perf_counter() - basis_start,
        skew_dimension=nw,
        mu_h_dimension=nh,
    )
    s_matrix = _s_params_to_dense_matrix(s_params, rank)
    fixed_a_matrix = -(s_matrix @ s_matrix.T)

    assembly_start = time.perf_counter()
    local_normal, local_rhs, local_target_sumsq, local_step_count = _assemble_dynamics_fit_system(
        manifest=manifest,
        rank=rank,
        w_basis=w_basis,
        h_basis_tensor=h_basis_tensor,
        context=context,
        fixed_a_matrix=fixed_a_matrix,
    )
    global_step_count = context.allreduce_int_sum(local_step_count)
    _write_opinf_debug(
        context,
        debug_log_path,
        "local_normal_assembly_done",
        attempt=attempt,
        elapsed_seconds=time.perf_counter() - assembly_start,
        regression_step_count=global_step_count,
        feature_dimension=local_normal.shape[0],
    )
    reduce_start = time.perf_counter()
    global_normal = context.allreduce_array_sum(local_normal)
    global_rhs = context.allreduce_array_sum(local_rhs)
    global_target_sumsq = context.allreduce_scalar_sum(local_target_sumsq)
    _write_opinf_debug(
        context,
        debug_log_path,
        "normal_allreduce_done",
        attempt=attempt,
        elapsed_seconds=time.perf_counter() - reduce_start,
        target_sumsq=float(global_target_sumsq),
    )

    input_dim = _infer_input_dimension(manifest)
    feature_dim = nw + nh + rank * input_dim + rank
    diagonal = np.zeros(feature_dim, dtype=np.float64)
    diagonal[:nw] = regularization.coeff_w
    diagonal[nw : nw + nh] = regularization.coeff_h
    offset = nw + nh
    diagonal[offset : offset + rank * input_dim] = regularization.coeff_b
    diagonal[offset + rank * input_dim :] = regularization.coeff_c
    solve_start = time.perf_counter()
    theta = np.linalg.solve(global_normal + np.diag(diagonal), global_rhs)
    _write_opinf_debug(
        context,
        debug_log_path,
        "linear_solve_done",
        attempt=attempt,
        elapsed_seconds=time.perf_counter() - solve_start,
        feature_dimension=feature_dim,
    )

    w_params = theta[:nw].copy()
    mu_h = theta[nw : nw + nh].copy()
    input_dim = _infer_input_dimension(manifest)
    offset = nw + nh
    if input_dim > 0:
        b = theta[offset : offset + rank * input_dim].reshape((rank, input_dim), order="F").copy()
    else:
        b = None
    c = theta[offset + rank * input_dim :].copy()
    dynamics = StabilizedQuadraticDynamics(
        s_params=np.asarray(s_params, dtype=np.float64).copy(),
        w_params=w_params,
        mu_h=mu_h,
        b=b,
        c=c,
    )
    residual_start = time.perf_counter()
    residual_sumsq = _compute_dynamics_fit_residual_sumsq(manifest, dynamics, context=context)
    relative_residual = 0.0 if global_target_sumsq <= 0.0 else np.sqrt(residual_sumsq / global_target_sumsq)
    _write_opinf_debug(
        context,
        debug_log_path,
        "fit_done",
        attempt=attempt,
        elapsed_seconds=time.perf_counter() - fit_start,
        residual_elapsed_seconds=time.perf_counter() - residual_start,
        relative_residual=float(relative_residual),
    )
    return dynamics, float(relative_residual)


def _fit_linear_dynamics_from_latent_dataset(
    manifest: NpzSampleManifest,
    rank: int,
    regularization: OpInfInitializationRegularization,
    context: DistributedContext,
    debug_log_path: Path | None = None,
    attempt: int = 0,
) -> tuple[LinearDynamics, float]:
    fit_start = time.perf_counter()
    _write_opinf_debug(context, debug_log_path, "linear_fit_start", attempt=attempt, rank=rank)
    local_normal, local_rhs, local_target_sumsq, local_step_count = _assemble_linear_dynamics_fit_system(
        manifest=manifest,
        rank=rank,
        context=context,
    )
    global_step_count = context.allreduce_int_sum(local_step_count)
    global_normal = context.allreduce_array_sum(local_normal)
    global_rhs = context.allreduce_array_sum(local_rhs)
    global_target_sumsq = context.allreduce_scalar_sum(local_target_sumsq)
    input_dim = _infer_input_dimension(manifest)
    feature_dim = rank * rank + rank * input_dim + rank
    diagonal = np.zeros(feature_dim, dtype=np.float64)
    diagonal[: rank * rank] = regularization.coeff_w
    offset = rank * rank
    diagonal[offset : offset + rank * input_dim] = regularization.coeff_b
    diagonal[offset + rank * input_dim :] = regularization.coeff_c
    theta = np.linalg.solve(global_normal + np.diag(diagonal), global_rhs)
    a = theta[: rank * rank].reshape((rank, rank)).copy()
    offset = rank * rank
    if input_dim > 0:
        b = theta[offset : offset + rank * input_dim].reshape((rank, input_dim)).copy()
    else:
        b = None
    c = theta[offset + rank * input_dim :].copy()
    dynamics = LinearDynamics(a=a, b=b, c=c)
    residual_sumsq = _compute_dynamics_fit_residual_sumsq(manifest, dynamics, context=context)
    relative_residual = 0.0 if global_target_sumsq <= 0.0 else np.sqrt(residual_sumsq / global_target_sumsq)
    _write_opinf_debug(
        context,
        debug_log_path,
        "linear_fit_done",
        attempt=attempt,
        elapsed_seconds=time.perf_counter() - fit_start,
        regression_step_count=global_step_count,
        feature_dimension=feature_dim,
        relative_residual=float(relative_residual),
    )
    return dynamics, float(relative_residual)


def _assemble_linear_dynamics_fit_system(
    manifest: NpzSampleManifest,
    rank: int,
    context: DistributedContext,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    input_dim = _infer_input_dimension(manifest)
    feature_dim = rank * rank + rank * input_dim + rank
    local_normal = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    local_rhs = np.zeros(feature_dim, dtype=np.float64)
    local_target_sumsq = 0.0
    local_step_count = 0

    for _, sample_path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(sample_path)
        latent_trajectory = _reconstruct_latent_trajectory_from_sample(sample, rank=rank)
        u_mid, du, p_mid = _midpoint_regression_arrays(sample, latent_trajectory)
        for step_idx in range(u_mid.shape[0]):
            u = u_mid[step_idx]
            p = None if p_mid is None else p_mid[step_idx]
            for output_idx in range(rank):
                row = np.zeros(feature_dim, dtype=np.float64)
                a_offset = output_idx * rank
                row[a_offset : a_offset + rank] = u
                b_offset = rank * rank + output_idx * input_dim
                if input_dim > 0 and p is not None:
                    row[b_offset : b_offset + input_dim] = p
                c_offset = rank * rank + rank * input_dim + output_idx
                row[c_offset] = 1.0
                local_normal += np.outer(row, row)
                local_rhs += row * du[step_idx, output_idx]
        local_target_sumsq += float(np.sum(du * du))
        local_step_count += int(u_mid.shape[0])
    return local_normal, local_rhs, local_target_sumsq, local_step_count


def _assemble_dynamics_fit_system(
    manifest: NpzSampleManifest,
    rank: int,
    w_basis: np.ndarray,
    h_basis_tensor: np.ndarray,
    context: DistributedContext,
    fixed_a_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    input_dim = _infer_input_dimension(manifest)
    nw = w_basis.shape[2]
    nh = h_basis_tensor.shape[2]
    feature_dim = nw + nh + rank * input_dim + rank
    local_normal = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    local_rhs = np.zeros(feature_dim, dtype=np.float64)
    local_target_sumsq = 0.0
    local_step_count = 0

    for _, sample_path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(sample_path)
        latent_trajectory = _reconstruct_latent_trajectory_from_sample(sample, rank=rank)
        u_mid, du, p_mid = _midpoint_regression_arrays(sample, latent_trajectory)
        adjusted_target = du - u_mid @ fixed_a_matrix.T
        for start in range(0, u_mid.shape[0], _DYNAMICS_ASSEMBLY_CHUNK_SIZE):
            stop = min(start + _DYNAMICS_ASSEMBLY_CHUNK_SIZE, u_mid.shape[0])
            p_chunk = None if p_mid is None else p_mid[start:stop]
            phi = _dynamics_feature_actions(
                u=u_mid[start:stop],
                p=p_chunk,
                w_basis=w_basis,
                h_basis_tensor=h_basis_tensor,
            )
            phi_2d = phi.reshape((-1, feature_dim))
            target_vector = adjusted_target[start:stop].reshape(-1)
            local_normal += phi_2d.T @ phi_2d
            local_rhs += phi_2d.T @ target_vector
        local_target_sumsq += float(np.sum(du * du))
        local_step_count += int(u_mid.shape[0])
    return local_normal, local_rhs, local_target_sumsq, local_step_count


def _compute_dynamics_fit_residual_sumsq(
    manifest: NpzSampleManifest,
    dynamics: LinearDynamics | StabilizedQuadraticDynamics,
    context: DistributedContext,
) -> float:
    local_residual_sumsq = 0.0
    for _, sample_path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(sample_path)
        latent_trajectory = _reconstruct_latent_trajectory_from_sample(sample, rank=dynamics.dimension)
        u_mid, du, p_mid = _midpoint_regression_arrays(sample, latent_trajectory)
        prediction = _evaluate_dynamics_rhs_batch(dynamics, u_mid, p_mid)
        residual = prediction - du
        local_residual_sumsq += float(np.sum(residual * residual))
    return context.allreduce_scalar_sum(local_residual_sumsq)


def _validate_opinf_forward_rollouts(
    manifests: tuple[tuple[str, NpzSampleManifest], ...],
    dynamics: LinearDynamics | StabilizedQuadraticDynamics,
    context: DistributedContext,
    max_dt: float,
    time_integrator: TimeIntegrator,
) -> dict[str, object]:
    local_failure: str | None = None
    local_checked_count = 0
    for dataset_name, manifest in manifests:
        for sample_id, sample_path in manifest.entries_for_rank(context.rank, context.size):
            sample = load_npz_qoi_sample(sample_path)
            local_checked_count += 1
            try:
                result, states = rollout_to_observation_times(
                    dynamics=dynamics,
                    u0=np.asarray(sample.u0, dtype=np.float64),
                    observation_times=np.asarray(sample.observation_times, dtype=np.float64),
                    max_dt=max_dt,
                    input_function=sample.build_input_function(),
                    time_integrator=time_integrator,
                )
            except Exception as exc:  # pragma: no cover - exercised by retry behavior in integration use.
                local_failure = (
                    f"rank={context.rank} dataset={dataset_name} sample_id={sample_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                break
            if not result.success:
                local_failure = f"rank={context.rank} dataset={dataset_name} sample_id={sample_id}: rollout reported failure"
                break
            if not np.all(np.isfinite(states)):
                local_failure = f"rank={context.rank} dataset={dataset_name} sample_id={sample_id}: rollout produced nonfinite states"
                break
        if local_failure is not None:
            break

    global_failure_count = context.allreduce_int_sum(1 if local_failure is not None else 0)
    global_checked_count = context.allreduce_int_sum(local_checked_count)
    gathered_failures = context.allgather_object(local_failure)
    failures = [str(failure) for failure in gathered_failures if failure is not None]
    return {
        "success": global_failure_count == 0,
        "global_failure_count": int(global_failure_count),
        "global_checked_sample_count": int(global_checked_count),
        "local_checked_sample_count": int(local_checked_count),
        "datasets": [dataset_name for dataset_name, _ in manifests],
        "time_integrator": time_integrator,
        "max_dt": float(max_dt),
        "first_failure": None if not failures else failures[0],
        "failures": failures,
    }


def _iter_midpoint_regression_data(
    sample: NpzQoiSample,
    latent_trajectory: np.ndarray,
):
    u_mid, du, p_mid = _midpoint_regression_arrays(sample, latent_trajectory)
    for step_idx in range(u_mid.shape[0]):
        p_value = None if p_mid is None else p_mid[step_idx]
        yield u_mid[step_idx], du[step_idx], p_value


def _midpoint_regression_arrays(
    sample: NpzQoiSample,
    latent_trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    times = np.asarray(sample.observation_times, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] != latent_trajectory.shape[0]:
        raise ValueError(
            f"observation_times must have shape ({latent_trajectory.shape[0]},), got {times.shape}"
        )
    if latent_trajectory.ndim != 2:
        raise ValueError(f"latent_trajectory must have shape (Nt, r), got {latent_trajectory.shape}")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("observation_times must be strictly increasing.")
    u_left = np.asarray(latent_trajectory[:-1], dtype=np.float64)
    u_right = np.asarray(latent_trajectory[1:], dtype=np.float64)
    u_mid = 0.5 * (u_left + u_right)
    du = (u_right - u_left) / dt[:, None]
    input_function = sample.build_input_function()
    if input_function is None:
        return u_mid, du, None
    t_mid = times[:-1] + 0.5 * dt
    p_mid = np.vstack([np.asarray(input_function(float(t)), dtype=np.float64) for t in t_mid])
    return u_mid, du, p_mid


def _identity_s_params(rank: int) -> np.ndarray:
    values = np.zeros(upper_triangular_dimension(rank), dtype=np.float64)
    idx = 0
    for i in range(rank):
        for j in range(i, rank):
            values[idx] = 1.0 if i == j else 0.0
            idx += 1
    return values


def _s_params_to_dense_matrix(s_params: np.ndarray, rank: int) -> np.ndarray:
    s_params = np.asarray(s_params, dtype=np.float64)
    if s_params.shape != (upper_triangular_dimension(rank),):
        raise ValueError(f"s_params must have shape ({upper_triangular_dimension(rank)},), got {s_params.shape}")
    return s_params_to_matrix(s_params, rank)


def _regularization_to_record(regularization: OpInfInitializationRegularization) -> dict[str, float]:
    return {
        "coeff_w": float(regularization.coeff_w),
        "coeff_h": float(regularization.coeff_h),
        "coeff_b": float(regularization.coeff_b),
        "coeff_c": float(regularization.coeff_c),
    }


def _dynamics_feature_action(
    u: np.ndarray,
    p: np.ndarray | None,
    w_basis: np.ndarray,
    h_basis: np.ndarray,
) -> np.ndarray:
    h_basis_tensor = _compressed_h_basis_tensor(h_basis, u.shape[0])
    p_batch = None if p is None else np.asarray(p, dtype=np.float64)[None, :]
    return _dynamics_feature_actions(
        u=np.asarray(u, dtype=np.float64)[None, :],
        p=p_batch,
        w_basis=w_basis,
        h_basis_tensor=h_basis_tensor,
    )[0]


def _dynamics_feature_actions(
    u: np.ndarray,
    p: np.ndarray | None,
    w_basis: np.ndarray,
    h_basis_tensor: np.ndarray,
) -> np.ndarray:
    sample_count, rank = u.shape
    nw = w_basis.shape[2]
    nh = h_basis_tensor.shape[2]
    input_dim = 0 if p is None else p.shape[1]
    total_dim = nw + nh + rank * input_dim + rank
    phi = np.zeros((sample_count, rank, total_dim), dtype=np.float64)

    p_array = np.empty((sample_count, 0), dtype=np.float64) if p is None else np.asarray(p, dtype=np.float64)
    _fill_dynamics_feature_actions_numba(
        np.asarray(u, dtype=np.float64),
        p_array,
        np.asarray(w_basis, dtype=np.float64),
        np.asarray(h_basis_tensor, dtype=np.float64),
        phi,
    )
    return phi


@njit(cache=True)
def _fill_dynamics_feature_actions_numba(
    u: np.ndarray,
    p: np.ndarray,
    w_basis: np.ndarray,
    h_basis_tensor: np.ndarray,
    phi: np.ndarray,
) -> None:
    sample_count = u.shape[0]
    rank = u.shape[1]
    nw = w_basis.shape[2]
    nh = h_basis_tensor.shape[2]
    input_dim = p.shape[1]
    zeta_dim = (rank * (rank + 1)) // 2
    zeta = np.empty(zeta_dim, dtype=np.float64)

    for sample_index in range(sample_count):
        zeta_index = 0
        for i in range(rank):
            for j in range(i + 1):
                zeta[zeta_index] = u[sample_index, i] * u[sample_index, j]
                zeta_index += 1

        for row in range(rank):
            for feature_index in range(nw):
                value = 0.0
                for col in range(rank):
                    value += w_basis[row, col, feature_index] * u[sample_index, col]
                phi[sample_index, row, feature_index] = value

            for feature_index in range(nh):
                value = 0.0
                for zeta_index in range(zeta_dim):
                    value += h_basis_tensor[row, zeta_index, feature_index] * zeta[zeta_index]
                phi[sample_index, row, nw + feature_index] = value

            offset = nw + nh
            for input_index in range(input_dim):
                for identity_col in range(rank):
                    value = 0.0
                    if identity_col == row:
                        value = p[sample_index, input_index]
                    phi[sample_index, row, offset + input_index * rank + identity_col] = value
            offset += rank * input_dim

            for identity_col in range(rank):
                value = 0.0
                if identity_col == row:
                    value = 1.0
                phi[sample_index, row, offset + identity_col] = value


def _compressed_h_basis_tensor(h_basis: np.ndarray, rank: int) -> np.ndarray:
    s = compressed_quadratic_dimension(rank)
    return np.stack(
        [h_basis[:, idx].reshape((rank, s), order="F") for idx in range(h_basis.shape[1])],
        axis=2,
    )


def _evaluate_dynamics_rhs_batch(
    dynamics: LinearDynamics | StabilizedQuadraticDynamics,
    u: np.ndarray,
    p: np.ndarray | None,
) -> np.ndarray:
    prediction = u @ dynamics.a.T
    zeta = _quadratic_features_matrix(u)
    prediction = prediction + zeta @ dynamics.h_matrix.T
    if dynamics.b is not None:
        if p is None:
            raise ValueError("Dynamics has an input matrix but midpoint inputs are missing.")
        prediction = prediction + p @ dynamics.b.T
    prediction = prediction + dynamics.c[None, :]
    return prediction


def _reconstruct_latent_trajectory_from_sample(sample: NpzQoiSample, rank: int) -> np.ndarray:
    metadata = sample.metadata or {}
    if "latent_trajectory" in metadata:
        latent = np.asarray(metadata["latent_trajectory"], dtype=np.float64)
        if latent.shape != (sample.observation_times.shape[0], rank):
            raise ValueError(f"latent_trajectory must have shape {(sample.observation_times.shape[0], rank)}, got {latent.shape}")
        return latent
    raise ValueError("Processed OpInf sample is missing required metadata field 'latent_trajectory'.")


def _quadratic_features_column(u: np.ndarray) -> np.ndarray:
    values = np.empty(compressed_quadratic_dimension(u.shape[0]), dtype=np.float64)
    idx = 0
    for i in range(u.shape[0]):
        for j in range(i + 1):
            values[idx] = u[i] * u[j]
            idx += 1
    return values


def _quadratic_features_matrix(u: np.ndarray) -> np.ndarray:
    values = np.empty((u.shape[0], compressed_quadratic_dimension(u.shape[1])), dtype=np.float64)
    idx = 0
    for i in range(u.shape[1]):
        for j in range(i + 1):
            values[:, idx] = u[:, i] * u[:, j]
            idx += 1
    return values


def _build_skew_symmetric_basis(rank: int) -> np.ndarray:
    dim = skew_symmetric_dimension(rank)
    basis = np.zeros((rank, rank, dim), dtype=np.float64)
    for idx in range(dim):
        params = np.zeros(dim, dtype=np.float64)
        params[idx] = 1.0
        basis[:, :, idx] = w_params_to_matrix(params, rank)
    return basis


def _infer_input_dimension(manifest: NpzSampleManifest) -> int:
    sample = load_npz_qoi_sample(manifest.absolute_paths()[0])
    return sample.input_dimension


def _finite_difference_time_derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"values must have shape (Nt, r), got {values.shape}")
    if times.ndim != 1 or times.shape[0] != values.shape[0]:
        raise ValueError(f"times must have shape ({values.shape[0]},), got {times.shape}")
    edge_order = 2 if values.shape[0] >= 3 else 1
    return np.gradient(values, times, axis=0, edge_order=edge_order)


def _regression_time_indices(sample_count: int) -> range:
    if sample_count >= 3:
        return range(1, sample_count - 1)
    return range(sample_count)
