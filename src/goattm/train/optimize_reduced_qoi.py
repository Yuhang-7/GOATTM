from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from goattm.data import NpzSampleManifest, load_npz_sample_manifest, make_npz_train_test_split
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization
from goattm.runtime import DistributedContext
from goattm.train.reduced_qoi_trainer import ReducedQoiTrainer, ReducedQoiTrainerConfig, ReducedQoiTrainingResult


DynamicsLike = QuadraticDynamics | StabilizedQuadraticDynamics


@dataclass(frozen=True)
class ReducedQoiOptimizationRun:
    train_manifest: NpzSampleManifest
    test_manifest: NpzSampleManifest | None
    result: ReducedQoiTrainingResult


def optimize_reduced_qoi_from_manifest(
    manifest: str | Path | NpzSampleManifest,
    initial_dynamics: DynamicsLike,
    decoder_template: QuadraticDecoder,
    regularization: DecoderTikhonovRegularization,
    max_dt: float,
    trainer_config: ReducedQoiTrainerConfig,
    dynamics_regularization: DynamicsTikhonovRegularization | None = None,
    preprocess_record: dict | None = None,
    train_sample_ids: list[str] | tuple[str, ...] | None = None,
    test_sample_ids: list[str] | tuple[str, ...] | None = None,
    sample_seed: int | None = None,
    train_fraction: float = 0.8,
    context: DistributedContext | None = None,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter_newton: int = 25,
) -> ReducedQoiOptimizationRun:
    if isinstance(manifest, (str, Path)):
        manifest = load_npz_sample_manifest(manifest)

    if train_sample_ids is not None or test_sample_ids is not None or sample_seed is not None:
        split = make_npz_train_test_split(
            manifest,
            train_sample_ids=train_sample_ids,
            test_sample_ids=test_sample_ids,
            sample_seed=sample_seed,
            train_fraction=train_fraction,
        )
        train_manifest = split.train_manifest
        test_manifest = split.test_manifest if len(split.test_manifest.sample_ids) > 0 else None
    else:
        train_manifest = manifest
        test_manifest = None

    trainer = ReducedQoiTrainer(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        decoder_template=decoder_template,
        regularization=regularization,
        dynamics_regularization=dynamics_regularization,
        max_dt=max_dt,
        config=trainer_config,
        preprocess_record=preprocess_record,
        context=context,
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter_newton=max_iter_newton,
    )
    result = trainer.train(initial_dynamics)
    return ReducedQoiOptimizationRun(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        result=result,
    )
