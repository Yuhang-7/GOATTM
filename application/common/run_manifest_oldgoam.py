from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.data import NpzSampleManifest, load_npz_sample_manifest, make_npz_train_test_split  # noqa: E402
from goattm.preprocess import OpInfInitializationRegularization, OpInfLatentEmbeddingConfig, initialize_reduced_model_via_opinf  # noqa: E402
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization  # noqa: E402
from goattm.runtime import DistributedContext  # noqa: E402
from goattm.train import BfgsUpdaterConfig, LbfgsUpdaterConfig, ReducedQoiTrainer, ReducedQoiTrainerConfig  # noqa: E402


@dataclass(frozen=True)
class ManifestOldGoamConfig:
    dataset_name: str
    manifest_path: Path
    sample_count: int
    ntrain: int
    ntest: int
    latent_rank: int
    dynamic_form: str
    decoder_form: str
    oldgoam_mode: bool
    latent_embedding_mode: str
    latent_embedding_augmentation_seed: int
    latent_embedding_augmentation_scale: float
    max_dt: float
    time_integrator: str
    optimizer: str
    max_iterations: int
    normalization_target_max_abs: float
    lbfgs_maxcor: int
    lbfgs_ftol: float
    lbfgs_gtol: float
    lbfgs_maxls: int
    bfgs_gtol: float
    bfgs_c1: float
    bfgs_c2: float
    bfgs_xrtol: float
    opinf_reg_w: float
    opinf_reg_h: float
    opinf_reg_b: float
    opinf_reg_c: float
    decoder_reg_v1: float
    decoder_reg_v2: float
    decoder_reg_v0: float
    dynamics_reg_a: float
    dynamics_reg_s: float
    dynamics_reg_w: float
    dynamics_reg_mu_h: float
    dynamics_reg_b: float
    dynamics_reg_c: float
    dynamics_reg_spectral_abscissa: float
    dynamics_reg_spectral_alpha: float
    output_dir: Path


def parse_args() -> ManifestOldGoamConfig:
    parser = argparse.ArgumentParser(description="Run GOATTM oldGOAM/direct-A training from an NPZ manifest.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--ntest", type=int, required=True)
    parser.add_argument("--latent-rank", type=int, required=True)
    parser.add_argument("--dynamic-form", choices=("ABc", "AHBc"), default="AHBc")
    parser.add_argument("--decoder-form", choices=("V1v", "V1V2v"), default="V1V2v")
    parser.add_argument("--oldgoam", dest="oldgoam_mode", action="store_true")
    parser.add_argument(
        "--latent-embedding-mode",
        choices=("pod_projection", "qoi_augmentation"),
        default="pod_projection",
    )
    parser.add_argument("--latent-embedding-augmentation-seed", type=int, default=12345)
    parser.add_argument("--latent-embedding-augmentation-scale", type=float, default=0.1)
    parser.add_argument("--max-dt", type=float, default=0.01)
    parser.add_argument("--time-integrator", choices=("implicit_midpoint", "explicit_euler", "rk4"), default="rk4")
    parser.add_argument("--optimizer", choices=("lbfgs", "bfgs", "adam", "gradient_descent", "newton_action"), default="bfgs")
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--normalization-target-max-abs", type=float, default=0.9)
    parser.add_argument("--lbfgs-maxcor", type=int, default=20)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-12)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    parser.add_argument("--lbfgs-maxls", type=int, default=30)
    parser.add_argument("--bfgs-gtol", type=float, default=1e-6)
    parser.add_argument("--bfgs-c1", type=float, default=1e-4)
    parser.add_argument("--bfgs-c2", type=float, default=0.9)
    parser.add_argument("--bfgs-xrtol", type=float, default=1e-7)
    parser.add_argument("--opinf-reg-w", type=float, default=1e-9)
    parser.add_argument("--opinf-reg-h", type=float, default=1e-9)
    parser.add_argument("--opinf-reg-b", type=float, default=1e-9)
    parser.add_argument("--opinf-reg-c", type=float, default=1e-9)
    parser.add_argument("--decoder-reg-v1", type=float, default=1e-7)
    parser.add_argument("--decoder-reg-v2", type=float, default=1e-7)
    parser.add_argument("--decoder-reg-v0", type=float, default=1e-7)
    parser.add_argument("--dynamics-reg-a", type=float, default=1e-9)
    parser.add_argument("--dynamics-reg-s", type=float, default=0.0)
    parser.add_argument("--dynamics-reg-w", type=float, default=0.0)
    parser.add_argument("--dynamics-reg-mu-h", type=float, default=1e-9)
    parser.add_argument("--dynamics-reg-b", type=float, default=1e-9)
    parser.add_argument("--dynamics-reg-c", type=float, default=1e-9)
    parser.add_argument("--dynamics-reg-spectral-abscissa", type=float, default=0.0)
    parser.add_argument("--dynamics-reg-spectral-alpha", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    config = ManifestOldGoamConfig(**vars(parser.parse_args()))
    validate_config(config)
    return config


def validate_config(config: ManifestOldGoamConfig) -> None:
    if config.sample_count <= 1:
        raise ValueError("sample_count must be greater than 1")
    if config.ntrain <= 0:
        raise ValueError("ntrain must be positive")
    if config.ntest < 0:
        raise ValueError("ntest must be nonnegative")
    if config.ntrain + config.ntest != config.sample_count:
        raise ValueError("ntrain + ntest must equal sample_count")
    if config.latent_rank <= 0:
        raise ValueError("latent_rank must be positive")
    if config.max_dt <= 0.0:
        raise ValueError("max_dt must be positive")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not (0.0 < config.bfgs_c1 < config.bfgs_c2 < 1.0):
        raise ValueError("BFGS line-search parameters must satisfy 0 < c1 < c2 < 1")


def distributed_context_from_environment() -> DistributedContext:
    mpi_markers = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "MPI_LOCALNRANKS", "SLURM_NTASKS")
    if not any(name in os.environ for name in mpi_markers):
        return DistributedContext()
    from mpi4py import MPI  # type: ignore

    return DistributedContext.from_comm(MPI.COMM_WORLD)


def first_n_manifest(manifest: NpzSampleManifest, sample_count: int) -> NpzSampleManifest:
    if len(manifest) < sample_count:
        raise ValueError(f"Manifest has {len(manifest)} samples, requested {sample_count}.")
    return manifest.subset_by_indices(tuple(range(sample_count)))


def split_manifest(manifest: NpzSampleManifest, config: ManifestOldGoamConfig):
    train_ids = manifest.sample_ids[: config.ntrain]
    test_ids = manifest.sample_ids[config.ntrain : config.ntrain + config.ntest]
    return make_npz_train_test_split(manifest, train_sample_ids=train_ids, test_sample_ids=test_ids)


def run(config: ManifestOldGoamConfig) -> dict[str, object] | None:
    context = distributed_context_from_environment()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = first_n_manifest(load_npz_sample_manifest(config.manifest_path), config.sample_count)
    split = split_manifest(manifest, config)
    test_manifest = None if config.ntest == 0 else split.test_manifest
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "oldgoam" if config.oldgoam_mode else "stabilized"
    run_root = config.output_dir / f"{config.dataset_name}_{mode_tag}_r{config.latent_rank}_n{config.sample_count}_{run_stamp}"

    opinf_result = initialize_reduced_model_via_opinf(
        train_manifest=split.train_manifest,
        test_manifest=test_manifest,
        output_dir=run_root / "opinf_init",
        rank=config.latent_rank,
        context=context,
        apply_normalization=True,
        normalization_target_max_abs=config.normalization_target_max_abs,
        time_rescale_to_unit_interval=True,
        max_dt=config.max_dt,
        regularization=OpInfInitializationRegularization(
            coeff_w=config.opinf_reg_w,
            coeff_h=config.opinf_reg_h,
            coeff_b=config.opinf_reg_b,
            coeff_c=config.opinf_reg_c,
        ),
        decoder_regularization=DecoderTikhonovRegularization(
            coeff_v1=config.decoder_reg_v1,
            coeff_v2=config.decoder_reg_v2,
            coeff_v0=config.decoder_reg_v0,
        ),
        validation_time_integrator=config.time_integrator,
        max_regularization_retries=8,
        dynamic_form=config.dynamic_form,
        decoder_form=config.decoder_form,
        oldgoam_mode=config.oldgoam_mode,
        latent_embedding=OpInfLatentEmbeddingConfig(
            mode=config.latent_embedding_mode,
            augmentation_seed=config.latent_embedding_augmentation_seed,
            augmentation_scale=config.latent_embedding_augmentation_scale,
        ),
    )

    trainer_config = ReducedQoiTrainerConfig(
        output_dir=run_root / "runs",
        time_integrator=config.time_integrator,
        run_name_prefix=(
            f"{config.dataset_name}_{mode_tag}_{config.optimizer}_{config.time_integrator}"
            f"_r{config.latent_rank}_ntrain{config.ntrain}_ntest{config.ntest}"
        ),
        optimizer=config.optimizer,
        max_iterations=config.max_iterations,
        checkpoint_every=10,
        log_every=1,
        test_every=1,
        lbfgs=LbfgsUpdaterConfig(
            maxcor=config.lbfgs_maxcor,
            ftol=config.lbfgs_ftol,
            gtol=config.lbfgs_gtol,
            maxls=config.lbfgs_maxls,
        ),
        bfgs=BfgsUpdaterConfig(gtol=config.bfgs_gtol, c1=config.bfgs_c1, c2=config.bfgs_c2, xrtol=config.bfgs_xrtol),
    )
    trainer = ReducedQoiTrainer(
        train_manifest=opinf_result.latent_train_manifest,
        test_manifest=opinf_result.latent_test_manifest,
        decoder_template=opinf_result.decoder,
        regularization=DecoderTikhonovRegularization(
            coeff_v1=config.decoder_reg_v1,
            coeff_v2=config.decoder_reg_v2,
            coeff_v0=config.decoder_reg_v0,
        ),
        dynamics_regularization=DynamicsTikhonovRegularization(
            coeff_a=config.dynamics_reg_a,
            coeff_s=config.dynamics_reg_s,
            coeff_w=config.dynamics_reg_w,
            coeff_mu_h=config.dynamics_reg_mu_h,
            coeff_b=config.dynamics_reg_b,
            coeff_c=config.dynamics_reg_c,
            coeff_spectral_abscissa=config.dynamics_reg_spectral_abscissa,
            spectral_abscissa_alpha=config.dynamics_reg_spectral_alpha,
        ),
        max_dt=config.max_dt,
        config=trainer_config,
        preprocess_record=opinf_result.as_preprocess_record(),
        context=context,
        dt_shrink=0.5,
        dt_min=1e-5,
        tol=1e-12,
        max_iter_newton=40,
    )
    result = trainer.train(opinf_result.dynamics)
    if context.rank != 0:
        return None

    metrics_records = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    initial = metrics_records[0]
    final = metrics_records[-1]
    summary: dict[str, object] = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "sample_ids_first_last": [manifest.sample_ids[0], manifest.sample_ids[-1]],
        "mpi_world_size": context.size,
        "opinf_regression_relative_residual": float(opinf_result.regression_relative_residual),
        "opinf_summary_path": str(opinf_result.summary_path),
        "initial_train_objective": float(initial["train_objective"]),
        "initial_train_relative_error": float(initial["train_relative_error"]),
        "initial_test_relative_error": initial.get("test_relative_error"),
        "final_train_objective": float(final["train_objective"]),
        "final_train_relative_error": float(final["train_relative_error"]),
        "final_test_relative_error": final.get("test_relative_error"),
        "best_train_objective": float(result.best_snapshot.objective_value),
        "best_train_relative_error": float(result.best_snapshot.train_relative_error),
        "best_test_relative_error": result.best_snapshot.test_relative_error,
        "run_root": str(run_root),
        "run_output_dir": str(result.output_dir),
        "best_checkpoint_path": str(result.best_checkpoint_path),
        "metrics_path": str(result.metrics_path),
        "training_summary_path": str(result.summary_path),
    }
    summary_path = config.output_dir / f"latest_{config.dataset_name}_r{config.latent_rank}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
