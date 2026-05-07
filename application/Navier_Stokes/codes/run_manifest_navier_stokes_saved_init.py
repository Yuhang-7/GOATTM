from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension
from goattm.data import (
    NpzQoiSample,
    NpzSampleManifest,
    load_npz_qoi_sample,
    load_npz_sample_manifest,
    make_npz_train_test_split,
    save_npz_qoi_sample,
    save_npz_sample_manifest,
)
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization
from goattm.runtime import DistributedContext
from goattm.train import BfgsUpdaterConfig, LbfgsUpdaterConfig, ReducedQoiTrainer, ReducedQoiTrainerConfig


@dataclass(frozen=True)
class SavedInitRunConfig:
    dataset_name: str
    manifest_path: Path
    sample_count: int
    ntrain: int
    ntest: int
    latent_rank: int
    decoder_form: str
    initial_value_dir: Path
    max_dt: float
    time_integrator: str
    optimizer: str
    max_iterations: int
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
    lbfgs_maxcor: int
    lbfgs_ftol: float
    lbfgs_gtol: float
    lbfgs_maxls: int
    bfgs_gtol: float
    bfgs_c1: float
    bfgs_c2: float
    bfgs_xrtol: float
    output_dir: Path
    run_stamp: str | None


def parse_args() -> SavedInitRunConfig:
    parser = argparse.ArgumentParser(
        description="Run Navier-Stokes oldGOAM training from saved GOAM muf/mug initialization parameters."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--ntrain", type=int, required=True)
    parser.add_argument("--ntest", type=int, required=True)
    parser.add_argument("--latent-rank", type=int, required=True)
    parser.add_argument("--decoder-form", choices=("V1V2v",), default="V1V2v")
    parser.add_argument("--initial-value-dir", type=Path, required=True)
    parser.add_argument("--max-dt", type=float, default=0.01)
    parser.add_argument("--time-integrator", choices=("implicit_midpoint", "explicit_euler", "rk4"), default="rk4")
    parser.add_argument(
        "--optimizer",
        choices=("lbfgs", "bfgs", "adam", "gradient_descent", "newton_action"),
        default="bfgs",
    )
    parser.add_argument("--max-iterations", type=int, default=20000)
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
    parser.add_argument("--lbfgs-maxcor", type=int, default=20)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-12)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    parser.add_argument("--lbfgs-maxls", type=int, default=30)
    parser.add_argument("--bfgs-gtol", type=float, default=1e-6)
    parser.add_argument("--bfgs-c1", type=float, default=1e-4)
    parser.add_argument("--bfgs-c2", type=float, default=0.9)
    parser.add_argument("--bfgs-xrtol", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-stamp", default=None)
    config = SavedInitRunConfig(**vars(parser.parse_args()))
    validate_config(config)
    return config


def validate_config(config: SavedInitRunConfig) -> None:
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


def split_manifest(manifest: NpzSampleManifest, config: SavedInitRunConfig):
    train_ids = manifest.sample_ids[: config.ntrain]
    test_ids = manifest.sample_ids[config.ntrain : config.ntrain + config.ntest]
    return make_npz_train_test_split(manifest, train_sample_ids=train_ids, test_sample_ids=test_ids)


def mug_path(config: SavedInitRunConfig) -> Path:
    return config.initial_value_dir / f"mug_{config.latent_rank}.npy"


def muf_path(config: SavedInitRunConfig) -> Path:
    return config.initial_value_dir / f"muf_{config.latent_rank}.npy"


def oldgoam_mug_to_dynamics(path: Path, rank: int, input_dim: int) -> QuadraticDynamics:
    theta = np.load(path).astype(np.float64).reshape(-1)
    mu_h_dim = mu_h_dimension(rank)
    expected = rank * rank + mu_h_dim + rank * input_dim + rank
    if theta.shape[0] != expected:
        raise ValueError(f"Expected saved GOAM mug length {expected}, got {theta.shape[0]} from {path}")
    marker = 0
    a = theta[marker : marker + rank * rank].reshape((rank, rank)).copy()
    marker += rank * rank
    mu_h = theta[marker : marker + mu_h_dim].copy()
    marker += mu_h_dim
    b = theta[marker : marker + rank * input_dim].reshape((rank, input_dim)).copy()
    marker += rank * input_dim
    c = theta[marker : marker + rank].copy()
    return QuadraticDynamics(a=a, mu_h=mu_h, b=b, c=c)


def oldgoam_muf_to_decoder(path: Path, rank: int, output_dim: int, form: str) -> QuadraticDecoder:
    theta = np.load(path).astype(np.float64).reshape(-1)
    quad_dim = compressed_quadratic_dimension(rank)
    expected = output_dim * rank + output_dim * quad_dim + output_dim
    if theta.shape[0] != expected:
        raise ValueError(f"Expected saved GOAM muf length {expected}, got {theta.shape[0]} from {path}")
    marker = 0
    v1 = theta[marker : marker + output_dim * rank].reshape((output_dim, rank)).copy()
    marker += output_dim * rank
    v2 = theta[marker : marker + output_dim * quad_dim].reshape((output_dim, quad_dim)).copy()
    marker += output_dim * quad_dim
    v0 = theta[marker : marker + output_dim].copy()
    return QuadraticDecoder(v1=v1, v2=v2, v0=v0, form=form)


def materialize_latent_zero_manifest(
    manifest: NpzSampleManifest,
    destination_root: Path,
    rank: int,
    context: DistributedContext,
) -> NpzSampleManifest:
    destination_root.mkdir(parents=True, exist_ok=True)
    output_paths = tuple(destination_root / f"{idx:06d}_{manifest.sample_ids[idx]}.npz" for idx in range(len(manifest)))
    for sample_id, sample_path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(sample_path)
        global_index = manifest.sample_ids.index(sample_id)
        metadata = dict(sample.metadata or {})
        metadata.update(
            {
                "oldgoam_saved_init_sample": 1,
                "oldgoam_latent_u0": "zero_initial_condition",
                "oldgoam_rescaled_like_goam": 0,
            }
        )
        processed = NpzQoiSample(
            sample_id=sample.sample_id,
            observation_times=np.asarray(sample.observation_times, dtype=np.float64),
            u0=np.zeros(rank, dtype=np.float64),
            qoi_observations=np.asarray(sample.qoi_observations, dtype=np.float64),
            input_times=None if sample.input_times is None else np.asarray(sample.input_times, dtype=np.float64),
            input_values=None if sample.input_values is None else np.asarray(sample.input_values, dtype=np.float64),
            metadata=metadata,
        )
        save_npz_qoi_sample(output_paths[global_index], processed)
    context.barrier()
    return NpzSampleManifest(
        root_dir=destination_root,
        sample_paths=tuple(Path(path.name) for path in output_paths),
        sample_ids=manifest.sample_ids,
    )


def run(config: SavedInitRunConfig) -> dict[str, object] | None:
    context = distributed_context_from_environment()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = first_n_manifest(load_npz_sample_manifest(config.manifest_path), config.sample_count)
    split = split_manifest(manifest, config)
    run_stamp = config.run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = config.output_dir / f"{config.dataset_name}_savedoldgoam_r{config.latent_rank}_n{config.sample_count}_{run_stamp}"
    train_manifest = materialize_latent_zero_manifest(
        split.train_manifest,
        destination_root=run_root / "oldgoam_latent_dataset" / "train",
        rank=config.latent_rank,
        context=context,
    )
    test_manifest = materialize_latent_zero_manifest(
        split.test_manifest,
        destination_root=run_root / "oldgoam_latent_dataset" / "test",
        rank=config.latent_rank,
        context=context,
    )
    train_manifest_path = run_root / "oldgoam_latent_train_manifest.npz"
    test_manifest_path = run_root / "oldgoam_latent_test_manifest.npz"
    if context.rank == 0:
        save_npz_sample_manifest(train_manifest_path, train_manifest)
        save_npz_sample_manifest(test_manifest_path, test_manifest)
    context.barrier()

    first_sample = load_npz_qoi_sample(train_manifest.absolute_paths()[0])
    input_dim = first_sample.input_dimension
    output_dim = first_sample.output_dimension
    init_mug_path = mug_path(config)
    init_muf_path = muf_path(config)
    dynamics = oldgoam_mug_to_dynamics(init_mug_path, config.latent_rank, input_dim)
    decoder_template = oldgoam_muf_to_decoder(init_muf_path, config.latent_rank, output_dim, config.decoder_form)
    trainer_config = ReducedQoiTrainerConfig(
        output_dir=run_root / "runs",
        time_integrator=config.time_integrator,
        run_name_prefix=(
            f"{config.dataset_name}_savedoldgoam_{config.optimizer}_{config.time_integrator}"
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
    preprocess_record = {
        "applied": True,
        "pipeline": "saved_goam_muf_mug_initialization",
        "saved_mug_path": str(init_mug_path),
        "saved_muf_path": str(init_muf_path),
        "train_manifest_path": str(train_manifest_path),
        "test_manifest_path": str(test_manifest_path),
        "time_rescaled_to_unit_interval": False,
        "normalization": "none",
        "validation_success": True,
        "validation_attempt_count": 0,
        "regression_relative_residual": float("nan"),
    }
    trainer = ReducedQoiTrainer(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        decoder_template=decoder_template,
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
        preprocess_record=preprocess_record,
        context=context,
        dt_shrink=0.5,
        dt_min=1e-5,
        tol=1e-12,
        max_iter_newton=40,
    )
    result = trainer.train(dynamics)
    if context.rank != 0:
        return None
    metrics_records = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    summary = {
        "config": asdict(config),
        "run_root": str(run_root),
        "saved_mug_path": str(init_mug_path),
        "saved_muf_path": str(init_muf_path),
        "initial_metrics": metrics_records[0],
        "final_metrics": metrics_records[-1],
        "metrics_path": str(result.metrics_path),
        "latest_checkpoint_path": str(result.latest_checkpoint_path),
        "best_checkpoint_path": str(result.best_checkpoint_path),
    }
    summary_path = run_root / "saved_oldgoam_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
