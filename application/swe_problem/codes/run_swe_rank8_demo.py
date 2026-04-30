from __future__ import annotations

"""Small SWE GOATTM demo using preprocessed SWE sensor QoI data.

Default setting:

* first 64 processed SWE samples;
* 48 train / 16 test split;
* latent rank 8;
* RK4 rollouts;
* physical QoI grid 0, 5, ..., 1500 rescaled to [0, 1];
* max_dt = 1 / 600, i.e. two RK4 steps per QoI interval after rescaling.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
SWE_PROBLEM_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.data import NpzSampleManifest, load_npz_sample_manifest, make_npz_train_test_split  # noqa: E402
from goattm.preprocess import OpInfInitializationRegularization, initialize_reduced_model_via_opinf  # noqa: E402
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization  # noqa: E402
from goattm.runtime import DistributedContext  # noqa: E402
from goattm.train import (  # noqa: E402
    AdamBfgsUpdaterConfig,
    AdamUpdaterConfig,
    BfgsUpdaterConfig,
    LbfgsUpdaterConfig,
    ReducedQoiTrainer,
    ReducedQoiTrainerConfig,
)


DEFAULT_MANIFEST_PATH = SWE_PROBLEM_ROOT / "data" / "processed_data" / "manifest.npz"
DEFAULT_OUTPUT_DIR = SWE_PROBLEM_ROOT / "outputs" / "swe_rank8_rk4_demo"


@dataclass(frozen=True)
class SweDemoConfig:
    manifest_path: Path
    sample_count: int
    ntrain: int
    ntest: int
    latent_rank: int
    max_dt: float
    time_integrator: str
    optimizer: str
    max_iterations: int
    normalization_target_max_abs: float
    adam_learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    adam_gradient_clip_norm: float | None
    lbfgs_maxcor: int
    lbfgs_ftol: float
    lbfgs_gtol: float
    lbfgs_maxls: int
    bfgs_gtol: float
    bfgs_c1: float
    bfgs_c2: float
    bfgs_xrtol: float
    adam_bfgs_adam_iterations: int
    opinf_reg_w: float
    opinf_reg_h: float
    opinf_reg_b: float
    opinf_reg_c: float
    decoder_reg_v1: float
    decoder_reg_v2: float
    decoder_reg_v0: float
    dynamics_reg_s: float
    dynamics_reg_w: float
    dynamics_reg_mu_h: float
    dynamics_reg_b: float
    dynamics_reg_c: float
    output_dir: Path


def parse_args() -> SweDemoConfig:
    parser = argparse.ArgumentParser(description="Run a 64-sample rank-8 RK4 GOATTM demo on SWE data.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--ntrain", type=int, default=48)
    parser.add_argument("--ntest", type=int, default=16)
    parser.add_argument("--latent-rank", type=int, default=8)
    parser.add_argument("--max-dt", type=float, default=1.0 / 600.0)
    parser.add_argument("--time-integrator", default="rk4", choices=("implicit_midpoint", "explicit_euler", "rk4"))
    parser.add_argument(
        "--optimizer",
        default="lbfgs",
        choices=("lbfgs", "bfgs", "adam_bfgs", "adam", "gradient_descent", "newton_action"),
    )
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--normalization-target-max-abs", type=float, default=0.9)
    parser.add_argument("--adam-learning-rate", type=float, default=1e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--adam-gradient-clip-norm", type=float, default=None)
    parser.add_argument("--lbfgs-maxcor", type=int, default=20)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-12)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    parser.add_argument("--lbfgs-maxls", type=int, default=30)
    parser.add_argument("--bfgs-gtol", type=float, default=1e-6)
    parser.add_argument("--bfgs-c1", type=float, default=1e-4)
    parser.add_argument("--bfgs-c2", type=float, default=0.9)
    parser.add_argument("--bfgs-xrtol", type=float, default=1e-7)
    parser.add_argument("--adam-bfgs-adam-iterations", type=int, default=100)
    parser.add_argument("--opinf-reg-w", type=float, default=1e-4)
    parser.add_argument("--opinf-reg-h", type=float, default=1e-4)
    parser.add_argument("--opinf-reg-b", type=float, default=1e-4)
    parser.add_argument("--opinf-reg-c", type=float, default=1e-6)
    parser.add_argument("--decoder-reg-v1", type=float, default=1e-7)
    parser.add_argument("--decoder-reg-v2", type=float, default=1e-7)
    parser.add_argument("--decoder-reg-v0", type=float, default=1e-7)
    parser.add_argument("--dynamics-reg-s", type=float, default=1e-4)
    parser.add_argument("--dynamics-reg-w", type=float, default=1e-4)
    parser.add_argument("--dynamics-reg-mu-h", type=float, default=1e-4)
    parser.add_argument("--dynamics-reg-b", type=float, default=1e-4)
    parser.add_argument("--dynamics-reg-c", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    config = SweDemoConfig(**vars(args))
    validate_config(config)
    return config


def validate_config(config: SweDemoConfig) -> None:
    if config.sample_count <= 1:
        raise ValueError(f"sample_count must be greater than 1, got {config.sample_count}")
    if config.ntrain <= 0:
        raise ValueError(f"ntrain must be positive, got {config.ntrain}")
    if config.ntest < 0:
        raise ValueError(f"ntest must be nonnegative, got {config.ntest}")
    if config.ntrain + config.ntest != config.sample_count:
        raise ValueError(
            f"ntrain + ntest must equal sample_count, got {config.ntrain}+{config.ntest}!={config.sample_count}"
        )
    if config.latent_rank <= 0:
        raise ValueError(f"latent_rank must be positive, got {config.latent_rank}")
    if config.max_dt <= 0.0:
        raise ValueError(f"max_dt must be positive, got {config.max_dt}")
    if config.max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {config.max_iterations}")
    if config.normalization_target_max_abs <= 0.0:
        raise ValueError("normalization_target_max_abs must be positive.")
    if config.adam_learning_rate <= 0.0:
        raise ValueError(f"adam_learning_rate must be positive, got {config.adam_learning_rate}")
    if not (0.0 < config.adam_beta1 < 1.0):
        raise ValueError(f"adam_beta1 must satisfy 0 < beta1 < 1, got {config.adam_beta1}")
    if not (0.0 < config.adam_beta2 < 1.0):
        raise ValueError(f"adam_beta2 must satisfy 0 < beta2 < 1, got {config.adam_beta2}")
    if config.adam_epsilon <= 0.0:
        raise ValueError(f"adam_epsilon must be positive, got {config.adam_epsilon}")
    if config.adam_gradient_clip_norm is not None and config.adam_gradient_clip_norm <= 0.0:
        raise ValueError(f"adam_gradient_clip_norm must be positive, got {config.adam_gradient_clip_norm}")
    if config.adam_bfgs_adam_iterations < 0:
        raise ValueError(
            f"adam_bfgs_adam_iterations must be nonnegative, got {config.adam_bfgs_adam_iterations}"
        )
    if config.lbfgs_maxcor <= 0:
        raise ValueError(f"lbfgs_maxcor must be positive, got {config.lbfgs_maxcor}")
    if config.lbfgs_maxls <= 0:
        raise ValueError(f"lbfgs_maxls must be positive, got {config.lbfgs_maxls}")
    if config.bfgs_gtol <= 0.0:
        raise ValueError(f"bfgs_gtol must be positive, got {config.bfgs_gtol}")
    if not (0.0 < config.bfgs_c1 < config.bfgs_c2 < 1.0):
        raise ValueError(
            f"BFGS line-search parameters must satisfy 0 < c1 < c2 < 1, got c1={config.bfgs_c1}, c2={config.bfgs_c2}"
        )
    if config.bfgs_xrtol < 0.0:
        raise ValueError(f"bfgs_xrtol must be nonnegative, got {config.bfgs_xrtol}")


def distributed_context_from_environment() -> DistributedContext:
    mpi_markers = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS",
    )
    if not any(name in os.environ for name in mpi_markers):
        return DistributedContext()

    from mpi4py import MPI  # type: ignore

    return DistributedContext.from_comm(MPI.COMM_WORLD)


def first_n_manifest(manifest: NpzSampleManifest, sample_count: int) -> NpzSampleManifest:
    if len(manifest) < sample_count:
        raise ValueError(f"Manifest has only {len(manifest)} samples, requested {sample_count}.")
    return manifest.subset_by_indices(tuple(range(sample_count)))


def split_swe_manifest(manifest: NpzSampleManifest, config: SweDemoConfig):
    train_ids = manifest.sample_ids[: config.ntrain]
    test_ids = manifest.sample_ids[config.ntrain : config.ntrain + config.ntest]
    if config.ntest == 0:
        return make_npz_train_test_split(manifest, train_sample_ids=train_ids, test_sample_ids=())
    return make_npz_train_test_split(manifest, train_sample_ids=train_ids, test_sample_ids=test_ids)


def write_loss_history(csv_path: Path, markdown_path: Path, records: list[dict[str, object]]) -> None:
    columns = (
        "iteration",
        "train_objective",
        "train_data_loss",
        "train_relative_error",
        "test_data_loss",
        "test_relative_error",
        "gradient_norm",
        "step_norm",
        "best_iteration",
        "best_train_objective",
        "best_train_relative_error",
        "best_test_data_loss",
        "best_test_relative_error",
    )
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for record in records:
            fh.write(",".join(str(record.get(column, "")) for column in columns) + "\n")

    lines = [
        "# SWE GOATTM Loss History",
        "",
        "| iter | train obj | train rel | test rel | grad norm | step norm |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.get('iteration', 'NA')} | "
            f"{_fmt(record.get('train_objective'))} | "
            f"{_fmt(record.get('train_relative_error'))} | "
            f"{_fmt(record.get('test_relative_error'))} | "
            f"{_fmt(record.get('gradient_norm'))} | "
            f"{_fmt(record.get('step_norm'))} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object, precision: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def run_swe_demo(config: SweDemoConfig) -> dict[str, object] | None:
    context = distributed_context_from_environment()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = first_n_manifest(load_npz_sample_manifest(config.manifest_path), config.sample_count)
    split = split_swe_manifest(manifest, config)
    test_manifest = None if config.ntest == 0 else split.test_manifest
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = config.output_dir / f"swe_rank{config.latent_rank}_n{config.sample_count}_{run_stamp}"

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
        validation_time_integrator=config.time_integrator,
        max_regularization_retries=8,
        decoder_regularization=DecoderTikhonovRegularization(
            coeff_v1=config.decoder_reg_v1,
            coeff_v2=config.decoder_reg_v2,
            coeff_v0=config.decoder_reg_v0,
        ),
    )

    trainer_config = ReducedQoiTrainerConfig(
        output_dir=run_root / "runs",
        time_integrator=config.time_integrator,
        run_name_prefix=(
            f"swe_{config.optimizer}_rk4_r{config.latent_rank}_ntrain{config.ntrain}_ntest{config.ntest}"
        ),
        optimizer=config.optimizer,
        max_iterations=config.max_iterations,
        checkpoint_every=10,
        log_every=1,
        test_every=1,
        adam=AdamUpdaterConfig(
            learning_rate=config.adam_learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon,
            gradient_clip_norm=config.adam_gradient_clip_norm,
        ),
        lbfgs=LbfgsUpdaterConfig(
            maxcor=config.lbfgs_maxcor,
            ftol=config.lbfgs_ftol,
            gtol=config.lbfgs_gtol,
            maxls=config.lbfgs_maxls,
        ),
        bfgs=BfgsUpdaterConfig(
            gtol=config.bfgs_gtol,
            c1=config.bfgs_c1,
            c2=config.bfgs_c2,
            xrtol=config.bfgs_xrtol,
        ),
        adam_bfgs=AdamBfgsUpdaterConfig(adam_iterations=config.adam_bfgs_adam_iterations),
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
            coeff_s=config.dynamics_reg_s,
            coeff_w=config.dynamics_reg_w,
            coeff_mu_h=config.dynamics_reg_mu_h,
            coeff_b=config.dynamics_reg_b,
            coeff_c=config.dynamics_reg_c,
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
    loss_history_csv_path = result.output_dir / "loss_history.csv"
    loss_history_markdown_path = result.output_dir / "loss_history.md"
    write_loss_history(loss_history_csv_path, loss_history_markdown_path, metrics_records)
    initial = metrics_records[0]
    final = metrics_records[-1]
    summary: dict[str, object] = {
        "dataset": "swe_problem",
        "manifest_path": str(config.manifest_path),
        "sample_count": config.sample_count,
        "ntrain": config.ntrain,
        "ntest": config.ntest,
        "sample_ids_first_last": [manifest.sample_ids[0], manifest.sample_ids[-1]],
        "latent_rank": config.latent_rank,
        "time_integrator": config.time_integrator,
        "time_rescale_to_unit_interval": True,
        "physical_qoi_times": "0, 5, 10, ..., 1500",
        "latent_qoi_dt": 1.0 / 300.0,
        "max_dt": config.max_dt,
        "optimizer": config.optimizer,
        "max_iterations": config.max_iterations,
        "adam": {
            "learning_rate": config.adam_learning_rate,
            "beta1": config.adam_beta1,
            "beta2": config.adam_beta2,
            "epsilon": config.adam_epsilon,
            "gradient_clip_norm": config.adam_gradient_clip_norm,
        },
        "lbfgs": {
            "maxcor": config.lbfgs_maxcor,
            "ftol": config.lbfgs_ftol,
            "gtol": config.lbfgs_gtol,
            "maxls": config.lbfgs_maxls,
        },
        "bfgs": {
            "gtol": config.bfgs_gtol,
            "c1": config.bfgs_c1,
            "c2": config.bfgs_c2,
            "xrtol": config.bfgs_xrtol,
        },
        "adam_bfgs": {
            "adam_iterations": config.adam_bfgs_adam_iterations,
        },
        "normalization_target_max_abs": config.normalization_target_max_abs,
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
        "loss_history_csv_path": str(loss_history_csv_path),
        "loss_history_markdown_path": str(loss_history_markdown_path),
        "training_summary_path": str(result.summary_path),
        "timing_summary_path": str(result.timing_summary_path),
        "stdout_log_path": str(result.stdout_log_path),
        "stderr_log_path": str(result.stderr_log_path),
    }
    summary_path = config.output_dir / "latest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> None:
    run_swe_demo(parse_args())


if __name__ == "__main__":
    main()
