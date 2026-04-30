from __future__ import annotations

"""MPI-ready GOATTM reduced-QoI demo with configurable train/test sizes.

The demo reproduces the small maintained problem setting at larger scale:

* synthetic scalar input p(t), sampled on t=0:0.1:1 and interpolated by cubic splines;
* 20-dimensional QoI observations
  q_j(t) = exp(a_j p(t)) + b_j (p(t)-a_j)^2;
* OpInf initialization on normalized data;
* stabilized quadratic reduced dynamics with S initialized as identity;
* implicit-midpoint rollouts for OpInf validation and training by default;
* L-BFGS optimization with global MPI reductions.

When launched with ``mpirun -n NTRAIN``, the train manifest is distributed so
that each MPI rank owns one train sample. If fewer or more ranks are used, the
same manifest partitioning logic still works.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.data import (  # noqa: E402
    NpzQoiSample,
    NpzSampleManifest,
    load_npz_sample_manifest,
    make_npz_train_test_split,
    save_npz_qoi_sample,
    save_npz_sample_manifest,
)
from goattm.preprocess import OpInfInitializationRegularization, initialize_reduced_model_via_opinf  # noqa: E402
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization  # noqa: E402
from goattm.runtime import DistributedContext  # noqa: E402
from goattm.train import BfgsUpdaterConfig, LbfgsUpdaterConfig, ReducedQoiTrainer, ReducedQoiTrainerConfig  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "demo" / "outputs" / "reduced_qoi_optimization_demo"


@dataclass(frozen=True)
class DemoConfig:
    # Problem setting.
    ntrain: int
    ntest: int
    observation_dt: float
    output_dimension: int
    seed: int

    # Model and solver setting.
    latent_rank: int
    max_dt: float
    time_integrator: str
    normalization_target_max_abs: float

    # Optimization setting.
    optimizer: str
    max_iterations: int
    lbfgs_maxcor: int
    lbfgs_ftol: float
    lbfgs_gtol: float
    lbfgs_maxls: int
    bfgs_gtol: float
    bfgs_c1: float
    bfgs_c2: float
    bfgs_xrtol: float

    # Regularization setting.
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

    @property
    def sample_count(self) -> int:
        return self.ntrain + self.ntest


@dataclass(frozen=True)
class RawDatasetArtifacts:
    manifest_path: Path
    qoi_stats: dict[str, object]


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        description="Run the configurable GOATTM reduced-QoI OpInf optimization demo."
    )
    parser.add_argument("--ntrain", type=int, default=10, help="Number of train samples.")
    parser.add_argument("--ntest", type=int, default=10, help="Number of test samples.")
    parser.add_argument("--observation-dt", type=float, default=0.1, help="Observation spacing.")
    parser.add_argument("--output-dimension", type=int, default=20, help="Synthetic QoI dimension.")
    parser.add_argument("--seed", type=int, default=20260428, help="Synthetic dataset RNG seed.")
    parser.add_argument("--latent-rank", type=int, default=4, help="Reduced latent dimension.")
    parser.add_argument("--max-dt", type=float, default=0.01, help="Maximum rollout time step.")
    parser.add_argument(
        "--normalization-target-max-abs",
        type=float,
        default=0.9,
        help="Per-dimension max absolute value after centered train-set normalization.",
    )
    parser.add_argument(
        "--time-integrator",
        default="implicit_midpoint",
        choices=("implicit_midpoint", "explicit_euler", "rk4"),
        help="Time integrator for OpInf validation and GOATTM training rollouts.",
    )
    parser.add_argument(
        "--optimizer",
        default="bfgs",
        choices=("lbfgs", "bfgs", "adam", "gradient_descent", "newton_action"),
    )
    parser.add_argument("--max-iterations", type=int, default=50, help="Optimizer max iterations.")
    parser.add_argument("--lbfgs-maxcor", type=int, default=20, help="L-BFGS memory size.")
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-12, help="L-BFGS ftol.")
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8, help="L-BFGS gtol.")
    parser.add_argument("--lbfgs-maxls", type=int, default=30, help="L-BFGS max line-search steps.")
    parser.add_argument("--bfgs-gtol", type=float, default=1e-6, help="BFGS gradient norm tolerance.")
    parser.add_argument("--bfgs-c1", type=float, default=1e-4, help="BFGS Armijo line-search parameter.")
    parser.add_argument("--bfgs-c2", type=float, default=0.9, help="BFGS curvature line-search parameter.")
    parser.add_argument("--bfgs-xrtol", type=float, default=1e-7, help="BFGS relative step-size tolerance.")
    parser.add_argument("--opinf-reg-w", type=float, default=1e-4, help="OpInf W regularization.")
    parser.add_argument("--opinf-reg-h", type=float, default=1e-4, help="OpInf H regularization.")
    parser.add_argument("--opinf-reg-b", type=float, default=1e-4, help="OpInf B regularization.")
    parser.add_argument("--opinf-reg-c", type=float, default=1e-6, help="OpInf c regularization.")
    parser.add_argument("--decoder-reg-v1", type=float, default=1e-7, help="Decoder V1 regularization.")
    parser.add_argument("--decoder-reg-v2", type=float, default=1e-7, help="Decoder V2 regularization.")
    parser.add_argument("--decoder-reg-v0", type=float, default=1e-7, help="Decoder v0 regularization.")
    parser.add_argument("--dynamics-reg-s", type=float, default=1e-4, help="Training dynamics S regularization.")
    parser.add_argument("--dynamics-reg-w", type=float, default=1e-4, help="Training dynamics W regularization.")
    parser.add_argument("--dynamics-reg-mu-h", type=float, default=1e-4, help="Training dynamics mu_H regularization.")
    parser.add_argument("--dynamics-reg-b", type=float, default=1e-4, help="Training dynamics B regularization.")
    parser.add_argument("--dynamics-reg-c", type=float, default=1e-4, help="Training dynamics c regularization.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()
    config = DemoConfig(
        ntrain=args.ntrain,
        ntest=args.ntest,
        observation_dt=args.observation_dt,
        output_dimension=args.output_dimension,
        seed=args.seed,
        latent_rank=args.latent_rank,
        max_dt=args.max_dt,
        normalization_target_max_abs=args.normalization_target_max_abs,
        time_integrator=args.time_integrator,
        optimizer=args.optimizer,
        max_iterations=args.max_iterations,
        lbfgs_maxcor=args.lbfgs_maxcor,
        lbfgs_ftol=args.lbfgs_ftol,
        lbfgs_gtol=args.lbfgs_gtol,
        lbfgs_maxls=args.lbfgs_maxls,
        bfgs_gtol=args.bfgs_gtol,
        bfgs_c1=args.bfgs_c1,
        bfgs_c2=args.bfgs_c2,
        bfgs_xrtol=args.bfgs_xrtol,
        opinf_reg_w=args.opinf_reg_w,
        opinf_reg_h=args.opinf_reg_h,
        opinf_reg_b=args.opinf_reg_b,
        opinf_reg_c=args.opinf_reg_c,
        decoder_reg_v1=args.decoder_reg_v1,
        decoder_reg_v2=args.decoder_reg_v2,
        decoder_reg_v0=args.decoder_reg_v0,
        dynamics_reg_s=args.dynamics_reg_s,
        dynamics_reg_w=args.dynamics_reg_w,
        dynamics_reg_mu_h=args.dynamics_reg_mu_h,
        dynamics_reg_b=args.dynamics_reg_b,
        dynamics_reg_c=args.dynamics_reg_c,
        output_dir=Path(args.output_dir),
    )
    validate_config(config)
    return config


def validate_config(config: DemoConfig) -> None:
    if config.ntrain <= 0:
        raise ValueError(f"ntrain must be positive, got {config.ntrain}")
    if config.ntest <= 0:
        raise ValueError(f"ntest must be positive, got {config.ntest}")
    if config.latent_rank <= 0:
        raise ValueError(f"latent_rank must be positive, got {config.latent_rank}")
    if config.latent_rank > config.output_dimension:
        raise ValueError(
            f"latent_rank={config.latent_rank} exceeds output_dimension={config.output_dimension}; "
            "this demo uses POD projection, so choose latent_rank <= output_dimension."
        )
    if config.max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {config.max_iterations}")
    if config.max_dt <= 0.0:
        raise ValueError(f"max_dt must be positive, got {config.max_dt}")
    if config.normalization_target_max_abs <= 0.0:
        raise ValueError(
            f"normalization_target_max_abs must be positive, got {config.normalization_target_max_abs}"
        )
    if config.observation_dt <= 0.0:
        raise ValueError(f"observation_dt must be positive, got {config.observation_dt}")
    if config.output_dimension <= 0:
        raise ValueError(f"output_dimension must be positive, got {config.output_dimension}")
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
    regularization_values = (
        config.opinf_reg_w,
        config.opinf_reg_h,
        config.opinf_reg_b,
        config.opinf_reg_c,
        config.decoder_reg_v1,
        config.decoder_reg_v2,
        config.decoder_reg_v0,
        config.dynamics_reg_s,
        config.dynamics_reg_w,
        config.dynamics_reg_mu_h,
        config.dynamics_reg_b,
        config.dynamics_reg_c,
    )
    if any(value < 0.0 for value in regularization_values):
        raise ValueError("regularization coefficients must be nonnegative.")


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


def build_raw_dataset(
    root: Path,
    config: DemoConfig,
    rng: np.random.Generator,
) -> RawDatasetArtifacts:
    """Materialize the synthetic raw dataset and its manifest."""
    observation_times = np.arange(0.0, 1.0 + 1e-12, config.observation_dt, dtype=float)
    input_dimension = 1
    ab_pairs = rng.uniform(-2.0, 2.0, size=(config.output_dimension, 2))

    sample_paths: list[Path] = []
    sample_ids: list[str] = []
    all_qoi_values: list[np.ndarray] = []
    for sample_idx in range(config.sample_count):
        p_values = rng.uniform(-2.0, 2.0, size=(observation_times.shape[0], input_dimension))
        p_column = p_values[:, 0]
        qoi_observations = np.column_stack(
            [
                np.exp(a_value * p_column) + b_value * np.square(p_column - a_value)
                for a_value, b_value in ab_pairs
            ]
        )
        sample = NpzQoiSample(
            sample_id=f"sample-{sample_idx:03d}",
            observation_times=observation_times,
            u0=qoi_observations[0].copy(),
            qoi_observations=qoi_observations,
            input_times=observation_times,
            input_values=p_values,
            metadata={"dataset_kind": "raw_qoi_exp_quadratic_of_input"},
        )
        sample_path = root / f"sample_{sample_idx:03d}.npz"
        save_npz_qoi_sample(sample_path, sample)
        sample_paths.append(Path(sample_path.name))
        sample_ids.append(sample.sample_id)
        all_qoi_values.append(qoi_observations)

    manifest = NpzSampleManifest(root_dir=root, sample_paths=tuple(sample_paths), sample_ids=tuple(sample_ids))
    manifest_path = root / "manifest.npz"
    save_npz_sample_manifest(manifest_path, manifest)

    qoi_stack = np.concatenate(all_qoi_values, axis=0)
    qoi_stats: dict[str, object] = {
        "output_dimension": config.output_dimension,
        "ab_pairs": ab_pairs.tolist(),
        "qoi_abs_mean": float(np.mean(np.abs(qoi_stack))),
        "qoi_abs_max": float(np.max(np.abs(qoi_stack))),
        "qoi_l2_mean_per_observation": float(np.mean(np.linalg.norm(qoi_stack, axis=1))),
        "qoi_l2_global": float(np.linalg.norm(qoi_stack)),
    }
    return RawDatasetArtifacts(manifest_path=manifest_path, qoi_stats=qoi_stats)


def split_train_test(manifest: NpzSampleManifest, config: DemoConfig):
    train_ids = [f"sample-{idx:03d}" for idx in range(config.ntrain)]
    test_ids = [f"sample-{idx:03d}" for idx in range(config.ntrain, config.sample_count)]
    return make_npz_train_test_split(
        manifest,
        train_sample_ids=train_ids,
        test_sample_ids=test_ids,
    )


def _format_optional_float(value: object, precision: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def _read_json_if_present(path: Path | str | None) -> dict[str, object]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _top_timing_records(path: Path, limit: int = 8) -> list[dict[str, object]]:
    timing = _read_json_if_present(path)
    records = timing.get("records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records[:limit] if isinstance(record, dict)]


LOSS_HISTORY_COLUMNS = (
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


def write_loss_history(
    csv_path: Path,
    markdown_path: Path,
    metrics_records: list[dict[str, object]],
) -> None:
    """Write per-iteration loss history in machine- and human-readable forms."""
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOSS_HISTORY_COLUMNS)
        writer.writeheader()
        for record in metrics_records:
            writer.writerow({column: record.get(column, "") for column in LOSS_HISTORY_COLUMNS})

    lines = [
        "# GOATTM Loss History",
        "",
        "| iter | train obj | train data | train rel | test data | test rel | grad norm | step norm | best iter |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in metrics_records:
        lines.append(
            "| "
            f"{record.get('iteration', 'NA')} | "
            f"{_format_optional_float(record.get('train_objective'))} | "
            f"{_format_optional_float(record.get('train_data_loss'))} | "
            f"{_format_optional_float(record.get('train_relative_error'))} | "
            f"{_format_optional_float(record.get('test_data_loss'))} | "
            f"{_format_optional_float(record.get('test_relative_error'))} | "
            f"{_format_optional_float(record.get('gradient_norm'))} | "
            f"{_format_optional_float(record.get('step_norm'))} | "
            f"{record.get('best_iteration', 'NA')} |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def write_optimization_report(
    report_path: Path,
    summary: dict[str, object],
    metrics_records: list[dict[str, object]],
    timing_json_path: Path,
    training_summary_path: Path,
) -> None:
    """Write a human-readable optimization report next to the raw logs."""
    initial_record = metrics_records[0] if metrics_records else {}
    final_record = metrics_records[-1] if metrics_records else {}
    best_test_record = (
        min(
            (record for record in metrics_records if record.get("test_data_loss") is not None),
            key=lambda record: float(record["test_data_loss"]),
        )
        if metrics_records
        else {}
    )
    opinf_summary = _read_json_if_present(summary.get("opinf_summary_path"))
    failure_dir = Path(str(summary["run_output_dir"])) / "failures"
    failure_count = len(tuple(failure_dir.glob("*.json"))) if failure_dir.exists() else 0
    timing_records = _top_timing_records(timing_json_path)

    lines = [
        "# GOATTM Optimization Report",
        "",
        "## Run",
        f"- Run output: `{summary['run_output_dir']}`",
        f"- MPI world size: `{summary['mpi_world_size']}`",
        f"- Train/test samples: `{summary['requested_ntrain']}` / `{summary['requested_ntest']}`",
        f"- Latent rank: `{summary['latent_rank']}`",
        f"- Time integrator: `{summary['time_integrator']}`",
        f"- Max dt: `{summary['max_dt']}`",
        f"- Optimizer: `{summary['optimizer']}`",
        f"- Max iterations: `{summary['max_iterations']}`",
        "",
        "## OpInf Initialization",
        f"- Regression relative residual: `{_format_optional_float(summary.get('opinf_regression_relative_residual'))}`",
        f"- Validation success: `{opinf_summary.get('validation_success', 'NA')}`",
        f"- Validation attempts: `{opinf_summary.get('validation_attempt_count', 'NA')}`",
        f"- Final regularization: `{json.dumps(opinf_summary.get('final_regularization', {}), sort_keys=True)}`",
        f"- OpInf summary: `{summary.get('opinf_summary_path', 'NA')}`",
        "",
        "## Optimization Metrics",
        "| metric | initial | final | best-test snapshot |",
        "| --- | ---: | ---: | ---: |",
        (
            "| train objective | "
            f"{_format_optional_float(initial_record.get('train_objective'))} | "
            f"{_format_optional_float(final_record.get('train_objective'))} | "
            f"{_format_optional_float(best_test_record.get('train_objective'))} |"
        ),
        (
            "| train data loss | "
            f"{_format_optional_float(initial_record.get('train_data_loss'))} | "
            f"{_format_optional_float(final_record.get('train_data_loss'))} | "
            f"{_format_optional_float(best_test_record.get('train_data_loss'))} |"
        ),
        (
            "| train relative error | "
            f"{_format_optional_float(initial_record.get('train_relative_error'))} | "
            f"{_format_optional_float(final_record.get('train_relative_error'))} | "
            f"{_format_optional_float(best_test_record.get('train_relative_error'))} |"
        ),
        (
            "| test data loss | "
            f"{_format_optional_float(initial_record.get('test_data_loss'))} | "
            f"{_format_optional_float(final_record.get('test_data_loss'))} | "
            f"{_format_optional_float(best_test_record.get('test_data_loss'))} |"
        ),
        (
            "| test relative error | "
            f"{_format_optional_float(initial_record.get('test_relative_error'))} | "
            f"{_format_optional_float(final_record.get('test_relative_error'))} | "
            f"{_format_optional_float(best_test_record.get('test_relative_error'))} |"
        ),
        (
            "| gradient norm | "
            f"{_format_optional_float(initial_record.get('gradient_norm'))} | "
            f"{_format_optional_float(final_record.get('gradient_norm'))} | "
            f"{_format_optional_float(best_test_record.get('gradient_norm'))} |"
        ),
        "",
        "## Stability And Failures",
        f"- L-BFGS penalty evaluation failures: `{failure_count}`",
        f"- Failure records: `{failure_dir}`",
        "",
        "## Runtime Hotspots",
        "| name | calls | total seconds | average seconds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for record in timing_records:
        lines.append(
            "| "
            f"`{record.get('name', 'unknown')}` | "
            f"{int(record.get('call_count', 0))} | "
            f"{_format_optional_float(record.get('total_seconds'))} | "
            f"{_format_optional_float(record.get('average_seconds'))} |"
        )
    lines.extend(
        [
            "",
            "## Raw Files",
            f"- Loss history CSV: `{summary.get('loss_history_csv_path', 'NA')}`",
            f"- Loss history Markdown: `{summary.get('loss_history_markdown_path', 'NA')}`",
            f"- Metrics JSONL: `{summary['metrics_path']}`",
            f"- Training summary: `{training_summary_path}`",
            f"- Timing summary: `{summary.get('timing_summary_path', timing_json_path.with_suffix('.txt'))}`",
            f"- Stdout log: `{summary['stdout_log_path']}`",
            f"- Stderr log: `{summary['stderr_log_path']}`",
            f"- Best checkpoint: `{summary['best_checkpoint_path']}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_demo(config: DemoConfig) -> dict[str, object] | None:
    context = distributed_context_from_environment()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if context.rank == 0:
        dataset_root = config.output_dir / "workflow_dataset" / datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_root.mkdir(parents=True, exist_ok=True)
        raw_artifacts = build_raw_dataset(dataset_root, config=config, rng=np.random.default_rng(config.seed))
    else:
        dataset_root = None
        raw_artifacts = None

    manifest_path = Path(context.bcast_object(None if raw_artifacts is None else str(raw_artifacts.manifest_path), root=0))
    qoi_stats = dict(context.bcast_object(None if raw_artifacts is None else raw_artifacts.qoi_stats, root=0))
    manifest = load_npz_sample_manifest(manifest_path)
    split = split_train_test(manifest, config)

    opinf_result = initialize_reduced_model_via_opinf(
        train_manifest=split.train_manifest,
        test_manifest=split.test_manifest,
        output_dir=manifest_path.parent / "opinf_init",
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
        output_dir=config.output_dir / "runs",
        time_integrator=config.time_integrator,
        run_name_prefix=f"goattm_demo_{config.optimizer}_r{config.latent_rank}_ntrain{config.ntrain}_ntest{config.ntest}",
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
        bfgs=BfgsUpdaterConfig(
            gtol=config.bfgs_gtol,
            c1=config.bfgs_c1,
            c2=config.bfgs_c2,
            xrtol=config.bfgs_xrtol,
        ),
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
    initial_record = metrics_records[0]
    final_record = metrics_records[-1]
    summary: dict[str, object] = {
        "mpi_world_size": context.size,
        "requested_ntrain": config.ntrain,
        "requested_ntest": config.ntest,
        "seed": config.seed,
        "time_horizon": [0.0, 1.0],
        "observation_dt": config.observation_dt,
        "max_dt": config.max_dt,
        "time_integrator": trainer_config.time_integrator,
        "normalization_target_max_abs": config.normalization_target_max_abs,
        "input_sampling": "uniform[-2,2] at t=0:0.1:1, cubic-spline interpolation",
        "output_definition": "q_j(t)=exp(a_j p(t)) + b_j (p(t)-a_j)^2",
        "latent_rank": config.latent_rank,
        "optimizer": config.optimizer,
        "max_iterations": trainer_config.max_iterations,
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
        "opinf_regularization": {
            "coeff_w": config.opinf_reg_w,
            "coeff_h": config.opinf_reg_h,
            "coeff_b": config.opinf_reg_b,
            "coeff_c": config.opinf_reg_c,
        },
        "decoder_regularization": {
            "coeff_v1": config.decoder_reg_v1,
            "coeff_v2": config.decoder_reg_v2,
            "coeff_v0": config.decoder_reg_v0,
        },
        "dynamics_regularization": {
            "coeff_s": config.dynamics_reg_s,
            "coeff_w": config.dynamics_reg_w,
            "coeff_mu_h": config.dynamics_reg_mu_h,
            "coeff_b": config.dynamics_reg_b,
            "coeff_c": config.dynamics_reg_c,
        },
        "train_sample_count": len(opinf_result.latent_train_manifest.sample_ids),
        "test_sample_count": 0 if opinf_result.latent_test_manifest is None else len(opinf_result.latent_test_manifest.sample_ids),
        "initialization": "OpInf initialization from normalized raw data",
        "qoi_stats": qoi_stats,
        "opinf_regression_relative_residual": float(opinf_result.regression_relative_residual),
        "opinf_summary_path": str(opinf_result.summary_path),
        "initial_train_objective": float(initial_record["train_objective"]),
        "initial_train_data_loss": float(initial_record["train_data_loss"]),
        "initial_train_relative_error": float(initial_record["train_relative_error"]),
        "initial_test_data_loss": float(initial_record["test_data_loss"]),
        "initial_test_relative_error": float(initial_record["test_relative_error"]),
        "final_train_objective": float(final_record["train_objective"]),
        "final_train_data_loss": float(final_record["train_data_loss"]),
        "final_train_relative_error": float(final_record["train_relative_error"]),
        "final_test_data_loss": float(final_record["test_data_loss"]),
        "final_test_relative_error": float(final_record["test_relative_error"]),
        "best_train_objective": float(result.best_snapshot.objective_value),
        "best_train_relative_error": float(result.best_snapshot.train_relative_error),
        "best_test_data_loss": None if result.best_snapshot.test_data_loss is None else float(result.best_snapshot.test_data_loss),
        "best_test_relative_error": None
        if result.best_snapshot.test_relative_error is None
        else float(result.best_snapshot.test_relative_error),
        "run_output_dir": str(result.output_dir),
        "best_checkpoint_path": str(result.best_checkpoint_path),
        "metrics_path": str(result.metrics_path),
        "loss_history_csv_path": str(loss_history_csv_path),
        "loss_history_markdown_path": str(loss_history_markdown_path),
        "training_summary_path": str(result.summary_path),
        "timing_summary_path": str(result.timing_summary_path),
        "timing_json_path": str(result.timing_json_path),
        "stdout_log_path": str(result.stdout_log_path),
        "stderr_log_path": str(result.stderr_log_path),
    }
    optimization_report_path = result.output_dir / "optimization_report.md"
    summary["optimization_report_path"] = str(optimization_report_path)
    write_optimization_report(
        optimization_report_path,
        summary=summary,
        metrics_records=metrics_records,
        timing_json_path=result.timing_json_path,
        training_summary_path=result.summary_path,
    )
    summary_path = config.output_dir / "latest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()
