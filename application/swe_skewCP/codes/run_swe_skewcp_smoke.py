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
APP_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.core import skew_cp_to_compressed_h  # noqa: E402
from goattm.data import NpzSampleManifest, load_npz_sample_manifest, make_npz_train_test_split  # noqa: E402
from goattm.data import NpzQoiSample, load_npz_qoi_sample, save_npz_qoi_sample, save_npz_sample_manifest  # noqa: E402
from goattm.models import QuadraticDecoder, SkewCPQuadraticDynamics  # noqa: E402
from goattm.preprocess import (  # noqa: E402
    OpInfInitializationRegularization,
    OpInfLatentEmbeddingConfig,
    initialize_reduced_model_via_opinf,
)
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization  # noqa: E402
from goattm.runtime import DistributedContext  # noqa: E402
from goattm.train import (  # noqa: E402
    GradientDescentUpdaterConfig,
    LbfgsUpdaterConfig,
    ReducedQoiTrainer,
    ReducedQoiTrainerConfig,
)


DEFAULT_MANIFEST_PATH = Path(
    "/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data/manifest.npz"
)
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs" / "login_smoke"


@dataclass(frozen=True)
class SweSkewCPSmokeConfig:
    manifest_path: Path
    output_dir: Path
    sample_count: int
    ntrain: int
    ntest: int
    latent_rank: int
    skew_cp_rank: int
    max_dt: float
    time_integrator: str
    optimizer: str
    max_iterations: int
    learning_rate: float
    gradient_clip_norm: float | None
    skew_cp_init_scale: float
    skew_cp_target_h_ratio: float
    skew_cp_zero_init: bool
    skew_cp_seed: int
    latent_embedding_mode: str
    latent_embedding_augmentation_seed: int
    latent_embedding_augmentation_scale: float
    normalization_target_max_abs: float
    qoi_stride: int
    initialization_mode: str
    dynamic_form_for_opinf: str
    decoder_form: str
    opinf_decoder_form: str
    opinf_reg_w: float
    opinf_reg_h: float
    opinf_reg_b: float
    opinf_reg_c: float
    decoder_reg_v1: float
    decoder_reg_v2: float
    decoder_reg_v0: float
    dynamics_reg_a: float
    dynamics_reg_skew_cp: float
    dynamics_reg_b: float
    dynamics_reg_c: float
    lbfgs_maxcor: int
    lbfgs_ftol: float
    lbfgs_gtol: float
    lbfgs_maxls: int
    checkpoint_every: int
    test_every: int
    keep_iteration_checkpoints: bool


def parse_args() -> SweSkewCPSmokeConfig:
    parser = argparse.ArgumentParser(description="Run a login-node SWE skewCP smoke test.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--ntrain", type=int, default=2)
    parser.add_argument("--ntest", type=int, default=1)
    parser.add_argument("--latent-rank", type=int, default=32)
    parser.add_argument("--skew-cp-rank", type=int, default=8)
    parser.add_argument("--max-dt", type=float, default=1.0 / 600.0)
    parser.add_argument("--time-integrator", choices=("implicit_midpoint", "explicit_euler", "rk4"), default="rk4")
    parser.add_argument("--optimizer", choices=("gradient_descent", "lbfgs"), default="gradient_descent")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--skew-cp-init-scale", type=float, default=1e-4)
    parser.add_argument("--skew-cp-target-h-ratio", type=float, default=1e-3)
    parser.add_argument("--skew-cp-zero-init", type=parse_bool, default=False)
    parser.add_argument("--skew-cp-seed", type=int, default=20260507)
    parser.add_argument("--latent-embedding-mode", choices=("pod_projection", "qoi_augmentation"), default="qoi_augmentation")
    parser.add_argument("--latent-embedding-augmentation-seed", type=int, default=20260507)
    parser.add_argument("--latent-embedding-augmentation-scale", type=float, default=0.1)
    parser.add_argument("--normalization-target-max-abs", type=float, default=0.9)
    parser.add_argument("--qoi-stride", type=int, default=100)
    parser.add_argument("--initialization-mode", choices=("opinf_abc",), default="opinf_abc")
    parser.add_argument("--dynamic-form-for-opinf", choices=("ABc",), default="ABc")
    parser.add_argument("--decoder-form", choices=("V1v", "V1V2v"), default="V1V2v")
    parser.add_argument("--opinf-decoder-form", choices=("V1v", "V1V2v"), default="V1v")
    parser.add_argument("--opinf-reg-w", type=float, default=1e-4)
    parser.add_argument("--opinf-reg-h", type=float, default=1e-4)
    parser.add_argument("--opinf-reg-b", type=float, default=1e-4)
    parser.add_argument("--opinf-reg-c", type=float, default=1e-6)
    parser.add_argument("--decoder-reg-v1", type=float, default=1e-7)
    parser.add_argument("--decoder-reg-v2", type=float, default=1e-7)
    parser.add_argument("--decoder-reg-v0", type=float, default=1e-7)
    parser.add_argument("--dynamics-reg-a", type=float, default=1e-6)
    parser.add_argument("--dynamics-reg-skew-cp", type=float, default=1e-4)
    parser.add_argument("--dynamics-reg-b", type=float, default=1e-4)
    parser.add_argument("--dynamics-reg-c", type=float, default=1e-4)
    parser.add_argument("--lbfgs-maxcor", type=int, default=20)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-12)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    parser.add_argument("--lbfgs-maxls", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=1)
    parser.add_argument("--keep-iteration-checkpoints", type=parse_bool, default=True)
    config = SweSkewCPSmokeConfig(**vars(parser.parse_args()))
    validate_config(config)
    return config


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def validate_config(config: SweSkewCPSmokeConfig) -> None:
    if not config.manifest_path.exists():
        raise FileNotFoundError(f"manifest does not exist: {config.manifest_path}")
    if config.ntrain <= 0 or config.ntest < 0:
        raise ValueError("ntrain must be positive and ntest must be nonnegative")
    if config.ntrain + config.ntest != config.sample_count:
        raise ValueError("sample_count must equal ntrain + ntest")
    if config.latent_rank <= 0 or config.skew_cp_rank <= 0:
        raise ValueError("latent_rank and skew_cp_rank must be positive")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.max_dt <= 0.0 or config.skew_cp_init_scale <= 0.0:
        raise ValueError("max_dt and skew_cp_init_scale must be positive")
    if config.skew_cp_target_h_ratio <= 0.0:
        raise ValueError("skew_cp_target_h_ratio must be positive")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when set")
    if config.qoi_stride <= 0:
        raise ValueError("qoi_stride must be positive")
    if config.checkpoint_every <= 0 or config.test_every <= 0:
        raise ValueError("checkpoint_every and test_every must be positive")


def distributed_context_from_environment() -> DistributedContext:
    mpi_markers = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "MPI_LOCALNRANKS", "SLURM_NTASKS")
    if not any(name in os.environ for name in mpi_markers):
        return DistributedContext()
    from mpi4py import MPI  # type: ignore

    return DistributedContext.from_comm(MPI.COMM_WORLD)


def synchronized_run_stamp(context: DistributedContext) -> str:
    """Return one rank-0 timestamp shared by every MPI rank."""
    local_stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if context.rank == 0 else ""
    return str(context.allgather_object(local_stamp)[0])


def first_n_manifest(manifest: NpzSampleManifest, sample_count: int) -> NpzSampleManifest:
    if len(manifest) < sample_count:
        raise ValueError(f"Manifest has {len(manifest)} samples, requested {sample_count}.")
    return manifest.subset_by_indices(tuple(range(sample_count)))


def split_manifest(manifest: NpzSampleManifest, config: SweSkewCPSmokeConfig):
    train_ids = manifest.sample_ids[: config.ntrain]
    test_ids = manifest.sample_ids[config.ntrain : config.ntrain + config.ntest]
    return make_npz_train_test_split(manifest, train_sample_ids=train_ids, test_sample_ids=test_ids)


def materialize_strided_manifest(
    manifest: NpzSampleManifest,
    destination_root: Path,
    qoi_stride: int,
    context: DistributedContext,
) -> NpzSampleManifest:
    manifest_path = destination_root / "manifest.npz"
    if context.rank == 0:
        destination_root.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        for index, (sample_id, sample_path) in enumerate(zip(manifest.sample_ids, manifest.absolute_paths(), strict=True)):
            sample = load_npz_qoi_sample(sample_path)
            selected = np.arange(0, sample.observation_times.shape[0], qoi_stride, dtype=int)
            if selected[-1] != sample.observation_times.shape[0] - 1:
                selected = np.concatenate([selected, np.array([sample.observation_times.shape[0] - 1], dtype=int)])
            output_path = destination_root / f"{index:06d}_{sample_id}.npz"
            metadata = dict(sample.metadata or {})
            metadata.update(
                {
                    "swe_skewcp_smoke_qoi_stride": int(qoi_stride),
                    "swe_skewcp_smoke_source_path": str(sample_path),
                }
            )
            processed = NpzQoiSample(
                sample_id=sample.sample_id,
                observation_times=np.asarray(sample.observation_times[selected], dtype=np.float64),
                u0=np.asarray(sample.u0, dtype=np.float64),
                qoi_observations=np.asarray(sample.qoi_observations[selected], dtype=np.float64),
                input_times=None if sample.input_times is None else np.asarray(sample.input_times, dtype=np.float64),
                input_values=None if sample.input_values is None else np.asarray(sample.input_values, dtype=np.float64),
                metadata=metadata,
            )
            save_npz_qoi_sample(output_path, processed)
            output_paths.append(output_path)
        latent_manifest = NpzSampleManifest(
            root_dir=destination_root,
            sample_paths=tuple(Path(path.name) for path in output_paths),
            sample_ids=manifest.sample_ids,
        )
        save_npz_sample_manifest(manifest_path, latent_manifest)
    context.barrier()
    return load_npz_sample_manifest(manifest_path)


def maybe_stride_manifest(
    manifest: NpzSampleManifest,
    run_root: Path,
    qoi_stride: int,
    context: DistributedContext,
) -> NpzSampleManifest:
    if qoi_stride == 1:
        return manifest
    return materialize_strided_manifest(
        manifest=manifest,
        destination_root=run_root / "strided_dataset",
        qoi_stride=qoi_stride,
        context=context,
    )


def make_skew_cp_initial_dynamics(base_dynamics, config: SweSkewCPSmokeConfig) -> tuple[SkewCPQuadraticDynamics, dict[str, float]]:
    rng = np.random.default_rng(config.skew_cp_seed)
    d = base_dynamics.dimension
    a_matrix = np.asarray(base_dynamics.a, dtype=np.float64).copy()
    target_h_norm = config.skew_cp_target_h_ratio * float(np.linalg.norm(a_matrix))
    if config.skew_cp_zero_init:
        skew_u = np.zeros((d, config.skew_cp_rank), dtype=np.float64)
        skew_v = np.zeros((d, config.skew_cp_rank), dtype=np.float64)
        skew_z = np.zeros((d, config.skew_cp_rank), dtype=np.float64)
        raw_h_norm = 0.0
        normalization_factor = 0.0
    else:
        skew_u = config.skew_cp_init_scale * rng.standard_normal((d, config.skew_cp_rank))
        skew_v = config.skew_cp_init_scale * rng.standard_normal((d, config.skew_cp_rank))
        skew_z = config.skew_cp_init_scale * rng.standard_normal((d, config.skew_cp_rank))
        raw_h_norm = float(np.linalg.norm(skew_cp_to_compressed_h(skew_u, skew_v, skew_z)))
        if raw_h_norm <= 0.0 or target_h_norm <= 0.0:
            raise ValueError(
                f"Cannot normalize skewCP initialization: raw_h_norm={raw_h_norm}, target_h_norm={target_h_norm}"
            )
        # The skewCP H map is cubic under common scaling of u, v, and z.
        normalization_factor = float((target_h_norm / raw_h_norm) ** (1.0 / 3.0))
        skew_u *= normalization_factor
        skew_v *= normalization_factor
        skew_z *= normalization_factor
    h_norm = float(np.linalg.norm(skew_cp_to_compressed_h(skew_u, skew_v, skew_z)))
    init_info = {
        "a_fro_norm": float(np.linalg.norm(a_matrix)),
        "target_h_fro_norm": float(target_h_norm),
        "raw_h_fro_norm": float(raw_h_norm),
        "h_fro_norm": float(h_norm),
        "normalization_factor": float(normalization_factor),
        "skew_factor_l2_norm": float(np.sqrt(np.sum(skew_u * skew_u) + np.sum(skew_v * skew_v) + np.sum(skew_z * skew_z))),
    }
    return SkewCPQuadraticDynamics(
        a=a_matrix,
        skew_u=skew_u,
        skew_v=skew_v,
        skew_z=skew_z,
        b=None if base_dynamics.b is None else np.asarray(base_dynamics.b, dtype=np.float64).copy(),
        c=np.asarray(base_dynamics.c, dtype=np.float64).copy(),
    ), init_info


def write_loss_history(metrics_path: Path, output_dir: Path) -> tuple[Path, Path]:
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    csv_path = output_dir / "loss_history.csv"
    md_path = output_dir / "loss_history.md"
    columns = ("iteration", "train_objective", "train_relative_error", "test_relative_error", "gradient_norm", "step_norm")
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for record in records:
            fh.write(",".join(str(record.get(column, "")) for column in columns) + "\n")
    lines = [
        "# SWE skewCP smoke loss history",
        "",
        "| iter | train obj | train rel | test rel | grad norm | step norm |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record.get('iteration')} | {record.get('train_objective')} | {record.get('train_relative_error')} | "
            f"{record.get('test_relative_error')} | {record.get('gradient_norm')} | {record.get('step_norm')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run(config: SweSkewCPSmokeConfig) -> dict[str, object] | None:
    context = distributed_context_from_environment()
    run_stamp = synchronized_run_stamp(context)
    run_root = config.output_dir / f"swe_skewcp_smoke_r{config.latent_rank}_R{config.skew_cp_rank}_{run_stamp}"
    if context.rank == 0:
        run_root.mkdir(parents=True, exist_ok=True)
    context.barrier()

    manifest = first_n_manifest(load_npz_sample_manifest(config.manifest_path), config.sample_count)
    strided_manifest = maybe_stride_manifest(
        manifest,
        run_root=run_root,
        qoi_stride=config.qoi_stride,
        context=context,
    )
    split = split_manifest(strided_manifest, config)
    opinf_result = initialize_reduced_model_via_opinf(
        train_manifest=split.train_manifest,
        test_manifest=split.test_manifest,
        output_dir=run_root / "opinf_abc_init",
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
        latent_embedding=OpInfLatentEmbeddingConfig(
            mode=config.latent_embedding_mode,
            augmentation_seed=config.latent_embedding_augmentation_seed,
            augmentation_scale=config.latent_embedding_augmentation_scale,
        ),
        validation_time_integrator=config.time_integrator,
        max_regularization_retries=3,
        decoder_regularization=DecoderTikhonovRegularization(
            coeff_v1=config.decoder_reg_v1,
            coeff_v2=config.decoder_reg_v2,
            coeff_v0=config.decoder_reg_v0,
        ),
        dynamic_form=config.dynamic_form_for_opinf,
        decoder_form=config.opinf_decoder_form,
    )
    initial_dynamics, skew_cp_init_info = make_skew_cp_initial_dynamics(opinf_result.dynamics, config=config)
    if context.rank == 0:
        print("[skew_cp_init] " + " ".join(f"{key}={value:.16e}" for key, value in skew_cp_init_info.items()), flush=True)
    decoder_template = opinf_result.decoder
    if config.decoder_form != config.opinf_decoder_form:
        decoder_template = QuadraticDecoder(
            v1=opinf_result.decoder.v1,
            v2=opinf_result.decoder.v2,
            v0=opinf_result.decoder.v0,
            form=config.decoder_form,
        )

    trainer_config = ReducedQoiTrainerConfig(
        output_dir=run_root / "runs",
        time_integrator=config.time_integrator,
        run_name_prefix=f"swe_skewcp_{config.optimizer}_r{config.latent_rank}_R{config.skew_cp_rank}_ntrain{config.ntrain}_ntest{config.ntest}",
        optimizer=config.optimizer,
        max_iterations=config.max_iterations,
        checkpoint_every=config.checkpoint_every,
        log_every=1,
        test_every=config.test_every,
        keep_iteration_checkpoints=config.keep_iteration_checkpoints,
        gradient_descent=GradientDescentUpdaterConfig(
            learning_rate=config.learning_rate,
            gradient_clip_norm=config.gradient_clip_norm,
        ),
        lbfgs=LbfgsUpdaterConfig(
            maxcor=config.lbfgs_maxcor,
            ftol=config.lbfgs_ftol,
            gtol=config.lbfgs_gtol,
            maxls=config.lbfgs_maxls,
        ),
    )
    trainer = ReducedQoiTrainer(
        train_manifest=opinf_result.latent_train_manifest,
        test_manifest=opinf_result.latent_test_manifest,
        decoder_template=decoder_template,
        regularization=DecoderTikhonovRegularization(
            coeff_v1=config.decoder_reg_v1,
            coeff_v2=config.decoder_reg_v2,
            coeff_v0=config.decoder_reg_v0,
        ),
        dynamics_regularization=DynamicsTikhonovRegularization(
            coeff_a=config.dynamics_reg_a,
            coeff_mu_h=config.dynamics_reg_skew_cp,
            coeff_b=config.dynamics_reg_b,
            coeff_c=config.dynamics_reg_c,
        ),
        max_dt=config.max_dt,
        config=trainer_config,
        preprocess_record={
            **opinf_result.as_preprocess_record(),
            "swe_skewcp_initialization": "ABc OpInf for A,B,c; skewCP H initialized from factors",
            "qoi_stride_for_login_smoke": int(config.qoi_stride),
            "skew_cp_zero_init": bool(config.skew_cp_zero_init),
            "skew_cp_target_h_ratio": float(config.skew_cp_target_h_ratio),
            "skew_cp_initialization_info": skew_cp_init_info,
        },
        context=context,
        dt_shrink=0.5,
        dt_min=1e-5,
        tol=1e-12,
        max_iter_newton=40,
    )
    result = trainer.train(initial_dynamics)
    cleanup_record = opinf_result.cleanup_materialized_sample_data(context=context)
    if context.rank != 0:
        return None

    loss_csv_path, loss_md_path = write_loss_history(result.metrics_path, result.output_dir)
    records = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = records[-1]
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "run_root": str(run_root),
        "run_output_dir": str(result.output_dir),
        "opinf_summary_path": str(opinf_result.summary_path),
        "opinf_regression_relative_residual": float(opinf_result.regression_relative_residual),
        "qoi_stride": int(config.qoi_stride),
        "initial_dynamics_type": "skew_cp",
        "initialization_note": "A,B,c copied from ordinary OpInf ABc initialization; skewCP factors are random and commonly rescaled so ||H||_F matches skew_cp_target_h_ratio * ||A||_F when skew_cp_zero_init=false.",
        "skew_cp_initialization_info": skew_cp_init_info,
        "final_iteration": int(final["iteration"]),
        "final_train_objective": float(final["train_objective"]),
        "final_train_relative_error": float(final["train_relative_error"]),
        "final_test_relative_error": final.get("test_relative_error"),
        "best_train_objective": float(result.best_snapshot.objective_value),
        "best_train_relative_error": float(result.best_snapshot.train_relative_error),
        "best_test_relative_error": result.best_snapshot.test_relative_error,
        "metrics_path": str(result.metrics_path),
        "loss_history_csv_path": str(loss_csv_path),
        "loss_history_markdown_path": str(loss_md_path),
        "best_checkpoint_path": str(result.best_checkpoint_path),
        "opinf_materialized_sample_cleanup": cleanup_record,
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    latest_path = config.output_dir / "latest_swe_skewcp_smoke_summary.json"
    latest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
