from __future__ import annotations

"""First-order skew-CP demo for two prescribed parameter regimes."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo"
SRC = ROOT / "src"
for path in (str(SRC), str(DEMO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from goattm.data import load_npz_sample_manifest  # noqa: E402
from goattm.models import SkewCPQuadraticDynamics  # noqa: E402
from goattm.preprocess import OpInfInitializationRegularization, initialize_reduced_model_via_opinf  # noqa: E402
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization  # noqa: E402
from goattm.train import (  # noqa: E402
    GradientDescentUpdaterConfig,
    LbfgsUpdaterConfig,
    ReducedQoiTrainer,
    ReducedQoiTrainerConfig,
)
from run_reduced_qoi_demo import (  # noqa: E402
    build_raw_dataset,
    distributed_context_from_environment,
    split_train_test,
    write_loss_history,
)


DEFAULT_OUTPUT_DIR = ROOT / "demo" / "outputs" / "skew_cp_gradient_descent_demo"


@dataclass(frozen=True)
class RegimeConfig:
    latent_dimension: int
    skew_cp_rank: int
    ntrain: int
    ntest: int
    observation_dt: float
    seed: int
    output_dir: Path

    @property
    def sample_count(self) -> int:
        return self.ntrain + self.ntest

    @property
    def latent_rank(self) -> int:
        return self.latent_dimension

    @property
    def output_dimension(self) -> int:
        return self.latent_dimension


@dataclass(frozen=True)
class SkewCPGradientDescentConfig:
    ntrain: int
    ntest: int
    observation_dt: float
    seed: int
    max_dt: float
    time_integrator: str
    optimizer: str
    max_iterations: int
    learning_rate: float
    gradient_clip_norm: float | None
    lbfgs_maxcor: int
    lbfgs_ftol: float
    lbfgs_gtol: float
    lbfgs_maxls: int
    skew_cp_init_scale: float
    normalization_target_max_abs: float
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
    output_dir: Path


REGIMES: tuple[tuple[int, int], ...] = (
    (15, 5),
    (40, 10),
)


def parse_args() -> SkewCPGradientDescentConfig:
    parser = argparse.ArgumentParser(
        description="Run GOATTM skew-CP gradient-descent optimization for two demo regimes."
    )
    parser.add_argument("--ntrain", type=int, default=12)
    parser.add_argument("--ntest", type=int, default=4)
    parser.add_argument("--observation-dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--max-dt", type=float, default=0.01)
    parser.add_argument(
        "--time-integrator",
        default="rk4",
        choices=("implicit_midpoint", "explicit_euler", "rk4"),
    )
    parser.add_argument("--optimizer", default="gradient_descent", choices=("gradient_descent", "lbfgs"))
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--lbfgs-maxcor", type=int, default=20)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-12)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    parser.add_argument("--lbfgs-maxls", type=int, default=30)
    parser.add_argument("--skew-cp-init-scale", type=float, default=1e-3)
    parser.add_argument("--normalization-target-max-abs", type=float, default=0.9)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    config = SkewCPGradientDescentConfig(
        ntrain=args.ntrain,
        ntest=args.ntest,
        observation_dt=args.observation_dt,
        seed=args.seed,
        max_dt=args.max_dt,
        time_integrator=args.time_integrator,
        optimizer=args.optimizer,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        lbfgs_maxcor=args.lbfgs_maxcor,
        lbfgs_ftol=args.lbfgs_ftol,
        lbfgs_gtol=args.lbfgs_gtol,
        lbfgs_maxls=args.lbfgs_maxls,
        skew_cp_init_scale=args.skew_cp_init_scale,
        normalization_target_max_abs=args.normalization_target_max_abs,
        opinf_reg_w=args.opinf_reg_w,
        opinf_reg_h=args.opinf_reg_h,
        opinf_reg_b=args.opinf_reg_b,
        opinf_reg_c=args.opinf_reg_c,
        decoder_reg_v1=args.decoder_reg_v1,
        decoder_reg_v2=args.decoder_reg_v2,
        decoder_reg_v0=args.decoder_reg_v0,
        dynamics_reg_a=args.dynamics_reg_a,
        dynamics_reg_skew_cp=args.dynamics_reg_skew_cp,
        dynamics_reg_b=args.dynamics_reg_b,
        dynamics_reg_c=args.dynamics_reg_c,
        output_dir=args.output_dir,
    )
    validate_config(config)
    return config


def validate_config(config: SkewCPGradientDescentConfig) -> None:
    if config.ntrain <= 0:
        raise ValueError(f"ntrain must be positive, got {config.ntrain}")
    if config.ntest < 0:
        raise ValueError(f"ntest must be nonnegative, got {config.ntest}")
    if config.observation_dt <= 0.0:
        raise ValueError(f"observation_dt must be positive, got {config.observation_dt}")
    if config.max_dt <= 0.0:
        raise ValueError(f"max_dt must be positive, got {config.max_dt}")
    if config.max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {config.max_iterations}")
    if config.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0.0:
        raise ValueError(f"gradient_clip_norm must be positive, got {config.gradient_clip_norm}")
    if config.lbfgs_maxcor <= 0:
        raise ValueError(f"lbfgs_maxcor must be positive, got {config.lbfgs_maxcor}")
    if config.lbfgs_maxls <= 0:
        raise ValueError(f"lbfgs_maxls must be positive, got {config.lbfgs_maxls}")
    if config.skew_cp_init_scale <= 0.0:
        raise ValueError(f"skew_cp_init_scale must be positive, got {config.skew_cp_init_scale}")
    if config.normalization_target_max_abs <= 0.0:
        raise ValueError("normalization_target_max_abs must be positive.")
    regularization_values = (
        config.opinf_reg_w,
        config.opinf_reg_h,
        config.opinf_reg_b,
        config.opinf_reg_c,
        config.decoder_reg_v1,
        config.decoder_reg_v2,
        config.decoder_reg_v0,
        config.dynamics_reg_a,
        config.dynamics_reg_skew_cp,
        config.dynamics_reg_b,
        config.dynamics_reg_c,
    )
    if any(value < 0.0 for value in regularization_values):
        raise ValueError("regularization coefficients must be nonnegative.")

    observation_count = int(np.floor(1.0 / config.observation_dt + 1e-12)) + 1
    max_latent_dimension = max(latent_dimension for latent_dimension, _ in REGIMES)
    if config.ntrain * observation_count < max_latent_dimension:
        raise ValueError(
            "ntrain * observation_count must be at least the largest latent dimension. "
            f"Got {config.ntrain} * {observation_count} < {max_latent_dimension}."
        )


def make_regime_config(
    config: SkewCPGradientDescentConfig,
    latent_dimension: int,
    skew_cp_rank: int,
    regime_root: Path,
    regime_index: int,
) -> RegimeConfig:
    return RegimeConfig(
        latent_dimension=latent_dimension,
        skew_cp_rank=skew_cp_rank,
        ntrain=config.ntrain,
        ntest=config.ntest,
        observation_dt=config.observation_dt,
        seed=config.seed + 100 * regime_index + latent_dimension,
        output_dir=regime_root,
    )


def make_skew_cp_initial_dynamics(
    base_dynamics,
    skew_cp_rank: int,
    init_scale: float,
    seed: int,
) -> SkewCPQuadraticDynamics:
    rng = np.random.default_rng(seed)
    d = base_dynamics.dimension
    return SkewCPQuadraticDynamics(
        a=base_dynamics.a.copy(),
        skew_u=init_scale * rng.standard_normal((d, skew_cp_rank)),
        skew_v=init_scale * rng.standard_normal((d, skew_cp_rank)),
        skew_z=init_scale * rng.standard_normal((d, skew_cp_rank)),
        b=None if base_dynamics.b is None else base_dynamics.b.copy(),
        c=base_dynamics.c.copy(),
    )


def run_regime(
    config: SkewCPGradientDescentConfig,
    regime: RegimeConfig,
    context,
) -> dict[str, object] | None:
    regime.output_dir.mkdir(parents=True, exist_ok=True)
    if context.rank == 0:
        dataset_root = regime.output_dir / "workflow_dataset"
        dataset_root.mkdir(parents=True, exist_ok=True)
        raw_artifacts = build_raw_dataset(dataset_root, config=regime, rng=np.random.default_rng(regime.seed))
    else:
        raw_artifacts = None

    manifest_path = Path(context.bcast_object(None if raw_artifacts is None else str(raw_artifacts.manifest_path), root=0))
    qoi_stats = dict(context.bcast_object(None if raw_artifacts is None else raw_artifacts.qoi_stats, root=0))
    manifest = load_npz_sample_manifest(manifest_path)
    split = split_train_test(manifest, regime)

    opinf_result = initialize_reduced_model_via_opinf(
        train_manifest=split.train_manifest,
        test_manifest=split.test_manifest,
        output_dir=regime.output_dir / "opinf_init",
        rank=regime.latent_dimension,
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
        dynamic_form="ABc",
        decoder_form="V1V2v",
    )

    initial_dynamics = make_skew_cp_initial_dynamics(
        base_dynamics=opinf_result.dynamics,
        skew_cp_rank=regime.skew_cp_rank,
        init_scale=config.skew_cp_init_scale,
        seed=regime.seed + 17,
    )
    trainer_config = ReducedQoiTrainerConfig(
        output_dir=regime.output_dir / "runs",
        time_integrator=config.time_integrator,
        run_name_prefix=(
            f"skewcp_{config.optimizer}_d{regime.latent_dimension}_R{regime.skew_cp_rank}"
            f"_ntrain{regime.ntrain}_ntest{regime.ntest}"
        ),
        optimizer=config.optimizer,
        max_iterations=config.max_iterations,
        checkpoint_every=max(1, min(10, config.max_iterations)),
        log_every=1,
        test_every=1,
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
        decoder_template=opinf_result.decoder,
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
        preprocess_record=opinf_result.as_preprocess_record(),
        context=context,
        dt_shrink=0.5,
        dt_min=1e-5,
        tol=1e-12,
        max_iter_newton=40,
    )
    result = trainer.train(initial_dynamics)
    if context.rank != 0:
        return None

    metrics_records = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
    loss_history_csv_path = result.output_dir / "loss_history.csv"
    loss_history_markdown_path = result.output_dir / "loss_history.md"
    write_loss_history(loss_history_csv_path, loss_history_markdown_path, metrics_records)
    initial = metrics_records[0]
    final = metrics_records[-1]
    summary = {
        "regime": {
            "latent_dimension": regime.latent_dimension,
            "skew_cp_rank": regime.skew_cp_rank,
            "ntrain": regime.ntrain,
            "ntest": regime.ntest,
            "output_dimension": regime.output_dimension,
            "observation_dt": regime.observation_dt,
        },
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "gradient_clip_norm": config.gradient_clip_norm,
        "lbfgs": {
            "maxcor": config.lbfgs_maxcor,
            "ftol": config.lbfgs_ftol,
            "gtol": config.lbfgs_gtol,
            "maxls": config.lbfgs_maxls,
        },
        "skew_cp_init_scale": config.skew_cp_init_scale,
        "max_iterations": config.max_iterations,
        "max_dt": config.max_dt,
        "time_integrator": config.time_integrator,
        "normalization_target_max_abs": config.normalization_target_max_abs,
        "qoi_stats": qoi_stats,
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
        "regime_output_dir": str(regime.output_dir),
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
    summary_path = regime.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def run_demo(config: SkewCPGradientDescentConfig) -> list[dict[str, object]] | None:
    context = distributed_context_from_environment()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = config.output_dir / f"skewcp_gd_{run_stamp}"
    if context.rank == 0:
        root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for index, (latent_dimension, skew_cp_rank) in enumerate(REGIMES):
        regime_root = root / f"d{latent_dimension}_R{skew_cp_rank}"
        regime = make_regime_config(config, latent_dimension, skew_cp_rank, regime_root, index)
        summary = run_regime(config, regime, context)
        if summary is not None:
            summaries.append(summary)

    if context.rank != 0:
        return None
    aggregate = {
        "run_root": str(root),
        "regimes": summaries,
    }
    latest_summary = config.output_dir / "latest_skew_cp_gradient_descent_summary.json"
    latest_summary.parent.mkdir(parents=True, exist_ok=True)
    latest_summary.write_text(json.dumps(aggregate, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=True))
    return summaries


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()
