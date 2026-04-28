from __future__ import annotations

import argparse
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

from goattm.data import NpzQoiSample, NpzSampleManifest, make_npz_train_test_split, save_npz_qoi_sample, save_npz_sample_manifest
from goattm.preprocess import OpInfInitializationRegularization, initialize_reduced_model_via_opinf
from goattm.problems import DecoderTikhonovRegularization, DynamicsTikhonovRegularization
from goattm.runtime import DistributedContext
from goattm.train import LbfgsUpdaterConfig, ReducedQoiTrainer, ReducedQoiTrainerConfig


OUTPUT_DIR = ROOT / "module_test" / "output_plots" / "reduced_qoi_optimization_demo"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GOATTM reduced-QoI OpInf optimization demo with configurable train/test sizes."
    )
    parser.add_argument("--ntrain", type=int, default=10, help="Number of training samples to generate and use.")
    parser.add_argument("--ntest", type=int, default=10, help="Number of test samples to generate and use.")
    parser.add_argument("--latent-rank", type=int, default=4, help="Reduced latent rank.")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum L-BFGS iterations.")
    parser.add_argument("--seed", type=int, default=20260428, help="Random seed for the synthetic problem.")
    parser.add_argument("--max-dt", type=float, default=0.01, help="Maximum solver time step.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for demo outputs.")
    args = parser.parse_args()
    if args.ntrain <= 0:
        raise ValueError(f"--ntrain must be positive, got {args.ntrain}")
    if args.ntest < 0:
        raise ValueError(f"--ntest must be nonnegative, got {args.ntest}")
    if args.latent_rank <= 0:
        raise ValueError(f"--latent-rank must be positive, got {args.latent_rank}")
    if args.max_iterations < 0:
        raise ValueError(f"--max-iterations must be nonnegative, got {args.max_iterations}")
    if args.max_dt <= 0.0:
        raise ValueError(f"--max-dt must be positive, got {args.max_dt}")
    return args


def build_raw_dataset(
    root: Path,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[Path, dict[str, float]]:
    observation_times = np.arange(0.0, 1.0 + 1e-12, 0.1, dtype=float)
    input_dimension = 1
    output_dimension = 20
    ab_pairs = rng.uniform(-2.0, 2.0, size=(output_dimension, 2))

    sample_paths: list[str] = []
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
            metadata={"dataset_kind": "raw_qoi_exp_of_input"},
        )
        sample_path = root / f"sample_{sample_idx:03d}.npz"
        save_npz_qoi_sample(sample_path, sample)
        sample_paths.append(sample_path.name)
        sample_ids.append(sample.sample_id)
        all_qoi_values.append(qoi_observations)

    manifest = NpzSampleManifest(
        root_dir=root,
        sample_paths=tuple(Path(path) for path in sample_paths),
        sample_ids=tuple(sample_ids),
    )
    manifest_path = root / "manifest.npz"
    save_npz_sample_manifest(manifest_path, manifest)

    qoi_stack = np.concatenate(all_qoi_values, axis=0)
    qoi_stats = {
        "output_dimension": output_dimension,
        "ab_pairs": ab_pairs.tolist(),
        "qoi_abs_mean": float(np.mean(np.abs(qoi_stack))),
        "qoi_abs_max": float(np.max(np.abs(qoi_stack))),
        "qoi_l2_mean_per_observation": float(np.mean(np.linalg.norm(qoi_stack, axis=1))),
        "qoi_l2_global": float(np.linalg.norm(qoi_stack)),
    }
    return manifest_path, qoi_stats


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpi_markers = [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS",
    ]
    if any(name in os.environ for name in mpi_markers):
        from mpi4py import MPI  # type: ignore

        comm = MPI.COMM_WORLD
        context = DistributedContext.from_comm(comm)
    else:
        context = DistributedContext()

    rng = np.random.default_rng(args.seed)
    dataset_root = None
    if context.rank == 0:
        dataset_root = output_dir / "workflow_dataset" / datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_root.mkdir(parents=True, exist_ok=True)
        manifest_path, qoi_stats = build_raw_dataset(
            dataset_root,
            sample_count=args.ntrain + args.ntest,
            rng=rng,
        )
    else:
        manifest_path = None
        qoi_stats = None

    manifest_path = Path(context.bcast_object(str(manifest_path) if manifest_path is not None else None, root=0))
    qoi_stats = dict(context.bcast_object(qoi_stats, root=0))
    manifest = NpzSampleManifest.from_path(manifest_path) if hasattr(NpzSampleManifest, "from_path") else None
    if manifest is None:
        from goattm.data import load_npz_sample_manifest

        manifest = load_npz_sample_manifest(manifest_path)

    train_ids = [f"sample-{idx:03d}" for idx in range(args.ntrain)]
    test_ids = [f"sample-{idx:03d}" for idx in range(args.ntrain, args.ntrain + args.ntest)]
    split = make_npz_train_test_split(
        manifest,
        train_sample_ids=train_ids,
        test_sample_ids=test_ids,
    )

    latent_rank = args.latent_rank
    opinf_result = initialize_reduced_model_via_opinf(
        train_manifest=split.train_manifest,
        test_manifest=split.test_manifest,
        output_dir=dataset_root / "opinf_init" if context.rank == 0 else manifest_path.parent / "opinf_init",
        rank=latent_rank,
        context=context,
        apply_normalization=True,
        time_rescale_to_unit_interval=True,
        max_dt=args.max_dt,
        regularization=OpInfInitializationRegularization(
            coeff_w=1.0e-4,
            coeff_h=1.0e-4,
            coeff_b=1.0e-4,
            coeff_c=1.0e-6,
        ),
        validation_time_integrator="implicit_midpoint",
        max_regularization_retries=8,
        decoder_regularization=DecoderTikhonovRegularization(coeff_v1=1e-7, coeff_v2=1e-7, coeff_v0=1e-7),
    )

    trainer_config = ReducedQoiTrainerConfig(
        output_dir=output_dir / "runs",
        time_integrator="implicit_midpoint",
        run_name_prefix=f"small_opt_demo_opinf_implicit_ntrain{args.ntrain}_ntest{args.ntest}",
        optimizer="lbfgs",
        max_iterations=args.max_iterations,
        checkpoint_every=10,
        log_every=1,
        test_every=1,
        lbfgs=LbfgsUpdaterConfig(
            maxcor=20,
            ftol=1e-12,
            gtol=1e-8,
            maxls=30,
        ),
    )
    trainer = ReducedQoiTrainer(
        train_manifest=opinf_result.latent_train_manifest,
        test_manifest=opinf_result.latent_test_manifest,
        decoder_template=opinf_result.decoder,
        regularization=DecoderTikhonovRegularization(coeff_v1=1e-7, coeff_v2=1e-7, coeff_v0=1e-7),
        dynamics_regularization=DynamicsTikhonovRegularization(
            coeff_s=1.0e-4,
            coeff_w=1.0e-4,
            coeff_mu_h=1.0e-4,
            coeff_b=1.0e-4,
            coeff_c=1.0e-4,
        ),
        max_dt=args.max_dt,
        config=trainer_config,
        preprocess_record=opinf_result.as_preprocess_record(),
        context=context,
        dt_shrink=0.5,
        dt_min=1e-5,
        tol=1e-12,
        max_iter_newton=40,
    )
    result = trainer.train(opinf_result.dynamics)

    if context.rank == 0:
        metrics_records = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]
        initial_record = metrics_records[0]
        final_record = metrics_records[-1]
        summary = {
            "mpi_world_size": context.size,
            "requested_ntrain": args.ntrain,
            "requested_ntest": args.ntest,
            "seed": args.seed,
            "time_horizon": [0.0, 1.0],
            "observation_dt": 0.1,
            "max_dt": args.max_dt,
            "time_integrator": trainer_config.time_integrator,
            "input_sampling": "uniform[-2,2] at t=0:0.1:1, cubic-spline interpolation",
            "output_definition": "q_j(t)=exp(a_j p(t)) + b_j (p(t)-a_j)^2",
            "latent_rank": latent_rank,
            "optimizer": "lbfgs",
            "max_iterations": trainer_config.max_iterations,
            "train_sample_count": len(opinf_result.latent_train_manifest.sample_ids),
            "test_sample_count": 0 if opinf_result.latent_test_manifest is None else len(opinf_result.latent_test_manifest.sample_ids),
            "decoder_regularization": {
                "coeff_v1": 1e-7,
                "coeff_v2": 1e-7,
                "coeff_v0": 1e-7,
            },
            "dynamics_regularization": {
                "coeff_s": 1e-4,
                "coeff_w": 1e-4,
                "coeff_mu_h": 1e-4,
                "coeff_b": 1e-4,
                "coeff_c": 1e-4,
            },
            "initialization": "OpInf initialization from normalized raw data",
            "qoi_stats": qoi_stats,
            "opinf_regression_relative_residual": float(opinf_result.regression_relative_residual),
            "normalization_stats_path": str(opinf_result.normalized_artifacts.stats_path),
            "latent_train_manifest_path": str(opinf_result.latent_train_manifest_path),
            "latent_test_manifest_path": None
            if opinf_result.latent_test_manifest_path is None
            else str(opinf_result.latent_test_manifest_path),
            "preprocess_record_path": str(result.preprocess_path),
            "opinf_summary_path": str(opinf_result.summary_path),
            "initial_train_objective": float(initial_record["train_objective"]),
            "initial_train_data_loss": float(initial_record["train_data_loss"]),
            "initial_train_relative_error": float(initial_record["train_relative_error"]),
            "initial_test_data_loss": None if initial_record["test_data_loss"] is None else float(initial_record["test_data_loss"]),
            "initial_test_relative_error": None
            if initial_record["test_relative_error"] is None
            else float(initial_record["test_relative_error"]),
            "initial_decoder_regularization_loss": float(initial_record["train_decoder_regularization_loss"]),
            "final_train_objective": float(final_record["train_objective"]),
            "final_train_data_loss": float(final_record["train_data_loss"]),
            "final_train_relative_error": float(final_record["train_relative_error"]),
            "final_test_data_loss": None if final_record["test_data_loss"] is None else float(final_record["test_data_loss"]),
            "final_test_relative_error": None
            if final_record["test_relative_error"] is None
            else float(final_record["test_relative_error"]),
            "final_decoder_regularization_loss": float(final_record["train_decoder_regularization_loss"]),
            "best_train_objective": float(result.best_snapshot.objective_value),
            "best_train_relative_error": float(result.best_snapshot.train_relative_error),
            "best_test_data_loss": None if result.best_snapshot.test_data_loss is None else float(result.best_snapshot.test_data_loss),
            "best_test_relative_error": None
            if result.best_snapshot.test_relative_error is None
            else float(result.best_snapshot.test_relative_error),
            "run_output_dir": str(result.output_dir),
            "best_checkpoint_path": str(result.best_checkpoint_path),
            "metrics_path": str(result.metrics_path),
            "stdout_log_path": str(result.stdout_log_path),
            "stderr_log_path": str(result.stderr_log_path),
        }
        summary_path = output_dir / "small_optimization_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
