from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import s_params_to_matrix
from goattm.data import load_npz_qoi_sample, load_npz_sample_manifest
from goattm.models import QuadraticDecoder, StabilizedQuadraticDynamics
from goattm.problems import DecoderTikhonovRegularization
from goattm.runtime import DistributedContext
from goattm.solvers.implicit_midpoint import (
    rollout_implicit_midpoint,
    rollout_implicit_midpoint_to_observation_times,
    solve_implicit_midpoint_step_with_retry,
)
from goattm.train import LbfgsUpdaterConfig, ReducedQoiTrainer, ReducedQoiTrainerConfig


FAILED_RUN_DIR = ROOT / "module_test" / "output_plots" / "reduced_qoi_optimization_demo" / "runs" / "small_opt_demo_opinf_20260428_013157_8fe9b3f0"
OUTPUT_DIR = ROOT / "module_test" / "output_plots" / "forward_rollout_failure_debug"


def load_initial_dynamics_and_decoder(initial_parameters_path: Path) -> tuple[StabilizedQuadraticDynamics, QuadraticDecoder]:
    payload = np.load(initial_parameters_path, allow_pickle=True)
    dynamics = StabilizedQuadraticDynamics(
        s_params=np.asarray(payload["s_params"], dtype=float),
        w_params=np.asarray(payload["w_params"], dtype=float),
        mu_h=np.asarray(payload["mu_h"], dtype=float),
        b=np.asarray(payload["b_matrix"], dtype=float) if "b_matrix" in payload.files else None,
        c=np.asarray(payload["c_vector"], dtype=float),
    )
    decoder = QuadraticDecoder(
        v1=np.asarray(payload["decoder_template_v1"], dtype=float),
        v2=np.asarray(payload["decoder_template_v2"], dtype=float),
        v0=np.asarray(payload["decoder_template_v0"], dtype=float),
    )
    return dynamics, decoder


def load_failure_dynamics(failure_npz_path: Path) -> StabilizedQuadraticDynamics:
    payload = np.load(failure_npz_path, allow_pickle=True)
    return StabilizedQuadraticDynamics(
        s_params=np.asarray(payload["s_params"], dtype=float),
        w_params=np.asarray(payload["w_params"], dtype=float),
        mu_h=np.asarray(payload["mu_h"], dtype=float),
        b=np.asarray(payload["b_matrix"], dtype=float) if "b_matrix" in payload.files else None,
        c=np.asarray(payload["c_vector"], dtype=float),
    )


def rerun_until_failure(debug_root: Path) -> tuple[Path, Path]:
    preprocess_record = json.loads((FAILED_RUN_DIR / "preprocess.json").read_text(encoding="utf-8"))
    train_manifest = load_npz_sample_manifest(preprocess_record["latent_train_manifest_path"])
    test_manifest = load_npz_sample_manifest(preprocess_record["latent_test_manifest_path"])
    initial_dynamics, decoder_template = load_initial_dynamics_and_decoder(FAILED_RUN_DIR / "initial_parameters.npz")

    trainer = ReducedQoiTrainer(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        decoder_template=decoder_template,
        regularization=DecoderTikhonovRegularization(coeff_v1=1e-7, coeff_v2=1e-7, coeff_v0=1e-7),
        max_dt=0.04,
        config=ReducedQoiTrainerConfig(
            output_dir=debug_root / "rerun_runs",
            run_name_prefix="forward_failure_debug",
            optimizer="lbfgs",
            max_iterations=30,
            checkpoint_every=10,
            log_every=1,
            test_every=1,
            echo_progress=False,
            lbfgs=LbfgsUpdaterConfig(maxcor=20, ftol=1e-12, gtol=1e-8, maxls=30),
        ),
        preprocess_record=preprocess_record,
        context=DistributedContext(),
        dt_shrink=0.5,
        dt_min=1e-12,
        tol=1e-12,
        max_iter_newton=40,
    )
    trainer.train(initial_dynamics)
    failure_dir = trainer.logger.failure_dir
    failure_jsons = sorted(failure_dir.glob("lbfgs_eval_failure_*.json"))
    failure_npzs = sorted(failure_dir.glob("lbfgs_eval_failure_*.npz"))
    if not failure_jsons or not failure_npzs:
        raise RuntimeError(f"No L-BFGS failure artifact was produced in {failure_dir}.")
    return failure_jsons[0], failure_npzs[0]


def get_or_create_failure_artifact(debug_root: Path) -> tuple[Path, Path]:
    existing_jsons = sorted(debug_root.glob("rerun_runs/*/failures/lbfgs_eval_failure_*.json"))
    existing_npzs = sorted(debug_root.glob("rerun_runs/*/failures/lbfgs_eval_failure_*.npz"))
    if existing_jsons and existing_npzs:
        return existing_jsons[-1], existing_npzs[-1]
    return rerun_until_failure(debug_root)


def verify_dynamics_conditions(dynamics: StabilizedQuadraticDynamics) -> dict[str, float]:
    s_matrix = s_params_to_matrix(dynamics.s_params, dynamics.dimension)
    a_matrix = dynamics.a
    w_matrix = a_matrix + s_matrix @ s_matrix.T
    sym_part = 0.5 * (a_matrix + a_matrix.T)
    eigvals_a = np.linalg.eigvals(a_matrix)
    sym_eigvals = np.linalg.eigvalsh(sym_part)
    rng = np.random.default_rng(20260428)
    defects = []
    for _ in range(64):
        u = rng.standard_normal(dynamics.dimension)
        defects.append(abs(dynamics.energy_preserving_defect(u)))
    return {
        "a_symmetric_part_max_eigenvalue": float(np.max(sym_eigvals)),
        "a_max_real_eigenvalue": float(np.max(np.real(eigvals_a))),
        "w_skew_symmetry_defect_fro": float(np.linalg.norm(w_matrix + w_matrix.T)),
        "s_param_norm": float(np.linalg.norm(dynamics.s_params)),
        "w_param_norm": float(np.linalg.norm(dynamics.w_params)),
        "mu_h_norm": float(np.linalg.norm(dynamics.mu_h)),
        "b_norm": 0.0 if dynamics.b is None else float(np.linalg.norm(dynamics.b)),
        "c_norm": float(np.linalg.norm(dynamics.c)),
        "max_energy_preserving_defect_randomized": float(np.max(defects)),
    }


def plot_latent_norm(sample_path: Path, output_path: Path) -> dict[str, float]:
    sample = load_npz_qoi_sample(sample_path)
    latent = np.asarray((sample.metadata or {})["latent_trajectory"], dtype=float)
    times = np.asarray(sample.observation_times, dtype=float)
    norms = np.linalg.norm(latent, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, norms, marker="o", linewidth=1.8)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|u_r(t)\|_2$")
    ax.set_title("Latent-State Norm for Failing Sample")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "latent_norm_min": float(np.min(norms)),
        "latent_norm_max": float(np.max(norms)),
        "latent_norm_mean": float(np.mean(norms)),
    }


def try_smaller_time_steps(
    dynamics: StabilizedQuadraticDynamics,
    sample_path: Path,
) -> list[dict[str, float | bool]]:
    sample = load_npz_qoi_sample(sample_path)
    input_function = sample.build_input_function()
    prefix_rollout, _ = rollout_implicit_midpoint_to_observation_times(
        dynamics=dynamics,
        u0=sample.u0,
        observation_times=sample.observation_times[:2],
        max_dt=0.04,
        input_function=input_function,
        dt_shrink=0.5,
        dt_min=1e-14,
        tol=1e-12,
        max_iter=80,
    )
    u_prev = prefix_rollout.states[-1]
    trial_max_dts = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]
    results: list[dict[str, float | bool]] = []
    for max_dt in trial_max_dts:
        step_result = solve_implicit_midpoint_step_with_retry(
            dynamics=dynamics,
            u_prev=u_prev,
            dt_initial=max_dt,
            t_prev=0.1,
            input_function=input_function,
            dt_shrink=0.5,
            dt_min=1e-14,
            tol=1e-12,
            max_iter=80,
        )
        results.append(
            {
                "max_dt": max_dt,
                "success": bool(step_result.success),
                "dt_used": float(step_result.dt_used),
                "dt_reductions": float(step_result.dt_reductions),
                "newton_failures": float(step_result.newton_failures),
                "residual_norm": float(step_result.residual_norm),
                "u_next_norm": float(np.linalg.norm(step_result.u_next)),
            }
        )
        if step_result.success:
            break
    return results


def plot_multidt_norm_growth(
    dynamics: StabilizedQuadraticDynamics,
    sample_path: Path,
    failure_time: float,
    output_path: Path,
) -> list[dict[str, float | bool]]:
    sample = load_npz_qoi_sample(sample_path)
    input_function = sample.build_input_function()
    final_time = min(float(sample.observation_times[-1]), float(failure_time) + 5.0e-5)
    trial_max_dts = [0.04, 0.02, 0.01, 0.005, 0.0025]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    records: list[dict[str, float | bool]] = []
    for max_dt in trial_max_dts:
        rollout = rollout_implicit_midpoint(
            dynamics=dynamics,
            u0=sample.u0,
            t_final=final_time,
            dt_initial=max_dt,
            input_function=input_function,
            dt_shrink=0.5,
            dt_min=1e-14,
            tol=1e-12,
            max_iter=80,
        )
        norms = np.linalg.norm(rollout.states, axis=1)
        label = f"max_dt={max_dt:g}, {'ok' if rollout.success else 'fail'}"
        ax.plot(rollout.times, norms, linewidth=1.6, label=label)
        records.append(
            {
                "max_dt": max_dt,
                "success": bool(rollout.success),
                "final_time": float(rollout.final_time),
                "accepted_steps": float(rollout.accepted_steps),
                "dt_reductions": float(rollout.dt_reductions),
                "newton_failures": float(rollout.newton_failures),
                "max_norm": float(np.max(norms)),
            }
        )
    ax.axvline(float(failure_time), color="k", linestyle="--", linewidth=1.0, label="original failure time")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|u_r(t)\|_2$")
    ax.set_title("Norm Growth Near the Failure Time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return records


def main() -> None:
    debug_root = OUTPUT_DIR / "failed_case_20260428_013157"
    debug_root.mkdir(parents=True, exist_ok=True)

    failure_json_path, failure_npz_path = get_or_create_failure_artifact(debug_root)
    failure_record = json.loads(failure_json_path.read_text(encoding="utf-8"))
    failure_dynamics = load_failure_dynamics(failure_npz_path)

    failing_sample_id = failure_record["extra"]["failing_sample_id"]
    failing_sample_index = int(failure_record["extra"]["failing_sample_index"])
    latent_sample_path = Path(failure_record["extra"]["failing_sample_path"])

    preprocess_record = json.loads((FAILED_RUN_DIR / "preprocess.json").read_text(encoding="utf-8"))
    normalized_root = Path(preprocess_record["normalized_output_dir"])
    raw_root = normalized_root.parents[1]
    raw_sample_path = raw_root / f"sample_{failing_sample_index:03d}.npz"

    condition_summary = verify_dynamics_conditions(failure_dynamics)
    latent_norm_plot_path = debug_root / "latent_norm_plot.png"
    latent_norm_summary = plot_latent_norm(latent_sample_path, latent_norm_plot_path)
    dt_results = try_smaller_time_steps(failure_dynamics, latent_sample_path)
    multidt_plot_path = debug_root / "multidt_norm_growth.png"
    multidt_records = plot_multidt_norm_growth(
        failure_dynamics,
        latent_sample_path,
        float(failure_record["extra"]["failing_sample_final_time"]),
        multidt_plot_path,
    )

    summary = {
        "source_run_dir": str(FAILED_RUN_DIR),
        "debug_root": str(debug_root),
        "failure_json_path": str(failure_json_path),
        "failure_npz_path": str(failure_npz_path),
        "failing_sample_id": failing_sample_id,
        "failing_sample_index": failing_sample_index,
        "latent_sample_path": str(latent_sample_path),
        "raw_sample_path": str(raw_sample_path),
        "failure_message": failure_record["message"],
        "failure_extra": failure_record["extra"],
        "preprocess_record": preprocess_record,
        "condition_summary": condition_summary,
        "latent_norm_plot_path": str(latent_norm_plot_path),
        "latent_norm_summary": latent_norm_summary,
        "smaller_dt_trials": dt_results,
        "multidt_norm_growth_plot_path": str(multidt_plot_path),
        "multidt_norm_growth_records": multidt_records,
    }
    summary_path = debug_root / "debug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
