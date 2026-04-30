from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.optimize import minimize

from goattm.data import NpzSampleManifest
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems import (
    DecoderTikhonovRegularization,
    DatasetQoiLossGradientResult,
    DynamicsTikhonovRegularization,
    ForwardRolloutFailure,
    ObservationAlignedBestResponseEvaluator,
    ReducedObjectivePreparedState,
    ReducedQoiBestResponseResult,
    decoder_parameter_matrix,
    dynamics_from_parameter_vector,
    dynamics_parameter_key,
    dynamics_parameter_vector,
)
from goattm.runtime import DistributedContext, FunctionTimer, timed, use_function_timer
from goattm.solvers import TimeIntegrator, validate_time_integrator


DynamicsLike = QuadraticDynamics | StabilizedQuadraticDynamics


@dataclass(frozen=True)
class AdamUpdaterConfig:
    learning_rate: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip_norm: float | None = None


@dataclass(frozen=True)
class GradientDescentUpdaterConfig:
    learning_rate: float = 1e-2
    gradient_clip_norm: float | None = None


@dataclass(frozen=True)
class LbfgsUpdaterConfig:
    maxcor: int = 10
    ftol: float = 1e-12
    gtol: float = 1e-8
    maxls: int = 20


@dataclass(frozen=True)
class BfgsUpdaterConfig:
    gtol: float = 1e-6
    c1: float = 1e-4
    c2: float = 0.9
    xrtol: float = 1e-7


@dataclass(frozen=True)
class AdamBfgsUpdaterConfig:
    adam_iterations: int = 100


@dataclass(frozen=True)
class NewtonActionUpdaterConfig:
    hessian_mode: str = "action"
    damping: float = 1e-4
    line_search_shrink: float = 0.5
    max_backtracks: int = 8
    min_step_scale: float = 1e-6
    cg_tolerance: float = 1e-8
    cg_max_iterations: int | None = None


@dataclass(frozen=True)
class ReducedQoiTrainerConfig:
    output_dir: str | Path
    time_integrator: TimeIntegrator = "rk4"
    run_name_prefix: str = "train_run"
    enable_function_timing: bool = True
    echo_progress: bool = True
    max_iterations: int = 100
    optimizer: str = "adam"
    checkpoint_every: int = 10
    log_every: int = 1
    test_every: int = 1
    keep_iteration_checkpoints: bool = True
    solve_root: int = 0
    adam: AdamUpdaterConfig = field(default_factory=AdamUpdaterConfig)
    gradient_descent: GradientDescentUpdaterConfig = field(default_factory=GradientDescentUpdaterConfig)
    lbfgs: LbfgsUpdaterConfig = field(default_factory=LbfgsUpdaterConfig)
    bfgs: BfgsUpdaterConfig = field(default_factory=BfgsUpdaterConfig)
    adam_bfgs: AdamBfgsUpdaterConfig = field(default_factory=AdamBfgsUpdaterConfig)
    newton_action: NewtonActionUpdaterConfig = field(default_factory=NewtonActionUpdaterConfig)


@dataclass(frozen=True)
class ReducedQoiTrainingSnapshot:
    iteration: int
    dynamics: DynamicsLike
    decoder: QuadraticDecoder
    train_result: ReducedQoiBestResponseResult
    test_data_loss: float | None
    train_relative_error: float
    test_relative_error: float | None
    step_norm: float
    gradient_norm: float
    dynamic_parameter_norm: float
    decoder_parameter_norm: float

    @property
    def objective_value(self) -> float:
        return self.train_result.objective_value


@dataclass(frozen=True)
class ReducedQoiTrainingResult:
    final_snapshot: ReducedQoiTrainingSnapshot
    best_snapshot: ReducedQoiTrainingSnapshot
    output_dir: Path
    metrics_path: Path
    summary_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    timing_json_path: Path
    timing_summary_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    preprocess_path: Path


def _dynamics_component_norms(dynamics: DynamicsLike) -> dict[str, float]:
    b_norm = 0.0 if dynamics.b is None else float(np.linalg.norm(dynamics.b))
    norms = {
        "a_fro_norm": float(np.linalg.norm(dynamics.a)),
        "h_fro_norm": float(np.linalg.norm(dynamics.h_matrix)),
        "b_fro_norm": b_norm,
        "c_l2_norm": float(np.linalg.norm(dynamics.c)),
    }
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        norms.update(
            {
                "s_param_l2_norm": float(np.linalg.norm(dynamics.s_params)),
                "w_param_l2_norm": float(np.linalg.norm(dynamics.w_params)),
                "mu_h_l2_norm": float(np.linalg.norm(dynamics.mu_h)),
            }
        )
    else:
        norms.update(
            {
                "s_param_l2_norm": 0.0,
                "w_param_l2_norm": 0.0,
                "mu_h_l2_norm": float(np.linalg.norm(dynamics.mu_h)),
            }
        )
    return norms


def _relative_error_from_dataset_result(
    dataset_result: DatasetQoiLossGradientResult,
    context: DistributedContext,
) -> float:
    local_residual_sumsq = 0.0
    local_target_sumsq = 0.0
    for sample_result in dataset_result.local_sample_results:
        partials = sample_result.decoder_partials
        weights = partials.quadrature_weights
        for residual, prediction, weight in zip(
            partials.residuals,
            partials.qoi_predictions,
            weights,
            strict=True,
        ):
            target = prediction - residual
            local_residual_sumsq += float(weight) * float(np.dot(residual, residual))
            local_target_sumsq += float(weight) * float(np.dot(target, target))
    residual_sumsq = context.allreduce_scalar_sum(local_residual_sumsq)
    target_sumsq = context.allreduce_scalar_sum(local_target_sumsq)
    if target_sumsq <= 0.0:
        return 0.0 if residual_sumsq <= 0.0 else float("inf")
    return float(np.sqrt(residual_sumsq / target_sumsq))


@dataclass
class AdamUpdater:
    config: AdamUpdaterConfig
    m: np.ndarray | None = None
    v: np.ndarray | None = None
    t: int = 0

    @timed("goattm.train.AdamUpdater.step")
    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, float]:
        grad = np.asarray(gradient, dtype=np.float64)
        params = np.asarray(parameters, dtype=np.float64)
        if self.m is None or self.v is None:
            self.m = np.zeros_like(params, dtype=np.float64)
            self.v = np.zeros_like(params, dtype=np.float64)

        if self.config.gradient_clip_norm is not None:
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > self.config.gradient_clip_norm > 0.0:
                grad = grad * (self.config.gradient_clip_norm / grad_norm)

        self.t += 1
        self.m = self.config.beta1 * self.m + (1.0 - self.config.beta1) * grad
        self.v = self.config.beta2 * self.v + (1.0 - self.config.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.config.beta1**self.t)
        v_hat = self.v / (1.0 - self.config.beta2**self.t)
        update = -self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
        return params + update, float(np.linalg.norm(update))


@dataclass
class GradientDescentUpdater:
    config: GradientDescentUpdaterConfig

    @timed("goattm.train.GradientDescentUpdater.step")
    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, float]:
        grad = np.asarray(gradient, dtype=np.float64)
        params = np.asarray(parameters, dtype=np.float64)
        if self.config.gradient_clip_norm is not None:
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > self.config.gradient_clip_norm > 0.0:
                grad = grad * (self.config.gradient_clip_norm / grad_norm)
        update = -self.config.learning_rate * grad
        return params + update, float(np.linalg.norm(update))


@dataclass
class NewtonActionUpdater:
    config: NewtonActionUpdaterConfig

    @timed("goattm.train.NewtonActionUpdater.step")
    def step(
        self,
        current_dynamics: DynamicsLike,
        prepared_state,
        workflow,
    ) -> tuple[np.ndarray, float]:
        gradient = np.asarray(prepared_state.gradient, dtype=np.float64)
        damping = float(self.config.damping)
        parameter_vector = dynamics_parameter_vector(current_dynamics)
        if self.config.hessian_mode == "explicit":
            step = self._compute_explicit_step(prepared_state, workflow, damping)
        elif self.config.hessian_mode == "action":
            step = self._compute_action_step(prepared_state, workflow, damping)
        else:
            raise ValueError(
                f"Unsupported Hessian mode '{self.config.hessian_mode}'. Supported modes are 'action' and 'explicit'."
            )

        base_objective = prepared_state.objective_value
        step_scale = 1.0
        candidate_vector = parameter_vector.copy()
        step_norm = 0.0
        for _ in range(self.config.max_backtracks + 1):
            trial_vector = parameter_vector + step_scale * step
            trial_dynamics = dynamics_from_parameter_vector(current_dynamics, trial_vector)
            trial_objective = workflow.evaluate_objective(trial_dynamics)
            if np.isfinite(trial_objective) and trial_objective <= base_objective:
                candidate_vector = trial_vector
                step_norm = float(np.linalg.norm(step_scale * step))
                break
            step_scale *= self.config.line_search_shrink
            if step_scale < self.config.min_step_scale:
                candidate_vector = parameter_vector.copy()
                step_norm = 0.0
                break
        return candidate_vector, step_norm

    @timed("goattm.train.NewtonActionUpdater.compute_explicit_step")
    def _compute_explicit_step(
        self,
        prepared_state: ReducedObjectivePreparedState,
        workflow,
        damping: float,
    ) -> np.ndarray:
        explicit = workflow.evaluate_explicit_hessian_from_prepared_state(prepared_state)
        hessian = np.asarray(explicit.hessian, dtype=np.float64)
        gradient = np.asarray(prepared_state.gradient, dtype=np.float64)
        dimension = int(gradient.shape[0])
        current_damping = damping
        while True:
            try:
                system_matrix = hessian + current_damping * np.eye(dimension, dtype=np.float64)
                return np.linalg.solve(system_matrix, -gradient)
            except np.linalg.LinAlgError:
                current_damping = max(10.0 * current_damping, 1e-8)
                if current_damping > 1e8:
                    raise

    @timed("goattm.train.NewtonActionUpdater.compute_action_step")
    def _compute_action_step(
        self,
        prepared_state: ReducedObjectivePreparedState,
        workflow,
        damping: float,
    ) -> np.ndarray:
        gradient = np.asarray(prepared_state.gradient, dtype=np.float64)
        rhs = -gradient
        dimension = int(rhs.shape[0])
        max_iterations = self.config.cg_max_iterations
        if max_iterations is None:
            max_iterations = max(2 * dimension, 25)
        tolerance = float(self.config.cg_tolerance)

        def matvec(vector: np.ndarray) -> np.ndarray:
            action = workflow.evaluate_hessian_action_from_prepared_state(
                prepared_state=prepared_state,
                direction=vector,
            ).action
            return np.asarray(action, dtype=np.float64) + damping * np.asarray(vector, dtype=np.float64)

        x = np.zeros_like(rhs, dtype=np.float64)
        residual = rhs.copy()
        direction = residual.copy()
        residual_norm_sq = float(np.dot(residual, residual))
        if residual_norm_sq <= tolerance * tolerance:
            return x

        for _ in range(max_iterations):
            matrix_direction = matvec(direction)
            curvature = float(np.dot(direction, matrix_direction))
            if not np.isfinite(curvature) or curvature <= 0.0:
                if np.linalg.norm(x) > 0.0:
                    break
                return rhs / max(damping, 1e-8)
            alpha = residual_norm_sq / curvature
            x = x + alpha * direction
            residual = residual - alpha * matrix_direction
            new_residual_norm_sq = float(np.dot(residual, residual))
            if new_residual_norm_sq <= tolerance * tolerance:
                break
            beta = new_residual_norm_sq / residual_norm_sq
            direction = residual + beta * direction
            residual_norm_sq = new_residual_norm_sq
        return x


class ReducedQoiGradientCalculator:
    def __init__(
        self,
        evaluator: ObservationAlignedBestResponseEvaluator,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization,
        dynamics_regularization: DynamicsTikhonovRegularization,
        solve_root: int = 0,
    ) -> None:
        self.evaluator = evaluator
        self.decoder_template = decoder_template
        self.regularization = regularization
        self.dynamics_regularization = dynamics_regularization
        self.solve_root = solve_root

    @timed("goattm.train.ReducedQoiGradientCalculator.evaluate")
    def evaluate(self, dynamics: DynamicsLike) -> ReducedQoiBestResponseResult:
        return self.evaluator.evaluate_reduced_objective_and_gradient(
            dynamics=dynamics,
            decoder_template=self.decoder_template,
            regularization=self.regularization,
            dynamics_regularization=self.dynamics_regularization,
            solve_root=self.solve_root,
        )


class ReducedQoiDatasetEvaluator:
    def __init__(
        self,
        evaluator: ObservationAlignedBestResponseEvaluator,
    ) -> None:
        self.evaluator = evaluator

    @timed("goattm.train.ReducedQoiDatasetEvaluator.evaluate_data_loss")
    def evaluate_data_loss(self, dynamics: DynamicsLike, decoder: QuadraticDecoder) -> float:
        return self.evaluator.evaluate_dataset_data_loss(dynamics, decoder)

    @timed("goattm.train.ReducedQoiDatasetEvaluator.evaluate_relative_error")
    def evaluate_relative_error(self, dynamics: DynamicsLike, decoder: QuadraticDecoder) -> float:
        return self.evaluator.evaluate_dataset_relative_error(dynamics, decoder)


class ReducedQoiTrainingLogger:
    def __init__(
        self,
        output_root: str | Path,
        run_name_prefix: str = "train_run",
        keep_iteration_checkpoints: bool = True,
        active: bool = True,
        fixed_output_dir: str | Path | None = None,
    ) -> None:
        self.active = bool(active)
        self.output_root = Path(output_root)
        if self.active:
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.output_dir = self._create_run_directory(run_name_prefix) if fixed_output_dir is None else Path(fixed_output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            if fixed_output_dir is None:
                raise ValueError("fixed_output_dir must be provided when logger is inactive.")
            self.output_dir = Path(fixed_output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.failure_dir = self.output_dir / "failures"
        if self.active:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.failure_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.txt"
        self.stdout_log_path = self.output_dir / "stdout.log"
        self.stderr_log_path = self.output_dir / "stderr.log"
        self.timing_json_path = self.output_dir / "timing_summary.json"
        self.timing_summary_path = self.output_dir / "timing_summary.txt"
        self.config_path = self.output_dir / "config.json"
        self.split_path = self.output_dir / "split.json"
        self.preprocess_path = self.output_dir / "preprocess.json"
        self.initial_parameters_path = self.output_dir / "initial_parameters.npz"
        self.latest_checkpoint_path = self.checkpoint_dir / "latest.npz"
        self.best_checkpoint_path = self.checkpoint_dir / "best.npz"
        self.keep_iteration_checkpoints = keep_iteration_checkpoints
        if self.active:
            self.stdout_log_path.write_text("", encoding="utf-8")
            self.stderr_log_path.write_text("", encoding="utf-8")

    def write_run_metadata(
        self,
        config: ReducedQoiTrainerConfig,
        regularization: DecoderTikhonovRegularization,
        dynamics_regularization: DynamicsTikhonovRegularization,
        train_manifest: NpzSampleManifest,
        test_manifest: NpzSampleManifest | None,
        preprocess_record: dict[str, Any] | None,
        dt_shrink: float,
        dt_min: float,
        tol: float,
        max_iter_newton: int,
        max_dt: float,
        time_integrator: TimeIntegrator,
    ) -> None:
        if not self.active:
            return
        config_record = {
            "run_name_prefix": config.run_name_prefix,
            "time_integrator": time_integrator,
            "enable_function_timing": config.enable_function_timing,
            "echo_progress": config.echo_progress,
            "max_iterations": config.max_iterations,
            "optimizer": config.optimizer,
            "checkpoint_every": config.checkpoint_every,
            "log_every": config.log_every,
            "test_every": config.test_every,
            "keep_iteration_checkpoints": config.keep_iteration_checkpoints,
            "solve_root": config.solve_root,
            "adam": {
                "learning_rate": config.adam.learning_rate,
                "beta1": config.adam.beta1,
                "beta2": config.adam.beta2,
                "epsilon": config.adam.epsilon,
                "gradient_clip_norm": config.adam.gradient_clip_norm,
            },
            "gradient_descent": {
                "learning_rate": config.gradient_descent.learning_rate,
                "gradient_clip_norm": config.gradient_descent.gradient_clip_norm,
            },
            "lbfgs": {
                "maxcor": config.lbfgs.maxcor,
                "ftol": config.lbfgs.ftol,
                "gtol": config.lbfgs.gtol,
                "maxls": config.lbfgs.maxls,
            },
            "bfgs": {
                "gtol": config.bfgs.gtol,
                "c1": config.bfgs.c1,
                "c2": config.bfgs.c2,
                "xrtol": config.bfgs.xrtol,
            },
            "adam_bfgs": {
                "adam_iterations": config.adam_bfgs.adam_iterations,
            },
            "newton_action": {
                "hessian_mode": config.newton_action.hessian_mode,
                "damping": config.newton_action.damping,
                "line_search_shrink": config.newton_action.line_search_shrink,
                "max_backtracks": config.newton_action.max_backtracks,
                "min_step_scale": config.newton_action.min_step_scale,
                "cg_tolerance": config.newton_action.cg_tolerance,
                "cg_max_iterations": config.newton_action.cg_max_iterations,
            },
            "decoder_regularization": {
                "coeff_v1": regularization.coeff_v1,
                "coeff_v2": regularization.coeff_v2,
                "coeff_v0": regularization.coeff_v0,
            },
            "dynamics_regularization": {
                "coeff_a": dynamics_regularization.coeff_a,
                "coeff_s": dynamics_regularization.coeff_s,
                "coeff_w": dynamics_regularization.coeff_w,
                "coeff_mu_h": dynamics_regularization.coeff_mu_h,
                "coeff_b": dynamics_regularization.coeff_b,
                "coeff_c": dynamics_regularization.coeff_c,
            },
            "solver": {
                "max_dt": max_dt,
                "dt_shrink": dt_shrink,
                "dt_min": dt_min,
                "tol": tol,
                "max_iter_newton": max_iter_newton,
            },
        }
        split_record = {
            "train_sample_ids": list(train_manifest.sample_ids),
            "train_sample_paths": [str(path) for path in train_manifest.sample_paths],
            "test_sample_ids": [] if test_manifest is None else list(test_manifest.sample_ids),
            "test_sample_paths": [] if test_manifest is None else [str(path) for path in test_manifest.sample_paths],
        }
        self.config_path.write_text(json.dumps(config_record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        self.split_path.write_text(json.dumps(split_record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        preprocess_payload = {"applied": False} if preprocess_record is None else preprocess_record
        self.preprocess_path.write_text(json.dumps(preprocess_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def save_failure_artifact(
        self,
        name: str,
        dynamics: DynamicsLike,
        parameter_vector: np.ndarray,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> tuple[Path, Path] | None:
        if not self.active:
            return None
        stem = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        json_path = self.failure_dir / f"{stem}.json"
        npz_path = self.failure_dir / f"{stem}.npz"
        payload = {
            "message": message,
            "parameter_norm": float(np.linalg.norm(parameter_vector)),
            "dynamics_type": "stabilized" if isinstance(dynamics, StabilizedQuadraticDynamics) else "general",
            "dynamics_key": dynamics_parameter_key(dynamics),
        }
        if extra is not None:
            payload["extra"] = extra
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        npz_payload: dict[str, Any] = {
            "parameter_vector": np.asarray(parameter_vector, dtype=np.float64),
            "dynamics_type": np.array(payload["dynamics_type"]),
            "dynamics_key": np.array(payload["dynamics_key"]),
        }
        if isinstance(dynamics, StabilizedQuadraticDynamics):
            npz_payload["s_params"] = dynamics.s_params
            npz_payload["w_params"] = dynamics.w_params
        else:
            npz_payload["a_matrix"] = dynamics.a
        npz_payload["mu_h"] = dynamics.mu_h
        npz_payload["c_vector"] = dynamics.c
        if dynamics.b is not None:
            npz_payload["b_matrix"] = dynamics.b
        np.savez(npz_path, **npz_payload)
        return json_path, npz_path

    def write_initial_parameters(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
    ) -> None:
        if not self.active:
            return
        payload: dict[str, Any] = {
            "dynamics_type": np.array("stabilized" if isinstance(dynamics, StabilizedQuadraticDynamics) else "general"),
            "decoder_template_v1": decoder_template.v1,
            "decoder_template_v2": decoder_template.v2,
            "decoder_template_v0": decoder_template.v0,
        }
        if isinstance(dynamics, StabilizedQuadraticDynamics):
            payload["s_params"] = dynamics.s_params
            payload["w_params"] = dynamics.w_params
        else:
            payload["a_matrix"] = dynamics.a
        payload["mu_h"] = dynamics.mu_h
        payload["c_vector"] = dynamics.c
        if dynamics.b is not None:
            payload["b_matrix"] = dynamics.b
        np.savez(self.initial_parameters_path, **payload)

    def log_iteration(
        self,
        snapshot: ReducedQoiTrainingSnapshot,
        best_snapshot: ReducedQoiTrainingSnapshot,
    ) -> None:
        if not self.active:
            return
        component_norms = _dynamics_component_norms(snapshot.dynamics)
        record = {
            "iteration": snapshot.iteration,
            "train_data_loss": float(snapshot.train_result.data_loss),
            "train_relative_error": float(snapshot.train_relative_error),
            "train_decoder_regularization_loss": float(snapshot.train_result.decoder_regularization_loss),
            "train_dynamics_regularization_loss": float(snapshot.train_result.dynamics_regularization_loss),
            "train_objective": float(snapshot.objective_value),
            "test_data_loss": None if snapshot.test_data_loss is None else float(snapshot.test_data_loss),
            "test_relative_error": None if snapshot.test_relative_error is None else float(snapshot.test_relative_error),
            "gradient_norm": float(snapshot.gradient_norm),
            "step_norm": float(snapshot.step_norm),
            "dynamic_parameter_norm": float(snapshot.dynamic_parameter_norm),
            "decoder_parameter_norm": float(snapshot.decoder_parameter_norm),
            "direct_dynamics_gradient_norm": float(np.linalg.norm(snapshot.train_result.direct_dynamics_gradient)),
            "decoder_chain_gradient_norm": float(np.linalg.norm(snapshot.train_result.decoder_chain_gradient)),
            "best_iteration": int(best_snapshot.iteration),
            "best_train_objective": float(best_snapshot.objective_value),
            "best_train_relative_error": float(best_snapshot.train_relative_error),
            "best_test_data_loss": None if best_snapshot.test_data_loss is None else float(best_snapshot.test_data_loss),
            "best_test_relative_error": None if best_snapshot.test_relative_error is None else float(best_snapshot.test_relative_error),
        }
        record.update(
            {
                "dynamics_a_fro_norm": component_norms["a_fro_norm"],
                "dynamics_h_fro_norm": component_norms["h_fro_norm"],
                "dynamics_b_fro_norm": component_norms["b_fro_norm"],
                "dynamics_c_l2_norm": component_norms["c_l2_norm"],
                "dynamics_s_param_l2_norm": component_norms["s_param_l2_norm"],
                "dynamics_w_param_l2_norm": component_norms["w_param_l2_norm"],
                "dynamics_mu_h_l2_norm": component_norms["mu_h_l2_norm"],
            }
        )
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._write_summary(snapshot, best_snapshot)

    def save_checkpoint(
        self,
        snapshot: ReducedQoiTrainingSnapshot,
        best_snapshot: ReducedQoiTrainingSnapshot,
        is_best: bool,
    ) -> Path:
        if not self.active:
            return self.latest_checkpoint_path
        latest_path = self.latest_checkpoint_path
        write_training_checkpoint(latest_path, snapshot, best_snapshot)
        if self.keep_iteration_checkpoints:
            iter_path = self.checkpoint_dir / f"iter_{snapshot.iteration:06d}.npz"
            write_training_checkpoint(iter_path, snapshot, best_snapshot)
        if is_best:
            write_training_checkpoint(self.best_checkpoint_path, snapshot, best_snapshot)
        return latest_path

    def write_timing_summary(self, function_timer: FunctionTimer) -> None:
        if not self.active:
            return
        function_timer.write_json(self.timing_json_path)
        function_timer.write_text(self.timing_summary_path)

    def log_stdout(self, message: str, echo: bool = True) -> None:
        if not self.active:
            return
        line = message.rstrip("\n")
        with self.stdout_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if echo:
            print(line, file=sys.stdout, flush=True)

    def log_stderr(self, message: str, echo: bool = True) -> None:
        if not self.active:
            return
        line = message.rstrip("\n")
        with self.stderr_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if echo:
            print(line, file=sys.stderr, flush=True)

    def format_iteration_message(
        self,
        snapshot: ReducedQoiTrainingSnapshot,
        best_snapshot: ReducedQoiTrainingSnapshot,
    ) -> str:
        test_piece = "NA" if snapshot.test_data_loss is None else f"{snapshot.test_data_loss:.6e}"
        test_rel_piece = "NA" if snapshot.test_relative_error is None else f"{snapshot.test_relative_error:.6e}"
        best_test_piece = "NA" if best_snapshot.test_data_loss is None else f"{best_snapshot.test_data_loss:.6e}"
        component_norms = _dynamics_component_norms(snapshot.dynamics)
        return (
            f"[iter {snapshot.iteration:04d}] "
            f"train_obj={snapshot.objective_value:.6e} "
            f"train_data={snapshot.train_result.data_loss:.6e} "
            f"train_rel={snapshot.train_relative_error:.6e} "
            f"decoder_reg={snapshot.train_result.decoder_regularization_loss:.6e} "
            f"dynamics_reg={snapshot.train_result.dynamics_regularization_loss:.6e} "
            f"grad={snapshot.gradient_norm:.6e} "
            f"step={snapshot.step_norm:.6e} "
            f"|A|={component_norms['a_fro_norm']:.3e} "
            f"|H|={component_norms['h_fro_norm']:.3e} "
            f"|B|={component_norms['b_fro_norm']:.3e} "
            f"|c|={component_norms['c_l2_norm']:.3e} "
            f"test_data={test_piece} "
            f"test_rel={test_rel_piece} "
            f"best_iter={best_snapshot.iteration:04d} "
            f"best_obj={best_snapshot.objective_value:.6e} "
            f"best_test={best_test_piece}"
        )

    def _write_summary(
        self,
        snapshot: ReducedQoiTrainingSnapshot,
        best_snapshot: ReducedQoiTrainingSnapshot,
    ) -> None:
        if not self.active:
            return
        component_norms = _dynamics_component_norms(snapshot.dynamics)
        lines = [
            "GOATTM Reduced QoI Training Summary",
            f"latest_iteration={snapshot.iteration}",
            f"latest_train_objective={snapshot.objective_value:.16e}",
            f"latest_train_data_loss={snapshot.train_result.data_loss:.16e}",
            f"latest_train_relative_error={snapshot.train_relative_error:.16e}",
            f"latest_train_decoder_regularization_loss={snapshot.train_result.decoder_regularization_loss:.16e}",
            f"latest_train_dynamics_regularization_loss={snapshot.train_result.dynamics_regularization_loss:.16e}",
            f"latest_dynamics_a_fro_norm={component_norms['a_fro_norm']:.16e}",
            f"latest_dynamics_h_fro_norm={component_norms['h_fro_norm']:.16e}",
            f"latest_dynamics_b_fro_norm={component_norms['b_fro_norm']:.16e}",
            f"latest_dynamics_c_l2_norm={component_norms['c_l2_norm']:.16e}",
            f"latest_dynamics_s_param_l2_norm={component_norms['s_param_l2_norm']:.16e}",
            f"latest_dynamics_w_param_l2_norm={component_norms['w_param_l2_norm']:.16e}",
            f"latest_dynamics_mu_h_l2_norm={component_norms['mu_h_l2_norm']:.16e}",
            f"latest_test_data_loss={snapshot.test_data_loss if snapshot.test_data_loss is not None else 'NA'}",
            f"latest_test_relative_error={snapshot.test_relative_error if snapshot.test_relative_error is not None else 'NA'}",
            f"latest_gradient_norm={snapshot.gradient_norm:.16e}",
            f"latest_step_norm={snapshot.step_norm:.16e}",
            f"best_iteration={best_snapshot.iteration}",
            f"best_train_objective={best_snapshot.objective_value:.16e}",
            f"best_train_relative_error={best_snapshot.train_relative_error:.16e}",
            f"best_test_data_loss={best_snapshot.test_data_loss if best_snapshot.test_data_loss is not None else 'NA'}",
            f"best_test_relative_error={best_snapshot.test_relative_error if best_snapshot.test_relative_error is not None else 'NA'}",
            f"latest_checkpoint={self.latest_checkpoint_path}",
            f"best_checkpoint={self.best_checkpoint_path}",
        ]
        self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _create_run_directory(self, run_name_prefix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.output_root / f"{run_name_prefix}_{timestamp}_{uuid4().hex[:8]}"
        return candidate


class ReducedQoiTrainer:
    def __init__(
        self,
        train_manifest: str | Path | NpzSampleManifest,
        test_manifest: str | Path | NpzSampleManifest | None,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization,
        max_dt: float,
        config: ReducedQoiTrainerConfig,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        preprocess_record: dict[str, Any] | None = None,
        context: DistributedContext | None = None,
        dt_shrink: float = 0.8,
        dt_min: float = 1e-10,
        tol: float = 1e-10,
        max_iter_newton: int = 25,
    ) -> None:
        if context is None:
            context = DistributedContext.from_comm()
        self.context = context
        self.decoder_template = decoder_template
        self.regularization = regularization
        self.dynamics_regularization = (
            DynamicsTikhonovRegularization() if dynamics_regularization is None else dynamics_regularization
        )
        self.preprocess_record = preprocess_record
        self.config = config
        self.max_dt = float(max_dt)
        self.time_integrator = validate_time_integrator(config.time_integrator)
        self.dt_shrink = float(dt_shrink)
        self.dt_min = float(dt_min)
        self.tol = float(tol)
        self.max_iter_newton = int(max_iter_newton)
        self.train_evaluator = ObservationAlignedBestResponseEvaluator(
            manifest=train_manifest,
            max_dt=max_dt,
            context=context,
            time_integrator=self.time_integrator,
            dt_shrink=dt_shrink,
            dt_min=dt_min,
            tol=tol,
            max_iter=max_iter_newton,
        )
        self.test_evaluator = None
        if test_manifest is not None:
            self.test_evaluator = ObservationAlignedBestResponseEvaluator(
                manifest=test_manifest,
                max_dt=max_dt,
                context=context,
                time_integrator=self.time_integrator,
                dt_shrink=dt_shrink,
                dt_min=dt_min,
                tol=tol,
                max_iter=max_iter_newton,
            )
        self.gradient_calculator = ReducedQoiGradientCalculator(
            evaluator=self.train_evaluator,
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=self.dynamics_regularization,
            solve_root=config.solve_root,
        )
        self.reduced_workflow = self.train_evaluator.build_reduced_objective_workflow(
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=self.dynamics_regularization,
            solve_root=config.solve_root,
        )
        self.train_dataset_evaluator = ReducedQoiDatasetEvaluator(self.train_evaluator)
        self.test_dataset_evaluator = None if self.test_evaluator is None else ReducedQoiDatasetEvaluator(self.test_evaluator)
        if config.optimizer == "adam":
            self.updater = AdamUpdater(config.adam)
        elif config.optimizer == "gradient_descent":
            self.updater = GradientDescentUpdater(config.gradient_descent)
        elif config.optimizer in {"lbfgs", "bfgs", "adam_bfgs", "adam+bfgs"}:
            self.updater = None
        elif config.optimizer == "newton_action":
            self.updater = NewtonActionUpdater(config.newton_action)
        else:
            raise ValueError(
                f"Unsupported optimizer '{config.optimizer}'. Supported optimizers are 'adam', 'gradient_descent', 'lbfgs', 'bfgs', 'adam_bfgs', and 'newton_action'."
            )
        logger = None
        if self.context.rank == 0:
            logger = ReducedQoiTrainingLogger(
                config.output_dir,
                run_name_prefix=config.run_name_prefix,
                keep_iteration_checkpoints=config.keep_iteration_checkpoints,
                active=True,
            )
            shared_output_dir = str(logger.output_dir)
        else:
            shared_output_dir = None
        shared_output_dir = self.context.bcast_object(shared_output_dir, root=0)
        if logger is None:
            logger = ReducedQoiTrainingLogger(
                config.output_dir,
                run_name_prefix=config.run_name_prefix,
                keep_iteration_checkpoints=config.keep_iteration_checkpoints,
                active=False,
                fixed_output_dir=str(shared_output_dir),
            )
        self.logger = logger
        self.logger.write_run_metadata(
            config=config,
            regularization=regularization,
            dynamics_regularization=self.dynamics_regularization,
            train_manifest=self.train_evaluator.manifest,
            test_manifest=None if self.test_evaluator is None else self.test_evaluator.manifest,
            preprocess_record=self.preprocess_record,
            dt_shrink=self.dt_shrink,
            dt_min=self.dt_min,
            tol=self.tol,
            max_iter_newton=self.max_iter_newton,
            max_dt=self.max_dt,
            time_integrator=self.time_integrator,
        )

    def _evaluate_snapshot(
        self,
        iteration: int,
        dynamics: DynamicsLike,
        step_norm: float,
        train_result: ReducedQoiBestResponseResult | None = None,
    ) -> ReducedQoiTrainingSnapshot:
        if train_result is None:
            train_result = self.gradient_calculator.evaluate(dynamics)
        train_relative_error = _relative_error_from_dataset_result(train_result.dataset_result, self.context)
        test_data_loss = None
        test_relative_error = None
        if self.test_dataset_evaluator is not None and iteration % self.config.test_every == 0:
            test_data_loss = self.test_dataset_evaluator.evaluate_data_loss(dynamics, train_result.decoder)
            test_relative_error = self.test_dataset_evaluator.evaluate_relative_error(dynamics, train_result.decoder)
        gradient = np.asarray(train_result.reduced_objective_gradient, dtype=np.float64)
        parameter_vector = dynamics_parameter_vector(dynamics)
        dynamic_parameter_norm = float(np.linalg.norm(parameter_vector))
        decoder_parameter_norm = float(np.linalg.norm(decoder_parameter_matrix(train_result.decoder)))
        gradient_norm = float(np.linalg.norm(gradient))
        return ReducedQoiTrainingSnapshot(
            iteration=iteration,
            dynamics=dynamics,
            decoder=train_result.decoder,
            train_result=train_result,
            test_data_loss=test_data_loss,
            train_relative_error=train_relative_error,
            test_relative_error=test_relative_error,
            step_norm=step_norm,
            gradient_norm=gradient_norm,
            dynamic_parameter_norm=dynamic_parameter_norm,
            decoder_parameter_norm=decoder_parameter_norm,
        )

    def _record_snapshot(
        self,
        snapshot: ReducedQoiTrainingSnapshot,
        best_snapshot: ReducedQoiTrainingSnapshot,
        is_best: bool,
    ) -> None:
        if snapshot.iteration % self.config.log_every == 0:
            self.logger.log_iteration(snapshot, best_snapshot)
            self.logger.log_stdout(
                self.logger.format_iteration_message(snapshot, best_snapshot),
                echo=self.config.echo_progress and self.context.rank == 0,
            )
        if snapshot.iteration % self.config.checkpoint_every == 0 or snapshot.iteration == self.config.max_iterations or is_best:
            self.logger.save_checkpoint(snapshot, best_snapshot, is_best=is_best)

    def _lbfgs_evaluate_vector(self, dynamics_template: DynamicsLike, vector: np.ndarray) -> ReducedQoiBestResponseResult:
        candidate = np.asarray(vector, dtype=np.float64).copy()
        dynamics = dynamics_from_parameter_vector(dynamics_template, candidate)
        return self.gradient_calculator.evaluate(dynamics)

    def _lbfgs_snapshot_from_vector(
        self,
        dynamics_template: DynamicsLike,
        vector: np.ndarray,
        iteration: int,
        step_norm: float,
        train_result: ReducedQoiBestResponseResult | None = None,
    ) -> ReducedQoiTrainingSnapshot:
        candidate = np.asarray(vector, dtype=np.float64).copy()
        dynamics = dynamics_from_parameter_vector(dynamics_template, candidate)
        if train_result is None:
            train_result = self.gradient_calculator.evaluate(dynamics)
        return self._evaluate_snapshot(
            iteration=iteration,
            dynamics=dynamics,
            step_norm=step_norm,
            train_result=train_result,
        )

    def _lbfgs_command_failure_record(self, exc: Exception) -> dict[str, Any]:
        record: dict[str, Any] = {
            "rank": self.context.rank,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, ForwardRolloutFailure):
            record.update(
                {
                    "sample_index": exc.sample_index,
                    "sample_id": exc.sample_id,
                    "sample_path": exc.sample_path,
                    "final_time": exc.final_time,
                    "reason": exc.reason,
                }
            )
        return record

    def _lbfgs_collect_command_failures(
        self,
        local_failure: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.context.allgather_object(local_failure) if item is not None)

    @staticmethod
    def _format_lbfgs_command_failures(failures: tuple[dict[str, Any], ...]) -> str:
        pieces = []
        for failure in failures:
            sample_piece = ""
            if "sample_id" in failure:
                sample_piece = f" sample={failure['sample_id']} index={failure['sample_index']}"
            pieces.append(
                f"rank={failure['rank']} {failure['type']}: {failure['message']}{sample_piece}"
            )
        return " | ".join(pieces)

    def _run_lbfgs_worker_loop(
        self,
        initial_dynamics: DynamicsLike,
        optimizer_root: int,
    ) -> tuple[ReducedQoiTrainingSnapshot, ReducedQoiTrainingSnapshot]:
        final_snapshot: ReducedQoiTrainingSnapshot | None = None
        best_snapshot: ReducedQoiTrainingSnapshot | None = None
        while True:
            command = self.context.bcast_object(None, root=optimizer_root)
            if not isinstance(command, dict):
                raise RuntimeError(f"Invalid root-led L-BFGS command: {command!r}")
            op = command.get("op")
            if op == "evaluate":
                try:
                    self._lbfgs_evaluate_vector(initial_dynamics, np.asarray(command["vector"], dtype=np.float64))
                    local_failure = None
                except Exception as exc:
                    local_failure = self._lbfgs_command_failure_record(exc)
                self._lbfgs_collect_command_failures(local_failure)
            elif op == "snapshot":
                try:
                    final_snapshot = self._lbfgs_snapshot_from_vector(
                        initial_dynamics,
                        np.asarray(command["vector"], dtype=np.float64),
                        iteration=int(command["iteration"]),
                        step_norm=float(command["step_norm"]),
                    )
                    if best_snapshot is None or self._is_better(final_snapshot, best_snapshot):
                        best_snapshot = final_snapshot
                    local_failure = None
                except Exception as exc:
                    local_failure = self._lbfgs_command_failure_record(exc)
                self._lbfgs_collect_command_failures(local_failure)
            elif op == "stop":
                if final_snapshot is None or best_snapshot is None:
                    raise RuntimeError("Root-led L-BFGS worker stopped before receiving any successful snapshot.")
                return final_snapshot, best_snapshot
            elif op == "abort":
                message = command.get("message", "unknown root-side failure")
                raise RuntimeError(f"Root-led L-BFGS aborted on optimizer root: {message}")
            else:
                raise RuntimeError(f"Unknown root-led L-BFGS command: {op!r}")

    def _scipy_quasi_newton_options(
        self,
        optimizer: str | None = None,
        max_iterations: int | None = None,
    ) -> tuple[str, dict[str, float | int]]:
        method_name = self.config.optimizer if optimizer is None else optimizer
        iteration_limit = self.config.max_iterations if max_iterations is None else max_iterations
        if method_name == "lbfgs":
            return (
                "L-BFGS-B",
                {
                    "maxiter": iteration_limit,
                    "maxcor": self.config.lbfgs.maxcor,
                    "ftol": self.config.lbfgs.ftol,
                    "gtol": self.config.lbfgs.gtol,
                    "maxls": self.config.lbfgs.maxls,
                },
            )
        if method_name == "bfgs":
            return (
                "BFGS",
                {
                    "maxiter": iteration_limit,
                    "gtol": self.config.bfgs.gtol,
                    "xrtol": self.config.bfgs.xrtol,
                },
            )
        raise RuntimeError(f"Unsupported SciPy quasi-Newton optimizer '{method_name}'.")

    def _train_with_scipy_quasi_newton(
        self,
        initial_dynamics: DynamicsLike,
        optimizer: str | None = None,
        max_iterations: int | None = None,
        iteration_offset: int = 0,
        initial_best: ReducedQoiTrainingSnapshot | None = None,
        record_initial: bool = True,
    ) -> tuple[ReducedQoiTrainingSnapshot, ReducedQoiTrainingSnapshot]:
        scipy_optimizer = self.config.optimizer if optimizer is None else optimizer
        iteration_limit = self.config.max_iterations if max_iterations is None else max_iterations
        optimizer_root = 0
        using_workers = self.context.size > 1
        if using_workers and self.context.rank != optimizer_root:
            return self._run_lbfgs_worker_loop(initial_dynamics, optimizer_root=optimizer_root)

        current_template = initial_dynamics
        last_eval_vector: np.ndarray | None = None
        last_eval_result: ReducedQoiBestResponseResult | None = None
        x0 = dynamics_parameter_vector(initial_dynamics)
        last_logged_vector = x0.copy()
        previous_callback_vector = last_logged_vector.copy()
        callback_iteration = 0
        lbfgs_failure_count = 0

        def local_evaluate(vector: np.ndarray) -> ReducedQoiBestResponseResult:
            nonlocal last_eval_vector, last_eval_result
            candidate = np.asarray(vector, dtype=np.float64).copy()
            if last_eval_vector is not None and np.array_equal(candidate, last_eval_vector):
                return last_eval_result  # type: ignore[return-value]
            result = self._lbfgs_evaluate_vector(current_template, candidate)
            last_eval_vector = candidate
            last_eval_result = result
            return result

        def dispatch_evaluate(vector: np.ndarray) -> ReducedQoiBestResponseResult:
            candidate = np.asarray(vector, dtype=np.float64).copy()
            if last_eval_vector is not None and np.array_equal(candidate, last_eval_vector):
                return last_eval_result  # type: ignore[return-value]
            if using_workers:
                self.context.bcast_object({"op": "evaluate", "vector": candidate}, root=optimizer_root)
            local_failure = None
            result = None
            try:
                result = local_evaluate(candidate)
            except Exception as exc:
                local_failure = self._lbfgs_command_failure_record(exc)
                local_exception = exc
            else:
                local_exception = None
            if using_workers:
                failures = self._lbfgs_collect_command_failures(local_failure)
                if failures:
                    if local_exception is not None:
                        raise local_exception
                    raise RuntimeError(
                        f"{scipy_optimizer.upper()} evaluation failed on worker rank: "
                        + self._format_lbfgs_command_failures(failures)
                    )
            if local_exception is not None:
                raise local_exception
            return result  # type: ignore[return-value]

        def dispatch_snapshot(iteration: int, vector: np.ndarray, step_norm: float) -> ReducedQoiTrainingSnapshot:
            candidate = np.asarray(vector, dtype=np.float64).copy()
            if using_workers:
                self.context.bcast_object(
                    {
                        "op": "snapshot",
                        "vector": candidate,
                        "iteration": int(iteration),
                        "step_norm": float(step_norm),
                    },
                    root=optimizer_root,
                )
            local_failure = None
            snapshot = None
            try:
                train_result = local_evaluate(candidate)
                snapshot = self._lbfgs_snapshot_from_vector(
                    current_template,
                    candidate,
                    iteration=iteration,
                    step_norm=step_norm,
                    train_result=train_result,
                )
            except Exception as exc:
                local_failure = self._lbfgs_command_failure_record(exc)
                local_exception = exc
            else:
                local_exception = None
            if using_workers:
                failures = self._lbfgs_collect_command_failures(local_failure)
                if failures:
                    if local_exception is not None:
                        raise local_exception
                    raise RuntimeError(
                        f"{scipy_optimizer.upper()} snapshot evaluation failed on worker rank: "
                        + self._format_lbfgs_command_failures(failures)
                    )
            if local_exception is not None:
                raise local_exception
            return snapshot  # type: ignore[return-value]

        def objective_and_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal lbfgs_failure_count
            candidate = np.asarray(vector, dtype=np.float64).copy()
            try:
                result = dispatch_evaluate(candidate)
                return float(result.objective_value), np.asarray(result.gradient, dtype=np.float64)
            except Exception as exc:
                lbfgs_failure_count += 1
                reference = x0 if last_eval_vector is None else last_eval_vector
                delta = candidate - reference
                penalty_scale = 1.0e6
                objective = 1.0e12 + penalty_scale * float(np.dot(delta, delta))
                gradient = 2.0 * penalty_scale * delta
                extra = {
                    "failure_count": lbfgs_failure_count,
                    "reference_parameter_norm": float(np.linalg.norm(reference)),
                    "candidate_parameter_norm": float(np.linalg.norm(candidate)),
                    "delta_norm": float(np.linalg.norm(delta)),
                    "optimizer": scipy_optimizer,
                }
                if isinstance(exc, ForwardRolloutFailure):
                    extra.update(
                        {
                            "failing_sample_id": exc.sample_id,
                            "failing_sample_index": exc.sample_index,
                            "failing_sample_path": exc.sample_path,
                            "failing_sample_final_time": exc.final_time,
                            "failing_sample_reason": exc.reason,
                        }
                    )
                if self.context.rank == 0:
                    dynamics = dynamics_from_parameter_vector(current_template, candidate)
                    artifact_paths = self.logger.save_failure_artifact(
                        name=f"{scipy_optimizer}_eval_failure_{lbfgs_failure_count:04d}",
                        dynamics=dynamics,
                        parameter_vector=candidate,
                        message=f"{type(exc).__name__}: {exc}",
                        extra=extra,
                    )
                else:
                    artifact_paths = None
                if self.context.rank == 0 and (lbfgs_failure_count <= 5 or lbfgs_failure_count % 10 == 0):
                    artifact_text = "" if artifact_paths is None else f" Saved to {artifact_paths[0]} and {artifact_paths[1]}."
                    self.logger.log_stderr(
                        (
                            f"{scipy_optimizer.upper()} evaluation failure #{lbfgs_failure_count}: "
                            f"{type(exc).__name__}: {exc}. Returning penalty objective.{artifact_text}"
                        ),
                        echo=self.config.echo_progress,
                    )
                return objective, gradient

        def callback(vector: np.ndarray) -> None:
            nonlocal callback_iteration, best_snapshot, last_logged_vector, previous_callback_vector
            callback_iteration += 1
            candidate = np.asarray(vector, dtype=np.float64).copy()
            step_norm = float(np.linalg.norm(candidate - previous_callback_vector))
            previous_callback_vector = candidate
            snapshot = dispatch_snapshot(iteration_offset + callback_iteration, candidate, step_norm)
            if self._is_better(snapshot, best_snapshot):
                best_snapshot = snapshot
                is_best = True
            else:
                is_best = False
            self._record_snapshot(snapshot, best_snapshot, is_best=is_best)
            last_logged_vector = candidate

        try:
            initial_snapshot = dispatch_snapshot(iteration=0, vector=x0, step_norm=0.0)
            if iteration_offset != 0:
                initial_snapshot = ReducedQoiTrainingSnapshot(
                    iteration=iteration_offset,
                    dynamics=initial_snapshot.dynamics,
                    decoder=initial_snapshot.decoder,
                    train_result=initial_snapshot.train_result,
                    test_data_loss=initial_snapshot.test_data_loss,
                    train_relative_error=initial_snapshot.train_relative_error,
                    test_relative_error=initial_snapshot.test_relative_error,
                    step_norm=initial_snapshot.step_norm,
                    gradient_norm=initial_snapshot.gradient_norm,
                    dynamic_parameter_norm=initial_snapshot.dynamic_parameter_norm,
                    decoder_parameter_norm=initial_snapshot.decoder_parameter_norm,
                )
            best_snapshot = initial_snapshot if initial_best is None else initial_best
            if record_initial:
                if self._is_better(initial_snapshot, best_snapshot):
                    best_snapshot = initial_snapshot
                    is_best = True
                else:
                    is_best = initial_best is None
                self._record_snapshot(initial_snapshot, best_snapshot, is_best=is_best)
            scipy_method, scipy_options = self._scipy_quasi_newton_options(
                optimizer=scipy_optimizer,
                max_iterations=iteration_limit,
            )

            opt_result = minimize(
                objective_and_gradient,
                x0=x0,
                method=scipy_method,
                jac=True,
                callback=callback,
                options=scipy_options,
            )

            final_vector = np.asarray(opt_result.x, dtype=np.float64).copy()
            final_iteration = iteration_offset + callback_iteration
            final_step_norm = float(np.linalg.norm(final_vector - previous_callback_vector))
            final_snapshot = dispatch_snapshot(
                iteration=final_iteration,
                vector=final_vector,
                step_norm=final_step_norm,
            )
            if callback_iteration == 0 or not np.array_equal(final_vector, last_logged_vector):
                if self._is_better(final_snapshot, best_snapshot):
                    best_snapshot = final_snapshot
                    is_best = True
                else:
                    is_best = False
                self._record_snapshot(final_snapshot, best_snapshot, is_best=is_best)
            if using_workers:
                self.context.bcast_object(
                    {"op": "stop"},
                    root=optimizer_root,
                )
            return final_snapshot, best_snapshot
        except Exception as exc:
            if using_workers:
                try:
                    self.context.bcast_object(
                        {
                            "op": "abort",
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                        root=optimizer_root,
                    )
                except Exception:
                    pass
            raise

    def _train_with_step_updater(
        self,
        initial_dynamics: DynamicsLike,
        updater: AdamUpdater | GradientDescentUpdater | NewtonActionUpdater,
        max_iterations: int,
        iteration_offset: int = 0,
        initial_best: ReducedQoiTrainingSnapshot | None = None,
    ) -> tuple[ReducedQoiTrainingSnapshot, ReducedQoiTrainingSnapshot]:
        current_dynamics = initial_dynamics
        best_snapshot = initial_best
        final_snapshot = None

        for local_iteration in range(max_iterations + 1):
            iteration = iteration_offset + local_iteration
            train_result = self.gradient_calculator.evaluate(current_dynamics)
            snapshot = self._evaluate_snapshot(
                iteration=iteration,
                dynamics=current_dynamics,
                step_norm=0.0,
                train_result=train_result,
            )

            step_norm = 0.0
            next_dynamics = current_dynamics
            if local_iteration < max_iterations:
                gradient = np.asarray(train_result.reduced_objective_gradient, dtype=np.float64)
                parameter_vector = dynamics_parameter_vector(current_dynamics)
                if isinstance(updater, (AdamUpdater, GradientDescentUpdater)):
                    next_vector, step_norm = updater.step(parameter_vector, gradient)
                else:
                    prepared_state = ReducedObjectivePreparedState(
                        dynamics=current_dynamics,
                        result=train_result,
                    )
                    next_vector, step_norm = updater.step(
                        current_dynamics=current_dynamics,
                        prepared_state=prepared_state,
                        workflow=self.reduced_workflow,
                    )
                next_dynamics = dynamics_from_parameter_vector(current_dynamics, next_vector)
                snapshot = ReducedQoiTrainingSnapshot(
                    iteration=snapshot.iteration,
                    dynamics=snapshot.dynamics,
                    decoder=snapshot.decoder,
                    train_result=snapshot.train_result,
                    test_data_loss=snapshot.test_data_loss,
                    train_relative_error=snapshot.train_relative_error,
                    test_relative_error=snapshot.test_relative_error,
                    step_norm=step_norm,
                    gradient_norm=snapshot.gradient_norm,
                    dynamic_parameter_norm=snapshot.dynamic_parameter_norm,
                    decoder_parameter_norm=snapshot.decoder_parameter_norm,
                )

            if best_snapshot is None or self._is_better(snapshot, best_snapshot):
                best_snapshot = snapshot
                is_best = True
            else:
                is_best = False

            self._record_snapshot(snapshot, best_snapshot, is_best=is_best)
            final_snapshot = snapshot
            current_dynamics = next_dynamics

        if final_snapshot is None or best_snapshot is None:
            raise RuntimeError("Training loop terminated before producing any snapshot.")
        return final_snapshot, best_snapshot

    def train(self, initial_dynamics: DynamicsLike) -> ReducedQoiTrainingResult:
        function_timer = FunctionTimer()
        train_start = time.perf_counter()
        try:
            timing_context = use_function_timer(function_timer) if self.config.enable_function_timing else contextlib.nullcontext()
            with timing_context:
                self.logger.log_stdout(
                    (
                        f"Starting GOATTM training run at {datetime.now().isoformat(timespec='seconds')} "
                        f"optimizer={self.config.optimizer} max_iterations={self.config.max_iterations}"
                    ),
                    echo=self.config.echo_progress and self.context.rank == 0,
                )
                self.logger.write_initial_parameters(initial_dynamics, self.decoder_template)
                if self.config.optimizer in {"lbfgs", "bfgs"}:
                    final_snapshot, best_snapshot = self._train_with_scipy_quasi_newton(
                        initial_dynamics,
                        optimizer=self.config.optimizer,
                        max_iterations=self.config.max_iterations,
                    )
                elif self.config.optimizer in {"adam_bfgs", "adam+bfgs"}:
                    adam_iterations = min(
                        max(0, int(self.config.adam_bfgs.adam_iterations)),
                        self.config.max_iterations,
                    )
                    self.logger.log_stdout(
                        (
                            f"Running Adam+BFGS schedule: adam_iterations={adam_iterations} "
                            f"bfgs_iterations={self.config.max_iterations - adam_iterations}"
                        ),
                        echo=self.config.echo_progress and self.context.rank == 0,
                    )
                    final_snapshot, best_snapshot = self._train_with_step_updater(
                        initial_dynamics,
                        updater=AdamUpdater(self.config.adam),
                        max_iterations=adam_iterations,
                    )
                    bfgs_iterations = self.config.max_iterations - adam_iterations
                    if bfgs_iterations > 0:
                        final_snapshot, best_snapshot = self._train_with_scipy_quasi_newton(
                            final_snapshot.dynamics,
                            optimizer="bfgs",
                            max_iterations=bfgs_iterations,
                            iteration_offset=final_snapshot.iteration,
                            initial_best=best_snapshot,
                            record_initial=False,
                        )
                else:
                    final_snapshot, best_snapshot = self._train_with_step_updater(
                        initial_dynamics,
                        updater=self.updater,
                        max_iterations=self.config.max_iterations,
                    )
                self.logger.log_stdout(
                    (
                        f"Completed GOATTM training run final_iter={final_snapshot.iteration} "
                        f"final_obj={final_snapshot.objective_value:.6e} "
                        f"best_iter={best_snapshot.iteration} "
                        f"best_obj={best_snapshot.objective_value:.6e}"
                    ),
                    echo=self.config.echo_progress and self.context.rank == 0,
                )
        except Exception:
            if self.context.rank == 0:
                try:
                    initial_vector = dynamics_parameter_vector(initial_dynamics)
                    self.logger.save_failure_artifact(
                        name="training_exception",
                        dynamics=initial_dynamics,
                        parameter_vector=initial_vector,
                        message="Unhandled training exception. See stderr.log for traceback.",
                        extra={
                            "optimizer": self.config.optimizer,
                            "max_dt": self.max_dt,
                            "dt_shrink": self.dt_shrink,
                            "dt_min": self.dt_min,
                            "tol": self.tol,
                            "max_iter_newton": self.max_iter_newton,
                        },
                    )
                except Exception:
                    pass
            self.logger.log_stderr(
                "GOATTM training run failed with exception:\n" + traceback.format_exc(),
                echo=self.config.echo_progress and self.context.rank == 0,
            )
            raise
        finally:
            if self.config.enable_function_timing:
                function_timer.record("goattm.train.ReducedQoiTrainer.train", time.perf_counter() - train_start)
            self.logger.write_timing_summary(function_timer)

        return ReducedQoiTrainingResult(
            final_snapshot=final_snapshot,
            best_snapshot=best_snapshot,
            output_dir=self.logger.output_dir,
            metrics_path=self.logger.metrics_path,
            summary_path=self.logger.summary_path,
            latest_checkpoint_path=self.logger.latest_checkpoint_path,
            best_checkpoint_path=self.logger.best_checkpoint_path,
            timing_json_path=self.logger.timing_json_path,
            timing_summary_path=self.logger.timing_summary_path,
            stdout_log_path=self.logger.stdout_log_path,
            stderr_log_path=self.logger.stderr_log_path,
            preprocess_path=self.logger.preprocess_path,
        )

    @staticmethod
    def _is_better(candidate: ReducedQoiTrainingSnapshot, incumbent: ReducedQoiTrainingSnapshot) -> bool:
        if candidate.test_data_loss is not None and incumbent.test_data_loss is not None:
            return candidate.test_data_loss < incumbent.test_data_loss
        if candidate.test_data_loss is not None and incumbent.test_data_loss is None:
            return True
        if candidate.test_data_loss is None and incumbent.test_data_loss is not None:
            return False
        return candidate.objective_value < incumbent.objective_value


def write_training_checkpoint(
    path: str | Path,
    snapshot: ReducedQoiTrainingSnapshot,
    best_snapshot: ReducedQoiTrainingSnapshot,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decoder = snapshot.decoder
    dynamics = snapshot.dynamics
    payload: dict[str, Any] = {
        "iteration": np.array(snapshot.iteration, dtype=np.int64),
        "train_data_loss": np.array(snapshot.train_result.data_loss, dtype=np.float64),
        "train_relative_error": np.array(snapshot.train_relative_error, dtype=np.float64),
        "train_decoder_regularization_loss": np.array(snapshot.train_result.decoder_regularization_loss, dtype=np.float64),
        "train_objective": np.array(snapshot.objective_value, dtype=np.float64),
        "gradient_norm": np.array(snapshot.gradient_norm, dtype=np.float64),
        "step_norm": np.array(snapshot.step_norm, dtype=np.float64),
        "dynamic_parameter_norm": np.array(snapshot.dynamic_parameter_norm, dtype=np.float64),
        "decoder_parameter_norm": np.array(snapshot.decoder_parameter_norm, dtype=np.float64),
        "best_iteration": np.array(best_snapshot.iteration, dtype=np.int64),
        "best_train_objective": np.array(best_snapshot.objective_value, dtype=np.float64),
        "best_train_relative_error": np.array(best_snapshot.train_relative_error, dtype=np.float64),
        "dynamics_type": np.array("stabilized" if isinstance(dynamics, StabilizedQuadraticDynamics) else "general"),
        "dynamics_key": np.array(dynamics_parameter_key(dynamics)),
        "decoder_v1": decoder.v1,
        "decoder_v2": decoder.v2,
        "decoder_v0": decoder.v0,
    }
    if snapshot.test_data_loss is not None:
        payload["test_data_loss"] = np.array(snapshot.test_data_loss, dtype=np.float64)
    if snapshot.test_relative_error is not None:
        payload["test_relative_error"] = np.array(snapshot.test_relative_error, dtype=np.float64)
    if best_snapshot.test_data_loss is not None:
        payload["best_test_data_loss"] = np.array(best_snapshot.test_data_loss, dtype=np.float64)
    if best_snapshot.test_relative_error is not None:
        payload["best_test_relative_error"] = np.array(best_snapshot.test_relative_error, dtype=np.float64)

    if isinstance(dynamics, StabilizedQuadraticDynamics):
        payload["s_params"] = dynamics.s_params
        payload["w_params"] = dynamics.w_params
    else:
        payload["a_matrix"] = dynamics.a
    payload["mu_h"] = dynamics.mu_h
    payload["c_vector"] = dynamics.c
    if dynamics.b is not None:
        payload["b_matrix"] = dynamics.b

    np.savez(path, **payload)
