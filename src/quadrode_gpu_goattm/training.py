from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from .adjoint import _supports_manual_lagged_adjoint
from .data import ContinuousBatch
from .reduced import ReducedObjective


@dataclass
class OptimizationRecord:
    step: int
    closure_call: int
    loss: float
    data_loss: float
    decoder_regularization_loss: float
    dynamics_regularization_loss: float
    normal_relative_residual: float
    closure_seconds: float
    elapsed_seconds: float
    gradient_mode: str
    adjoint_backend: str
    stepper_class: str
    solver_backend: str
    dynamics_backend: str
    grad_norm: float


class JsonlOptimizationLogger:
    """Append-only optimization logger for reduced-objective training."""

    def __init__(self, path: str | Path, *, metadata: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        if metadata is not None:
            self.write_event("metadata", metadata)

    def write_event(self, event: str, payload: dict) -> None:
        self._handle.write(json.dumps({"event": event, **payload}, sort_keys=True) + "\n")
        self._handle.flush()

    def write_record(self, record: OptimizationRecord) -> None:
        self.write_event("closure", asdict(record))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlOptimizationLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class LBFGSTrainingResult:
    final_loss: float
    closure_calls: int
    optimizer_steps: int
    elapsed_seconds: float
    log_path: Path | None


def dynamics_grad_norm(objective: ReducedObjective) -> float:
    total: torch.Tensor | None = None
    for param in objective.dynamics.parameters():
        if param.grad is None:
            continue
        piece = param.grad.detach().square().sum()
        total = piece if total is None else total + piece
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu())


def unwrap_reduced_objective(objective):
    return getattr(objective, "objective", objective)


def objective_backend_metadata(objective) -> dict[str, str | int]:
    base = unwrap_reduced_objective(objective)
    gradient_mode = getattr(base, "gradient_mode", "unknown")
    dynamics = getattr(base, "dynamics", None)
    stepper = getattr(base, "stepper", None)
    stepper_class = type(stepper).__name__ if stepper is not None else "unknown"
    if gradient_mode == "autograd":
        adjoint_backend = "torch_autograd_full_rollout"
    elif gradient_mode == "rk4_adjoint":
        adjoint_backend = "manual_runge_kutta4_discrete_adjoint_no_autograd"
    elif gradient_mode == "frozen_adjoint":
        adjoint_backend = "manual_frozen_step_adjoint_with_local_autograd_vjp"
    elif dynamics is not None and _supports_manual_lagged_adjoint(dynamics):
        adjoint_backend = "manual_lagged_discrete_adjoint"
    elif gradient_mode == "lagged_adjoint":
        adjoint_backend = "local_autograd_vjp_fallback"
    else:
        adjoint_backend = "unknown"

    quadratic_for_solver = getattr(dynamics, "quadratic", None)
    if (
        type(quadratic_for_solver).__name__ == "EnergyTuckerTTQuadratic"
        and int(getattr(quadratic_for_solver, "reduced_rank", 0)) < int(getattr(dynamics, "latent_dim", 0))
    ):
        solver_backend = "energy_tucker_nested_smw"
    elif "SMW" in stepper_class:
        solver_backend = "exact_smw_low_rank_solve"
    elif "Defect" in stepper_class:
        solver_backend = "defect_iteration"
    elif "RungeKutta4" in stepper_class:
        substeps = int(getattr(stepper, "substeps", 1))
        solver_backend = "explicit_runge_kutta4" if substeps == 1 else f"explicit_runge_kutta4_substeps_{substeps}"
    elif "Dense" in stepper_class:
        solver_backend = "dense_batched_gpu_solve"
    else:
        solver_backend = "unknown"

    linear = getattr(dynamics, "linear", None)
    quadratic = getattr(dynamics, "quadratic", None)
    source = getattr(dynamics, "source", None)
    return {
        "gradient_mode": str(gradient_mode),
        "adjoint_backend": adjoint_backend,
        "stepper_class": stepper_class,
        "solver_backend": solver_backend,
        "dynamics_backend": type(dynamics).__name__ if dynamics is not None else "unknown",
        "linear_backend": type(linear).__name__ if linear is not None else "unknown",
        "quadratic_backend": type(quadratic).__name__ if quadratic is not None else "unknown",
        "source_backend": type(source).__name__ if source is not None else "unknown",
        "picard_iters": int(getattr(stepper, "picard_iters", -1)) if stepper is not None else -1,
        "substeps": int(getattr(stepper, "substeps", 1)) if stepper is not None else 1,
    }


class LBFGSReducedTrainer:
    """LBFGS wrapper around `ReducedObjective.value_and_grad`.

    The decoder is eliminated inside the objective at each closure call.  The
    optimizer only updates the dynamics parameters.
    """

    def __init__(
        self,
        objective: ReducedObjective,
        *,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int | None = None,
        history_size: int = 50,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        self.objective = objective
        self.optimizer = torch.optim.LBFGS(
            list(objective.dynamics.parameters()),
            lr=float(lr),
            max_iter=int(max_iter),
            max_eval=max_eval,
            history_size=int(history_size),
            line_search_fn=line_search_fn,
        )

    def fit(
        self,
        batch: ContinuousBatch,
        *,
        optimizer_steps: int = 1,
        logger: JsonlOptimizationLogger | None = None,
        callback: Callable[[OptimizationRecord], None] | None = None,
    ) -> LBFGSTrainingResult:
        start = time.perf_counter()
        closure_calls = 0
        last_loss = float("nan")

        for step in range(int(optimizer_steps)):
            def closure() -> torch.Tensor:
                nonlocal closure_calls, last_loss
                closure_start = time.perf_counter()
                result = self.objective.value_and_grad(batch)
                closure_calls += 1
                last_loss = float(result.loss.detach().cpu())
                backend = objective_backend_metadata(self.objective)
                record = OptimizationRecord(
                    step=step,
                    closure_call=closure_calls,
                    loss=last_loss,
                    data_loss=float(result.data_loss.detach().cpu()),
                    decoder_regularization_loss=float(result.decoder_regularization_loss.detach().cpu()),
                    dynamics_regularization_loss=float(result.dynamics_regularization_loss.detach().cpu()),
                    normal_relative_residual=result.normal_solve.relative_residual,
                    closure_seconds=time.perf_counter() - closure_start,
                    elapsed_seconds=time.perf_counter() - start,
                    gradient_mode=str(backend["gradient_mode"]),
                    adjoint_backend=str(backend["adjoint_backend"]),
                    stepper_class=str(backend["stepper_class"]),
                    solver_backend=str(backend["solver_backend"]),
                    dynamics_backend=str(backend["dynamics_backend"]),
                    grad_norm=dynamics_grad_norm(self.objective),
                )
                if logger is not None:
                    logger.write_record(record)
                if callback is not None:
                    callback(record)
                return result.loss.detach()

            self.optimizer.step(closure)

        return LBFGSTrainingResult(
            final_loss=last_loss,
            closure_calls=closure_calls,
            optimizer_steps=int(optimizer_steps),
            elapsed_seconds=time.perf_counter() - start,
            log_path=None if logger is None else logger.path,
        )
