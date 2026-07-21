from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .data import ContinuousBatch
from .decoders import decoder_loss_and_state_grad
from .reduced import ReducedObjective, ReducedObjectiveResult
from .varpro import (
    DecoderNormalSolveResult,
    DecoderNormalTerms,
    assemble_decoder_normal_terms,
    solve_decoder_normal_terms,
)
from .adjoint import lagged_midpoint_rollout_adjoint, runge_kutta4_rollout_adjoint, runge_kutta4_substep_rollout_adjoint


def distributed_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return dist.get_rank() if distributed_is_active() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if distributed_is_active() else 1


def all_reduce_sum_(value: torch.Tensor) -> torch.Tensor:
    if distributed_world_size() > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


@dataclass
class DistributedNormalTerms:
    terms: DecoderNormalTerms


class DistributedReducedObjective:
    """Data-parallel variable-projection objective.

    Each rank owns a disjoint shard of trajectories.  The wrapper computes
    local rollout and local decoder normal terms, all-reduces the normal
    equation, solves the same decoder on every rank, computes local adjoint
    gradients, then all-reduces A/H/source gradients.  This is pure data
    parallelism over samples; no solver or latent dimension is split.
    """

    def __init__(self, objective: ReducedObjective) -> None:
        if objective.gradient_mode not in {"lagged_adjoint", "rk4_adjoint"}:
            raise ValueError("DistributedReducedObjective currently supports gradient_mode='lagged_adjoint' or 'rk4_adjoint'")
        self.objective = objective

    @property
    def dynamics(self):
        return self.objective.dynamics

    @property
    def decoder(self):
        return self.objective.decoder

    @property
    def gradient_mode(self) -> str:
        return self.objective.gradient_mode

    @gradient_mode.setter
    def gradient_mode(self, value: str) -> None:
        if value not in {"lagged_adjoint", "rk4_adjoint"}:
            raise ValueError("DistributedReducedObjective currently supports only lagged_adjoint or rk4_adjoint")
        self.objective.gradient_mode = value

    def zero_dynamics_grad(self) -> None:
        self.objective.zero_dynamics_grad()

    def _all_reduce_normal_terms(self, local: DecoderNormalTerms) -> DecoderNormalTerms:
        normal = local.normal_matrix.clone()
        rhs = local.rhs.clone()
        count = torch.tensor([local.observation_count], device=normal.device, dtype=torch.float64)
        all_reduce_sum_(normal)
        all_reduce_sum_(rhs)
        all_reduce_sum_(count)
        return DecoderNormalTerms(
            normal_matrix=normal,
            rhs=rhs,
            feature_dim=local.feature_dim,
            observation_count=int(count.item()),
        )

    def _all_reduce_grads(self, grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for grad in grads.values():
            all_reduce_sum_(grad)
        return grads

    def _set_param_grads(self, grads: dict[str, torch.Tensor]) -> None:
        params = dict(self.dynamics.named_parameters())
        for name, grad in grads.items():
            params[name].grad = grad.detach().clone()

    def value_and_grad(self, local_batch: ContinuousBatch) -> ReducedObjectiveResult:
        objective = self.objective
        with torch.no_grad():
            p_mid = objective._stepper_inputs(local_batch)
            if objective.gradient_mode == "rk4_adjoint":
                rollout = objective.stepper.rollout_with_lags(
                    self.dynamics,
                    objective._initial_state(local_batch),
                    local_batch.step_size,
                    local_batch.steps,
                    p_mid=p_mid,
                )
            else:
                rollout = objective.stepper.rollout_with_picard_history(
                    self.dynamics,
                    objective._initial_state(local_batch),
                    local_batch.step_size,
                    local_batch.steps,
                    p_mid=p_mid,
                )
            weights = objective._loss_weights(local_batch)
            local_terms = assemble_decoder_normal_terms(
                self.decoder,
                rollout.states,
                local_batch.qoi,
                weights=weights,
                chunk_size=objective.normal_chunk_size,
            )
            global_terms = self._all_reduce_normal_terms(local_terms)
            normal = solve_decoder_normal_terms(self.decoder, global_terms, ridge=objective.decoder_ridge)

        _, local_data_loss, state_grads = decoder_loss_and_state_grad(
            self.decoder,
            rollout.states.detach(),
            local_batch.qoi,
            weights=weights,
            chunk_size=min(objective.normal_chunk_size, 2048),
            return_prediction=False,
        )
        if state_grads is None:
            raise RuntimeError("state gradients were not returned")
        data_loss = all_reduce_sum_(local_data_loss.detach().clone())

        self.zero_dynamics_grad()
        if objective.gradient_mode == "rk4_adjoint":
            if int(getattr(objective.stepper, "substeps", 1)) > 1:
                adjoint = runge_kutta4_substep_rollout_adjoint(
                    self.dynamics,
                    objective._initial_state(local_batch),
                    local_batch.step_size,
                    state_grads,
                    p_mid.detach() if p_mid is not None else None,
                    states=rollout.states.detach(),
                    substeps=int(getattr(objective.stepper, "substeps", 1)),
                    return_input_adjoint=False,
                )
            else:
                adjoint = runge_kutta4_rollout_adjoint(
                    self.dynamics,
                    objective._initial_state(local_batch),
                    local_batch.step_size,
                    state_grads,
                    p_mid.detach() if p_mid is not None else None,
                    states=rollout.states.detach(),
                    return_input_adjoint=False,
                )
        else:
            adjoint = lagged_midpoint_rollout_adjoint(
                self.dynamics,
                objective._initial_state(local_batch),
                local_batch.step_size,
                state_grads,
                p_mid.detach() if p_mid is not None else None,
                picard_iters=objective.stepper.picard_iters,
                states=rollout.states.detach(),
                picard_iterates=rollout.picard_iterates.detach(),
                return_input_adjoint=False,
            )
        param_grads = self._all_reduce_grads(dict(adjoint.parameter_grads))
        objective._add_dynamics_regularization_gradient(param_grads)
        self._set_param_grads(param_grads)

        decoder_reg = objective._decoder_regularization(normal)
        dynamics_reg = objective._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=normal.coefficients.new_empty(0),
            states=rollout.states.detach(),
            lags=rollout.lags.detach(),
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=state_grads.detach(),
        )

    def evaluate(self, local_batch: ContinuousBatch, *, return_prediction: bool = False) -> ReducedObjectiveResult:
        """Evaluate without an adjoint sweep, using a globally fitted decoder.

        Each rank returns predictions only for its local shard, but the decoder
        normal equation and the scalar data loss are all-reduced across ranks.
        """

        objective = self.objective
        with torch.no_grad():
            p_mid = objective._stepper_inputs(local_batch)
            rollout = objective.stepper.rollout_with_lags(
                self.dynamics,
                objective._initial_state(local_batch),
                local_batch.step_size,
                local_batch.steps,
                p_mid=p_mid,
            )
            weights = objective._loss_weights(local_batch)
            local_terms = assemble_decoder_normal_terms(
                self.decoder,
                rollout.states,
                local_batch.qoi,
                weights=weights,
                chunk_size=objective.normal_chunk_size,
            )
            global_terms = self._all_reduce_normal_terms(local_terms)
            normal = solve_decoder_normal_terms(self.decoder, global_terms, ridge=objective.decoder_ridge)
            prediction, local_data_loss, _ = decoder_loss_and_state_grad(
                self.decoder,
                rollout.states.detach(),
                local_batch.qoi,
                weights=weights,
                chunk_size=min(objective.normal_chunk_size, 2048),
                return_prediction=return_prediction,
                return_state_grad=False,
            )
            data_loss = all_reduce_sum_(local_data_loss.detach().clone())
            decoder_reg = objective._decoder_regularization(normal)
            dynamics_reg = objective._dynamics_regularization()
            loss = data_loss + decoder_reg + dynamics_reg
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=normal.coefficients.new_empty(0) if prediction is None else prediction.detach(),
            states=rollout.states.detach(),
            lags=rollout.lags.detach(),
            normal_solve=normal,
            parameter_grads=None,
            state_grads=None,
        )
