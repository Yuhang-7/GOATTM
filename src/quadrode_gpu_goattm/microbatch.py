from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch

from .adjoint import exact_frozen_step_adjoint, lagged_midpoint_rollout_adjoint
from .data import ContinuousBatch, slice_batch
from .decoders import decoder_loss_and_state_grad
from .reduced import ReducedObjective, ReducedObjectiveResult
from .varpro import (
    DecoderNormalSolveResult,
    DecoderNormalTerms,
    assemble_decoder_normal_terms,
    solve_decoder_normal_terms,
)


BatchSource = ContinuousBatch | Iterable[ContinuousBatch] | Callable[[], Iterable[ContinuousBatch]]


@dataclass
class _CachedLaggedMicroBatch:
    qoi: torch.Tensor
    u0: torch.Tensor
    p_mid: torch.Tensor | None
    weights: torch.Tensor | None
    step_size: float
    steps: int
    states: torch.Tensor
    picard_iterates: torch.Tensor


@dataclass
class _CachedFrozenMicroBatch:
    qoi: torch.Tensor
    u0: torch.Tensor
    p_mid: torch.Tensor | None
    weights: torch.Tensor | None
    step_size: float
    steps: int
    states: torch.Tensor
    lags: torch.Tensor


class MicroBatchReducedObjective:
    """Memory-bounded wrapper for a full reduced objective.

    The decoder is eliminated once for the union of all micro-batches.  This is
    mathematically different from optimizing separate micro-batch decoders and
    is the path needed for large train sets.
    """

    def __init__(
        self,
        objective: ReducedObjective,
        micro_batch_size: int,
        *,
        cache_mode: str = "full",
    ) -> None:
        if int(micro_batch_size) <= 0:
            raise ValueError("micro_batch_size must be positive")
        if cache_mode not in {"none", "full"}:
            raise ValueError("cache_mode must be 'none' or 'full'")
        self.objective = objective
        self.micro_batch_size = int(micro_batch_size)
        self.cache_mode = cache_mode

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
        self.objective.gradient_mode = value

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.dynamics.parameters())
        return param.device, param.dtype

    def _prepare_micro_batch(self, batch: ContinuousBatch) -> ContinuousBatch:
        device, dtype = self._device_dtype()
        return batch.to(device=device, dtype=dtype)

    def iter_micro_batches(self, batch: BatchSource) -> Iterable[ContinuousBatch]:
        if isinstance(batch, ContinuousBatch):
            for start in range(0, batch.batch_size, self.micro_batch_size):
                yield self._prepare_micro_batch(
                    slice_batch(batch, start, min(start + self.micro_batch_size, batch.batch_size))
                )
            return
        if callable(batch):
            for micro in batch():
                yield self._prepare_micro_batch(micro)
            return
        iterator = iter(batch)
        if isinstance(iterator, Iterator) and iterator is batch:
            raise ValueError("single-use batch iterators cannot be scanned twice; pass a callable batch factory")
        for micro in batch:
            yield self._prepare_micro_batch(micro)

    def zero_dynamics_grad(self) -> None:
        self.objective.zero_dynamics_grad()

    def evaluate(self, batch: BatchSource, *, return_prediction: bool = False) -> ReducedObjectiveResult:
        normal = self._solve_global_decoder(batch)
        data_loss = None
        predictions = []
        with torch.no_grad():
            for micro in self.iter_micro_batches(batch):
                rollout = self.objective.rollout(micro)
                weights = self.objective._loss_weights(micro)
                prediction, chunk_loss, _ = decoder_loss_and_state_grad(
                    self.decoder,
                    rollout.states.detach(),
                    micro.qoi,
                    weights=weights,
                    chunk_size=min(self.objective.normal_chunk_size, 2048),
                    return_prediction=return_prediction,
                    return_state_grad=False,
                )
                data_loss = chunk_loss if data_loss is None else data_loss + chunk_loss
                if prediction is not None:
                    predictions.append(prediction.detach())
            if data_loss is None:
                data_loss = normal.coefficients.new_zeros(())
            decoder_reg = self.objective._decoder_regularization(normal)
            dynamics_reg = self.objective._dynamics_regularization()
            loss = data_loss + decoder_reg + dynamics_reg
            prediction_out = normal.coefficients.new_empty(0)
            if predictions:
                prediction_out = torch.cat(predictions, dim=1)
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=prediction_out.detach(),
            states=normal.coefficients.new_empty(0),
            lags=normal.coefficients.new_empty(0),
            normal_solve=normal,
        )

    def value_and_grad(self, batch: BatchSource) -> ReducedObjectiveResult:
        if self.gradient_mode == "autograd":
            raise ValueError("MicroBatchReducedObjective supports manual adjoint modes, not autograd")
        if self.gradient_mode == "lagged_adjoint":
            return self._value_and_grad_lagged_adjoint(batch)
        if self.gradient_mode == "frozen_adjoint":
            return self._value_and_grad_frozen_adjoint(batch)
        raise ValueError("unknown gradient_mode")

    def _solve_global_decoder(self, batch: BatchSource) -> DecoderNormalSolveResult:
        total_terms: DecoderNormalTerms | None = None
        with torch.no_grad():
            for micro in self.iter_micro_batches(batch):
                rollout = self.objective.rollout(micro)
                terms = assemble_decoder_normal_terms(
                    self.decoder,
                    rollout.states,
                    micro.qoi,
                    weights=self.objective._loss_weights(micro),
                    chunk_size=self.objective.normal_chunk_size,
                )
                if total_terms is None:
                    total_terms = DecoderNormalTerms(
                        normal_matrix=terms.normal_matrix.clone(),
                        rhs=terms.rhs.clone(),
                        feature_dim=terms.feature_dim,
                        observation_count=terms.observation_count,
                    )
                else:
                    if total_terms.feature_dim != terms.feature_dim:
                        raise ValueError("micro-batches produced incompatible decoder feature dimensions")
                    total_terms.normal_matrix.add_(terms.normal_matrix)
                    total_terms.rhs.add_(terms.rhs)
                    total_terms.observation_count += terms.observation_count
        if total_terms is None:
            raise ValueError("batch produced no micro-batches")
        return solve_decoder_normal_terms(self.decoder, total_terms, ridge=self.objective.decoder_ridge)

    def _accumulate_normal_terms(
        self,
        total_terms: DecoderNormalTerms | None,
        terms: DecoderNormalTerms,
    ) -> DecoderNormalTerms:
        if total_terms is None:
            return DecoderNormalTerms(
                normal_matrix=terms.normal_matrix.clone(),
                rhs=terms.rhs.clone(),
                feature_dim=terms.feature_dim,
                observation_count=terms.observation_count,
            )
        if total_terms.feature_dim != terms.feature_dim:
            raise ValueError("micro-batches produced incompatible decoder feature dimensions")
        total_terms.normal_matrix.add_(terms.normal_matrix)
        total_terms.rhs.add_(terms.rhs)
        total_terms.observation_count += terms.observation_count
        return total_terms

    def _empty_param_grads(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.zeros_like(param)
            for name, param in self.dynamics.named_parameters()
            if param.requires_grad
        }

    def _set_param_grads(self, grads: dict[str, torch.Tensor]) -> None:
        params = dict(self.dynamics.named_parameters())
        for name, grad in grads.items():
            params[name].grad = grad.detach().clone()

    def _value_and_grad_lagged_adjoint(self, batch: BatchSource) -> ReducedObjectiveResult:
        if self.cache_mode == "full":
            return self._value_and_grad_lagged_adjoint_cached(batch)
        normal = self._solve_global_decoder(batch)
        self.zero_dynamics_grad()
        param_grads = self._empty_param_grads()
        data_loss = None
        for micro in self.iter_micro_batches(batch):
            with torch.no_grad():
                p_mid = micro.midpoint_inputs()
                rollout = self.objective.stepper.rollout_with_picard_history(
                    self.dynamics,
                    self.objective._initial_state(micro),
                    micro.step_size,
                    micro.steps,
                    p_mid=p_mid,
                )
                weights = self.objective._loss_weights(micro)
            _, chunk_loss, state_grads = decoder_loss_and_state_grad(
                self.decoder,
                rollout.states.detach(),
                micro.qoi,
                weights=weights,
                chunk_size=min(self.objective.normal_chunk_size, 2048),
                return_prediction=False,
            )
            if state_grads is None:
                raise RuntimeError("state gradients were not returned")
            data_loss = chunk_loss if data_loss is None else data_loss + chunk_loss
            adjoint = lagged_midpoint_rollout_adjoint(
                self.dynamics,
                self.objective._initial_state(micro),
                micro.step_size,
                state_grads,
                p_mid.detach() if p_mid is not None else None,
                picard_iters=self.objective.stepper.picard_iters,
                states=rollout.states.detach(),
                picard_iterates=rollout.picard_iterates.detach(),
                return_input_adjoint=False,
            )
            for name, grad in adjoint.parameter_grads.items():
                param_grads[name] = param_grads[name] + grad.to(device=param_grads[name].device, dtype=param_grads[name].dtype)

        if data_loss is None:
            data_loss = normal.coefficients.new_zeros(())
        self.objective._add_dynamics_regularization_gradient(param_grads)
        self._set_param_grads(param_grads)
        decoder_reg = self.objective._decoder_regularization(normal)
        dynamics_reg = self.objective._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=normal.coefficients.new_empty(0),
            states=normal.coefficients.new_empty(0),
            lags=normal.coefficients.new_empty(0),
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=None,
        )

    def _value_and_grad_lagged_adjoint_cached(self, batch: BatchSource) -> ReducedObjectiveResult:
        cached: list[_CachedLaggedMicroBatch] = []
        total_terms: DecoderNormalTerms | None = None
        with torch.no_grad():
            for micro in self.iter_micro_batches(batch):
                p_mid = micro.midpoint_inputs()
                u0 = self.objective._initial_state(micro)
                rollout = self.objective.stepper.rollout_with_picard_history(
                    self.dynamics,
                    u0,
                    micro.step_size,
                    micro.steps,
                    p_mid=p_mid,
                )
                weights = self.objective._loss_weights(micro)
                terms = assemble_decoder_normal_terms(
                    self.decoder,
                    rollout.states,
                    micro.qoi,
                    weights=weights,
                    chunk_size=self.objective.normal_chunk_size,
                )
                total_terms = self._accumulate_normal_terms(total_terms, terms)
                cached.append(
                    _CachedLaggedMicroBatch(
                        qoi=micro.qoi.detach(),
                        u0=u0.detach(),
                        p_mid=None if p_mid is None else p_mid.detach(),
                        weights=None if weights is None else weights.detach(),
                        step_size=micro.step_size,
                        steps=micro.steps,
                        states=rollout.states.detach(),
                        picard_iterates=rollout.picard_iterates.detach(),
                    )
                )
        if total_terms is None:
            raise ValueError("batch produced no micro-batches")
        normal = solve_decoder_normal_terms(self.decoder, total_terms, ridge=self.objective.decoder_ridge)
        self.zero_dynamics_grad()
        param_grads = self._empty_param_grads()
        data_loss = None
        for item in cached:
            _, chunk_loss, state_grads = decoder_loss_and_state_grad(
                self.decoder,
                item.states,
                item.qoi,
                weights=item.weights,
                chunk_size=min(self.objective.normal_chunk_size, 2048),
                return_prediction=False,
            )
            if state_grads is None:
                raise RuntimeError("state gradients were not returned")
            data_loss = chunk_loss if data_loss is None else data_loss + chunk_loss
            adjoint = lagged_midpoint_rollout_adjoint(
                self.dynamics,
                item.u0,
                item.step_size,
                state_grads,
                item.p_mid,
                picard_iters=self.objective.stepper.picard_iters,
                states=item.states,
                picard_iterates=item.picard_iterates,
                return_input_adjoint=False,
            )
            for name, grad in adjoint.parameter_grads.items():
                param_grads[name] = param_grads[name] + grad.to(device=param_grads[name].device, dtype=param_grads[name].dtype)

        if data_loss is None:
            data_loss = normal.coefficients.new_zeros(())
        self.objective._add_dynamics_regularization_gradient(param_grads)
        self._set_param_grads(param_grads)
        decoder_reg = self.objective._decoder_regularization(normal)
        dynamics_reg = self.objective._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=normal.coefficients.new_empty(0),
            states=normal.coefficients.new_empty(0),
            lags=normal.coefficients.new_empty(0),
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=None,
        )

    def _value_and_grad_frozen_adjoint(self, batch: BatchSource) -> ReducedObjectiveResult:
        if self.cache_mode == "full":
            return self._value_and_grad_frozen_adjoint_cached(batch)
        normal = self._solve_global_decoder(batch)
        self.zero_dynamics_grad()
        param_grads = self._empty_param_grads()
        data_loss = None
        for micro in self.iter_micro_batches(batch):
            with torch.no_grad():
                p_mid = micro.midpoint_inputs()
                rollout = self.objective.stepper.rollout_with_lags(
                    self.dynamics,
                    self.objective._initial_state(micro),
                    micro.step_size,
                    micro.steps,
                    p_mid=p_mid,
                )
                weights = self.objective._loss_weights(micro)
            _, chunk_loss, state_grads = decoder_loss_and_state_grad(
                self.decoder,
                rollout.states.detach(),
                micro.qoi,
                weights=weights,
                chunk_size=min(self.objective.normal_chunk_size, 2048),
                return_prediction=False,
            )
            if state_grads is None:
                raise RuntimeError("state gradients were not returned")
            data_loss = chunk_loss if data_loss is None else data_loss + chunk_loss
            lambda_state = state_grads[-1]
            states = rollout.states.detach()
            lags = rollout.lags.detach()
            for n in reversed(range(micro.steps)):
                p_n = None if p_mid is None else p_mid[n].detach()
                adj = exact_frozen_step_adjoint(
                    self.dynamics,
                    states[n],
                    lags[n],
                    states[n + 1],
                    lambda_state,
                    micro.step_size,
                    p_mid=p_n,
                )
                for name, grad in adj.parameter_grads.items():
                    param_grads[name] = param_grads[name] + grad.to(device=param_grads[name].device, dtype=param_grads[name].dtype)
                lambda_state = adj.lambda_u + state_grads[n]

        if data_loss is None:
            data_loss = normal.coefficients.new_zeros(())
        self.objective._add_dynamics_regularization_gradient(param_grads)
        self._set_param_grads(param_grads)
        decoder_reg = self.objective._decoder_regularization(normal)
        dynamics_reg = self.objective._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=normal.coefficients.new_empty(0),
            states=normal.coefficients.new_empty(0),
            lags=normal.coefficients.new_empty(0),
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=None,
        )

    def _value_and_grad_frozen_adjoint_cached(self, batch: BatchSource) -> ReducedObjectiveResult:
        cached: list[_CachedFrozenMicroBatch] = []
        total_terms: DecoderNormalTerms | None = None
        with torch.no_grad():
            for micro in self.iter_micro_batches(batch):
                p_mid = micro.midpoint_inputs()
                u0 = self.objective._initial_state(micro)
                rollout = self.objective.stepper.rollout_with_lags(
                    self.dynamics,
                    u0,
                    micro.step_size,
                    micro.steps,
                    p_mid=p_mid,
                )
                weights = self.objective._loss_weights(micro)
                terms = assemble_decoder_normal_terms(
                    self.decoder,
                    rollout.states,
                    micro.qoi,
                    weights=weights,
                    chunk_size=self.objective.normal_chunk_size,
                )
                total_terms = self._accumulate_normal_terms(total_terms, terms)
                cached.append(
                    _CachedFrozenMicroBatch(
                        qoi=micro.qoi.detach(),
                        u0=u0.detach(),
                        p_mid=None if p_mid is None else p_mid.detach(),
                        weights=None if weights is None else weights.detach(),
                        step_size=micro.step_size,
                        steps=micro.steps,
                        states=rollout.states.detach(),
                        lags=rollout.lags.detach(),
                    )
                )
        if total_terms is None:
            raise ValueError("batch produced no micro-batches")
        normal = solve_decoder_normal_terms(self.decoder, total_terms, ridge=self.objective.decoder_ridge)
        self.zero_dynamics_grad()
        param_grads = self._empty_param_grads()
        data_loss = None
        for item in cached:
            _, chunk_loss, state_grads = decoder_loss_and_state_grad(
                self.decoder,
                item.states,
                item.qoi,
                weights=item.weights,
                chunk_size=min(self.objective.normal_chunk_size, 2048),
                return_prediction=False,
            )
            if state_grads is None:
                raise RuntimeError("state gradients were not returned")
            data_loss = chunk_loss if data_loss is None else data_loss + chunk_loss
            lambda_state = state_grads[-1]
            for n in reversed(range(item.steps)):
                p_n = None if item.p_mid is None else item.p_mid[n]
                adj = exact_frozen_step_adjoint(
                    self.dynamics,
                    item.states[n],
                    item.lags[n],
                    item.states[n + 1],
                    lambda_state,
                    item.step_size,
                    p_mid=p_n,
                )
                for name, grad in adj.parameter_grads.items():
                    param_grads[name] = param_grads[name] + grad.to(device=param_grads[name].device, dtype=param_grads[name].dtype)
                lambda_state = adj.lambda_u + state_grads[n]

        if data_loss is None:
            data_loss = normal.coefficients.new_zeros(())
        self.objective._add_dynamics_regularization_gradient(param_grads)
        self._set_param_grads(param_grads)
        decoder_reg = self.objective._decoder_regularization(normal)
        dynamics_reg = self.objective._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=normal.coefficients.new_empty(0),
            states=normal.coefficients.new_empty(0),
            lags=normal.coefficients.new_empty(0),
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=None,
        )
