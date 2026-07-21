from __future__ import annotations

from dataclasses import dataclass

import torch

from .adjoint import exact_frozen_step_adjoint, lagged_midpoint_rollout_adjoint, runge_kutta4_rollout_adjoint, runge_kutta4_substep_rollout_adjoint
from .data import ContinuousBatch
from .decoders import QuadraticReadoutDecoder, decoder_loss_and_state_grad
from .dynamics import QuadraticDynamics
from .steppers import DenseLaggedMidpointStepper, DenseRolloutResult
from .varpro import DecoderNormalSolveResult, solve_decoder_normal_equation


def trapezoidal_weights(times: torch.Tensor) -> torch.Tensor:
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be one-dimensional with at least two entries")
    weights = torch.empty_like(times)
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    if times.numel() > 2:
        weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    return weights


def weighted_trajectory_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    residual_sq = (prediction - target).square().sum(dim=-1)
    if weights is not None:
        residual_sq = residual_sq * weights.to(device=residual_sq.device, dtype=residual_sq.dtype)
    return 0.5 * residual_sq.sum()


@dataclass
class ReducedObjectiveResult:
    loss: torch.Tensor
    data_loss: torch.Tensor
    decoder_regularization_loss: torch.Tensor
    dynamics_regularization_loss: torch.Tensor
    prediction: torch.Tensor
    states: torch.Tensor
    lags: torch.Tensor
    normal_solve: DecoderNormalSolveResult
    parameter_grads: dict[str, torch.Tensor] | None = None
    state_grads: torch.Tensor | None = None


class ReducedObjective:
    """Variable-projection objective for dense GPU frozen-lag dynamics.

    For fixed dynamics parameters this class rolls out latent states, solves the
    decoder readout best response by normal equations, evaluates the QoI loss,
    and optionally applies the exact frozen-lag discrete adjoint to populate
    dynamics gradients.  The decoder readout is treated as an eliminated
    variable.
    """

    def __init__(
        self,
        dynamics: QuadraticDynamics,
        decoder: QuadraticReadoutDecoder,
        stepper: DenseLaggedMidpointStepper | None = None,
        *,
        decoder_ridge: float = 1.0e-8,
        dynamics_ridge: float = 1.0e-7,
        linear_ridge: float | None = None,
        quadratic_ridge: float | None = None,
        source_ridge: float | None = None,
        normal_chunk_size: int = 8192,
        use_trapezoidal_weights: bool = True,
        gradient_mode: str = "autograd",
    ) -> None:
        if gradient_mode not in {"autograd", "frozen_adjoint", "lagged_adjoint", "rk4_adjoint"}:
            raise ValueError("gradient_mode must be 'autograd', 'frozen_adjoint', 'lagged_adjoint', or 'rk4_adjoint'")
        self.dynamics = dynamics
        self.decoder = decoder
        self.stepper = stepper if stepper is not None else DenseLaggedMidpointStepper(picard_iters=2)
        self.decoder_ridge = float(decoder_ridge)
        self.dynamics_ridge = float(dynamics_ridge)
        self.linear_ridge = float(dynamics_ridge if linear_ridge is None else linear_ridge)
        self.quadratic_ridge = float(dynamics_ridge if quadratic_ridge is None else quadratic_ridge)
        self.source_ridge = float(dynamics_ridge if source_ridge is None else source_ridge)
        self.normal_chunk_size = int(normal_chunk_size)
        self.use_trapezoidal_weights = bool(use_trapezoidal_weights)
        self.gradient_mode = gradient_mode

    def _initial_state(self, batch: ContinuousBatch) -> torch.Tensor:
        if batch.u0 is not None:
            return batch.u0
        return batch.qoi.new_zeros(batch.batch_size, self.dynamics.latent_dim)

    def _loss_weights(self, batch: ContinuousBatch) -> torch.Tensor | None:
        if not self.use_trapezoidal_weights:
            return None
        time_weights = trapezoidal_weights(batch.observation_times)
        return time_weights[:, None].expand(batch.qoi.shape[:-1])

    def _stepper_inputs(self, batch: ContinuousBatch) -> torch.Tensor | None:
        substeps = int(getattr(self.stepper, "substeps", 1))
        if substeps > 1:
            return batch.substep_midpoint_inputs(substeps)
        return batch.midpoint_inputs()

    def rollout(self, batch: ContinuousBatch) -> DenseRolloutResult:
        p_mid = self._stepper_inputs(batch)
        return self.stepper.rollout_with_lags(
            self.dynamics,
            self._initial_state(batch),
            batch.step_size,
            batch.steps,
            p_mid=p_mid,
        )

    def evaluate(self, batch: ContinuousBatch, *, return_prediction: bool = False) -> ReducedObjectiveResult:
        with torch.no_grad():
            rollout = self.rollout(batch)
            weights = self._loss_weights(batch)
            normal = solve_decoder_normal_equation(
                self.decoder,
                rollout.states,
                batch.qoi,
                ridge=self.decoder_ridge,
                weights=weights,
                chunk_size=self.normal_chunk_size,
            )
            prediction, data_loss, _ = decoder_loss_and_state_grad(
                self.decoder,
                rollout.states.detach(),
                batch.qoi,
                weights=weights,
                chunk_size=min(self.normal_chunk_size, 2048),
                return_prediction=return_prediction,
                return_state_grad=False,
            )
            decoder_reg = self._decoder_regularization(normal)
            dynamics_reg = self._dynamics_regularization()
            loss = data_loss + decoder_reg + dynamics_reg
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=batch.qoi.new_empty(0) if prediction is None else prediction.detach(),
            states=rollout.states.detach(),
            lags=rollout.lags.detach(),
            normal_solve=normal,
        )

    def zero_dynamics_grad(self) -> None:
        for param in self.dynamics.parameters():
            param.grad = None

    def value_and_grad(self, batch: ContinuousBatch) -> ReducedObjectiveResult:
        if self.gradient_mode == "autograd":
            return self._value_and_grad_autograd(batch)
        if self.gradient_mode == "rk4_adjoint":
            return self._value_and_grad_rk4_adjoint(batch)
        if self.gradient_mode == "lagged_adjoint":
            return self._value_and_grad_lagged_adjoint(batch)
        return self._value_and_grad_frozen_adjoint(batch)

    def _value_and_grad_autograd(self, batch: ContinuousBatch) -> ReducedObjectiveResult:
        with torch.no_grad():
            rollout = self.rollout(batch)
            weights = self._loss_weights(batch)
            normal = solve_decoder_normal_equation(
                self.decoder,
                rollout.states,
                batch.qoi,
                ridge=self.decoder_ridge,
                weights=weights,
                chunk_size=self.normal_chunk_size,
            )

        self.zero_dynamics_grad()
        p_mid = self._stepper_inputs(batch)
        states = self.stepper.rollout(
            self.dynamics,
            self._initial_state(batch),
            batch.step_size,
            batch.steps,
            p_mid=p_mid,
        )
        prediction = self.decoder(states)
        data_loss = weighted_trajectory_mse(prediction, batch.qoi, weights=weights)
        decoder_reg = self._decoder_regularization(normal)
        dynamics_reg = self._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg
        named_params = [(name, param) for name, param in self.dynamics.named_parameters() if param.requires_grad]
        grad_values = torch.autograd.grad(loss, [param for _, param in named_params], allow_unused=True)
        param_grads: dict[str, torch.Tensor] = {}
        for (name, param), grad in zip(named_params, grad_values):
            value = torch.zeros_like(param) if grad is None else grad.detach()
            param.grad = value.clone()
            param_grads[name] = value.clone()
        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=prediction.detach(),
            states=states.detach(),
            lags=rollout.lags.detach(),
            normal_solve=normal,
            parameter_grads=param_grads,
            state_grads=None,
        )

    def _value_and_grad_frozen_adjoint(self, batch: ContinuousBatch) -> ReducedObjectiveResult:
        with torch.no_grad():
            p_mid = self._stepper_inputs(batch)
            rollout = self.stepper.rollout_with_lags(
                self.dynamics,
                self._initial_state(batch),
                batch.step_size,
                batch.steps,
                p_mid=p_mid,
            )
            weights = self._loss_weights(batch)
            normal = solve_decoder_normal_equation(
                self.decoder,
                rollout.states,
                batch.qoi,
                ridge=self.decoder_ridge,
                weights=weights,
                chunk_size=self.normal_chunk_size,
            )

        prediction, data_loss, state_grads = decoder_loss_and_state_grad(
            self.decoder,
            rollout.states.detach(),
            batch.qoi,
            weights=weights,
            chunk_size=min(self.normal_chunk_size, 2048),
            return_prediction=False,
        )
        decoder_reg = self._decoder_regularization(normal)
        dynamics_reg = self._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg

        self.zero_dynamics_grad()
        param_grads = {
            name: torch.zeros_like(param)
            for name, param in self.dynamics.named_parameters()
            if param.requires_grad
        }
        lambda_state = state_grads[-1]
        states = rollout.states.detach()
        lags = rollout.lags.detach()
        for n in reversed(range(batch.steps)):
            p_n = None if p_mid is None else p_mid[n].detach()
            adj = exact_frozen_step_adjoint(
                self.dynamics,
                states[n],
                lags[n],
                states[n + 1],
                lambda_state,
                batch.step_size,
                p_mid=p_n,
            )
            for name, grad in adj.parameter_grads.items():
                param_grads[name] = param_grads[name] + grad.to(device=param_grads[name].device, dtype=param_grads[name].dtype)
            lambda_state = adj.lambda_u + state_grads[n]

        self._add_dynamics_regularization_gradient(param_grads)
        params = dict(self.dynamics.named_parameters())
        for name, grad in param_grads.items():
            params[name].grad = grad.detach().clone()

        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=batch.qoi.new_empty(0),
            states=states,
            lags=lags,
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=state_grads,
        )

    def _value_and_grad_lagged_adjoint(self, batch: ContinuousBatch) -> ReducedObjectiveResult:
        with torch.no_grad():
            p_mid = self._stepper_inputs(batch)
            rollout = self.stepper.rollout_with_picard_history(
                self.dynamics,
                self._initial_state(batch),
                batch.step_size,
                batch.steps,
                p_mid=p_mid,
            )
            weights = self._loss_weights(batch)
            normal = solve_decoder_normal_equation(
                self.decoder,
                rollout.states,
                batch.qoi,
                ridge=self.decoder_ridge,
                weights=weights,
                chunk_size=self.normal_chunk_size,
            )

        prediction, data_loss, state_grads = decoder_loss_and_state_grad(
            self.decoder,
            rollout.states.detach(),
            batch.qoi,
            weights=weights,
            chunk_size=min(self.normal_chunk_size, 2048),
            return_prediction=False,
        )
        decoder_reg = self._decoder_regularization(normal)
        dynamics_reg = self._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg

        self.zero_dynamics_grad()
        adjoint = lagged_midpoint_rollout_adjoint(
            self.dynamics,
            self._initial_state(batch),
            batch.step_size,
            state_grads,
            p_mid.detach() if p_mid is not None else None,
            picard_iters=self.stepper.picard_iters,
            states=rollout.states.detach(),
            picard_iterates=rollout.picard_iterates.detach(),
            return_input_adjoint=False,
        )
        param_grads = dict(adjoint.parameter_grads)
        self._add_dynamics_regularization_gradient(param_grads)

        params = dict(self.dynamics.named_parameters())
        for name, grad in param_grads.items():
            params[name].grad = grad.detach().clone()

        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=batch.qoi.new_empty(0),
            states=rollout.states.detach(),
            lags=rollout.lags.detach(),
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=state_grads,
        )

    def _value_and_grad_rk4_adjoint(self, batch: ContinuousBatch) -> ReducedObjectiveResult:
        with torch.no_grad():
            p_mid = self._stepper_inputs(batch)
            rollout = self.stepper.rollout_with_lags(
                self.dynamics,
                self._initial_state(batch),
                batch.step_size,
                batch.steps,
                p_mid=p_mid,
            )
            weights = self._loss_weights(batch)
            normal = solve_decoder_normal_equation(
                self.decoder,
                rollout.states,
                batch.qoi,
                ridge=self.decoder_ridge,
                weights=weights,
                chunk_size=self.normal_chunk_size,
            )

        prediction, data_loss, state_grads = decoder_loss_and_state_grad(
            self.decoder,
            rollout.states.detach(),
            batch.qoi,
            weights=weights,
            chunk_size=min(self.normal_chunk_size, 2048),
            return_prediction=False,
        )
        decoder_reg = self._decoder_regularization(normal)
        dynamics_reg = self._dynamics_regularization()
        loss = data_loss + decoder_reg + dynamics_reg

        self.zero_dynamics_grad()
        if int(getattr(self.stepper, "substeps", 1)) > 1:
            adjoint = runge_kutta4_substep_rollout_adjoint(
                self.dynamics,
                self._initial_state(batch),
                batch.step_size,
                state_grads,
                p_mid.detach() if p_mid is not None else None,
                states=rollout.states.detach(),
                substeps=int(getattr(self.stepper, "substeps", 1)),
                return_input_adjoint=False,
            )
        else:
            adjoint = runge_kutta4_rollout_adjoint(
                self.dynamics,
                self._initial_state(batch),
                batch.step_size,
                state_grads,
                p_mid.detach() if p_mid is not None else None,
                states=rollout.states.detach(),
                return_input_adjoint=False,
            )
        param_grads = dict(adjoint.parameter_grads)
        self._add_dynamics_regularization_gradient(param_grads)

        params = dict(self.dynamics.named_parameters())
        for name, grad in param_grads.items():
            params[name].grad = grad.detach().clone()

        return ReducedObjectiveResult(
            loss=loss.detach(),
            data_loss=data_loss.detach(),
            decoder_regularization_loss=decoder_reg.detach(),
            dynamics_regularization_loss=dynamics_reg.detach(),
            prediction=batch.qoi.new_empty(0),
            states=rollout.states.detach(),
            lags=rollout.lags.detach(),
            normal_solve=normal,
            parameter_grads={name: grad.detach().clone() for name, grad in param_grads.items()},
            state_grads=state_grads,
        )

    def _decoder_regularization(self, normal: DecoderNormalSolveResult) -> torch.Tensor:
        if self.decoder_ridge <= 0.0:
            return normal.coefficients.new_zeros(())
        return 0.5 * float(self.decoder_ridge) * normal.coefficients.square().sum()

    def _dynamics_regularization(self) -> torch.Tensor:
        params = [param for param in self.dynamics.parameters() if param.requires_grad]
        if not params:
            return torch.zeros((), dtype=torch.get_default_dtype())
        first = params[0]
        total = first.new_zeros(())
        for name, param in self.dynamics.named_parameters():
            if not param.requires_grad:
                continue
            ridge = self._parameter_ridge(name)
            if ridge > 0.0:
                total = total + 0.5 * ridge * param.square().sum()
        return total

    def _add_dynamics_regularization_gradient(self, grads: dict[str, torch.Tensor]) -> None:
        for name, param in self.dynamics.named_parameters():
            if not param.requires_grad:
                continue
            ridge = self._parameter_ridge(name)
            if ridge <= 0.0:
                continue
            reg_grad = ridge * param.detach()
            if name in grads:
                grads[name] = grads[name] + reg_grad.to(device=grads[name].device, dtype=grads[name].dtype)
            else:
                grads[name] = reg_grad.clone()

    def _parameter_ridge(self, name: str) -> float:
        if name.startswith("linear."):
            return self.linear_ridge
        if name.startswith("quadratic."):
            return self.quadratic_ridge
        if name.startswith("source."):
            return self.source_ridge
        return self.dynamics_ridge
