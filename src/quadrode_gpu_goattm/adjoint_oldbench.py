from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .dynamics import QuadraticDynamics
from .linear import DenseLinearA, DissipativeSkewA
from .quadratic import DenseQuadratic, EnergyDenseQuadratic, EnergyTuckerTTQuadratic, SkewCPQuadratic
from .source import LinearSource, ZeroSource
from .steppers import exact_frozen_smw_solve, exact_frozen_solve, make_frozen_smw_cache, predictor


@dataclass
class FrozenStepAdjointResult:
    mu: torch.Tensor
    lambda_u: torch.Tensor
    lambda_p: torch.Tensor | None
    parameter_grads: dict[str, torch.Tensor]


@dataclass
class LaggedMidpointStepAdjointResult:
    u_next: torch.Tensor
    lambda_u: torch.Tensor
    lambda_p: torch.Tensor | None
    parameter_grads: dict[str, torch.Tensor]
    picard_iterates: tuple[torch.Tensor, ...]


@dataclass
class LaggedMidpointRolloutAdjointResult:
    states: torch.Tensor
    lambda_u0: torch.Tensor
    lambda_p_mid: torch.Tensor | None
    parameter_grads: dict[str, torch.Tensor]


@dataclass
class BatchedLaggedMidpointRolloutAdjointResult:
    states: torch.Tensor
    lambda_u0: torch.Tensor
    lambda_p_mid: torch.Tensor | None
    parameter_grads: dict[str, torch.Tensor]


@dataclass
class RungeKutta4StepAdjointResult:
    u_next: torch.Tensor
    lambda_u: torch.Tensor
    lambda_p: torch.Tensor | None
    parameter_grads: dict[str, torch.Tensor]


@dataclass
class RungeKutta4RolloutAdjointResult:
    states: torch.Tensor
    lambda_u0: torch.Tensor
    lambda_p_mid: torch.Tensor | None
    parameter_grads: dict[str, torch.Tensor]


@dataclass
class EnergyTuckerTTGradientAccumulator:
    quadratic: EnergyTuckerTTQuadratic
    basis_grad: torch.Tensor
    reduced_tensor_grad: torch.Tensor

    @classmethod
    def create(cls, quadratic: EnergyTuckerTTQuadratic, *, device: torch.device, dtype: torch.dtype) -> "EnergyTuckerTTGradientAccumulator":
        basis = quadratic.basis.to(device=device, dtype=dtype)
        return cls(
            quadratic=quadratic,
            basis_grad=torch.zeros_like(basis),
            reduced_tensor_grad=torch.zeros(
                quadratic.reduced_rank,
                quadratic.reduced_rank,
                quadratic.reduced_rank,
                device=device,
                dtype=dtype,
            ),
        )

    def add(self, grad_basis: torch.Tensor, grad_c: torch.Tensor) -> None:
        self.basis_grad.add_(grad_basis.detach().to(device=self.basis_grad.device, dtype=self.basis_grad.dtype))
        self.reduced_tensor_grad.add_(grad_c.detach().to(device=self.reduced_tensor_grad.device, dtype=self.reduced_tensor_grad.dtype))

    def flush_to(self, param_grads: dict[str, torch.Tensor]) -> None:
        quad = self.quadratic
        grad_c = self.reduced_tensor_grad
        _add_param_grad(param_grads, "quadratic.basis", self.basis_grad)

        coeff = quad.reconstruct_coeff.to(device=grad_c.device, dtype=grad_c.dtype)
        full_grad_flat = grad_c.reshape(-1)
        free_grad = grad_c.new_zeros(quad.free_dim)
        free_grad.index_add_(0, quad.reconstruct_source, coeff * full_grad_flat[quad.reconstruct_target])
        free_full_grad = grad_c.new_zeros(quad.reduced_rank, quad.reduced_rank, quad.reduced_rank)
        free_full_grad[quad.free_a, quad.free_b, quad.free_c] = free_grad

        core0 = quad.core0.to(device=grad_c.device, dtype=grad_c.dtype)
        core1 = quad.core1.to(device=grad_c.device, dtype=grad_c.dtype)
        core2 = quad.core2.to(device=grad_c.device, dtype=grad_c.dtype)
        _add_param_grad(param_grads, "quadratic.core0", torch.einsum("abc,rbs,sc->ar", free_full_grad, core1, core2))
        _add_param_grad(param_grads, "quadratic.core1", torch.einsum("abc,ar,sc->rbs", free_full_grad, core0, core2))
        _add_param_grad(param_grads, "quadratic.core2", torch.einsum("abc,ar,rbs->sc", free_full_grad, core0, core1))


@dataclass
class BatchedEnergyTuckerTTGradientAccumulator:
    quadratic: EnergyTuckerTTQuadratic
    basis_grad: torch.Tensor
    reduced_tensor_grad: torch.Tensor

    @classmethod
    def create(
        cls,
        quadratic: EnergyTuckerTTQuadratic,
        direction_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "BatchedEnergyTuckerTTGradientAccumulator":
        basis = quadratic.basis.to(device=device, dtype=dtype)
        return cls(
            quadratic=quadratic,
            basis_grad=torch.zeros(
                int(direction_count),
                *basis.shape,
                device=device,
                dtype=dtype,
            ),
            reduced_tensor_grad=torch.zeros(
                int(direction_count),
                quadratic.reduced_rank,
                quadratic.reduced_rank,
                quadratic.reduced_rank,
                device=device,
                dtype=dtype,
            ),
        )

    def add(self, grad_basis: torch.Tensor, grad_c: torch.Tensor) -> None:
        self.basis_grad.add_(grad_basis.detach().to(device=self.basis_grad.device, dtype=self.basis_grad.dtype))
        self.reduced_tensor_grad.add_(grad_c.detach().to(device=self.reduced_tensor_grad.device, dtype=self.reduced_tensor_grad.dtype))

    def flush_to(self, param_grads: dict[str, torch.Tensor]) -> None:
        quad = self.quadratic
        grad_c = self.reduced_tensor_grad
        _add_param_grad(param_grads, "quadratic.basis", self.basis_grad)

        coeff = quad.reconstruct_coeff.to(device=grad_c.device, dtype=grad_c.dtype)
        full_grad_flat = grad_c.reshape(grad_c.shape[0], -1)
        free_grad = grad_c.new_zeros(grad_c.shape[0], quad.free_dim)
        free_grad.index_add_(1, quad.reconstruct_source, coeff * full_grad_flat[:, quad.reconstruct_target])
        free_full_grad = grad_c.new_zeros(grad_c.shape[0], quad.reduced_rank, quad.reduced_rank, quad.reduced_rank)
        free_full_grad[:, quad.free_a, quad.free_b, quad.free_c] = free_grad

        core0 = quad.core0.to(device=grad_c.device, dtype=grad_c.dtype)
        core1 = quad.core1.to(device=grad_c.device, dtype=grad_c.dtype)
        core2 = quad.core2.to(device=grad_c.device, dtype=grad_c.dtype)
        _add_param_grad(param_grads, "quadratic.core0", torch.einsum("kabc,rbs,sc->kar", free_full_grad, core1, core2))
        _add_param_grad(param_grads, "quadratic.core1", torch.einsum("kabc,ar,sc->krbs", free_full_grad, core0, core2))
        _add_param_grad(param_grads, "quadratic.core2", torch.einsum("kabc,ar,rbs->ksc", free_full_grad, core0, core1))


@dataclass
class EnergyDenseGradientAccumulator:
    quadratic: EnergyDenseQuadratic
    tensor_grad: torch.Tensor

    @classmethod
    def create(cls, quadratic: EnergyDenseQuadratic, *, device: torch.device, dtype: torch.dtype) -> "EnergyDenseGradientAccumulator":
        return cls(
            quadratic=quadratic,
            tensor_grad=torch.zeros(
                quadratic.latent_dim,
                quadratic.latent_dim,
                quadratic.latent_dim,
                device=device,
                dtype=dtype,
            ),
        )

    def add(self, grad_c: torch.Tensor) -> None:
        self.tensor_grad.add_(grad_c.detach().to(device=self.tensor_grad.device, dtype=self.tensor_grad.dtype))

    def flush_to(self, param_grads: dict[str, torch.Tensor]) -> None:
        quad = self.quadratic
        grad_c = self.tensor_grad
        coeff = quad.reconstruct_coeff.to(device=grad_c.device, dtype=grad_c.dtype)
        full_grad_flat = grad_c.reshape(-1)
        free_grad = grad_c.new_zeros(quad.free_dim)
        free_grad.index_add_(0, quad.reconstruct_source, coeff * full_grad_flat[quad.reconstruct_target])
        _add_param_grad(param_grads, "quadratic.free_values", free_grad)


def exact_frozen_step(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    ell: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None = None,
) -> torch.Tensor:
    tau = 0.5 * float(h)
    rhs = u + tau * (dynamics.linear(u) + dynamics.quadratic.frozen_action(ell, u)) + float(h) * dynamics.source(p_mid, like=u)
    return exact_frozen_solve(dynamics, ell, rhs, tau, transpose=False)


def exact_frozen_step_adjoint(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    ell: torch.Tensor,
    u_next: torch.Tensor,
    lambda_next: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None = None,
) -> FrozenStepAdjointResult:
    """Exact adjoint of one frozen-lag step using dense GPU linear solves.

    This differentiates the frozen system with respect to u, A, H and source
    parameters. The lag ell is treated as frozen, matching the frozen-lag
    discrete adjoint used as the first correctness target.
    """

    tau = 0.5 * float(h)
    with torch.enable_grad():
        named = [(name, p) for name, p in dynamics.named_parameters() if p.requires_grad]
        params = [p for _, p in named]
        u_req = u.detach().requires_grad_(True)
        p_req = None if p_mid is None else p_mid.detach().requires_grad_(True)
        ell_det = ell.detach()
        y = exact_frozen_step(dynamics, u_req, ell_det, h, p_req)
        scalar = (y * lambda_next.detach()).sum()
        grad_values = torch.autograd.grad(scalar, [u_req] + ([] if p_req is None else [p_req]) + params, allow_unused=True)
    lambda_u = grad_values[0].detach()
    offset = 1
    lambda_p = None
    if p_req is not None:
        lambda_p = grad_values[1].detach()
        offset = 2
    param_grads = {}
    for (name, param), grad in zip(named, grad_values[offset:]):
        param_grads[name] = torch.zeros_like(param) if grad is None else grad.detach()
    mu = exact_frozen_solve(dynamics, ell.detach(), lambda_next.detach(), tau, transpose=True)
    return FrozenStepAdjointResult(mu=mu, lambda_u=lambda_u, lambda_p=lambda_p, parameter_grads=param_grads)


def _identity_batch(batch: int, r: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(r, device=device, dtype=dtype).expand(batch, r, r)


def _quadratic_cache_context(dynamics: QuadraticDynamics):
    cached = getattr(dynamics.quadratic, "cached_reduced_tensor", None)
    if cached is None:
        return nullcontext()
    return cached()


def _lagged_picard_update(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    y_prev: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None = None,
) -> torch.Tensor:
    tau = 0.5 * float(h)
    ell = 0.5 * (u + y_prev)
    force = dynamics.source(p_mid, like=u)
    rhs = u + tau * (dynamics.linear(u) + dynamics.quadratic.frozen_action(ell, u)) + float(h) * force
    if not hasattr(dynamics.quadratic, "smw_frame"):
        return exact_frozen_solve(dynamics, ell, rhs, tau, transpose=False)
    return exact_frozen_smw_solve(dynamics, ell, rhs, tau, transpose=False)


def lagged_midpoint_step_with_history(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None = None,
    *,
    picard_iters: int = 2,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Evaluate one finite-Picard lagged midpoint step and keep Picard states."""

    y = predictor(dynamics, u, h, p_mid)
    iterates = [y]
    for _ in range(int(picard_iters)):
        y = _lagged_picard_update(dynamics, u, y, h, p_mid)
        iterates.append(y)
    return y, tuple(iterates)


def _empty_param_grads(
    dynamics: QuadraticDynamics,
    *,
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(param)
        for name, param in dynamics.named_parameters()
        if param.requires_grad and not name.startswith(skip_prefixes)
    }


def _empty_param_grads_batched(
    dynamics: QuadraticDynamics,
    direction_count: int,
    *,
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(
            int(direction_count),
            *param.shape,
            device=param.device,
            dtype=param.dtype,
        )
        for name, param in dynamics.named_parameters()
        if param.requires_grad and not name.startswith(skip_prefixes)
    }


def _accumulate_param_grads(
    target: dict[str, torch.Tensor],
    named_params: list[tuple[str, torch.nn.Parameter]],
    grad_values: tuple[torch.Tensor | None, ...],
) -> None:
    for (name, param), grad in zip(named_params, grad_values):
        if grad is not None:
            target[name] = target[name] + grad.detach().to(device=param.device, dtype=param.dtype)


def _rhs_vjp(
    dynamics: QuadraticDynamics,
    x: torch.Tensor,
    p_mid: torch.Tensor | None,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    return_input_adjoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Manual VJP of ``f(x, p; theta)=A x + H(x,x) + Bp + c``.

    This is the GOATTM-style path used by the RK4 discrete adjoint.  It does
    not call PyTorch autograd; each dynamics block contributes its own
    Jacobian-transpose action and parameter-gradient assembly.
    """

    x_det = x.detach()
    cot_det = cot.detach()
    p_det = None if p_mid is None else p_mid.detach()
    lambda_x = torch.zeros_like(x_det)

    linear_grad = _linear_vjp(
        dynamics,
        x_det,
        cot_det,
        param_grads,
        include_x_grad=True,
    )
    if linear_grad is not None:
        lambda_x = lambda_x + linear_grad

    ell_grad, x_grad = _quadratic_frozen_vjp(
        dynamics,
        x_det,
        x_det,
        cot_det,
        param_grads,
        include_x_grad=True,
    )
    lambda_x = lambda_x + ell_grad
    if x_grad is not None:
        lambda_x = lambda_x + x_grad

    lambda_p = _source_vjp(
        dynamics,
        p_det,
        cot_det,
        1.0,
        param_grads,
        return_input_grad=return_input_adjoint,
    )
    return lambda_x.detach(), None if lambda_p is None else lambda_p.detach()


def runge_kutta4_step_with_stages(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Evaluate one RK4 step and return the four stage derivatives."""

    h_float = float(h)
    k1 = dynamics.rhs(u, p_mid)
    k2 = dynamics.rhs(u + 0.5 * h_float * k1, p_mid)
    k3 = dynamics.rhs(u + 0.5 * h_float * k2, p_mid)
    k4 = dynamics.rhs(u + h_float * k3, p_mid)
    u_next = u + (h_float / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return u_next, (k1, k2, k3, k4)


def runge_kutta4_step_adjoint(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    lambda_next: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None = None,
    *,
    stages: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_input_adjoint: bool = True,
) -> RungeKutta4StepAdjointResult:
    """Exact discrete adjoint of one classical RK4 step.

    The reverse sweep differentiates the four-stage RK4 map exactly.  Each
    stage RHS uses a local VJP, so the implementation supports every dynamics
    parametrization while avoiding a full rollout autograd graph.
    """

    h_float = float(h)
    u_det = u.detach()
    p_det = None if p_mid is None else p_mid.detach()
    with torch.no_grad():
        if stages is None:
            u_next, stages = runge_kutta4_step_with_stages(dynamics, u_det, h_float, p_det)
        else:
            stages = tuple(stage.detach() for stage in stages)
            k1, k2, k3, k4 = stages
            u_next = u_det + (h_float / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    k1, k2, k3, k4 = stages

    param_grads = _empty_param_grads(dynamics)
    lambda_u = lambda_next.detach().clone()
    lambda_p = None if p_det is None or not return_input_adjoint else torch.zeros_like(p_det)

    adj_k1 = (h_float / 6.0) * lambda_next.detach()
    adj_k2 = (h_float / 3.0) * lambda_next.detach()
    adj_k3 = (h_float / 3.0) * lambda_next.detach()
    adj_k4 = (h_float / 6.0) * lambda_next.detach()

    x4 = u_det + h_float * k3
    grad_x4, grad_p = _rhs_vjp(dynamics, x4, p_det, adj_k4, param_grads, return_input_adjoint=return_input_adjoint)
    lambda_u = lambda_u + grad_x4
    adj_k3 = adj_k3 + h_float * grad_x4
    if lambda_p is not None and grad_p is not None:
        lambda_p = lambda_p + grad_p

    x3 = u_det + 0.5 * h_float * k2
    grad_x3, grad_p = _rhs_vjp(dynamics, x3, p_det, adj_k3, param_grads, return_input_adjoint=return_input_adjoint)
    lambda_u = lambda_u + grad_x3
    adj_k2 = adj_k2 + 0.5 * h_float * grad_x3
    if lambda_p is not None and grad_p is not None:
        lambda_p = lambda_p + grad_p

    x2 = u_det + 0.5 * h_float * k1
    grad_x2, grad_p = _rhs_vjp(dynamics, x2, p_det, adj_k2, param_grads, return_input_adjoint=return_input_adjoint)
    lambda_u = lambda_u + grad_x2
    adj_k1 = adj_k1 + 0.5 * h_float * grad_x2
    if lambda_p is not None and grad_p is not None:
        lambda_p = lambda_p + grad_p

    grad_x1, grad_p = _rhs_vjp(dynamics, u_det, p_det, adj_k1, param_grads, return_input_adjoint=return_input_adjoint)
    lambda_u = lambda_u + grad_x1
    if lambda_p is not None and grad_p is not None:
        lambda_p = lambda_p + grad_p

    return RungeKutta4StepAdjointResult(
        u_next=u_next.detach(),
        lambda_u=lambda_u.detach(),
        lambda_p=None if lambda_p is None else lambda_p.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
    )


def runge_kutta4_rollout_adjoint(
    dynamics: QuadraticDynamics,
    u0: torch.Tensor,
    h: float,
    state_cotangents: torch.Tensor,
    p_mid: torch.Tensor | None = None,
    *,
    states: torch.Tensor | None = None,
    return_input_adjoint: bool = True,
) -> RungeKutta4RolloutAdjointResult:
    """Reverse a classical RK4 rollout.

    ``state_cotangents[n]`` is the direct objective cotangent with respect to
    ``u_n``.  The returned parameter gradients match differentiating the RK4
    rollout itself, but the reverse sweep only builds a one-step local graph.
    """

    steps = int(state_cotangents.shape[0] - 1)
    if steps < 0:
        raise ValueError("state_cotangents must contain at least u0")
    if p_mid is not None and p_mid.shape[0] != steps:
        raise ValueError("p_mid must have one entry per step")

    if states is None:
        state_list = [u0.detach()]
        u = u0.detach()
        with torch.no_grad():
            for n in range(steps):
                p_n = None if p_mid is None else p_mid[n].detach()
                u, _ = runge_kutta4_step_with_stages(dynamics, u, h, p_n)
                state_list.append(u.detach())
        states_tensor = torch.stack(state_list, dim=0)
    else:
        if states.shape != state_cotangents.shape:
            raise ValueError("states must match state_cotangents shape")
        states_tensor = states.detach()

    param_grads = _empty_param_grads(dynamics)
    lambda_p_mid = None if p_mid is None or not return_input_adjoint else torch.zeros_like(p_mid)
    lambda_state = state_cotangents[-1].detach()
    for n in reversed(range(steps)):
        p_n = None if p_mid is None else p_mid[n].detach()
        step_adj = runge_kutta4_step_adjoint(
            dynamics,
            states_tensor[n],
            lambda_state,
            h,
            p_n,
            return_input_adjoint=return_input_adjoint,
        )
        for name, grad in step_adj.parameter_grads.items():
            param_grads[name] = param_grads[name] + grad
        if lambda_p_mid is not None and step_adj.lambda_p is not None:
            lambda_p_mid[n] = step_adj.lambda_p
        lambda_state = step_adj.lambda_u + state_cotangents[n].detach()

    return RungeKutta4RolloutAdjointResult(
        states=states_tensor,
        lambda_u0=lambda_state.detach(),
        lambda_p_mid=None if lambda_p_mid is None else lambda_p_mid.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
    )


def _substep_input(p_mid: torch.Tensor | None, n: int, s: int) -> torch.Tensor | None:
    if p_mid is None:
        return None
    if p_mid.ndim >= 4:
        return p_mid[n, s]
    return p_mid[n]


def runge_kutta4_substep_rollout_adjoint(
    dynamics: QuadraticDynamics,
    u0: torch.Tensor,
    h: float,
    state_cotangents: torch.Tensor,
    p_mid: torch.Tensor | None = None,
    *,
    states: torch.Tensor | None = None,
    substeps: int = 10,
    return_input_adjoint: bool = True,
) -> RungeKutta4RolloutAdjointResult:
    """Reverse an RK4 rollout with substeps inside each observation interval."""

    substeps = int(substeps)
    if substeps <= 0:
        raise ValueError("substeps must be positive")
    steps = int(state_cotangents.shape[0] - 1)
    if steps < 0:
        raise ValueError("state_cotangents must contain at least u0")
    if p_mid is not None and p_mid.shape[0] != steps:
        raise ValueError("p_mid must have one entry per observation interval")

    if states is None:
        state_list = [u0.detach()]
        u = u0.detach()
        sub_h = float(h) / float(substeps)
        with torch.no_grad():
            for n in range(steps):
                for s in range(substeps):
                    u, _ = runge_kutta4_step_with_stages(dynamics, u, sub_h, _substep_input(p_mid, n, s))
                state_list.append(u.detach())
        states_tensor = torch.stack(state_list, dim=0)
    else:
        if states.shape != state_cotangents.shape:
            raise ValueError("states must match state_cotangents shape")
        states_tensor = states.detach()

    param_grads = _empty_param_grads(dynamics)
    lambda_p_mid = None if p_mid is None or not return_input_adjoint else torch.zeros_like(p_mid)
    lambda_state = state_cotangents[-1].detach()
    sub_h = float(h) / float(substeps)

    for n in reversed(range(steps)):
        sub_states = [states_tensor[n]]
        u = states_tensor[n]
        with torch.no_grad():
            for s in range(substeps):
                u, _ = runge_kutta4_step_with_stages(dynamics, u, sub_h, _substep_input(p_mid, n, s))
                sub_states.append(u.detach())
        lambda_sub = lambda_state
        for s in reversed(range(substeps)):
            step_adj = runge_kutta4_step_adjoint(
                dynamics,
                sub_states[s],
                lambda_sub,
                sub_h,
                _substep_input(p_mid, n, s),
                return_input_adjoint=return_input_adjoint,
            )
            for name, grad in step_adj.parameter_grads.items():
                param_grads[name] = param_grads[name] + grad
            if lambda_p_mid is not None and step_adj.lambda_p is not None:
                if lambda_p_mid.ndim >= 4:
                    lambda_p_mid[n, s] = step_adj.lambda_p
                else:
                    lambda_p_mid[n] = lambda_p_mid[n] + step_adj.lambda_p
            lambda_sub = step_adj.lambda_u
        lambda_state = lambda_sub + state_cotangents[n].detach()

    return RungeKutta4RolloutAdjointResult(
        states=states_tensor,
        lambda_u0=lambda_state.detach(),
        lambda_p_mid=None if lambda_p_mid is None else lambda_p_mid.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
    )


def _supports_manual_lagged_adjoint(dynamics: QuadraticDynamics) -> bool:
    return (
        isinstance(dynamics.linear, (DenseLinearA, DissipativeSkewA))
        and isinstance(dynamics.quadratic, (SkewCPQuadratic, EnergyTuckerTTQuadratic, DenseQuadratic, EnergyDenseQuadratic))
        and isinstance(dynamics.source, (LinearSource, ZeroSource))
    )


def _add_param_grad(target: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    if name in target:
        target[name] = target[name] + value.to(device=target[name].device, dtype=target[name].dtype)


def _frozen_transpose_action(dynamics: QuadraticDynamics, ell: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return dynamics.linear.apply_transpose(v) - dynamics.quadratic.frozen_action(ell, v)


def _source_vjp(
    dynamics: QuadraticDynamics,
    p_mid: torch.Tensor | None,
    cot: torch.Tensor,
    scale: float,
    param_grads: dict[str, torch.Tensor],
    *,
    return_input_grad: bool = True,
) -> torch.Tensor | None:
    if isinstance(dynamics.source, ZeroSource) or p_mid is None:
        if isinstance(dynamics.source, LinearSource) and dynamics.source.c is not None:
            _add_param_grad(param_grads, "source.c", float(scale) * cot.sum(dim=0))
        return None
    if not isinstance(dynamics.source, LinearSource):
        return None
    _add_param_grad(param_grads, "source.B", float(scale) * cot.T @ p_mid)
    if dynamics.source.c is not None:
        _add_param_grad(param_grads, "source.c", float(scale) * cot.sum(dim=0))
    if not return_input_grad:
        return None
    return float(scale) * (cot @ dynamics.source.B.to(device=cot.device, dtype=cot.dtype))


def _skewcp_frozen_vjp(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    quad = dynamics.quadratic
    u_fac = quad.U.to(device=ell.device, dtype=ell.dtype)
    v_fac = quad.V.to(device=ell.device, dtype=ell.dtype)
    w_fac = quad.W.to(device=ell.device, dtype=ell.dtype)

    gamma = ell @ w_fac
    alpha = x @ u_fac
    beta = x @ v_fac
    cot_u = cot @ u_fac
    cot_v = cot @ v_fac

    grad_gamma = beta * cot_u - alpha * cot_v
    grad_beta = gamma * cot_u
    grad_alpha = -gamma * cot_v

    ell_grad = grad_gamma @ w_fac.T
    x_grad = None
    if include_x_grad:
        x_grad = grad_beta @ v_fac.T + grad_alpha @ u_fac.T

    _add_param_grad(param_grads, "quadratic.U", cot.T @ (gamma * beta) + x.T @ grad_alpha)
    _add_param_grad(param_grads, "quadratic.V", -(cot.T @ (gamma * alpha)) + x.T @ grad_beta)
    _add_param_grad(param_grads, "quadratic.W", ell.T @ grad_gamma)
    return ell_grad, x_grad


def _dense_frozen_vjp(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    quad = dynamics.quadratic
    if not isinstance(quad, DenseQuadratic):
        raise TypeError("expected DenseQuadratic")
    c = quad.tensor.to(device=ell.device, dtype=ell.dtype)
    ell_grad = torch.einsum("na,abc,nc->nb", cot, c, x)
    x_grad = torch.einsum("na,abc,nb->nc", cot, c, ell) if include_x_grad else None
    _add_param_grad(param_grads, "quadratic.tensor", torch.einsum("na,nb,nc->abc", cot, ell, x))
    return ell_grad, x_grad


def _energy_dense_frozen_vjp(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
    accumulator: EnergyDenseGradientAccumulator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyDenseQuadratic):
        raise TypeError("expected EnergyDenseQuadratic")
    c = quad.reduced_tensor().to(device=ell.device, dtype=ell.dtype)
    ell_grad = torch.einsum("na,abc,nc->nb", cot, c, x)
    x_grad = torch.einsum("na,abc,nb->nc", cot, c, ell) if include_x_grad else None
    grad_c = torch.einsum("na,nb,nc->abc", cot, ell, x)
    if accumulator is not None:
        accumulator.add(grad_c)
        return ell_grad, x_grad
    coeff = quad.reconstruct_coeff.to(device=grad_c.device, dtype=grad_c.dtype)
    full_grad_flat = grad_c.reshape(-1)
    free_grad = grad_c.new_zeros(quad.free_dim)
    free_grad.index_add_(0, quad.reconstruct_source, coeff * full_grad_flat[quad.reconstruct_target])
    _add_param_grad(param_grads, "quadratic.free_values", free_grad)
    return ell_grad, x_grad


def _energy_tuckertt_frozen_vjp(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
    accumulator: EnergyTuckerTTGradientAccumulator | EnergyDenseGradientAccumulator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyTuckerTTQuadratic):
        raise TypeError("expected EnergyTuckerTTQuadratic")
    p = quad.basis.to(device=ell.device, dtype=ell.dtype)
    c = quad.reduced_tensor().to(device=ell.device, dtype=ell.dtype)
    z_lag = ell @ p.T
    z = x @ p.T
    reduced_out = torch.einsum("abc,nb,nc->na", c, z_lag, z)
    omega = cot @ p.T

    grad_c = torch.einsum("na,nb,nc->abc", omega, z_lag, z)
    grad_z_lag = torch.einsum("na,abc,nc->nb", omega, c, z)
    grad_z = torch.einsum("na,abc,nb->nc", omega, c, z_lag)
    ell_grad = grad_z_lag @ p
    x_grad = grad_z @ p if include_x_grad else None

    grad_basis = reduced_out.T @ cot + grad_z_lag.T @ ell
    if include_x_grad:
        grad_basis = grad_basis + grad_z.T @ x
    else:
        # Even when x is a fixed primal vector, the basis in z=x P^T is still trainable.
        grad_basis = grad_basis + grad_z.T @ x
    if accumulator is not None:
        accumulator.add(grad_basis, grad_c)
        return ell_grad, x_grad

    _add_param_grad(param_grads, "quadratic.basis", grad_basis)

    coeff = quad.reconstruct_coeff.to(device=grad_c.device, dtype=grad_c.dtype)
    full_grad_flat = grad_c.reshape(-1)
    free_grad = grad_c.new_zeros(quad.free_dim)
    free_grad.index_add_(0, quad.reconstruct_source, coeff * full_grad_flat[quad.reconstruct_target])
    free_full_grad = grad_c.new_zeros(quad.reduced_rank, quad.reduced_rank, quad.reduced_rank)
    free_full_grad[quad.free_a, quad.free_b, quad.free_c] = free_grad

    core0 = quad.core0.to(device=grad_c.device, dtype=grad_c.dtype)
    core1 = quad.core1.to(device=grad_c.device, dtype=grad_c.dtype)
    core2 = quad.core2.to(device=grad_c.device, dtype=grad_c.dtype)
    _add_param_grad(param_grads, "quadratic.core0", torch.einsum("abc,rbs,sc->ar", free_full_grad, core1, core2))
    _add_param_grad(param_grads, "quadratic.core1", torch.einsum("abc,ar,sc->rbs", free_full_grad, core0, core2))
    _add_param_grad(param_grads, "quadratic.core2", torch.einsum("abc,ar,rbs->sc", free_full_grad, core0, core1))
    return ell_grad, x_grad


def _quadratic_frozen_vjp(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
    accumulator: EnergyTuckerTTGradientAccumulator | EnergyDenseGradientAccumulator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(dynamics.quadratic, SkewCPQuadratic):
        return _skewcp_frozen_vjp(dynamics, ell, x, cot, param_grads, include_x_grad=include_x_grad)
    if isinstance(dynamics.quadratic, DenseQuadratic):
        return _dense_frozen_vjp(dynamics, ell, x, cot, param_grads, include_x_grad=include_x_grad)
    if isinstance(dynamics.quadratic, EnergyDenseQuadratic):
        return _energy_dense_frozen_vjp(
            dynamics,
            ell,
            x,
            cot,
            param_grads,
            include_x_grad=include_x_grad,
            accumulator=accumulator if isinstance(accumulator, EnergyDenseGradientAccumulator) else None,
        )
    if isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic):
        return _energy_tuckertt_frozen_vjp(
            dynamics,
            ell,
            x,
            cot,
            param_grads,
            include_x_grad=include_x_grad,
            accumulator=accumulator,
        )
    raise TypeError(f"unsupported quadratic type {type(dynamics.quadratic)!r}")


def _linear_vjp(
    dynamics: QuadraticDynamics,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
) -> torch.Tensor | None:
    linear = dynamics.linear
    x_grad = linear.apply_transpose(cot) if include_x_grad else None
    if isinstance(linear, DenseLinearA):
        _add_param_grad(param_grads, "linear.A", cot.T @ x)
        return x_grad
    if isinstance(linear, DissipativeSkewA):
        diagonal_cot = (cot * x).sum(dim=0)
        _add_param_grad(param_grads, "linear.raw_damping", -2.0 * linear.raw_damping * diagonal_cot)
        if linear.skew_rank > 0:
            p = linear.P.to(device=x.device, dtype=x.dtype)
            q = linear.Q.to(device=x.device, dtype=x.dtype)
            _add_param_grad(param_grads, "linear.P", cot.T @ (x @ q) - x.T @ (cot @ q))
            _add_param_grad(param_grads, "linear.Q", x.T @ (cot @ p) - cot.T @ (x @ p))
        return x_grad
    raise TypeError(f"unsupported linear type {type(linear)!r}")


def _exact_frozen_solve_batched(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    rhs: torch.Tensor,
    tau: float,
    *,
    transpose: bool,
    dense_a: torch.Tensor | None = None,
) -> torch.Tensor:
    sample_count, direction_count, latent_dim = rhs.shape
    ell_flat = ell[:, None, :].expand(sample_count, direction_count, latent_dim).reshape(sample_count * direction_count, latent_dim)
    rhs_flat = rhs.reshape(sample_count * direction_count, latent_dim)
    solved = exact_frozen_solve(
        dynamics,
        ell_flat,
        rhs_flat,
        tau,
        transpose=transpose,
        dense_a=dense_a,
        eye=None,
    )
    return solved.reshape(sample_count, direction_count, latent_dim)


def _linear_vjp_batched(
    dynamics: QuadraticDynamics,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
) -> torch.Tensor | None:
    linear = dynamics.linear
    sample_count, direction_count, latent_dim = cot.shape
    if include_x_grad:
        x_grad = linear.apply_transpose(cot.reshape(sample_count * direction_count, latent_dim)).reshape(
            sample_count,
            direction_count,
            latent_dim,
        )
    else:
        x_grad = None
    if isinstance(linear, DenseLinearA):
        _add_param_grad(param_grads, "linear.A", torch.einsum("mka,mb->kab", cot, x))
        return x_grad
    if isinstance(linear, DissipativeSkewA):
        diagonal_cot = torch.einsum("mka,ma->ka", cot, x)
        raw = linear.raw_damping.to(device=x.device, dtype=x.dtype)
        _add_param_grad(param_grads, "linear.raw_damping", -2.0 * raw[None, :] * diagonal_cot)
        if linear.skew_rank > 0:
            p = linear.P.to(device=x.device, dtype=x.dtype)
            q = linear.Q.to(device=x.device, dtype=x.dtype)
            xq = x @ q
            xp = x @ p
            cotq = torch.einsum("mka,as->mks", cot, q)
            cotp = torch.einsum("mka,as->mks", cot, p)
            _add_param_grad(param_grads, "linear.P", torch.einsum("mka,ms->kas", cot, xq) - torch.einsum("ma,mks->kas", x, cotq))
            _add_param_grad(param_grads, "linear.Q", torch.einsum("ma,mks->kas", x, cotp) - torch.einsum("mka,ms->kas", cot, xp))
        return x_grad
    raise TypeError(f"unsupported linear type {type(linear)!r}")


def _source_vjp_batched(
    dynamics: QuadraticDynamics,
    p_mid: torch.Tensor | None,
    cot: torch.Tensor,
    scale: float,
    param_grads: dict[str, torch.Tensor],
    *,
    return_input_grad: bool = True,
) -> torch.Tensor | None:
    if isinstance(dynamics.source, ZeroSource) or p_mid is None:
        if isinstance(dynamics.source, LinearSource) and dynamics.source.c is not None:
            _add_param_grad(param_grads, "source.c", float(scale) * cot.sum(dim=0))
        return None
    if not isinstance(dynamics.source, LinearSource):
        return None
    _add_param_grad(param_grads, "source.B", float(scale) * torch.einsum("mka,mi->kai", cot, p_mid))
    if dynamics.source.c is not None:
        _add_param_grad(param_grads, "source.c", float(scale) * cot.sum(dim=0))
    if not return_input_grad:
        return None
    b = dynamics.source.B.to(device=cot.device, dtype=cot.dtype)
    return float(scale) * torch.einsum("mka,ai->mki", cot, b)


def _energy_tuckertt_frozen_vjp_batched(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
    accumulator: BatchedEnergyTuckerTTGradientAccumulator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyTuckerTTQuadratic):
        raise TypeError("expected EnergyTuckerTTQuadratic")
    p = quad.basis.to(device=ell.device, dtype=ell.dtype)
    c = quad.reduced_tensor().to(device=ell.device, dtype=ell.dtype)
    z_lag = ell @ p.T
    z = x @ p.T
    reduced_out = torch.einsum("abc,mb,mc->ma", c, z_lag, z)
    omega = torch.einsum("mka,ba->mkb", cot, p)

    grad_c = torch.einsum("mka,mb,mc->kabc", omega, z_lag, z)
    grad_z_lag = torch.einsum("mka,abc,mc->mkb", omega, c, z)
    grad_z = torch.einsum("mka,abc,mb->mkc", omega, c, z_lag)
    ell_grad = torch.einsum("mkb,ba->mka", grad_z_lag, p)
    x_grad = torch.einsum("mkb,ba->mka", grad_z, p) if include_x_grad else None

    grad_basis = torch.einsum("mb,mka->kba", reduced_out, cot)
    grad_basis = grad_basis + torch.einsum("mkb,ma->kba", grad_z_lag, ell)
    grad_basis = grad_basis + torch.einsum("mkb,ma->kba", grad_z, x)
    if accumulator is not None:
        accumulator.add(grad_basis, grad_c)
        return ell_grad, x_grad

    _add_param_grad(param_grads, "quadratic.basis", grad_basis)
    temp_acc = BatchedEnergyTuckerTTGradientAccumulator.create(
        quad,
        int(cot.shape[1]),
        device=cot.device,
        dtype=cot.dtype,
    )
    temp_acc.add(torch.zeros_like(grad_basis), grad_c)
    temp_acc.flush_to(param_grads)
    return ell_grad, x_grad


def _quadratic_frozen_vjp_batched(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    cot: torch.Tensor,
    param_grads: dict[str, torch.Tensor],
    *,
    include_x_grad: bool,
    accumulator: BatchedEnergyTuckerTTGradientAccumulator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic):
        return _energy_tuckertt_frozen_vjp_batched(
            dynamics,
            ell,
            x,
            cot,
            param_grads,
            include_x_grad=include_x_grad,
            accumulator=accumulator,
        )
    raise TypeError(f"batched lagged adjoint does not support quadratic type {type(dynamics.quadratic)!r}")


def _lagged_midpoint_step_adjoint_manual_batched(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    lambda_next: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None,
    *,
    picard_iters: int,
    picard_iterates: tuple[torch.Tensor, ...],
    return_input_adjoint: bool = True,
    dense_a: torch.Tensor | None = None,
    energy_accumulator: BatchedEnergyTuckerTTGradientAccumulator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    tau = 0.5 * float(h)
    direction_count = int(lambda_next.shape[1])
    skip_prefixes = ("quadratic.basis", "quadratic.core") if isinstance(energy_accumulator, BatchedEnergyTuckerTTGradientAccumulator) else ()
    param_grads = _empty_param_grads_batched(dynamics, direction_count, skip_prefixes=skip_prefixes)
    lambda_u = torch.zeros_like(lambda_next)
    lambda_p = None
    if p_mid is not None and return_input_adjoint:
        lambda_p = torch.zeros(
            p_mid.shape[0],
            direction_count,
            p_mid.shape[1],
            device=p_mid.device,
            dtype=p_mid.dtype,
        )
    adj_y = lambda_next.detach()

    for k in reversed(range(int(picard_iters))):
        y_prev = picard_iterates[k].detach()
        y_next = picard_iterates[k + 1].detach()
        ell = 0.5 * (u + y_prev)
        with torch.no_grad():
            mu = _exact_frozen_solve_batched(
                dynamics,
                ell,
                adj_y,
                tau,
                transpose=True,
                dense_a=dense_a,
            )

        lambda_u = lambda_u + mu
        linear_x_grad = _linear_vjp_batched(
            dynamics,
            u + y_next,
            tau * mu,
            param_grads,
            include_x_grad=True,
        )
        if linear_x_grad is not None:
            lambda_u = lambda_u + linear_x_grad
        p_grad = _source_vjp_batched(
            dynamics,
            p_mid,
            mu,
            float(h),
            param_grads,
            return_input_grad=lambda_p is not None,
        )
        if lambda_p is not None and p_grad is not None:
            lambda_p = lambda_p + p_grad

        ell_grad, x_grad = _quadratic_frozen_vjp_batched(
            dynamics,
            ell,
            u + y_next,
            tau * mu,
            param_grads,
            include_x_grad=True,
            accumulator=energy_accumulator,
        )
        lambda_u = lambda_u + 0.5 * ell_grad
        if x_grad is not None:
            lambda_u = lambda_u + x_grad
        adj_y = 0.5 * ell_grad

    predictor_cot = adj_y
    lambda_u = lambda_u + predictor_cot
    linear_x_grad = _linear_vjp_batched(
        dynamics,
        u,
        float(h) * predictor_cot,
        param_grads,
        include_x_grad=True,
    )
    if linear_x_grad is not None:
        lambda_u = lambda_u + linear_x_grad
    p_grad = _source_vjp_batched(
        dynamics,
        p_mid,
        predictor_cot,
        float(h),
        param_grads,
        return_input_grad=lambda_p is not None,
    )
    if lambda_p is not None and p_grad is not None:
        lambda_p = lambda_p + p_grad
    ell_grad, x_grad = _quadratic_frozen_vjp_batched(
        dynamics,
        u,
        u,
        float(h) * predictor_cot,
        param_grads,
        include_x_grad=True,
        accumulator=energy_accumulator,
    )
    lambda_u = lambda_u + ell_grad
    if x_grad is not None:
        lambda_u = lambda_u + x_grad

    return lambda_u.detach(), None if lambda_p is None else lambda_p.detach(), {name: grad.detach() for name, grad in param_grads.items()}


def _lagged_midpoint_step_adjoint_manual(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    lambda_next: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None,
    *,
    picard_iters: int,
    picard_iterates: tuple[torch.Tensor, ...],
    return_input_adjoint: bool = True,
    transpose_cache=None,
    dense_a: torch.Tensor | None = None,
    solve_eye: torch.Tensor | None = None,
    energy_accumulator: EnergyTuckerTTGradientAccumulator | EnergyDenseGradientAccumulator | None = None,
) -> LaggedMidpointStepAdjointResult:
    tau = 0.5 * float(h)
    if isinstance(energy_accumulator, EnergyTuckerTTGradientAccumulator):
        skip_prefixes = ("quadratic.basis", "quadratic.core")
    elif isinstance(energy_accumulator, EnergyDenseGradientAccumulator):
        skip_prefixes = ("quadratic.free_values",)
    else:
        skip_prefixes = ()
    param_grads = _empty_param_grads(dynamics, skip_prefixes=skip_prefixes)
    lambda_u = torch.zeros_like(u)
    lambda_p = None if p_mid is None or not return_input_adjoint else torch.zeros_like(p_mid)
    adj_y = lambda_next.detach()
    if transpose_cache is None:
        if isinstance(dynamics.quadratic, SkewCPQuadratic):
            transpose_cache = make_frozen_smw_cache(dynamics, tau, device=u.device, dtype=u.dtype, transpose=True)

    for k in reversed(range(int(picard_iters))):
        y_prev = picard_iterates[k].detach()
        y_next = picard_iterates[k + 1].detach()
        ell = 0.5 * (u + y_prev)
        if isinstance(dynamics.quadratic, SkewCPQuadratic):
            mu = exact_frozen_smw_solve(dynamics, ell, adj_y, tau, transpose=True, cache=transpose_cache)
        else:
            with torch.no_grad():
                mu = exact_frozen_solve(
                    dynamics,
                    ell,
                    adj_y,
                    tau,
                    transpose=True,
                    dense_a=dense_a,
                    eye=solve_eye,
                )

        lambda_u = lambda_u + mu
        linear_x_grad = _linear_vjp(
            dynamics,
            u + y_next,
            tau * mu,
            param_grads,
            include_x_grad=True,
        )
        if linear_x_grad is not None:
            lambda_u = lambda_u + linear_x_grad
        p_grad = _source_vjp(
            dynamics,
            p_mid,
            mu,
            float(h),
            param_grads,
            return_input_grad=lambda_p is not None,
        )
        if lambda_p is not None and p_grad is not None:
            lambda_p = lambda_p + p_grad

        ell_grad, x_grad = _quadratic_frozen_vjp(
            dynamics,
            ell,
            u + y_next,
            tau * mu,
            param_grads,
            include_x_grad=True,
            accumulator=energy_accumulator,
        )
        lambda_u = lambda_u + 0.5 * ell_grad
        if x_grad is not None:
            lambda_u = lambda_u + x_grad
        adj_y = 0.5 * ell_grad

    predictor_cot = adj_y
    lambda_u = lambda_u + predictor_cot
    linear_x_grad = _linear_vjp(
        dynamics,
        u,
        float(h) * predictor_cot,
        param_grads,
        include_x_grad=True,
    )
    if linear_x_grad is not None:
        lambda_u = lambda_u + linear_x_grad
    p_grad = _source_vjp(
        dynamics,
        p_mid,
        predictor_cot,
        float(h),
        param_grads,
        return_input_grad=lambda_p is not None,
    )
    if lambda_p is not None and p_grad is not None:
        lambda_p = lambda_p + p_grad
    ell_grad, x_grad = _quadratic_frozen_vjp(
        dynamics,
        u,
        u,
        float(h) * predictor_cot,
        param_grads,
        include_x_grad=True,
        accumulator=energy_accumulator,
    )
    lambda_u = lambda_u + ell_grad
    if x_grad is not None:
        lambda_u = lambda_u + x_grad

    return LaggedMidpointStepAdjointResult(
        u_next=picard_iterates[-1].detach(),
        lambda_u=lambda_u.detach(),
        lambda_p=None if lambda_p is None else lambda_p.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
        picard_iterates=tuple(value.detach() for value in picard_iterates),
    )


def _lagged_midpoint_step_adjoint_dense_implicit(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    lambda_next: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None,
    *,
    picard_iters: int,
    picard_iterates: tuple[torch.Tensor, ...],
    return_input_adjoint: bool = True,
    dense_a: torch.Tensor | None = None,
    solve_eye: torch.Tensor | None = None,
) -> LaggedMidpointStepAdjointResult:
    """Dense finite-Picard adjoint without differentiating through solves.

    For each frozen Picard substep

        y = (I - tau L(ell))^{-1} [u + tau L(ell) u + h f],

    the reverse sweep first solves

        (I - tau L(ell))^T mu = lambda_y,

    then differentiates only

        u + tau L(ell) (u + y) + h f

    against ``mu``.  This is the same implicit discrete adjoint as the generic
    local-VJP fallback, but it avoids putting ``torch.linalg.solve`` in the
    autograd graph.
    """

    tau = 0.5 * float(h)
    named = [(name, p) for name, p in dynamics.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    param_grads = _empty_param_grads(dynamics)
    lambda_u = torch.zeros_like(u)
    lambda_p = None if p_mid is None or not return_input_adjoint else torch.zeros_like(p_mid)
    adj_y = lambda_next.detach()

    for k in reversed(range(int(picard_iters))):
        y_prev = picard_iterates[k].detach()
        y_next = picard_iterates[k + 1].detach()
        ell = 0.5 * (u.detach() + y_prev)
        with torch.no_grad():
            mu = exact_frozen_solve(
                dynamics,
                ell,
                adj_y,
                tau,
                transpose=True,
                dense_a=dense_a,
                eye=solve_eye,
            ).detach()

        with torch.enable_grad():
            u_req = u.detach().requires_grad_(True)
            y_prev_req = y_prev.requires_grad_(True)
            p_req = None if p_mid is None else p_mid.detach().requires_grad_(True)
            ell_req = 0.5 * (u_req + y_prev_req)
            x_req = u_req + y_next
            local_rhs = (
                u_req
                + tau * (dynamics.linear(x_req) + dynamics.quadratic.frozen_action(ell_req, x_req))
                + float(h) * dynamics.source(p_req, like=u_req)
            )
            scalar = (local_rhs * mu).sum()
            variables = [u_req, y_prev_req] + ([] if p_req is None else [p_req]) + params
            grad_values = torch.autograd.grad(scalar, variables, allow_unused=True)

        grad_u = grad_values[0]
        grad_y_prev = grad_values[1]
        if grad_u is not None:
            lambda_u = lambda_u + grad_u.detach()
        adj_y = torch.zeros_like(adj_y) if grad_y_prev is None else grad_y_prev.detach()
        offset = 2
        if lambda_p is not None:
            if grad_values[2] is not None:
                lambda_p = lambda_p + grad_values[2].detach()
            offset = 3
        elif p_mid is not None:
            offset = 3
        _accumulate_param_grads(param_grads, named, grad_values[offset:])

    with torch.enable_grad():
        u_req = u.detach().requires_grad_(True)
        p_req = None if p_mid is None else p_mid.detach().requires_grad_(True)
        y0 = predictor(dynamics, u_req, h, p_req)
        scalar = (y0 * adj_y).sum()
        variables = [u_req] + ([] if p_req is None else [p_req]) + params
        grad_values = torch.autograd.grad(scalar, variables, allow_unused=True)
    if grad_values[0] is not None:
        lambda_u = lambda_u + grad_values[0].detach()
    offset = 1
    if lambda_p is not None:
        if grad_values[1] is not None:
            lambda_p = lambda_p + grad_values[1].detach()
        offset = 2
    elif p_mid is not None:
        offset = 2
    _accumulate_param_grads(param_grads, named, grad_values[offset:])

    return LaggedMidpointStepAdjointResult(
        u_next=picard_iterates[-1].detach(),
        lambda_u=lambda_u.detach(),
        lambda_p=None if lambda_p is None else lambda_p.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
        picard_iterates=tuple(value.detach() for value in picard_iterates),
    )


def lagged_midpoint_step_adjoint(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    lambda_next: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None = None,
    *,
    picard_iters: int = 2,
    picard_iterates: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    return_input_adjoint: bool = True,
    transpose_cache=None,
    dense_a: torch.Tensor | None = None,
    solve_eye: torch.Tensor | None = None,
    energy_accumulator: EnergyTuckerTTGradientAccumulator | EnergyDenseGradientAccumulator | None = None,
) -> LaggedMidpointStepAdjointResult:
    """Exact discrete adjoint of the finite-Picard lagged midpoint step.

    This differentiates the actual map implemented by `DenseLaggedMidpointStepper`
    for a fixed number of Picard iterations.  It is intentionally dense and
    correctness-first: each Picard substep is reversed by a local VJP, so this is
    a reference implementation for future low-rank adjoints without saving a
    full rollout autograd graph.
    """

    named = [(name, p) for name, p in dynamics.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    if picard_iterates is None:
        with torch.no_grad():
            u_next, iterates = lagged_midpoint_step_with_history(
                dynamics,
                u.detach(),
                h,
                None if p_mid is None else p_mid.detach(),
                picard_iters=picard_iters,
            )
    else:
        if isinstance(picard_iterates, torch.Tensor):
            iterates = tuple(value.detach() for value in picard_iterates.unbind(0))
        else:
            iterates = tuple(value.detach() for value in picard_iterates)
        if len(iterates) != int(picard_iters) + 1:
            raise ValueError("picard_iterates must contain predictor plus one entry per Picard iteration")
        u_next = iterates[-1]

    if _supports_manual_lagged_adjoint(dynamics):
        return _lagged_midpoint_step_adjoint_manual(
            dynamics,
            u.detach(),
            lambda_next.detach(),
            h,
            None if p_mid is None else p_mid.detach(),
            picard_iters=picard_iters,
            picard_iterates=iterates,
            return_input_adjoint=return_input_adjoint,
            transpose_cache=transpose_cache,
            dense_a=dense_a,
            solve_eye=solve_eye,
            energy_accumulator=energy_accumulator,
        )

    return _lagged_midpoint_step_adjoint_dense_implicit(
        dynamics,
        u.detach(),
        lambda_next.detach(),
        h,
        None if p_mid is None else p_mid.detach(),
        picard_iters=picard_iters,
        picard_iterates=iterates,
        return_input_adjoint=return_input_adjoint,
        dense_a=dense_a,
        solve_eye=solve_eye,
    )

    param_grads = _empty_param_grads(dynamics)
    lambda_u = torch.zeros_like(u)
    lambda_p = None if p_mid is None or not return_input_adjoint else torch.zeros_like(p_mid)
    adj_y = lambda_next.detach()

    for k in reversed(range(int(picard_iters))):
        y_prev = iterates[k].detach()
        with torch.enable_grad():
            u_req = u.detach().requires_grad_(True)
            y_req = y_prev.requires_grad_(True)
            p_req = None if p_mid is None else p_mid.detach().requires_grad_(True)
            y = _lagged_picard_update(dynamics, u_req, y_req, h, p_req)
            scalar = (y * adj_y).sum()
            variables = [u_req, y_req] + ([] if p_req is None else [p_req]) + params
            grad_values = torch.autograd.grad(scalar, variables, allow_unused=True)
        lambda_u = lambda_u + grad_values[0].detach()
        adj_y = grad_values[1].detach()
        offset = 2
        if lambda_p is not None:
            if grad_values[2] is not None:
                lambda_p = lambda_p + grad_values[2].detach()
            offset = 3
        elif p_mid is not None:
            offset = 3
        _accumulate_param_grads(param_grads, named, grad_values[offset:])

    with torch.enable_grad():
        u_req = u.detach().requires_grad_(True)
        p_req = None if p_mid is None else p_mid.detach().requires_grad_(True)
        y0 = predictor(dynamics, u_req, h, p_req)
        scalar = (y0 * adj_y).sum()
        variables = [u_req] + ([] if p_req is None else [p_req]) + params
        grad_values = torch.autograd.grad(scalar, variables, allow_unused=True)
    lambda_u = lambda_u + grad_values[0].detach()
    offset = 1
    if lambda_p is not None:
        if grad_values[1] is not None:
            lambda_p = lambda_p + grad_values[1].detach()
        offset = 2
    elif p_mid is not None:
        offset = 2
    _accumulate_param_grads(param_grads, named, grad_values[offset:])

    return LaggedMidpointStepAdjointResult(
        u_next=u_next.detach(),
        lambda_u=lambda_u.detach(),
        lambda_p=None if lambda_p is None else lambda_p.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
        picard_iterates=tuple(value.detach() for value in iterates),
    )


def lagged_midpoint_rollout_adjoint(
    dynamics: QuadraticDynamics,
    u0: torch.Tensor,
    h: float,
    state_cotangents: torch.Tensor,
    p_mid: torch.Tensor | None = None,
    *,
    picard_iters: int = 2,
    states: torch.Tensor | None = None,
    picard_iterates: torch.Tensor | None = None,
    return_input_adjoint: bool = True,
) -> LaggedMidpointRolloutAdjointResult:
    """Reverse a finite-Picard lagged midpoint rollout.

    `state_cotangents[n]` is the contribution d Phi / d u_n from the objective.
    The returned gradients match differentiating `DenseLaggedMidpointStepper`
    with the same number of Picard iterations.
    """

    steps = int(state_cotangents.shape[0] - 1)
    if steps < 0:
        raise ValueError("state_cotangents must contain at least u0")
    if p_mid is not None and p_mid.shape[0] != steps:
        raise ValueError("p_mid must have one entry per step")
    if picard_iterates is not None and picard_iterates.shape[0] != steps:
        raise ValueError("picard_iterates must have one history per step")

    if states is None:
        state_list = [u0.detach()]
        u = u0.detach()
        with torch.no_grad():
            for n in range(steps):
                p_n = None if p_mid is None else p_mid[n].detach()
                u, _ = lagged_midpoint_step_with_history(
                    dynamics,
                    u,
                    h,
                    p_n,
                    picard_iters=picard_iters,
                )
                state_list.append(u.detach())
        states_tensor = torch.stack(state_list, dim=0)
    else:
        if states.shape != state_cotangents.shape:
            raise ValueError("states must match state_cotangents shape")
        states_tensor = states.detach()

    param_grads = _empty_param_grads(dynamics)
    lambda_p_mid = None if p_mid is None or not return_input_adjoint else torch.zeros_like(p_mid)
    lambda_state = state_cotangents[-1].detach()
    transpose_cache = None
    if hasattr(dynamics.quadratic, "smw_frame"):
        transpose_cache = make_frozen_smw_cache(
            dynamics,
            float(h) * 0.5,
            device=states_tensor.device,
            dtype=states_tensor.dtype,
            transpose=True,
        )
    dense_a = None
    solve_eye = None
    energy_accumulator = None
    if _supports_manual_lagged_adjoint(dynamics) and isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic):
        energy_accumulator = EnergyTuckerTTGradientAccumulator.create(
            dynamics.quadratic,
            device=states_tensor.device,
            dtype=states_tensor.dtype,
        )
    elif _supports_manual_lagged_adjoint(dynamics) and isinstance(dynamics.quadratic, EnergyDenseQuadratic):
        energy_accumulator = EnergyDenseGradientAccumulator.create(
            dynamics.quadratic,
            device=states_tensor.device,
            dtype=states_tensor.dtype,
        )
    if _supports_manual_lagged_adjoint(dynamics) and not isinstance(dynamics.quadratic, SkewCPQuadratic):
        dense_a = dynamics.dense_A().to(device=states_tensor.device, dtype=states_tensor.dtype)
        solve_eye = _identity_batch(states_tensor.shape[1], states_tensor.shape[2], states_tensor.device, states_tensor.dtype)

    cache_context = _quadratic_cache_context(dynamics) if _supports_manual_lagged_adjoint(dynamics) else nullcontext()
    with cache_context:
        for n in reversed(range(steps)):
            p_n = None if p_mid is None else p_mid[n].detach()
            step_adj = lagged_midpoint_step_adjoint(
                dynamics,
                states_tensor[n],
                lambda_state,
                h,
                p_n,
                picard_iters=picard_iters,
                picard_iterates=None if picard_iterates is None else picard_iterates[n],
                return_input_adjoint=return_input_adjoint,
                transpose_cache=transpose_cache,
                dense_a=dense_a,
                solve_eye=solve_eye,
                energy_accumulator=energy_accumulator,
            )
            for name, grad in step_adj.parameter_grads.items():
                param_grads[name] = param_grads[name] + grad
            if lambda_p_mid is not None and step_adj.lambda_p is not None:
                lambda_p_mid[n] = step_adj.lambda_p
            lambda_state = step_adj.lambda_u + state_cotangents[n].detach()

    if energy_accumulator is not None:
        energy_accumulator.flush_to(param_grads)

    return LaggedMidpointRolloutAdjointResult(
        states=states_tensor,
        lambda_u0=lambda_state.detach(),
        lambda_p_mid=None if lambda_p_mid is None else lambda_p_mid.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
    )


def lagged_midpoint_rollout_adjoint_batched(
    dynamics: QuadraticDynamics,
    u0: torch.Tensor,
    h: float,
    state_cotangents: torch.Tensor,
    p_mid: torch.Tensor | None = None,
    *,
    picard_iters: int = 2,
    states: torch.Tensor,
    picard_iterates: torch.Tensor,
    return_input_adjoint: bool = False,
) -> BatchedLaggedMidpointRolloutAdjointResult:
    """Batched reverse pass for several cotangent columns.

    ``state_cotangents`` has shape ``(steps + 1, sample, direction, latent)``.
    The returned parameter gradients keep the direction dimension as their
    leading axis, so flattening them columnwise gives ``J^T Z`` for all
    cotangent columns at once.

    This fast path is intentionally specialized to the current
    EnergyTuckerTT-based lagged-midpoint model.
    """

    if state_cotangents.ndim != 4:
        raise ValueError("state_cotangents must have shape time x sample x direction x latent")
    steps = int(state_cotangents.shape[0] - 1)
    if steps < 0:
        raise ValueError("state_cotangents must contain at least u0")
    if p_mid is not None and p_mid.shape[0] != steps:
        raise ValueError("p_mid must have one entry per step")
    if states.shape != (steps + 1, state_cotangents.shape[1], state_cotangents.shape[3]):
        raise ValueError("states must have shape time x sample x latent matching state_cotangents")
    if picard_iterates.shape[0] != steps:
        raise ValueError("picard_iterates must have one history per step")
    if not (
        isinstance(dynamics.linear, (DenseLinearA, DissipativeSkewA))
        and isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic)
        and isinstance(dynamics.source, (LinearSource, ZeroSource))
    ):
        raise TypeError("batched lagged adjoint currently supports EnergyTuckerTT lagged systems")

    states_tensor = states.detach()
    direction_count = int(state_cotangents.shape[2])
    param_grads = _empty_param_grads_batched(dynamics, direction_count)
    lambda_p_mid = None
    if p_mid is not None and return_input_adjoint:
        lambda_p_mid = torch.zeros(
            steps,
            state_cotangents.shape[1],
            direction_count,
            p_mid.shape[-1],
            device=p_mid.device,
            dtype=p_mid.dtype,
        )
    lambda_state = state_cotangents[-1].detach()
    dense_a = dynamics.dense_A().to(device=states_tensor.device, dtype=states_tensor.dtype)
    energy_accumulator = BatchedEnergyTuckerTTGradientAccumulator.create(
        dynamics.quadratic,
        direction_count,
        device=states_tensor.device,
        dtype=states_tensor.dtype,
    )

    with _quadratic_cache_context(dynamics):
        for n in reversed(range(steps)):
            p_n = None if p_mid is None else p_mid[n].detach()
            lambda_u, lambda_p, step_grads = _lagged_midpoint_step_adjoint_manual_batched(
                dynamics,
                states_tensor[n],
                lambda_state,
                h,
                p_n,
                picard_iters=picard_iters,
                picard_iterates=tuple(value.detach() for value in picard_iterates[n].unbind(0)),
                return_input_adjoint=return_input_adjoint,
                dense_a=dense_a,
                energy_accumulator=energy_accumulator,
            )
            for name, grad in step_grads.items():
                param_grads[name] = param_grads[name] + grad
            if lambda_p_mid is not None and lambda_p is not None:
                lambda_p_mid[n] = lambda_p
            lambda_state = lambda_u + state_cotangents[n].detach()

    energy_accumulator.flush_to(param_grads)

    return BatchedLaggedMidpointRolloutAdjointResult(
        states=states_tensor,
        lambda_u0=lambda_state.detach(),
        lambda_p_mid=None if lambda_p_mid is None else lambda_p_mid.detach(),
        parameter_grads={name: grad.detach() for name, grad in param_grads.items()},
    )
