from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch

from .dynamics import QuadraticDynamics
from .quadratic import EnergyTuckerTTQuadratic


def _identity_batch(batch: int, r: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(r, device=device, dtype=dtype).expand(batch, r, r)


def _base_solve_columns(
    dynamics: QuadraticDynamics,
    columns: torch.Tensor,
    tau: float,
    *,
    transpose: bool = False,
) -> torch.Tensor:
    return dynamics.linear.solve_base(columns.T, tau, transpose=transpose).T


def _skewcp_c_matrix(gamma: torch.Tensor, scale: float) -> torch.Tensor:
    batch, rank = gamma.shape
    if rank == 0:
        return gamma.new_empty(batch, 0, 0)
    c = gamma.new_zeros(batch, 2 * rank, 2 * rank)
    diag = torch.diag_embed(float(scale) * gamma)
    c[:, :rank, rank:] = diag
    c[:, rank:, :rank] = -diag
    return c


def make_frozen_smw_cache(
    dynamics: QuadraticDynamics,
    tau: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    transpose: bool = False,
) -> "FrozenSMWCache | None":
    rank = int(dynamics.quadratic.rank)
    if rank == 0:
        return None
    z = dynamics.quadratic.smw_frame().to(device=device, dtype=dtype)
    k_base = _base_solve_columns(dynamics, z, tau, transpose=transpose)
    return FrozenSMWCache(
        z=z,
        k_base=k_base,
        gram=z.T @ k_base,
        transpose=transpose,
    )


def exact_frozen_smw_solve(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    rhs: torch.Tensor,
    tau: float,
    transpose: bool = False,
    cache: "FrozenSMWCache | None" = None,
) -> torch.Tensor:
    """Solve a frozen skewCP system by exact Woodbury.

    The dense reference solves

    ``(I - tau * (A + F(ell))) x = rhs``.

    Here ``F(ell) = Z C(ell) Z^T`` with ``Z=[U,V]`` and a batch-dependent
    block-skew matrix ``C``.  The transpose solve uses
    ``(I - tau * A^T + tau * F(ell))`` because ``F(ell)^T=-F(ell)``.
    """

    rank = int(dynamics.quadratic.rank)
    if rank == 0:
        return dynamics.linear.solve_base(rhs, tau, transpose=transpose)

    if cache is None:
        cache = make_frozen_smw_cache(
            dynamics,
            tau,
            device=rhs.device,
            dtype=rhs.dtype,
            transpose=transpose,
        )
    if cache is None:
        return dynamics.linear.solve_base(rhs, tau, transpose=transpose)
    if cache.transpose != transpose:
        raise ValueError("SMW cache transpose flag does not match solve request")
    z = cache.z
    gamma = ell @ dynamics.quadratic.W.to(device=rhs.device, dtype=rhs.dtype)
    lowrank_scale = float(tau) if transpose else -float(tau)
    c = _skewcp_c_matrix(gamma, lowrank_scale)

    y = dynamics.linear.solve_base(rhs, tau, transpose=transpose)
    small_eye = torch.eye(c.shape[-1], device=rhs.device, dtype=rhs.dtype).expand(rhs.shape[0], -1, -1)
    small = small_eye + torch.bmm(c, cache.gram.expand(rhs.shape[0], -1, -1))
    zty = y @ z
    small_rhs = torch.bmm(c, zty.unsqueeze(-1))
    eta = torch.linalg.solve(small, small_rhs).squeeze(-1)
    return y - eta @ cache.k_base.T


def predictor(dynamics: QuadraticDynamics, u: torch.Tensor, h: float, p: torch.Tensor | None) -> torch.Tensor:
    return u + float(h) * dynamics.rhs(u, p)


def _quadratic_cache_context(dynamics: QuadraticDynamics):
    cached = getattr(dynamics.quadratic, "cached_reduced_tensor", None)
    if cached is None:
        return nullcontext()
    return cached()


class DenseLaggedMidpointStepper:
    """Correctness-first lagged-midpoint stepper.

    General quadratic backends use dense GPU solves.  The Picard-history path
    used by the manual lagged adjoint dispatches energy-Tucker systems to the
    exact nested-SMW solve when the reduced rank is smaller than the latent
    dimension.
    """

    def __init__(self, picard_iters: int = 2) -> None:
        self.picard_iters = int(picard_iters)

    def step(self, dynamics: QuadraticDynamics, u: torch.Tensor, h: float, p_mid: torch.Tensor | None = None) -> torch.Tensor:
        tau = 0.5 * float(h)
        batch, r = u.shape
        eye = _identity_batch(batch, r, u.device, u.dtype)
        a = dynamics.dense_A().to(device=u.device, dtype=u.dtype).unsqueeze(0)
        force = dynamics.source(p_mid, like=u)
        with _quadratic_cache_context(dynamics):
            u_next = predictor(dynamics, u, h, p_mid)
            u_col = u.unsqueeze(-1)
            for _ in range(self.picard_iters):
                ell = 0.5 * (u + u_next)
                l_mat = a + dynamics.frozen_matrix(ell)
                rhs = u + tau * torch.bmm(l_mat, u_col).squeeze(-1) + float(h) * force
                u_next = torch.linalg.solve(eye - tau * l_mat, rhs.unsqueeze(-1)).squeeze(-1)
        return u_next

    def rollout(self, dynamics: QuadraticDynamics, u0: torch.Tensor, h: float, steps: int, p_mid: torch.Tensor | None = None) -> torch.Tensor:
        states = [u0]
        u = u0
        with _quadratic_cache_context(dynamics):
            for n in range(int(steps)):
                p = None if p_mid is None else p_mid[n]
                u = self.step(dynamics, u, h, p)
                states.append(u)
        return torch.stack(states, dim=0)

    def rollout_with_lags(
        self,
        dynamics: QuadraticDynamics,
        u0: torch.Tensor,
        h: float,
        steps: int,
        p_mid: torch.Tensor | None = None,
    ) -> "DenseRolloutResult":
        states = [u0]
        lags = []
        u = u0
        tau = 0.5 * float(h)
        batch, r = u.shape
        eye = _identity_batch(batch, r, u.device, u.dtype)
        a = dynamics.dense_A().to(device=u.device, dtype=u.dtype).unsqueeze(0)
        with _quadratic_cache_context(dynamics):
            for n in range(int(steps)):
                p = None if p_mid is None else p_mid[n]
                force = dynamics.source(p, like=u)
                u_next = predictor(dynamics, u, h, p)
                lag = 0.5 * (u + u_next)
                u_col = u.unsqueeze(-1)
                for _ in range(self.picard_iters):
                    lag = 0.5 * (u + u_next)
                    l_mat = a + dynamics.frozen_matrix(lag)
                    rhs = u + tau * torch.bmm(l_mat, u_col).squeeze(-1) + float(h) * force
                    u_next = torch.linalg.solve(eye - tau * l_mat, rhs.unsqueeze(-1)).squeeze(-1)
                lags.append(lag)
                u = u_next
                states.append(u)
        return DenseRolloutResult(states=torch.stack(states, dim=0), lags=torch.stack(lags, dim=0))

    def rollout_with_picard_history(
        self,
        dynamics: QuadraticDynamics,
        u0: torch.Tensor,
        h: float,
        steps: int,
        p_mid: torch.Tensor | None = None,
    ) -> "DensePicardRolloutResult":
        steps = int(steps)
        picard_iters = int(self.picard_iters)
        u = u0
        tau = 0.5 * float(h)
        batch, r = u.shape
        eye = _identity_batch(batch, r, u.device, u.dtype)
        states = u0.new_empty(steps + 1, batch, r)
        lags = u0.new_empty(steps, batch, r)
        histories = u0.new_empty(steps, picard_iters + 1, batch, r)
        states[0].copy_(u0)
        eye = _identity_batch(batch, r, u.device, u.dtype)
        a = dynamics.dense_A().to(device=u.device, dtype=u.dtype).unsqueeze(0)
        with _quadratic_cache_context(dynamics):
            for n in range(steps):
                p = None if p_mid is None else p_mid[n]
                force = dynamics.source(p, like=u)
                u_next = predictor(dynamics, u, h, p)
                histories[n, 0].copy_(u_next)
                lag = 0.5 * (u + u_next)
                u_col = u.unsqueeze(-1)
                for k in range(picard_iters):
                    lag = 0.5 * (u + u_next)
                    if use_energy_tucker_smw(dynamics):
                        rhs = u + tau * (dynamics.linear(u) + dynamics.quadratic.frozen_action(lag, u)) + float(h) * force
                        u_next = exact_frozen_solve(dynamics, lag, rhs, tau, transpose=False)
                    else:
                        l_mat = a + dynamics.frozen_matrix(lag)
                        rhs = u + tau * torch.bmm(l_mat, u_col).squeeze(-1) + float(h) * force
                        u_next = torch.linalg.solve(eye - tau * l_mat, rhs.unsqueeze(-1)).squeeze(-1)
                    histories[n, k + 1].copy_(u_next)
                lags[n].copy_(lag)
                u = u_next
                states[n + 1].copy_(u)
        return DensePicardRolloutResult(
            states=states,
            lags=lags,
            picard_iterates=histories,
        )


class RungeKutta4Stepper:
    """Explicit fourth-order Runge-Kutta stepper.

    The current data interface supplies one input value per time step, evaluated
    at the midpoint.  RK4 therefore keeps that input frozen across the four
    stages.  No implicit linear solve is performed.
    """

    picard_iters = 0

    def step(self, dynamics: QuadraticDynamics, u: torch.Tensor, h: float, p_mid: torch.Tensor | None = None) -> torch.Tensor:
        h_float = float(h)
        k1 = dynamics.rhs(u, p_mid)
        k2 = dynamics.rhs(u + 0.5 * h_float * k1, p_mid)
        k3 = dynamics.rhs(u + 0.5 * h_float * k2, p_mid)
        k4 = dynamics.rhs(u + h_float * k3, p_mid)
        return u + (h_float / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rollout(self, dynamics: QuadraticDynamics, u0: torch.Tensor, h: float, steps: int, p_mid: torch.Tensor | None = None) -> torch.Tensor:
        states = [u0]
        u = u0
        with _quadratic_cache_context(dynamics):
            for n in range(int(steps)):
                p = None if p_mid is None else p_mid[n]
                u = self.step(dynamics, u, h, p)
                states.append(u)
        return torch.stack(states, dim=0)

    def rollout_with_lags(
        self,
        dynamics: QuadraticDynamics,
        u0: torch.Tensor,
        h: float,
        steps: int,
        p_mid: torch.Tensor | None = None,
    ) -> "DenseRolloutResult":
        states = [u0]
        lags = []
        u = u0
        with _quadratic_cache_context(dynamics):
            for n in range(int(steps)):
                p = None if p_mid is None else p_mid[n]
                u_next = self.step(dynamics, u, h, p)
                lags.append(0.5 * (u + u_next))
                u = u_next
                states.append(u)
        return DenseRolloutResult(states=torch.stack(states, dim=0), lags=torch.stack(lags, dim=0))

    def rollout_with_picard_history(
        self,
        dynamics: QuadraticDynamics,
        u0: torch.Tensor,
        h: float,
        steps: int,
        p_mid: torch.Tensor | None = None,
    ) -> "DensePicardRolloutResult":
        raise NotImplementedError("RungeKutta4Stepper has no Picard history; use gradient_mode='autograd' or implement an RK4 adjoint.")


class SubstepRungeKutta4Stepper(RungeKutta4Stepper):
    """Explicit RK4 with multiple latent substeps per observation interval."""

    def __init__(self, substeps: int = 10) -> None:
        self.substeps = int(substeps)
        if self.substeps <= 0:
            raise ValueError("substeps must be positive")

    def _p_for_substep(self, p_mid: torch.Tensor | None, n: int, s: int) -> torch.Tensor | None:
        if p_mid is None:
            return None
        if p_mid.ndim >= 4:
            return p_mid[n, s]
        return p_mid[n]

    def interval_step(
        self,
        dynamics: QuadraticDynamics,
        u: torch.Tensor,
        h: float,
        p_mid: torch.Tensor | None = None,
        *,
        interval_index: int = 0,
    ) -> torch.Tensor:
        sub_h = float(h) / float(self.substeps)
        out = u
        for s in range(self.substeps):
            out = self.step(dynamics, out, sub_h, self._p_for_substep(p_mid, interval_index, s))
        return out

    def rollout(self, dynamics: QuadraticDynamics, u0: torch.Tensor, h: float, steps: int, p_mid: torch.Tensor | None = None) -> torch.Tensor:
        states = [u0]
        u = u0
        with _quadratic_cache_context(dynamics):
            for n in range(int(steps)):
                u = self.interval_step(dynamics, u, h, p_mid, interval_index=n)
                states.append(u)
        return torch.stack(states, dim=0)

    def rollout_with_lags(
        self,
        dynamics: QuadraticDynamics,
        u0: torch.Tensor,
        h: float,
        steps: int,
        p_mid: torch.Tensor | None = None,
    ) -> "DenseRolloutResult":
        states = [u0]
        lags = []
        u = u0
        with _quadratic_cache_context(dynamics):
            for n in range(int(steps)):
                u_next = self.interval_step(dynamics, u, h, p_mid, interval_index=n)
                lags.append(0.5 * (u + u_next))
                u = u_next
                states.append(u)
        return DenseRolloutResult(states=torch.stack(states, dim=0), lags=torch.stack(lags, dim=0))


@dataclass(frozen=True)
class FrozenSMWCache:
    z: torch.Tensor
    k_base: torch.Tensor
    gram: torch.Tensor
    transpose: bool


@dataclass(frozen=True)
class DenseRolloutResult:
    states: torch.Tensor
    lags: torch.Tensor


@dataclass(frozen=True)
class DensePicardRolloutResult:
    states: torch.Tensor
    lags: torch.Tensor
    picard_iterates: torch.Tensor


def use_energy_tucker_smw(dynamics: QuadraticDynamics) -> bool:
    return (
        isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic)
        and int(dynamics.quadratic.reduced_rank) < int(dynamics.latent_dim)
    )


def exact_energy_tucker_smw_solve(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    rhs: torch.Tensor,
    tau: float,
    transpose: bool = False,
) -> torch.Tensor:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyTuckerTTQuadratic):
        raise TypeError("expected EnergyTuckerTTQuadratic")
    p = quad.basis.to(device=rhs.device, dtype=rhs.dtype)
    reduced = quad.reduced_frozen_matrix(ell).to(device=rhs.device, dtype=rhs.dtype)
    if transpose:
        reduced = reduced.transpose(-1, -2)

    y = dynamics.linear.solve_base(rhs, tau, transpose=transpose)
    k = _base_solve_columns(dynamics, p.T, tau, transpose=transpose)
    gram = p @ k
    small_eye = torch.eye(gram.shape[0], device=rhs.device, dtype=rhs.dtype).expand(rhs.shape[0], -1, -1)
    small = small_eye - float(tau) * torch.bmm(reduced, gram.expand(rhs.shape[0], -1, -1))
    py = y @ p.T
    reduced_rhs = torch.bmm(reduced, py.unsqueeze(-1))
    eta = torch.linalg.solve(small, reduced_rhs).squeeze(-1)
    return y + float(tau) * (eta @ k.T)


def exact_frozen_solve(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    rhs: torch.Tensor,
    tau: float,
    transpose: bool = False,
    *,
    dense_a: torch.Tensor | None = None,
    eye: torch.Tensor | None = None,
) -> torch.Tensor:
    if use_energy_tucker_smw(dynamics):
        return exact_energy_tucker_smw_solve(dynamics, ell, rhs, tau, transpose=transpose)
    batch, r = rhs.shape
    if eye is None:
        eye = _identity_batch(batch, r, rhs.device, rhs.dtype)
    if dense_a is None:
        dense_a = dynamics.dense_A().to(device=rhs.device, dtype=rhs.dtype)
    else:
        dense_a = dense_a.to(device=rhs.device, dtype=rhs.dtype)
    l_mat = dense_a.unsqueeze(0) + dynamics.frozen_matrix(ell)
    mat = eye - float(tau) * l_mat
    if transpose:
        mat = mat.transpose(-1, -2)
    return torch.linalg.solve(mat, rhs.unsqueeze(-1)).squeeze(-1)


class ExactSMWLaggedMidpointStepper(DenseLaggedMidpointStepper):
    """Exact low-rank Woodbury solve for skewCP frozen midpoint systems."""

    def step(self, dynamics: QuadraticDynamics, u: torch.Tensor, h: float, p_mid: torch.Tensor | None = None) -> torch.Tensor:
        tau = 0.5 * float(h)
        cache = make_frozen_smw_cache(dynamics, tau, device=u.device, dtype=u.dtype, transpose=False)
        force = dynamics.source(p_mid, like=u)
        u_next = predictor(dynamics, u, h, p_mid)
        for _ in range(self.picard_iters):
            ell = 0.5 * (u + u_next)
            rhs = u + tau * (dynamics.linear(u) + dynamics.quadratic.frozen_action(ell, u)) + float(h) * force
            u_next = exact_frozen_smw_solve(dynamics, ell, rhs, tau, transpose=False, cache=cache)
        return u_next

    def rollout_with_lags(
        self,
        dynamics: QuadraticDynamics,
        u0: torch.Tensor,
        h: float,
        steps: int,
        p_mid: torch.Tensor | None = None,
    ) -> "DenseRolloutResult":
        states = [u0]
        lags = []
        u = u0
        tau = 0.5 * float(h)
        cache = make_frozen_smw_cache(dynamics, tau, device=u0.device, dtype=u0.dtype, transpose=False)
        for n in range(int(steps)):
            p = None if p_mid is None else p_mid[n]
            force = dynamics.source(p, like=u)
            u_next = predictor(dynamics, u, h, p)
            lag = 0.5 * (u + u_next)
            for _ in range(self.picard_iters):
                lag = 0.5 * (u + u_next)
                rhs = u + tau * (dynamics.linear(u) + dynamics.quadratic.frozen_action(lag, u)) + float(h) * force
                u_next = exact_frozen_smw_solve(dynamics, lag, rhs, tau, transpose=False, cache=cache)
            lags.append(lag)
            u = u_next
            states.append(u)
        return DenseRolloutResult(states=torch.stack(states, dim=0), lags=torch.stack(lags, dim=0))

    def rollout_with_picard_history(
        self,
        dynamics: QuadraticDynamics,
        u0: torch.Tensor,
        h: float,
        steps: int,
        p_mid: torch.Tensor | None = None,
    ) -> "DensePicardRolloutResult":
        steps = int(steps)
        picard_iters = int(self.picard_iters)
        u = u0
        tau = 0.5 * float(h)
        batch, r = u.shape
        cache = make_frozen_smw_cache(dynamics, tau, device=u0.device, dtype=u0.dtype, transpose=False)
        states = u0.new_empty(steps + 1, batch, r)
        lags = u0.new_empty(steps, batch, r)
        histories = u0.new_empty(steps, picard_iters + 1, batch, r)
        states[0].copy_(u0)
        for n in range(steps):
            p = None if p_mid is None else p_mid[n]
            force = dynamics.source(p, like=u)
            u_next = predictor(dynamics, u, h, p)
            histories[n, 0].copy_(u_next)
            lag = 0.5 * (u + u_next)
            for k in range(picard_iters):
                lag = 0.5 * (u + u_next)
                rhs = u + tau * (dynamics.linear(u) + dynamics.quadratic.frozen_action(lag, u)) + float(h) * force
                u_next = exact_frozen_smw_solve(dynamics, lag, rhs, tau, transpose=False, cache=cache)
                histories[n, k + 1].copy_(u_next)
            lags[n].copy_(lag)
            u = u_next
            states[n + 1].copy_(u)
        return DensePicardRolloutResult(
            states=states,
            lags=lags,
            picard_iterates=histories,
        )


@dataclass(frozen=True)
class AdaptiveDefectConfig:
    max_iters: int = 20
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    fallback_to_exact: bool = True


class SkewCPDefectLaggedMidpointStepper(DenseLaggedMidpointStepper):
    """Defect-iteration API with dense exact fallback.

    The initial rewrite keeps the same external contract but routes through the
    dense exact solve until the low-rank reduced solver is reintroduced and
    tested against this reference.
    """

    def __init__(self, picard_iters: int = 2, defect: AdaptiveDefectConfig | None = None) -> None:
        super().__init__(picard_iters=picard_iters)
        self.defect = defect or AdaptiveDefectConfig()
