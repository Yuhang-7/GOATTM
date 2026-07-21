from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class LinearOperatorA(nn.Module, ABC):
    latent_dim: int

    @abstractmethod
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply A to row-batch states u."""

    @abstractmethod
    def apply_transpose(self, v: torch.Tensor) -> torch.Tensor:
        """Apply A^T to row-batch cotangent vectors v."""

    @abstractmethod
    def dense_matrix(self) -> torch.Tensor:
        """Return the dense A matrix."""

    def solve_base(self, rhs: torch.Tensor, tau: float, transpose: bool = False) -> torch.Tensor:
        a = self.dense_matrix().to(device=rhs.device, dtype=rhs.dtype)
        eye = torch.eye(self.latent_dim, device=rhs.device, dtype=rhs.dtype)
        mat = eye - float(tau) * a
        if transpose:
            mat = mat.T
        return torch.linalg.solve(mat, rhs.T).T


class DenseLinearA(LinearOperatorA):
    """Unconstrained dense A, matching GOATTM's skewCP dynamics model."""

    def __init__(self, latent_dim: int, init_scale: float = 0.02, matrix: torch.Tensor | None = None) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if matrix is None:
            value = init_scale * torch.randn(self.latent_dim, self.latent_dim)
        else:
            if tuple(matrix.shape) != (self.latent_dim, self.latent_dim):
                raise ValueError("matrix shape does not match latent_dim")
            value = matrix.detach().clone()
        self.A = nn.Parameter(value)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u @ self.A.T

    def apply_transpose(self, v: torch.Tensor) -> torch.Tensor:
        return v @ self.A

    def dense_matrix(self) -> torch.Tensor:
        return self.A


class DissipativeSkewA(LinearOperatorA):
    """A = diag(shift - d^2) + P Q^T - Q P^T.

    The default ``shift=0`` gives the dissipative form
    ``A = -D + P Q^T - Q P^T``.  Positive shifts, for example ``shift=0.5``,
    allow selected diagonal growth rates to be positive while retaining the
    low-rank skew part.
    """

    def __init__(
        self,
        latent_dim: int,
        skew_rank: int,
        damping_init: float = 0.1,
        factor_scale: float = 0.02,
        damping_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.skew_rank = int(skew_rank)
        self.damping_shift = float(damping_shift)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.skew_rank < 0:
            raise ValueError("skew_rank must be nonnegative")
        raw = torch.full((self.latent_dim,), float(damping_init)).sqrt()
        self.raw_damping = nn.Parameter(raw)
        scale = factor_scale / max(self.latent_dim, 1) ** 0.5
        self.P = nn.Parameter(scale * torch.randn(self.latent_dim, self.skew_rank))
        self.Q = nn.Parameter(scale * torch.randn(self.latent_dim, self.skew_rank))

    @property
    def damping(self) -> torch.Tensor:
        return self.raw_damping.square()

    @property
    def diagonal(self) -> torch.Tensor:
        return self.damping_shift - self.damping

    def skew_action(self, u: torch.Tensor) -> torch.Tensor:
        if self.skew_rank == 0:
            return torch.zeros_like(u)
        return (u @ self.Q) @ self.P.T - (u @ self.P) @ self.Q.T

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.diagonal * u + self.skew_action(u)

    def apply_transpose(self, v: torch.Tensor) -> torch.Tensor:
        return self.diagonal * v - self.skew_action(v)

    def dense_matrix(self) -> torch.Tensor:
        diag = torch.diag(self.diagonal)
        if self.skew_rank == 0:
            return diag
        return diag + self.P @ self.Q.T - self.Q @ self.P.T

    def solve_base(self, rhs: torch.Tensor, tau: float, transpose: bool = False) -> torch.Tensor:
        diagonal = self.diagonal.to(device=rhs.device, dtype=rhs.dtype)
        scale = 1.0 - float(tau) * diagonal
        if self.skew_rank == 0:
            return rhs / scale

        p = self.P.to(device=rhs.device, dtype=rhs.dtype)
        q = self.Q.to(device=rhs.device, dtype=rhs.dtype)
        if transpose:
            # A^T = diag(d) + [Q, -P] [P, Q]^T.
            u = torch.cat((q, -p), dim=1)
            v = torch.cat((p, q), dim=1)
        else:
            # A = diag(d) + [P, -Q] [Q, P]^T.
            u = torch.cat((p, -q), dim=1)
            v = torch.cat((q, p), dim=1)

        y = rhs / scale
        k = u / scale[:, None]
        gram = v.T @ k
        small = torch.eye(gram.shape[0], device=rhs.device, dtype=rhs.dtype) - float(tau) * gram
        rhs_small = y @ v
        coeff = torch.linalg.solve(small, rhs_small.T).T
        return y + float(tau) * (coeff @ k.T)
