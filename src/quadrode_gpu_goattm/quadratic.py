from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class SkewCPQuadratic(nn.Module):
    """Energy-preserving skewCP quadratic term.

    H(u,u) = U ((W^T u) * (V^T u)) - V ((W^T u) * (U^T u)).
    """

    def __init__(self, latent_dim: int, rank: int, scale: float = 0.02) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        if self.latent_dim <= 0 or self.rank < 0:
            raise ValueError("invalid latent_dim or rank")
        factor_scale = scale / max(self.latent_dim, 1) ** 0.5
        self.U = nn.Parameter(factor_scale * torch.randn(self.latent_dim, self.rank))
        self.V = nn.Parameter(factor_scale * torch.randn(self.latent_dim, self.rank))
        self.W = nn.Parameter(factor_scale * torch.randn(self.latent_dim, self.rank))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        alpha = u @ self.U
        beta = u @ self.V
        gamma = u @ self.W
        return (gamma * beta) @ self.U.T - (gamma * alpha) @ self.V.T

    def frozen_action(self, ell: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        gamma = ell @ self.W
        alpha = u @ self.U
        beta = u @ self.V
        return (gamma * beta) @ self.U.T - (gamma * alpha) @ self.V.T

    def frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        gamma = ell @ self.W
        u = self.U.to(device=ell.device, dtype=ell.dtype)
        v = self.V.to(device=ell.device, dtype=ell.dtype)
        return torch.einsum("bk,ik,jk->bij", gamma, u, v) - torch.einsum("bk,ik,jk->bij", gamma, v, u)

    def smw_frame(self) -> torch.Tensor:
        return torch.cat((self.U, self.V), dim=1)

    def apply_J(self, coeff: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        rank = self.rank
        cu = coeff[:, :rank]
        cv = coeff[:, rank:]
        return torch.cat((-gamma * cv, gamma * cu), dim=1)


class DenseQuadratic(nn.Module):
    """Unconstrained dense quadratic tensor H(ell, u)."""

    def __init__(self, latent_dim: int, scale: float = 0.02) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        tensor_scale = float(scale) / max(self.latent_dim, 1)
        self.tensor = nn.Parameter(tensor_scale * torch.randn(self.latent_dim, self.latent_dim, self.latent_dim))

    @property
    def rank(self) -> int:
        return self.latent_dim

    def frozen_action(self, ell: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        c = self.tensor.to(device=u.device, dtype=u.dtype)
        return torch.einsum("abc,nb,nc->na", c, ell, u)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.frozen_action(u, u)

    def frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        c = self.tensor.to(device=ell.device, dtype=ell.dtype)
        return torch.einsum("abc,nb->nac", c, ell)


class EnergyDenseQuadratic(nn.Module):
    """Dense energy-preserving quadratic tensor with cyclic constraint."""

    def __init__(self, latent_dim: int, scale: float = 0.02) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reduced_rank = int(latent_dim)
        if self.latent_dim <= 1:
            raise ValueError("latent_dim must exceed 1")
        free_triples = [
            (a, b, c)
            for a in range(self.latent_dim)
            for b in range(self.latent_dim)
            for c in range(b, self.latent_dim)
            if a > b
        ]
        self.free_dim = len(free_triples)
        free_index = {triple: i for i, triple in enumerate(free_triples)}
        target: list[int] = []
        source: list[int] = []
        coeffs: list[float] = []

        def add_term(out_a: int, out_b: int, out_c: int, src: tuple[int, int, int], coeff: float) -> None:
            target.append((out_a * self.latent_dim + out_b) * self.latent_dim + out_c)
            source.append(free_index[src])
            coeffs.append(float(coeff))

        for a in range(self.latent_dim):
            for b in range(self.latent_dim):
                for c in range(self.latent_dim):
                    out_b, out_c = b, c
                    if out_b > out_c:
                        out_b, out_c = out_c, out_b
                    if a > out_b:
                        add_term(a, b, c, (a, out_b, out_c), 1.0)
                    elif a == out_b == out_c:
                        continue
                    elif a == out_b:
                        add_term(a, b, c, (out_c, a, a), -0.5)
                    elif out_b == out_c:
                        add_term(a, b, c, (out_b, a, out_b), -2.0)
                    else:
                        add_term(a, b, c, (out_b, a, out_c), -1.0)
                        add_term(a, b, c, (out_c, a, out_b), -1.0)

        self.register_buffer("free_a", torch.tensor([t[0] for t in free_triples], dtype=torch.long))
        self.register_buffer("free_b", torch.tensor([t[1] for t in free_triples], dtype=torch.long))
        self.register_buffer("free_c", torch.tensor([t[2] for t in free_triples], dtype=torch.long))
        self.register_buffer("reconstruct_target", torch.tensor(target, dtype=torch.long))
        self.register_buffer("reconstruct_source", torch.tensor(source, dtype=torch.long))
        self.register_buffer("reconstruct_coeff", torch.tensor(coeffs, dtype=torch.float64))
        tensor_scale = float(scale) / max(self.latent_dim, 1)
        self.free_values = nn.Parameter(tensor_scale * torch.randn(self.free_dim))
        self._reduced_tensor_cache: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        return self.latent_dim

    @contextmanager
    def cached_reduced_tensor(self):
        if self._reduced_tensor_cache is not None:
            yield
            return
        self._reduced_tensor_cache = self._compute_reduced_tensor()
        try:
            yield
        finally:
            self._reduced_tensor_cache = None

    def _compute_reduced_tensor(self) -> torch.Tensor:
        flat = self.free_values.new_zeros(self.latent_dim**3)
        coeff = self.reconstruct_coeff.to(device=self.free_values.device, dtype=self.free_values.dtype)
        flat.index_add_(0, self.reconstruct_target, coeff * self.free_values[self.reconstruct_source])
        return flat.reshape(self.latent_dim, self.latent_dim, self.latent_dim)

    def reduced_tensor(self) -> torch.Tensor:
        if self._reduced_tensor_cache is not None:
            return self._reduced_tensor_cache
        return self._compute_reduced_tensor()

    def frozen_action(self, ell: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        c = self.reduced_tensor().to(device=u.device, dtype=u.dtype)
        return torch.einsum("abc,nb,nc->na", c, ell, u)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.frozen_action(u, u)

    def frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        c = self.reduced_tensor().to(device=ell.device, dtype=ell.dtype)
        return torch.einsum("abc,nb->nac", c, ell)


class TuckerTTQuadratic(nn.Module):
    """Tucker-compressed quadratic map with a 3-core tensor-train core.

    The map is

        H(ell, u) = O^T C(P ell, P u),

    where P and O have orthonormal rows at initialization and the reduced core
    C is stored in tensor-train form over output, first input, and second input
    modes.  This backend is intended for dense frozen solves: it controls the
    number of trainable H parameters, but it does not expose the skewCP SMW
    structure.
    """

    def __init__(
        self,
        latent_dim: int,
        input_rank: int,
        output_rank: int | None = None,
        tt_rank: int = 16,
        scale: float = 0.02,
        *,
        basis_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_rank = int(input_rank)
        self.output_rank = int(output_rank if output_rank is not None else input_rank)
        self.tt_rank = int(tt_rank)
        if self.latent_dim <= 0 or self.input_rank <= 0 or self.output_rank <= 0 or self.tt_rank <= 0:
            raise ValueError("latent_dim, input_rank, output_rank, and tt_rank must be positive")
        if self.input_rank > self.latent_dim or self.output_rank > self.latent_dim:
            raise ValueError("Tucker ranks cannot exceed latent_dim")

        input_basis = self._orthonormal_rows(self.input_rank, self.latent_dim)
        output_basis = self._orthonormal_rows(self.output_rank, self.latent_dim)
        if basis_trainable:
            self.input_basis = nn.Parameter(input_basis)
            self.output_basis = nn.Parameter(output_basis)
        else:
            self.register_buffer("input_basis", input_basis)
            self.register_buffer("output_basis", output_basis)

        core_scale = float(scale) / max(self.input_rank * self.tt_rank, 1) ** 0.5
        self.core0 = nn.Parameter(core_scale * torch.randn(self.output_rank, self.tt_rank))
        self.core1 = nn.Parameter(core_scale * torch.randn(self.tt_rank, self.input_rank, self.tt_rank))
        self.core2 = nn.Parameter(core_scale * torch.randn(self.tt_rank, self.input_rank))

    @staticmethod
    def _orthonormal_rows(rows: int, cols: int) -> torch.Tensor:
        q, _ = torch.linalg.qr(torch.randn(cols, rows), mode="reduced")
        return q.T.contiguous()

    @property
    def rank(self) -> int:
        return self.tt_rank

    def reduced_frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        p = self.input_basis.to(device=ell.device, dtype=ell.dtype)
        x = ell @ p.T
        mid = torch.einsum("bi,aic->bac", x, self.core1.to(device=ell.device, dtype=ell.dtype))
        return torch.einsum(
            "ma,bac,cn->bmn",
            self.core0.to(device=ell.device, dtype=ell.dtype),
            mid,
            self.core2.to(device=ell.device, dtype=ell.dtype),
        )

    def frozen_action(self, ell: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        p = self.input_basis.to(device=u.device, dtype=u.dtype)
        o = self.output_basis.to(device=u.device, dtype=u.dtype)
        y = u @ p.T
        reduced_matrix = self.reduced_frozen_matrix(ell)
        reduced_out = torch.bmm(reduced_matrix, y.unsqueeze(-1)).squeeze(-1)
        return reduced_out @ o

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.frozen_action(u, u)

    def frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        p = self.input_basis.to(device=ell.device, dtype=ell.dtype)
        o = self.output_basis.to(device=ell.device, dtype=ell.dtype)
        reduced_matrix = self.reduced_frozen_matrix(ell)
        return torch.einsum("mi,bmn,nj->bij", o, reduced_matrix, p)


class EnergyTuckerTTQuadratic(nn.Module):
    """Energy-preserving Tucker/TT quadratic map.

    This is the hard tensor reparameterization used in stabilized OpInf.  The
    reduced tensor C satisfies

        C[a,b,c] = C[a,c,b],
        C[a,b,c] + C[b,c,a] + C[c,a,b] = 0,

    hence z^T C(z,z)=0.  Trainable TT cores define only the free entries
    K_sym \\ K_ep, equivalently indices (a,b,c) with a>b and b<=c.  A fixed
    sparse linear reconstruction fills the full reduced tensor.
    """

    def __init__(
        self,
        latent_dim: int,
        reduced_rank: int,
        tt_rank: int = 16,
        scale: float = 0.02,
        *,
        basis_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reduced_rank = int(reduced_rank)
        self.tt_rank = int(tt_rank)
        if self.latent_dim <= 0 or self.reduced_rank <= 1 or self.tt_rank <= 0:
            raise ValueError("latent_dim must be positive, reduced_rank must exceed 1, and tt_rank must be positive")
        if self.reduced_rank > self.latent_dim:
            raise ValueError("reduced_rank cannot exceed latent_dim")
        basis = self._orthonormal_rows(self.reduced_rank, self.latent_dim)
        if basis_trainable:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis)
        free_triples = [
            (a, b, c)
            for a in range(self.reduced_rank)
            for b in range(self.reduced_rank)
            for c in range(b, self.reduced_rank)
            if a > b
        ]
        self.free_dim = len(free_triples)
        free_index = {triple: i for i, triple in enumerate(free_triples)}
        target: list[int] = []
        source: list[int] = []
        coeffs: list[float] = []

        def add_term(out_a: int, out_b: int, out_c: int, src: tuple[int, int, int], coeff: float) -> None:
            target.append((out_a * self.reduced_rank + out_b) * self.reduced_rank + out_c)
            source.append(free_index[src])
            coeffs.append(float(coeff))

        for a in range(self.reduced_rank):
            for b in range(self.reduced_rank):
                for c in range(self.reduced_rank):
                    out_b, out_c = b, c
                    if out_b > out_c:
                        out_b, out_c = out_c, out_b
                    if a > out_b:
                        add_term(a, b, c, (a, out_b, out_c), 1.0)
                    elif a == out_b == out_c:
                        continue
                    elif a == out_b:
                        add_term(a, b, c, (out_c, a, a), -0.5)
                    elif out_b == out_c:
                        add_term(a, b, c, (out_b, a, out_b), -2.0)
                    else:
                        add_term(a, b, c, (out_b, a, out_c), -1.0)
                        add_term(a, b, c, (out_c, a, out_b), -1.0)

        self.register_buffer("free_a", torch.tensor([t[0] for t in free_triples], dtype=torch.long))
        self.register_buffer("free_b", torch.tensor([t[1] for t in free_triples], dtype=torch.long))
        self.register_buffer("free_c", torch.tensor([t[2] for t in free_triples], dtype=torch.long))
        self.register_buffer("reconstruct_target", torch.tensor(target, dtype=torch.long))
        self.register_buffer("reconstruct_source", torch.tensor(source, dtype=torch.long))
        self.register_buffer("reconstruct_coeff", torch.tensor(coeffs, dtype=torch.float64))
        core_scale = float(scale) / max(self.reduced_rank * self.tt_rank, 1) ** 0.5
        self.core0 = nn.Parameter(core_scale * torch.randn(self.reduced_rank, self.tt_rank))
        self.core1 = nn.Parameter(core_scale * torch.randn(self.tt_rank, self.reduced_rank, self.tt_rank))
        self.core2 = nn.Parameter(core_scale * torch.randn(self.tt_rank, self.reduced_rank))
        self._reduced_tensor_cache: torch.Tensor | None = None

    @staticmethod
    def _orthonormal_rows(rows: int, cols: int) -> torch.Tensor:
        q, _ = torch.linalg.qr(torch.randn(cols, rows), mode="reduced")
        return q.T.contiguous()

    @property
    def rank(self) -> int:
        return self.tt_rank

    @property
    def input_rank(self) -> int:
        return self.reduced_rank

    @property
    def output_rank(self) -> int:
        return self.reduced_rank

    def free_tensor(self) -> torch.Tensor:
        return torch.einsum("ar,rbs,sc->abc", self.core0, self.core1, self.core2)

    @contextmanager
    def cached_reduced_tensor(self):
        if self._reduced_tensor_cache is not None:
            yield
            return
        self._reduced_tensor_cache = self._compute_reduced_tensor()
        try:
            yield
        finally:
            self._reduced_tensor_cache = None

    def _compute_reduced_tensor(self) -> torch.Tensor:
        free_full = self.free_tensor()
        free_values = free_full[self.free_a, self.free_b, self.free_c]
        flat = free_values.new_zeros(self.reduced_rank**3)
        coeff = self.reconstruct_coeff.to(device=free_values.device, dtype=free_values.dtype)
        flat.index_add_(0, self.reconstruct_target, coeff * free_values[self.reconstruct_source])
        return flat.reshape(self.reduced_rank, self.reduced_rank, self.reduced_rank)

    def reduced_tensor(self) -> torch.Tensor:
        if self._reduced_tensor_cache is not None:
            return self._reduced_tensor_cache
        return self._compute_reduced_tensor()

    def reduced_action(self, z_lag: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.reduced_tensor().to(device=z.device, dtype=z.dtype)
        return torch.einsum("abc,nb,nc->na", c, z_lag, z)

    def reduced_frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        p = self.basis.to(device=ell.device, dtype=ell.dtype)
        z_lag = ell @ p.T
        c = self.reduced_tensor().to(device=ell.device, dtype=ell.dtype)
        return torch.einsum("abc,nb->nac", c, z_lag)

    def frozen_action(self, ell: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        p = self.basis.to(device=u.device, dtype=u.dtype)
        z_lag = ell @ p.T
        z = u @ p.T
        reduced_out = self.reduced_action(z_lag, z)
        return reduced_out @ p

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.frozen_action(u, u)

    def frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        p = self.basis.to(device=ell.device, dtype=ell.dtype)
        reduced_matrix = self.reduced_frozen_matrix(ell)
        return torch.einsum("ai,nab,bj->nij", p, reduced_matrix, p)
