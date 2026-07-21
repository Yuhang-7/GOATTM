from __future__ import annotations

import torch
from torch import nn

from .linear import LinearOperatorA
from .quadratic import SkewCPQuadratic
from .source import LinearSource, ZeroSource


class QuadraticDynamics(nn.Module):
    def __init__(
        self,
        linear: LinearOperatorA,
        quadratic: SkewCPQuadratic,
        source: LinearSource | ZeroSource | None = None,
    ) -> None:
        super().__init__()
        if linear.latent_dim != quadratic.latent_dim:
            raise ValueError("linear and quadratic dimensions must match")
        self.linear = linear
        self.quadratic = quadratic
        self.source = source if source is not None else ZeroSource(linear.latent_dim)
        self.latent_dim = linear.latent_dim

    def rhs(self, u: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        return self.linear(u) + self.quadratic(u) + self.source(p, like=u)

    def dense_A(self) -> torch.Tensor:
        return self.linear.dense_matrix()

    def frozen_matrix(self, ell: torch.Tensor) -> torch.Tensor:
        return self.quadratic.frozen_matrix(ell)
