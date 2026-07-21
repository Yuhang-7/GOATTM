from __future__ import annotations

import torch
from torch import nn


class ZeroSource(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_dim = 0

    def forward(self, p: torch.Tensor | None, like: torch.Tensor | None = None) -> torch.Tensor:
        if like is None:
            raise ValueError("ZeroSource requires like for shape")
        return torch.zeros_like(like)


class LinearSource(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, init_scale: float = 0.02, bias: bool = True) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        scale = init_scale / max(self.input_dim, 1) ** 0.5
        self.B = nn.Parameter(scale * torch.randn(self.latent_dim, self.input_dim))
        self.c = nn.Parameter(torch.zeros(self.latent_dim)) if bias else None

    def forward(self, p: torch.Tensor | None, like: torch.Tensor | None = None) -> torch.Tensor:
        if p is None:
            if like is None:
                raise ValueError("LinearSource requires p or like")
            out = torch.zeros_like(like)
        else:
            out = p @ self.B.T
        if self.c is not None:
            out = out + self.c.to(device=out.device, dtype=out.dtype)
        return out
