from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class TorchQuadraticLatentDynamics(nn.Module):
    """Differentiable latent dynamics du/dt = Au + H(u,u) + Bp + c."""

    def __init__(
        self,
        latent_dim: int,
        input_dim: int = 0,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        h_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        factory = {"dtype": dtype, "device": device}
        self.a = nn.Parameter(torch.zeros(self.latent_dim, self.latent_dim, **factory))
        self.h = nn.Parameter(h_scale * torch.randn(self.latent_dim, self.latent_dim, self.latent_dim, **factory))
        self.c = nn.Parameter(torch.zeros(self.latent_dim, **factory))
        if self.input_dim > 0:
            self.b = nn.Parameter(torch.zeros(self.latent_dim, self.input_dim, **factory))
        else:
            self.register_parameter("b", None)

    @classmethod
    def from_numpy(
        cls,
        a: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
        b: np.ndarray | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> "TorchQuadraticLatentDynamics":
        model = cls(a.shape[0], 0 if b is None else b.shape[1], dtype=dtype, device=device)
        with torch.no_grad():
            model.a.copy_(torch.as_tensor(a, dtype=dtype, device=device))
            model.h.copy_(torch.as_tensor(h, dtype=dtype, device=device))
            model.c.copy_(torch.as_tensor(c, dtype=dtype, device=device))
            if b is not None and model.b is not None:
                model.b.copy_(torch.as_tensor(b, dtype=dtype, device=device))
        return model

    def rhs(self, u: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        if u.shape[-1] != self.latent_dim:
            raise ValueError(f"u last dimension must be {self.latent_dim}, got {u.shape}")
        linear = torch.einsum("ij,...j->...i", self.a, u)
        quadratic = torch.einsum("ijk,...j,...k->...i", self.h, u, u)
        forcing = self.c
        if self.b is not None and p is not None:
            forcing = forcing + torch.einsum("ij,...j->...i", self.b, p)
        return linear + quadratic + forcing

    def component_norms(self) -> dict[str, float]:
        with torch.no_grad():
            norms = {
                "a_fro_norm": float(torch.linalg.norm(self.a).detach().cpu()),
                "h_fro_norm": float(torch.linalg.norm(self.h).detach().cpu()),
                "c_l2_norm": float(torch.linalg.norm(self.c).detach().cpu()),
            }
            norms["b_fro_norm"] = 0.0 if self.b is None else float(torch.linalg.norm(self.b).detach().cpu())
        return norms


def latent_rollout_euler(
    dynamics: TorchQuadraticLatentDynamics,
    u0: torch.Tensor,
    times: torch.Tensor,
    inputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll out latent states at observation times using explicit Euler substeps."""

    if times.ndim != 1:
        raise ValueError(f"times must have shape (T,), got {times.shape}")
    if u0.ndim != 2:
        raise ValueError(f"u0 must have shape (batch, latent_dim), got {u0.shape}")
    if inputs is not None and inputs.shape[:2] != (u0.shape[0], times.shape[0]):
        raise ValueError("inputs must have shape (batch, T, input_dim)")

    states = [u0]
    current = u0
    for step in range(times.shape[0] - 1):
        dt = times[step + 1] - times[step]
        p = None if inputs is None else inputs[:, step, :]
        current = current + dt * dynamics.rhs(current, p=p)
        states.append(current)
    return torch.stack(states, dim=1)


@dataclass(frozen=True)
class LatentRolloutSummary:
    max_abs_state: float
    final_state_norm: float
