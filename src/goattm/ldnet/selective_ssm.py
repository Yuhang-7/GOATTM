from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SelectiveSSMDecoderConfig:
    latent_dim: int
    qoi_dim: int
    state_dim: int = 32
    dropout: float = 0.0
    dtype: torch.dtype = torch.float64
    device: torch.device | str | None = None


class SelectiveSSMDecoder(nn.Module):
    """Small selective-SSM decoder for u(0:T) -> q(0:T).

    The map is deliberately close to a state-space decoder:

        z_{k+1} = exp(dt_k A) * z_k + dt_k * B(u_k)
        qhat_t  = C_t z_t + D u_t + b

    where B(u_k) and C(u_k) are simple affine functions of u_k. The physical
    time increment dt_k comes from the input time grid; there is no learned
    Delta_t = f(u_t) gate. This keeps a small state-dependent SSM decoder
    without bringing in the full Mamba block or CUDA selective-scan kernels.
    """

    def __init__(self, config: SelectiveSSMDecoderConfig) -> None:
        super().__init__()
        self.config = config
        factory = {"dtype": config.dtype, "device": config.device}
        self.b_proj = nn.Linear(config.latent_dim, config.state_dim, **factory)
        self.c_proj = nn.Linear(config.latent_dim, config.qoi_dim * config.state_dim, **factory)
        self.skip = nn.Linear(config.latent_dim, config.qoi_dim, **factory)
        self.output_bias = nn.Parameter(torch.zeros(config.qoi_dim, **factory))
        self.log_a = nn.Parameter(torch.zeros(config.state_dim, **factory))
        self.dropout = nn.Dropout(config.dropout)

    def initial_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.state_dim, dtype=dtype, device=device)

    def output_from_state(self, u: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        c_t = self.c_proj(u).reshape(u.shape[0], self.config.qoi_dim, self.config.state_dim)
        y = torch.einsum("bqs,bs->bq", c_t, state)
        return y + self.skip(u) + self.output_bias[None, :]

    def advance_state(self, u: torch.Tensor, state: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        b_t = self.b_proj(u)
        a_diag = -F.softplus(self.log_a)
        decay = torch.exp(dt * a_diag)[None, :]
        return decay * state + dt * b_t

    def forward(self, latent_sequence: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        if latent_sequence.ndim != 3:
            raise ValueError(f"latent_sequence must have shape (batch, T, r), got {latent_sequence.shape}")
        batch, steps, _ = latent_sequence.shape
        if times is None:
            dts = torch.ones(steps - 1, dtype=latent_sequence.dtype, device=latent_sequence.device)
        else:
            if times.ndim != 1 or times.shape[0] != steps:
                raise ValueError(f"times must have shape ({steps},), got {times.shape}")
            dts = times[1:] - times[:-1]
            if torch.any(dts <= 0):
                raise ValueError("times must be strictly increasing.")
        u = self.dropout(latent_sequence)
        state = self.initial_state(batch, latent_sequence.dtype, latent_sequence.device)
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            outputs.append(self.output_from_state(u[:, step, :], state))
            if step < steps - 1:
                state = self.advance_state(u[:, step, :], state, dts[step])
        return torch.stack(outputs, dim=1)
