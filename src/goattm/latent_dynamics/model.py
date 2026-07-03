from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LDNetConfig:
    latent_dim: int
    input_dim: int
    qoi_dim: int
    hidden_dim: int = 128
    depth: int = 3
    activation: str = "tanh"
    include_time: bool = True
    residual_scale: float = 1.0

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'.")


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int, activation: str) -> nn.Sequential:
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    layers: list[nn.Module] = []
    width_in = input_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(width_in, hidden_dim))
        layers.append(_activation(activation))
        width_in = hidden_dim
    layers.append(nn.Linear(width_in, output_dim))
    return nn.Sequential(*layers)


class LDNetModel(nn.Module):
    """Latent residual dynamics baseline for QoI sequences.

    The model uses a shared initial latent state and evolves

        z_{n+1} = z_n + residual_scale * dt_n * f_theta(z_n, b_n, t_n),
        q_n = g_phi(z_n).

    This is deliberately less structured than the OpInf quadratic model, so it
    gives a clean neural latent-dynamics baseline on the same manifest data.
    """

    def __init__(self, config: LDNetConfig) -> None:
        super().__init__()
        self.config = config
        field_input_dim = config.latent_dim + config.input_dim + (1 if config.include_time else 0)
        self.vector_field = _mlp(
            input_dim=field_input_dim,
            output_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
        )
        self.decoder = _mlp(
            input_dim=config.latent_dim,
            output_dim=config.qoi_dim,
            hidden_dim=config.hidden_dim,
            depth=max(2, config.depth),
            activation=config.activation,
        )
        self.z0 = nn.Parameter(torch.zeros(config.latent_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.z0)

    def rollout(self, times: torch.Tensor, inputs: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if times.ndim != 1:
            raise ValueError(f"times must have shape (T,), got {tuple(times.shape)}")
        if inputs is not None and inputs.shape[:2] != (batch_size, times.numel()):
            raise ValueError("inputs must have shape (batch,T,input_dim)")
        if self.config.input_dim == 0 and inputs is not None:
            raise ValueError("model was created with input_dim=0 but inputs were provided")
        if self.config.input_dim > 0 and inputs is None:
            raise ValueError("model expects inputs but none were provided")

        z = self.z0.expand(batch_size, -1)
        states = [z]
        final_time = torch.clamp(times[-1], min=torch.finfo(times.dtype).eps)
        for step in range(times.numel() - 1):
            dt = times[step + 1] - times[step]
            features = [z]
            if inputs is not None:
                features.append(inputs[:, step, :])
            if self.config.include_time:
                tau = (times[step] / final_time).expand(batch_size, 1)
                features.append(tau)
            dz = self.vector_field(torch.cat(features, dim=-1))
            z = z + self.config.residual_scale * dt * dz
            states.append(z)
        return torch.stack(states, dim=1)

    def forward(self, times: torch.Tensor, inputs: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        states = self.rollout(times=times, inputs=inputs, batch_size=batch_size)
        flat_outputs = self.decoder(states.reshape(-1, self.config.latent_dim))
        return flat_outputs.reshape(batch_size, times.numel(), self.config.qoi_dim)
