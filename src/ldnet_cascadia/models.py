from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from quadrode_gpu_goattm import DenseLinearA, EnergyTuckerTTQuadratic, LinearSource, QuadraticDynamics, RungeKutta4Stepper

from .data import CascadiaPackedBatch


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name!r}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Iterable[int] = (256, 256),
        *,
        activation: str = "silu",
        layer_norm: bool = False,
        final_scale: float | None = None,
        final_zero: bool = False,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *(int(x) for x in hidden), int(output_dim)]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(left, right))
            if layer_norm:
                layers.append(nn.LayerNorm(right))
            layers.append(_activation(activation))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        if final_zero:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
        elif final_scale is not None:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.mul_(float(final_scale))
                    if last.bias is not None:
                        last.bias.mul_(float(final_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InitialConditionEncoder(MLP):
    """Map a seafloor displacement context to the latent initial condition."""


class NeuralDecoder(nn.Module):
    """Pointwise decoder q(t) = linear_readout(z(t)) + correction(z(t))."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden: Iterable[int] = (256, 256),
        *,
        activation: str = "silu",
        layer_norm: bool = True,
        correction_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.linear_readout = nn.Linear(self.latent_dim, self.output_dim)
        self.correction = MLP(
            self.latent_dim,
            self.output_dim,
            hidden,
            activation=activation,
            layer_norm=layer_norm,
            final_zero=correction_zero_init,
        )

    def init_linear_readout(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        rows = min(int(weight.shape[0]), self.output_dim)
        cols = min(int(weight.shape[1]), self.latent_dim)
        with torch.no_grad():
            self.linear_readout.weight.zero_()
            self.linear_readout.bias.zero_()
            self.linear_readout.weight[:rows, :cols].copy_(
                weight[:rows, :cols].to(device=self.linear_readout.weight.device, dtype=self.linear_readout.weight.dtype)
            )
            if bias is not None:
                self.linear_readout.bias[:rows].copy_(
                    bias[:rows].to(device=self.linear_readout.bias.device, dtype=self.linear_readout.bias.dtype)
                )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear_readout(z) + self.correction(z)


@dataclass
class LDNetForwardResult:
    prediction: torch.Tensor
    states: torch.Tensor
    z0: torch.Tensor
    midpoint_inputs: torch.Tensor | None


class CascadiaLDNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        dynamics: QuadraticDynamics,
        decoder: nn.Module,
        *,
        input_context_mode: str = "final",
        use_time_dependent_source: bool = True,
        stepper: RungeKutta4Stepper | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.input_context_mode = input_context_mode
        self.use_time_dependent_source = bool(use_time_dependent_source)
        self.stepper = stepper if stepper is not None else RungeKutta4Stepper()

    def encode(self, batch: CascadiaPackedBatch) -> torch.Tensor:
        return self.encoder(batch.input_context(self.input_context_mode))

    def rollout_latent(self, batch: CascadiaPackedBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z0 = self.encode(batch)
        p_mid = batch.midpoint_inputs() if self.use_time_dependent_source else None
        states = self.stepper.rollout(self.dynamics, z0, batch.step_size, batch.steps, p_mid=p_mid)
        return states, z0, p_mid

    def decode_states(self, states: torch.Tensor, *, chunk_size: int = 0) -> torch.Tensor:
        flat = states.reshape(-1, states.shape[-1])
        if int(chunk_size) <= 0 or flat.shape[0] <= int(chunk_size):
            decoded = self.decoder(flat)
        else:
            pieces = [self.decoder(flat[i : i + int(chunk_size)]) for i in range(0, flat.shape[0], int(chunk_size))]
            decoded = torch.cat(pieces, dim=0)
        return decoded.reshape(*states.shape[:-1], decoded.shape[-1])

    def forward(self, batch: CascadiaPackedBatch, *, decode_chunk_size: int = 0) -> LDNetForwardResult:
        states, z0, p_mid = self.rollout_latent(batch)
        prediction = self.decode_states(states, chunk_size=decode_chunk_size)
        return LDNetForwardResult(prediction=prediction, states=states, z0=z0, midpoint_inputs=p_mid)


def make_denseA_lowrankH_neural_decoder_model(
    *,
    input_dim: int,
    output_dim: int,
    encoder_input_dim: int | None = None,
    latent_dim: int = 120,
    h_rank: int = 50,
    h_tt_rank: int | None = None,
    encoder_hidden: Iterable[int] = (256, 256),
    decoder_hidden: Iterable[int] = (256, 256),
    activation: str = "silu",
    linear_init_scale: float = 1.0e-3,
    linear_damping: float = 1.0e-3,
    quadratic_scale: float = 2.0e-2,
    source_init_scale: float = 2.0e-2,
    input_context_mode: str = "final",
    use_time_dependent_source: bool = True,
    dtype: torch.dtype = torch.float64,
) -> CascadiaLDNet:
    latent_dim = int(latent_dim)
    input_dim = int(input_dim)
    encoder_input_dim = int(input_dim if encoder_input_dim is None else encoder_input_dim)
    output_dim = int(output_dim)
    h_rank = int(h_rank)
    h_tt_rank = int(h_rank if h_tt_rank is None else h_tt_rank)
    a0 = linear_init_scale * torch.randn(latent_dim, latent_dim, dtype=dtype)
    a0 = a0 - float(linear_damping) * torch.eye(latent_dim, dtype=dtype)
    linear = DenseLinearA(latent_dim, matrix=a0)
    quadratic = EnergyTuckerTTQuadratic(
        latent_dim,
        reduced_rank=h_rank,
        tt_rank=h_tt_rank,
        scale=quadratic_scale,
    )
    source = LinearSource(latent_dim, input_dim, init_scale=source_init_scale, bias=True)
    dynamics = QuadraticDynamics(linear, quadratic, source)
    encoder = InitialConditionEncoder(
        encoder_input_dim,
        latent_dim,
        encoder_hidden,
        activation=activation,
        layer_norm=True,
        final_zero=True,
    )
    decoder = NeuralDecoder(
        latent_dim,
        output_dim,
        decoder_hidden,
        activation=activation,
        layer_norm=True,
    )
    model = CascadiaLDNet(
        encoder,
        dynamics,
        decoder,
        input_context_mode=input_context_mode,
        use_time_dependent_source=use_time_dependent_source,
    )
    return model.to(dtype=dtype)
