from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FiniteDifferenceRow:
    epsilon: float
    fd_slope: float
    autograd_slope: float
    abs_error: float
    rel_error: float


def _parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def finite_difference_check(
    model: nn.Module,
    closure,
    *,
    epsilons: tuple[float, ...] = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5),
    seed: int = 0,
) -> list[FiniteDifferenceRow]:
    params = _parameters(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    directions = []
    for param in params:
        direction = torch.randn(param.shape, generator=generator, dtype=param.dtype).to(device=param.device)
        directions.append(direction)
    norm_sq = sum(float(direction.square().sum().detach().cpu()) for direction in directions)
    norm = max(norm_sq, 1.0e-300) ** 0.5
    directions = [direction / norm for direction in directions]

    model.zero_grad(set_to_none=True)
    base = closure()
    base.backward()
    autograd = sum(
        (param.grad.detach() * direction).sum()
        for param, direction in zip(params, directions)
        if param.grad is not None
    )
    saved = [param.detach().clone() for param in params]
    rows: list[FiniteDifferenceRow] = []
    with torch.no_grad():
        for eps in epsilons:
            for param, value, direction in zip(params, saved, directions):
                param.copy_(value + float(eps) * direction)
            plus = closure()
            for param, value, direction in zip(params, saved, directions):
                param.copy_(value - float(eps) * direction)
            minus = closure()
            fd = (plus - minus) / (2.0 * float(eps))
            abs_error = torch.abs(fd - autograd)
            rel_error = abs_error / torch.clamp(torch.abs(autograd), min=torch.finfo(autograd.dtype).tiny)
            rows.append(
                FiniteDifferenceRow(
                    epsilon=float(eps),
                    fd_slope=float(fd.detach().cpu()),
                    autograd_slope=float(autograd.detach().cpu()),
                    abs_error=float(abs_error.detach().cpu()),
                    rel_error=float(rel_error.detach().cpu()),
                )
            )
        for param, value in zip(params, saved):
            param.copy_(value)
    model.zero_grad(set_to_none=True)
    return rows
