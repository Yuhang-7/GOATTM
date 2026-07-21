from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


def grad_norm(parameters) -> torch.Tensor:
    total: torch.Tensor | None = None
    for param in parameters:
        if param.grad is None:
            continue
        term = param.grad.detach().square().sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros(())
    return torch.sqrt(total)


def named_parameter_blocks(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    blocks: dict[str, list[nn.Parameter]] = defaultdict(list)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            block = "encoder"
        elif name.startswith("decoder."):
            block = "decoder"
        elif ".linear." in name or name.startswith("dynamics.linear."):
            block = "linear"
        elif ".quadratic." in name or name.startswith("dynamics.quadratic."):
            block = "quadratic"
        elif ".source." in name or name.startswith("dynamics.source."):
            block = "source"
        else:
            block = "other"
        blocks[block].append(param)
    return dict(blocks)


@dataclass(frozen=True)
class ArmijoResult:
    accepted: bool
    alpha: float
    next_initial_alpha: float
    initial_loss: float
    trial_loss: float
    grad_norm: float
    directional_derivative: float


@dataclass(frozen=True)
class BlockwiseArmijoResult:
    accepted: bool
    alpha: float
    next_initial_alpha: float
    initial_loss: float
    trial_loss: float
    grad_norm: float


def _snapshot(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def _restore(params: list[nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)


def _directional_derivative(params: list[nn.Parameter], directions: list[torch.Tensor]) -> torch.Tensor:
    total: torch.Tensor | None = None
    for param, direction in zip(params, directions):
        if param.grad is None:
            continue
        term = (param.grad.detach() * direction).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros(())
    return total


def _global_steepest_descent_direction(
    params: list[nn.Parameter],
    *,
    trust: float | None,
    min_norm: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    grad_sq: torch.Tensor | None = None
    param_sq: torch.Tensor | None = None
    directions: list[torch.Tensor] = []
    for param in params:
        if param.grad is None:
            directions.append(torch.zeros_like(param))
            continue
        gterm = param.grad.detach().square().sum()
        pterm = param.detach().square().sum()
        grad_sq = gterm if grad_sq is None else grad_sq + gterm
        param_sq = pterm if param_sq is None else param_sq + pterm
    if grad_sq is None:
        return directions, torch.zeros(())
    grad_norm_value = torch.sqrt(grad_sq)
    scale = 1.0
    if trust is not None and float(trust) > 0.0:
        param_norm = 0.0 if param_sq is None else float(torch.sqrt(param_sq).detach().cpu())
        radius = float(trust) * max(param_norm, float(min_norm))
        scale = min(1.0, radius / max(float(grad_norm_value.detach().cpu()), torch.finfo(grad_norm_value.dtype).tiny))
    directions = [
        torch.zeros_like(param) if param.grad is None else -scale * param.grad.detach().clone()
        for param in params
    ]
    return directions, grad_norm_value


def armijo_step(
    model: nn.Module,
    closure: Callable[[], torch.Tensor],
    *,
    initial_alpha: float,
    max_alpha: float = 1.0,
    shrink: float = 0.5,
    grow: float = 2.0,
    c1: float = 1.0e-4,
    max_trials: int = 20,
    trust: float | None = 1.0e-4,
    min_norm: float = 1.0,
) -> ArmijoResult:
    model.zero_grad(set_to_none=True)
    base_loss = closure()
    base_loss.backward()
    params = [param for param in model.parameters() if param.requires_grad]
    directions, total_grad = _global_steepest_descent_direction(params, trust=trust, min_norm=min_norm)
    if not params or float(total_grad.detach().cpu()) == 0.0:
        value = float(base_loss.detach().cpu())
        return ArmijoResult(False, 0.0, initial_alpha, value, value, 0.0, 0.0)

    base_values = _snapshot(params)
    directional = _directional_derivative(params, directions)
    alpha = min(float(initial_alpha), float(max_alpha))
    accepted = False
    trial_value = float("nan")
    for _ in range(int(max_trials)):
        _restore(params, base_values)
        with torch.no_grad():
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=alpha)
        with torch.no_grad():
            trial_loss = closure()
        trial_value = float(trial_loss.detach().cpu())
        sufficient = base_loss + float(c1) * alpha * directional
        if torch.isfinite(trial_loss) and bool((trial_loss <= sufficient).detach().cpu()):
            accepted = True
            break
        alpha *= float(shrink)
    if not accepted:
        _restore(params, base_values)
    next_alpha = min(float(max_alpha), alpha * float(grow)) if accepted else max(alpha, float(initial_alpha) * float(shrink))
    model.zero_grad(set_to_none=True)
    return ArmijoResult(
        accepted=accepted,
        alpha=float(alpha if accepted else 0.0),
        next_initial_alpha=float(next_alpha),
        initial_loss=float(base_loss.detach().cpu()),
        trial_loss=float(trial_value),
        grad_norm=float(total_grad.detach().cpu()),
        directional_derivative=float(directional.detach().cpu()),
    )


def _make_blockwise_descent(
    blocks: dict[str, list[nn.Parameter]],
    trust: dict[str, float],
    *,
    min_norm: float,
) -> tuple[list[nn.Parameter], list[torch.Tensor], torch.Tensor]:
    params: list[nn.Parameter] = []
    directions: list[torch.Tensor] = []
    total_grad_sq: torch.Tensor | None = None
    for block_name, block_params in blocks.items():
        grad_sq: torch.Tensor | None = None
        param_sq: torch.Tensor | None = None
        for param in block_params:
            if param.grad is None:
                continue
            gterm = param.grad.detach().square().sum()
            pterm = param.detach().square().sum()
            grad_sq = gterm if grad_sq is None else grad_sq + gterm
            param_sq = pterm if param_sq is None else param_sq + pterm
        if grad_sq is None or param_sq is None:
            continue
        grad_norm_block = torch.sqrt(grad_sq)
        total_grad_sq = grad_sq if total_grad_sq is None else total_grad_sq + grad_sq
        radius = float(trust.get(block_name, trust.get("default", 1.0e-4))) * max(
            float(torch.sqrt(param_sq).detach().cpu()),
            float(min_norm),
        )
        scale = min(1.0, radius / max(float(grad_norm_block.detach().cpu()), torch.finfo(param_sq.dtype).tiny))
        for param in block_params:
            if param.grad is None:
                continue
            params.append(param)
            directions.append(-scale * param.grad.detach().clone())
    if total_grad_sq is None:
        total_grad = torch.zeros(())
    else:
        total_grad = torch.sqrt(total_grad_sq)
    return params, directions, total_grad


def blockwise_armijo_step(
    model: nn.Module,
    closure: Callable[[], torch.Tensor],
    *,
    initial_alpha: float,
    max_alpha: float = 1.0,
    shrink: float = 0.5,
    grow: float = 2.0,
    c1: float = 1.0e-4,
    max_trials: int = 20,
    trust: dict[str, float] | None = None,
    min_norm: float = 1.0,
) -> BlockwiseArmijoResult:
    if trust is None:
        trust = {"default": 1.0e-4}
    model.zero_grad(set_to_none=True)
    base_loss = closure()
    base_loss.backward()
    blocks = named_parameter_blocks(model)
    params, directions, total_grad = _make_blockwise_descent(blocks, trust, min_norm=min_norm)
    if not params:
        return BlockwiseArmijoResult(False, 0.0, initial_alpha, float(base_loss.detach().cpu()), float(base_loss.detach().cpu()), 0.0)
    base_values = _snapshot(params)
    directional = _directional_derivative(params, directions)
    alpha = min(float(initial_alpha), float(max_alpha))
    accepted = False
    trial_value = float("nan")
    with torch.no_grad():
        for _ in range(int(max_trials)):
            _restore(params, base_values)
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=alpha)
            trial_loss = closure()
            trial_value = float(trial_loss.detach().cpu())
            sufficient = base_loss + float(c1) * alpha * directional
            if torch.isfinite(trial_loss) and bool((trial_loss <= sufficient).detach().cpu()):
                accepted = True
                break
            alpha *= float(shrink)
        if not accepted:
            _restore(params, base_values)
    next_alpha = min(float(max_alpha), alpha * float(grow)) if accepted else max(alpha, float(initial_alpha) * float(shrink))
    return BlockwiseArmijoResult(
        accepted=accepted,
        alpha=float(alpha if accepted else 0.0),
        next_initial_alpha=float(next_alpha),
        initial_loss=float(base_loss.detach().cpu()),
        trial_loss=float(trial_value),
        grad_norm=float(total_grad.detach().cpu()),
    )
