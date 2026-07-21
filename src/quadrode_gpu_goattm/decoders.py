from __future__ import annotations

import torch
from torch import nn


def symmetric_quadratic_features(u: torch.Tensor) -> torch.Tensor:
    r = u.shape[-1]
    pieces = []
    for i in range(r):
        pieces.append(u[..., i:] * u[..., i : i + 1])
    return torch.cat(pieces, dim=-1)


def symmetric_quadratic_feature_indices(
    latent_dim: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(int(latent_dim), int(latent_dim), device=device)


def quadratic_feature_state_grad(u: torch.Tensor, quadratic_cotangent: torch.Tensor) -> torch.Tensor:
    """Apply the transpose Jacobian of upper-triangular quadratic features."""

    r = u.shape[-1]
    idx_i, idx_j = symmetric_quadratic_feature_indices(r, device=u.device)
    grad = torch.zeros_like(u)
    gather_i = u.index_select(1, idx_i)
    gather_j = u.index_select(1, idx_j)
    scatter_i = idx_i.unsqueeze(0).expand(u.shape[0], -1)
    scatter_j = idx_j.unsqueeze(0).expand(u.shape[0], -1)
    grad.scatter_add_(1, scatter_i, quadratic_cotangent * gather_j)
    grad.scatter_add_(1, scatter_j, quadratic_cotangent * gather_i)
    return grad


def decoder_loss_and_state_grad(
    decoder: "QuadraticReadoutDecoder",
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    chunk_size: int = 8192,
    return_prediction: bool = True,
    return_state_grad: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Evaluate decoder loss and d loss / d states without a feature graph."""

    if states.shape[:-1] != targets.shape[:-1]:
        raise ValueError("states and targets must have matching leading dimensions")
    leading = int(states.shape[:-1].numel())
    flat_states = states.detach().reshape(leading, states.shape[-1])
    flat_targets = targets.detach().reshape(leading, targets.shape[-1])
    flat_weights = None
    if weights is not None:
        if weights.shape != states.shape[:-1]:
            raise ValueError("weights must match states leading dimensions")
        flat_weights = weights.reshape(leading).to(device=states.device, dtype=states.dtype)

    weight = decoder.readout.weight.detach().to(device=states.device, dtype=states.dtype)
    bias = decoder.readout.bias
    if bias is not None:
        bias = bias.detach().to(device=states.device, dtype=states.dtype)
    state_grads = torch.empty_like(flat_states) if return_state_grad else None
    prediction = torch.empty_like(flat_targets) if return_prediction else None
    data_loss = states.new_zeros(())
    chunk_size = max(1, int(chunk_size))
    r = int(decoder.latent_dim)
    for start in range(0, leading, chunk_size):
        end = min(start + chunk_size, leading)
        u = flat_states[start:end]
        features = decoder.features(u)
        pred = features @ weight.T
        if bias is not None:
            pred = pred + bias
        residual = pred - flat_targets[start:end]
        if flat_weights is None:
            weighted_residual = residual
            data_loss = data_loss + 0.5 * residual.square().sum()
        else:
            w = flat_weights[start:end]
            weighted_residual = residual * w[:, None]
            data_loss = data_loss + 0.5 * (residual.square().sum(dim=-1) * w).sum()
        if state_grads is not None:
            feature_cotangent = weighted_residual @ weight
            grad = feature_cotangent[:, :r].clone()
            if decoder.include_quadratic:
                grad = grad + quadratic_feature_state_grad(u, feature_cotangent[:, r:])
            state_grads[start:end] = grad
        if prediction is not None:
            prediction[start:end] = pred
    return (
        None if prediction is None else prediction.reshape_as(targets),
        data_loss,
        None if state_grads is None else state_grads.reshape_as(states),
    )


class QuadraticReadoutDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, include_quadratic: bool = True, bias: bool = True) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.include_quadratic = bool(include_quadratic)
        quad_dim = self.latent_dim * (self.latent_dim + 1) // 2 if self.include_quadratic else 0
        self.feature_dim = self.latent_dim + quad_dim
        self.readout = nn.Linear(self.feature_dim, self.output_dim, bias=bias)

    def features(self, u: torch.Tensor) -> torch.Tensor:
        if not self.include_quadratic:
            return u
        return torch.cat((u, symmetric_quadratic_features(u)), dim=-1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        flat = u.reshape(-1, u.shape[-1])
        out = self.readout(self.features(flat))
        return out.reshape(*u.shape[:-1], self.output_dim)
