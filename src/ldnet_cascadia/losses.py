from __future__ import annotations

from dataclasses import dataclass

import torch

from quadrode_gpu_goattm import trapezoidal_weights

from .data import CascadiaPackedBatch
from .models import CascadiaLDNet, LDNetForwardResult


@dataclass(frozen=True)
class LDNetLossConfig:
    data_weight: float = 1.0
    decoder_l2: float = 0.0
    latent_l2: float = 0.0
    dense_a_symmetric_penalty: float = 0.0
    dense_a_symmetric_temperature: float = 1.0e-2
    time_weighted: bool = True
    normalize_by_batch: bool = False


def smooth_positive_part(value: torch.Tensor, temperature: float) -> torch.Tensor:
    tau = float(temperature)
    if tau <= 0.0:
        return torch.clamp(value, min=0.0)
    return tau * torch.nn.functional.softplus(value / tau)


def dense_a_symmetric_energy_penalty(
    linear_or_matrix: object,
    *,
    weight: float,
    temperature: float = 1.0e-2,
) -> dict[str, torch.Tensor]:
    if hasattr(linear_or_matrix, "dense_matrix"):
        matrix = linear_or_matrix.dense_matrix()
    else:
        matrix = torch.as_tensor(linear_or_matrix)
    sym = 0.5 * (matrix + matrix.T)
    lambda_max = torch.linalg.eigvalsh(sym).amax()
    smooth_pos = smooth_positive_part(lambda_max, temperature)
    penalty = 0.5 * float(weight) * smooth_pos.square()
    return {
        "loss": penalty,
        "lambda_max": lambda_max,
        "smooth_positive": smooth_pos,
    }


def trajectory_data_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    times: torch.Tensor,
    *,
    time_weighted: bool = True,
    normalize_by_batch: bool = False,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}")
    residual_sq = (prediction - target).square().sum(dim=-1)
    if time_weighted:
        weights = trapezoidal_weights(times).to(device=residual_sq.device, dtype=residual_sq.dtype)
        residual_sq = residual_sq * weights[:, None]
    loss = 0.5 * residual_sq.sum()
    if normalize_by_batch:
        loss = loss / max(1, int(target.shape[1]))
    return loss


def relative_error(prediction: torch.Tensor, target: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
    residual_sq = (prediction - target).square().sum(dim=-1)
    target_sq = target.square().sum(dim=-1)
    if times is not None:
        weights = trapezoidal_weights(times).to(device=target.device, dtype=target.dtype)
        residual_sq = residual_sq * weights[:, None]
        target_sq = target_sq * weights[:, None]
    denom = target_sq.sum().clamp_min(torch.finfo(target.dtype).tiny)
    return torch.sqrt(residual_sq.sum() / denom)


def _module_l2(module: torch.nn.Module) -> torch.Tensor:
    total: torch.Tensor | None = None
    for param in module.parameters():
        term = param.square().sum()
        total = term if total is None else total + term
    if total is None:
        first = next(module.buffers(), None)
        if first is not None:
            return first.new_zeros(())
        return torch.zeros(())
    return total


def ldnet_loss(
    model: CascadiaLDNet,
    batch: CascadiaPackedBatch,
    result: LDNetForwardResult,
    config: LDNetLossConfig,
) -> dict[str, torch.Tensor]:
    data = trajectory_data_loss(
        result.prediction,
        batch.qoi,
        batch.observation_times,
        time_weighted=config.time_weighted,
        normalize_by_batch=config.normalize_by_batch,
    )
    total = float(config.data_weight) * data
    decoder_l2 = data.new_zeros(())
    latent_l2 = data.new_zeros(())
    if config.decoder_l2 > 0.0:
        decoder_l2 = 0.5 * float(config.decoder_l2) * _module_l2(model.decoder)
        total = total + decoder_l2
    if config.latent_l2 > 0.0:
        latent_l2 = 0.5 * float(config.latent_l2) * result.states.square().mean()
        total = total + latent_l2
    energy = {
        "loss": data.new_zeros(()),
        "lambda_max": data.new_zeros(()),
        "smooth_positive": data.new_zeros(()),
    }
    if config.dense_a_symmetric_penalty > 0.0:
        energy = dense_a_symmetric_energy_penalty(
            model.dynamics.linear,
            weight=config.dense_a_symmetric_penalty,
            temperature=config.dense_a_symmetric_temperature,
        )
        total = total + energy["loss"]
    rel = relative_error(result.prediction.detach(), batch.qoi, batch.observation_times)
    return {
        "loss": total,
        "data_loss": data.detach(),
        "decoder_l2": decoder_l2.detach(),
        "latent_l2": latent_l2.detach(),
        "a_symmetric_penalty": energy["loss"].detach(),
        "a_symmetric_lambda_max": energy["lambda_max"].detach(),
        "a_symmetric_smooth_positive": energy["smooth_positive"].detach(),
        "relative_error": rel.detach(),
    }
