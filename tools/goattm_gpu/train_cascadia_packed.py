from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


DEFAULT_GOATTM_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm")
GOATTM_ROOT = Path(os.environ.get("QUAD_GOATTM_ROOT", str(DEFAULT_GOATTM_ROOT))).expanduser()
sys.path.insert(0, str(GOATTM_ROOT))

from quadrode_gpu_goattm import (  # noqa: E402
    ContinuousBatch,
    DenseLaggedMidpointStepper,
    DenseLinearA,
    DissipativeSkewA,
    DistributedReducedObjective,
    EnergyDenseQuadratic,
    EnergyTuckerTTQuadratic,
    LinearSource,
    MicroBatchReducedObjective,
    QuadraticDynamics,
    QuadraticReadoutDecoder,
    ReducedObjective,
    RungeKutta4Stepper,
    SubstepRungeKutta4Stepper,
    decoder_loss_and_state_grad,
    trapezoidal_weights,
)
import quadrode_gpu_goattm.distributed as goattm_distributed  # noqa: E402
import quadrode_gpu_goattm.microbatch as goattm_microbatch  # noqa: E402
import quadrode_gpu_goattm.reduced as goattm_reduced  # noqa: E402
from quadrode_gpu_goattm.varpro import DecoderNormalTerms, solve_decoder_normal_terms  # noqa: E402
from ldnet_cascadia.models import MLP  # noqa: E402

_library_decoder_loss_and_state_grad = decoder_loss_and_state_grad
_library_solve_decoder_normal_equation = goattm_reduced.solve_decoder_normal_equation
_library_distributed_assemble_decoder_normal_terms = goattm_distributed.assemble_decoder_normal_terms


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_reduce_max(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    return list(range(int(rank), int(total), int(world_size)))


def batch_from_payload(
    payload: dict,
    device: torch.device,
    *,
    sample_slice: slice | None = None,
    sample_indices: list[int] | None = None,
    time_mode: str = "normalized",
) -> ContinuousBatch:
    if sample_slice is not None and sample_indices is not None:
        raise ValueError("pass sample_slice or sample_indices, not both")
    if sample_indices is not None:
        qoi = payload["qoi"][:, sample_indices]
        inp = payload["input_values"][:, sample_indices]
        sample_ids = [payload["sample_ids"][i] for i in sample_indices]
    elif sample_slice is not None:
        qoi = payload["qoi"][:, sample_slice]
        inp = payload["input_values"][:, sample_slice]
        sample_ids = payload["sample_ids"][sample_slice]
    else:
        qoi = payload["qoi"]
        inp = payload["input_values"]
        sample_ids = payload["sample_ids"]
    midpoint_input = 0.5 * (inp[:-1] + inp[1:])
    observation_times = payload["observation_times"]
    if time_mode == "step_index":
        observation_times = torch.arange(observation_times.numel(), dtype=observation_times.dtype)
    elif time_mode != "normalized":
        raise ValueError(f"unknown time_mode {time_mode!r}")
    input_times = 0.5 * (observation_times[:-1] + observation_times[1:])
    return ContinuousBatch(
        sample_ids=tuple(str(x) for x in sample_ids),
        observation_times=observation_times.to(device=device, dtype=torch.float64),
        qoi=qoi.to(device=device, dtype=torch.float64),
        u0=None,
        input_times=input_times.to(device=device, dtype=torch.float64),
        input_values=midpoint_input.to(device=device, dtype=torch.float64),
    )


def load_packed_payload(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu")
    return payload, dict(payload.get("metadata", {}))


def load_packed_batch(
    path: Path,
    device: torch.device,
    *,
    sample_slice: slice | None = None,
    time_mode: str = "normalized",
) -> tuple[ContinuousBatch, dict]:
    payload, metadata = load_packed_payload(path)
    batch = batch_from_payload(payload, device, sample_slice=sample_slice, time_mode=time_mode)
    return batch, metadata


def limit_packed_payload_samples(payload: dict, limit: int) -> dict:
    if int(limit) <= 0:
        return payload
    total = int(payload["qoi"].shape[1])
    count = min(int(limit), total)
    limited = dict(payload)
    limited["sample_ids"] = list(payload["sample_ids"][:count])
    limited["qoi"] = payload["qoi"][:, :count]
    limited["input_values"] = payload["input_values"][:, :count]
    metadata = dict(payload.get("metadata", {}))
    metadata["sample_count_before_limit"] = total
    metadata["sample_count"] = count
    metadata["sample_limit"] = int(limit)
    if "qoi_shape" in metadata:
        metadata["qoi_shape_before_limit"] = metadata["qoi_shape"]
    if "input_shape" in metadata:
        metadata["input_shape_before_limit"] = metadata["input_shape"]
    metadata["qoi_shape"] = list(limited["qoi"].shape)
    metadata["input_shape"] = list(limited["input_values"].shape)
    limited["metadata"] = metadata
    return limited


def random_cross_pair_indices(latent_dim: int, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [(i, j) for i in range(int(latent_dim)) for j in range(i + 1, int(latent_dim))]
    if int(count) <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    if int(count) >= len(pairs):
        chosen = pairs
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(len(pairs), generator=generator)[: int(count)].tolist()
        chosen = [pairs[k] for k in sorted(perm)]
    idx_i = torch.tensor([p[0] for p in chosen], dtype=torch.long)
    idx_j = torch.tensor([p[1] for p in chosen], dtype=torch.long)
    return idx_i, idx_j


class MaskedCrossQuadraticReadoutDecoder(nn.Module):
    """Readout with all linear terms and a fixed random subset of cross terms."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        *,
        cross_terms: int,
        mask_seed: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        idx_i, idx_j = random_cross_pair_indices(self.latent_dim, int(cross_terms), int(mask_seed))
        self.register_buffer("quadratic_i", idx_i)
        self.register_buffer("quadratic_j", idx_j)
        self.include_quadratic = bool(idx_i.numel() > 0)
        self.feature_dim = self.latent_dim + int(idx_i.numel())
        self.readout = nn.Linear(self.feature_dim, self.output_dim, bias=bias)

    @property
    def quadratic_pairs(self) -> list[tuple[int, int]]:
        return [(int(i), int(j)) for i, j in zip(self.quadratic_i.cpu(), self.quadratic_j.cpu())]

    def features(self, u: torch.Tensor) -> torch.Tensor:
        if not self.include_quadratic:
            return u
        idx_i = self.quadratic_i.to(device=u.device)
        idx_j = self.quadratic_j.to(device=u.device)
        q = u.index_select(-1, idx_i) * u.index_select(-1, idx_j)
        return torch.cat((u, q), dim=-1)

    def feature_state_grad(self, u: torch.Tensor, feature_cotangent: torch.Tensor) -> torch.Tensor:
        grad = feature_cotangent[:, : self.latent_dim].clone()
        if not self.include_quadratic:
            return grad
        idx_i = self.quadratic_i.to(device=u.device)
        idx_j = self.quadratic_j.to(device=u.device)
        quad_cot = feature_cotangent[:, self.latent_dim :]
        scatter_i = idx_i.unsqueeze(0).expand(u.shape[0], -1)
        scatter_j = idx_j.unsqueeze(0).expand(u.shape[0], -1)
        grad.scatter_add_(1, scatter_i, quad_cot * u.index_select(1, idx_j))
        grad.scatter_add_(1, scatter_j, quad_cot * u.index_select(1, idx_i))
        return grad

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        flat = u.reshape(-1, u.shape[-1])
        out = self.readout(self.features(flat))
        return out.reshape(*u.shape[:-1], self.output_dim)


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    if text.strip() == "":
        return ()
    return tuple(int(part) for part in text.split(",") if part.strip())


class FixedCorrectionReadoutDecoder(nn.Module):
    """Linear varpro readout plus a frozen NN correction q = W z + b + N(z)."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden: tuple[int, ...],
        *,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.include_quadratic = False
        self.feature_dim = self.latent_dim
        self.readout = nn.Linear(self.feature_dim, self.output_dim, bias=bias)
        self.fixed_correction = MLP(
            self.latent_dim,
            self.output_dim,
            hidden,
            activation=activation,
            layer_norm=True,
            final_zero=True,
        )
        for param in self.fixed_correction.parameters():
            param.requires_grad_(False)

    def features(self, u: torch.Tensor) -> torch.Tensor:
        return u

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        flat = u.reshape(-1, u.shape[-1])
        out = self.readout(self.features(flat)) + self.fixed_correction(flat)
        return out.reshape(*u.shape[:-1], self.output_dim)


def _has_fixed_correction(decoder: nn.Module) -> bool:
    return hasattr(decoder, "fixed_correction") and getattr(decoder, "fixed_correction") is not None


def fixed_correction_assemble_decoder_normal_terms(
    decoder,
    states: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    chunk_size: int = 8192,
) -> DecoderNormalTerms:
    if not _has_fixed_correction(decoder):
        return _library_distributed_assemble_decoder_normal_terms(
            decoder,
            states,
            targets,
            weights=weights,
            chunk_size=chunk_size,
        )
    with torch.no_grad():
        if states.shape[:-1] != targets.shape[:-1]:
            raise ValueError("states and targets must have matching leading dimensions")
        leading = int(states.shape[:-1].numel())
        flat_states = states.reshape(leading, states.shape[-1])
        flat_targets = targets.reshape(leading, targets.shape[-1])
        feature_dim = int(decoder.feature_dim) + int(decoder.readout.bias is not None)
        output_dim = int(targets.shape[-1])
        normal = states.new_zeros(feature_dim, feature_dim)
        rhs = states.new_zeros(output_dim, feature_dim)
        flat_weights = None
        if weights is not None:
            if weights.shape != states.shape[:-1]:
                raise ValueError("weights must match states leading dimensions")
            flat_weights = weights.reshape(leading).to(device=states.device, dtype=states.dtype)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, leading, chunk_size):
            end = min(start + chunk_size, leading)
            u = flat_states[start:end]
            features = decoder.features(u)
            if decoder.readout.bias is not None:
                ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
                features = torch.cat((features, ones), dim=1)
            adjusted_targets = flat_targets[start:end] - decoder.fixed_correction(u)
            weighted = features if flat_weights is None else features * flat_weights[start:end, None]
            normal.add_(features.T @ weighted)
            rhs.add_(adjusted_targets.T @ weighted)
    return DecoderNormalTerms(
        normal_matrix=normal,
        rhs=rhs,
        feature_dim=feature_dim,
        observation_count=leading,
    )


def fixed_correction_solve_decoder_normal_equation(
    decoder,
    states: torch.Tensor,
    targets: torch.Tensor,
    ridge: float = 1.0e-8,
    weights: torch.Tensor | None = None,
    chunk_size: int = 8192,
):
    if not _has_fixed_correction(decoder):
        return _library_solve_decoder_normal_equation(
            decoder,
            states,
            targets,
            ridge=ridge,
            weights=weights,
            chunk_size=chunk_size,
        )
    terms = fixed_correction_assemble_decoder_normal_terms(
        decoder,
        states,
        targets,
        weights=weights,
        chunk_size=chunk_size,
    )
    return solve_decoder_normal_terms(decoder, terms, ridge=ridge)


def fixed_correction_decoder_loss_and_state_grad(
    decoder,
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    chunk_size: int = 8192,
    return_prediction: bool = True,
    return_state_grad: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
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
    prediction = torch.empty_like(flat_targets) if return_prediction else None
    state_grads = torch.empty_like(flat_states) if return_state_grad else None
    data_loss = states.new_zeros(())
    chunk_size = max(1, int(chunk_size))
    for start in range(0, leading, chunk_size):
        end = min(start + chunk_size, leading)
        u_base = flat_states[start:end]
        if return_state_grad:
            with torch.enable_grad():
                u = u_base.detach().requires_grad_(True)
                pred = decoder.features(u) @ weight.T
                if bias is not None:
                    pred = pred + bias
                pred = pred + decoder.fixed_correction(u)
                residual = pred - flat_targets[start:end]
                if flat_weights is None:
                    chunk_loss = 0.5 * residual.square().sum()
                else:
                    chunk_loss = 0.5 * (residual.square().sum(dim=-1) * flat_weights[start:end]).sum()
                grad = torch.autograd.grad(chunk_loss, u, retain_graph=False, create_graph=False)[0]
            state_grads[start:end] = grad.detach()
        else:
            with torch.no_grad():
                pred = decoder.features(u_base) @ weight.T
                if bias is not None:
                    pred = pred + bias
                pred = pred + decoder.fixed_correction(u_base)
                residual = pred - flat_targets[start:end]
                if flat_weights is None:
                    chunk_loss = 0.5 * residual.square().sum()
                else:
                    chunk_loss = 0.5 * (residual.square().sum(dim=-1) * flat_weights[start:end]).sum()
        data_loss = data_loss + chunk_loss.detach()
        if prediction is not None:
            prediction[start:end] = pred.detach()
    return (
        None if prediction is None else prediction.reshape_as(targets),
        data_loss,
        None if state_grads is None else state_grads.reshape_as(states),
    )


def masked_aware_decoder_loss_and_state_grad(
    decoder,
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    chunk_size: int = 8192,
    return_prediction: bool = True,
    return_state_grad: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    if _has_fixed_correction(decoder):
        return fixed_correction_decoder_loss_and_state_grad(
            decoder,
            states,
            targets,
            weights=weights,
            chunk_size=chunk_size,
            return_prediction=return_prediction,
            return_state_grad=return_state_grad,
        )
    if not hasattr(decoder, "feature_state_grad"):
        return _library_decoder_loss_and_state_grad(
            decoder,
            states,
            targets,
            weights=weights,
            chunk_size=chunk_size,
            return_prediction=return_prediction,
            return_state_grad=return_state_grad,
        )
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
            state_grads[start:end] = decoder.feature_state_grad(u, weighted_residual @ weight)
        if prediction is not None:
            prediction[start:end] = pred
    return (
        None if prediction is None else prediction.reshape_as(targets),
        data_loss,
        None if state_grads is None else state_grads.reshape_as(states),
    )


decoder_loss_and_state_grad = masked_aware_decoder_loss_and_state_grad
goattm_reduced.decoder_loss_and_state_grad = masked_aware_decoder_loss_and_state_grad
goattm_reduced.solve_decoder_normal_equation = fixed_correction_solve_decoder_normal_equation
goattm_distributed.assemble_decoder_normal_terms = fixed_correction_assemble_decoder_normal_terms
goattm_distributed.decoder_loss_and_state_grad = masked_aware_decoder_loss_and_state_grad
goattm_microbatch.assemble_decoder_normal_terms = fixed_correction_assemble_decoder_normal_terms
goattm_microbatch.decoder_loss_and_state_grad = masked_aware_decoder_loss_and_state_grad


def make_dynamics(
    latent_dim: int,
    input_dim: int,
    init_scale: float,
    device: torch.device,
    *,
    linear_a: str,
    a_rank: int,
    damping_init: float,
    damping_shift: float,
    quadratic: str,
    h_reduced_rank: int,
    h_tt_rank: int,
) -> QuadraticDynamics:
    if linear_a == "dense":
        linear = DenseLinearA(latent_dim, init_scale=init_scale)
    elif linear_a == "dissipative_skew":
        linear = DissipativeSkewA(
            latent_dim,
            skew_rank=int(a_rank),
            damping_init=float(damping_init),
            factor_scale=float(init_scale),
            damping_shift=float(damping_shift),
        )
    else:
        raise ValueError(f"unknown linear_a {linear_a!r}")

    if quadratic == "energy_dense":
        quad = EnergyDenseQuadratic(latent_dim, scale=init_scale)
    elif quadratic == "energy_tucker":
        quad = EnergyTuckerTTQuadratic(
            latent_dim,
            reduced_rank=int(h_reduced_rank),
            tt_rank=int(h_tt_rank),
            scale=init_scale,
            basis_trainable=True,
        )
    else:
        raise ValueError(f"unknown quadratic {quadratic!r}")

    return QuadraticDynamics(
        linear,
        quad,
        LinearSource(latent_dim, input_dim, init_scale=init_scale),
    ).double().to(device)


def full_quadratic_pair_index(i: int, j: int, latent_dim: int) -> int:
    if i > j:
        i, j = j, i
    return i * latent_dim - (i * (i - 1)) // 2 + (j - i)


def apply_pod_initializer(dynamics: QuadraticDynamics, decoder: QuadraticReadoutDecoder, path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    source_dim = int(data["a_matrix"].shape[0])
    if source_dim > int(dynamics.latent_dim):
        raise ValueError("initializer latent_dim exceeds model latent_dim")
    with torch.no_grad():
        a_matrix = torch.as_tensor(data["a_matrix"], device=next(dynamics.linear.parameters()).device, dtype=torch.float64)
        if isinstance(dynamics.linear, DenseLinearA):
            dynamics.linear.A.zero_()
            dynamics.linear.A[:source_dim, :source_dim].copy_(
                a_matrix.to(device=dynamics.linear.A.device, dtype=dynamics.linear.A.dtype)
            )
            linear_init = "dense_opinf_block"
        elif isinstance(dynamics.linear, DissipativeSkewA):
            diagonal = torch.diag(a_matrix).to(device=dynamics.linear.raw_damping.device, dtype=dynamics.linear.raw_damping.dtype)
            damping = torch.clamp(float(dynamics.linear.damping_shift) - diagonal, min=1.0e-6)
            dynamics.linear.raw_damping[:source_dim].copy_(torch.sqrt(damping))
            linear_init = "dissipative_skew_diagonal_from_opinf"
        else:
            linear_init = "skipped"
        dynamics.source.B.zero_()
        dynamics.source.B[:source_dim, : data["b_matrix"].shape[1]].copy_(
            torch.as_tensor(data["b_matrix"], device=dynamics.source.B.device, dtype=dynamics.source.B.dtype)
        )
        if dynamics.source.c is not None:
            dynamics.source.c.zero_()
            dynamics.source.c[:source_dim].copy_(
                torch.as_tensor(data["c_vector"], device=dynamics.source.c.device, dtype=dynamics.source.c.dtype)
            )
        if isinstance(dynamics.quadratic, EnergyDenseQuadratic) and "mu_h" in data:
            mu_h = torch.as_tensor(data["mu_h"], device=dynamics.quadratic.free_values.device, dtype=dynamics.quadratic.free_values.dtype)
            dynamics.quadratic.free_values.zero_()
            dynamics.quadratic.free_values[: mu_h.numel()].copy_(mu_h)
            quadratic_init = "energy_dense_zero_mu_h"
        elif isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic):
            quadratic_init = "energy_tucker_small_random"
        else:
            quadratic_init = "skipped"
        decoder.readout.weight.zero_()
        if decoder.readout.bias is not None:
            decoder.readout.bias.zero_()
            decoder.readout.bias.copy_(
                torch.as_tensor(data["decoder_template_v0"], device=decoder.readout.bias.device, dtype=decoder.readout.bias.dtype)
            )
        v1 = torch.as_tensor(data["decoder_template_v1"], device=decoder.readout.weight.device, dtype=decoder.readout.weight.dtype)
        decoder.readout.weight[:, :source_dim].copy_(v1[:, :source_dim])
        if decoder.include_quadratic and "decoder_template_v2" in data:
            v2 = np.asarray(data["decoder_template_v2"], dtype=np.float64)
            if hasattr(decoder, "quadratic_pairs"):
                for offset, (i, j) in enumerate(decoder.quadratic_pairs):
                    if i < source_dim and j < source_dim:
                        src = full_quadratic_pair_index(i, j, source_dim)
                        dst = int(dynamics.latent_dim) + offset
                        decoder.readout.weight[:, dst].copy_(
                            torch.as_tensor(v2[:, src], device=decoder.readout.weight.device, dtype=decoder.readout.weight.dtype)
                        )
            else:
                dst = int(dynamics.latent_dim)
                for i in range(int(dynamics.latent_dim)):
                    for j in range(i, int(dynamics.latent_dim)):
                        if i < source_dim and j < source_dim:
                            src = full_quadratic_pair_index(i, j, source_dim)
                            decoder.readout.weight[:, dst].copy_(
                                torch.as_tensor(v2[:, src], device=decoder.readout.weight.device, dtype=decoder.readout.weight.dtype)
                            )
                        dst += 1
    return {
        "path": str(path),
        "source_latent_dim": source_dim,
        "max_real_before_shift": float(data["max_real_before_shift"]) if "max_real_before_shift" in data else None,
        "max_real_after_shift": float(data["max_real_after_shift"]) if "max_real_after_shift" in data else None,
        "linear_init": linear_init,
        "quadratic_init": quadratic_init,
    }


def grad_norm(dynamics: QuadraticDynamics) -> float:
    total = None
    for param in dynamics.parameters():
        if param.grad is None:
            continue
        piece = param.grad.detach().square().sum()
        total = piece if total is None else total + piece
    return 0.0 if total is None else float(torch.sqrt(total).detach().cpu())


def a_symmetric_spectral_terms(
    dynamics: QuadraticDynamics,
    temperature: float,
    tolerance: float,
) -> dict[str, torch.Tensor]:
    a_matrix = dynamics.linear.dense_matrix()
    symmetric_part = 0.5 * (a_matrix + a_matrix.T)
    eigvals = torch.linalg.eigvalsh(symmetric_part)
    lambda_max = eigvals[-1]
    excess = lambda_max - float(tolerance)
    if float(temperature) > 0.0:
        smooth_positive = float(temperature) * torch.nn.functional.softplus(excess / float(temperature))
    else:
        smooth_positive = torch.clamp(excess, min=0.0)
    return {
        "lambda_max": lambda_max,
        "excess": excess,
        "smooth_positive": smooth_positive,
    }


def a_symmetric_spectral_report(
    dynamics: QuadraticDynamics,
    *,
    weight: float,
    temperature: float,
    tolerance: float,
) -> dict[str, float]:
    if float(weight) <= 0.0:
        return {}
    with torch.no_grad():
        terms = a_symmetric_spectral_terms(dynamics, float(temperature), float(tolerance))
        penalty = 0.5 * float(weight) * terms["smooth_positive"].square()
    return {
        "a_symmetric_lambda_max": float(terms["lambda_max"].detach().cpu()),
        "a_symmetric_tolerance": float(tolerance),
        "a_symmetric_excess": float(terms["excess"].detach().cpu()),
        "a_symmetric_smooth_positive": float(terms["smooth_positive"].detach().cpu()),
        "a_symmetric_penalty": float(penalty.detach().cpu()),
    }


def install_a_symmetric_spectral_penalty(
    objective: ReducedObjective,
    *,
    weight: float,
    temperature: float,
    tolerance: float,
    require_dense: bool = False,
) -> None:
    if float(weight) <= 0.0:
        return
    if require_dense and not isinstance(objective.dynamics.linear, DenseLinearA):
        raise ValueError("--dense-a-symmetric-penalty requires --linear-a dense")

    original_dynamics_regularization = objective._dynamics_regularization
    original_add_dynamics_regularization_gradient = objective._add_dynamics_regularization_gradient

    def spectral_penalty() -> torch.Tensor:
        terms = a_symmetric_spectral_terms(objective.dynamics, float(temperature), float(tolerance))
        return 0.5 * float(weight) * terms["smooth_positive"].square()

    def patched_dynamics_regularization() -> torch.Tensor:
        return original_dynamics_regularization() + spectral_penalty()

    def patched_add_dynamics_regularization_gradient(grads: dict[str, torch.Tensor]) -> None:
        original_add_dynamics_regularization_gradient(grads)
        linear_named_params = list(objective.dynamics.linear.named_parameters())
        linear_params = [param for _, param in linear_named_params]
        with torch.enable_grad():
            penalty = spectral_penalty()
            param_grads = torch.autograd.grad(
                penalty,
                linear_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        for local_name, param_grad in zip((name for name, _ in linear_named_params), param_grads):
            if param_grad is None:
                continue
            param_name = f"linear.{local_name}"
            if param_name in grads:
                grads[param_name] = grads[param_name] + param_grad.to(
                    device=grads[param_name].device,
                    dtype=grads[param_name].dtype,
                )
            else:
                grads[param_name] = param_grad.detach().clone()

    objective._dynamics_regularization = patched_dynamics_regularization
    objective._add_dynamics_regularization_gradient = patched_add_dynamics_regularization_gradient


def project_dense_a_symmetric_positive_part(
    dynamics: QuadraticDynamics,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    if not isinstance(dynamics.linear, DenseLinearA):
        return {"applied": False}
    with torch.no_grad():
        a_matrix = dynamics.linear.A
        symmetric = 0.5 * (a_matrix + a_matrix.T)
        skew = 0.5 * (a_matrix - a_matrix.T)
        eigvals, eigvecs = torch.linalg.eigh(symmetric)
        clipped = torch.clamp(eigvals, max=float(tolerance))
        projected_symmetric = (eigvecs * clipped.unsqueeze(0)) @ eigvecs.T
        a_matrix.copy_(skew + projected_symmetric)
        positive = torch.clamp(eigvals - float(tolerance), min=0.0)
        return {
            "applied": True,
            "tolerance": float(tolerance),
            "lambda_max_before": float(eigvals[-1].detach().cpu()),
            "lambda_max_after": float(clipped[-1].detach().cpu()),
            "positive_fro_norm_removed": float(torch.linalg.norm(positive).detach().cpu()),
            "positive_rank_removed": int((positive > 0).sum().detach().cpu()),
        }


def dense_a_symmetric_spectral_terms(
    dynamics: QuadraticDynamics,
    temperature: float,
) -> dict[str, torch.Tensor]:
    if not isinstance(dynamics.linear, DenseLinearA):
        raise TypeError("dense A symmetric spectral penalty requires --linear-a dense")
    return a_symmetric_spectral_terms(dynamics, float(temperature), 0.0)


def dense_a_symmetric_spectral_report(
    dynamics: QuadraticDynamics,
    *,
    weight: float,
    temperature: float,
) -> dict[str, float]:
    return a_symmetric_spectral_report(dynamics, weight=weight, temperature=temperature, tolerance=0.0)


def install_dense_a_symmetric_spectral_penalty(
    objective: ReducedObjective,
    *,
    weight: float,
    temperature: float,
) -> None:
    install_a_symmetric_spectral_penalty(
        objective,
        weight=weight,
        temperature=temperature,
        tolerance=0.0,
        require_dense=True,
    )


def flatten_parameters(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in parameters])


def flatten_gradients(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for param in parameters:
        if param.grad is None:
            pieces.append(torch.zeros_like(param.detach()).reshape(-1))
        else:
            pieces.append(param.grad.detach().reshape(-1))
    return torch.cat(pieces)


def assign_flat_parameters(parameters: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in parameters:
            count = param.numel()
            param.copy_(flat[offset : offset + count].view_as(param))
            offset += count


def lbfgs_direction_from_history(
    grad: torch.Tensor,
    history_s: list[torch.Tensor],
    history_y: list[torch.Tensor],
    history_rho: list[torch.Tensor],
) -> torch.Tensor:
    if not history_s:
        return -grad
    q = grad.clone()
    alpha_values = []
    for s_vec, y_vec, rho in zip(reversed(history_s), reversed(history_y), reversed(history_rho)):
        alpha = rho * torch.dot(s_vec, q)
        alpha_values.append(alpha)
        q = q - alpha * y_vec
    y_last = history_y[-1]
    s_last = history_s[-1]
    yy = torch.dot(y_last, y_last)
    gamma = torch.dot(s_last, y_last) / yy if float(yy.detach().cpu()) > 0.0 else grad.new_tensor(1.0)
    r = gamma * q
    for s_vec, y_vec, rho, alpha in zip(history_s, history_y, history_rho, reversed(alpha_values)):
        beta = rho * torch.dot(y_vec, r)
        r = r + s_vec * (alpha - beta)
    return -r


def weighted_target_energy(batch: ContinuousBatch) -> torch.Tensor:
    weights = trapezoidal_weights(batch.observation_times)[:, None].expand(batch.qoi.shape[:-1])
    return 0.5 * (batch.qoi.square().sum(dim=-1) * weights).sum()


def raw_qoi_error_terms(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    observation_times: torch.Tensor,
    qoi_stats: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_norm = target_norm.to(device=prediction_norm.device, dtype=prediction_norm.dtype)
    observation_times = observation_times.to(device=prediction_norm.device, dtype=prediction_norm.dtype)
    scale = qoi_stats["scale"].to(device=prediction_norm.device, dtype=prediction_norm.dtype)
    mean = qoi_stats["mean"].to(device=prediction_norm.device, dtype=prediction_norm.dtype)
    weights = trapezoidal_weights(observation_times).to(device=prediction_norm.device, dtype=prediction_norm.dtype)
    residual_raw = (prediction_norm - target_norm) * scale
    target_raw = target_norm * scale + mean
    numerator = (residual_raw.square().sum(dim=-1) * weights[:, None]).sum()
    denominator = (target_raw.square().sum(dim=-1) * weights[:, None]).sum()
    return numerator, denominator


def validate_fixed_decoder(
    train_objective,
    base_objective: ReducedObjective,
    train_batch: ContinuousBatch,
    test_payload: dict,
    device: torch.device,
    *,
    validation_chunk_samples: int,
    rank: int = 0,
    world_size: int = 1,
    time_mode: str = "normalized",
) -> dict:
    start = time.perf_counter()
    qoi_stats = test_payload["qoi_stats"]
    train_eval = train_objective.evaluate(train_batch, return_prediction=True)
    train_energy = weighted_target_energy(train_batch).to(device=train_eval.data_loss.device, dtype=train_eval.data_loss.dtype)
    train_raw_num, train_raw_den = raw_qoi_error_terms(
        train_eval.prediction,
        train_batch.qoi,
        train_batch.observation_times,
        qoi_stats,
    )
    all_reduce_sum(train_energy)
    all_reduce_sum(train_raw_num)
    all_reduce_sum(train_raw_den)
    train_rel = float(torch.sqrt(train_eval.data_loss / train_energy).detach().cpu())
    train_raw_rel = float(torch.sqrt(train_raw_num / train_raw_den).detach().cpu())
    total = int(test_payload["qoi"].shape[1])
    test_loss = train_eval.data_loss.new_zeros(())
    test_energy = train_eval.data_loss.new_zeros(())
    test_raw_num = train_eval.data_loss.new_zeros(())
    test_raw_den = train_eval.data_loss.new_zeros(())
    local_test_indices = shard_indices(total, rank, world_size)
    for start_idx in range(0, len(local_test_indices), int(validation_chunk_samples)):
        chunk_indices = local_test_indices[start_idx : start_idx + int(validation_chunk_samples)]
        chunk = batch_from_payload(test_payload, device, sample_indices=chunk_indices, time_mode=time_mode)
        with torch.no_grad():
            rollout = base_objective.rollout(chunk)
            weights = base_objective._loss_weights(chunk)
            prediction, loss, _ = decoder_loss_and_state_grad(
                base_objective.decoder,
                rollout.states.detach(),
                chunk.qoi,
                weights=weights,
                chunk_size=min(base_objective.normal_chunk_size, 2048),
                return_prediction=True,
                return_state_grad=False,
            )
            test_loss = test_loss + loss
            test_energy = test_energy + weighted_target_energy(chunk).to(device=test_energy.device, dtype=test_energy.dtype)
            raw_num, raw_den = raw_qoi_error_terms(
                prediction,
                chunk.qoi,
                chunk.observation_times,
                qoi_stats,
            )
            test_raw_num = test_raw_num + raw_num
            test_raw_den = test_raw_den + raw_den
        del chunk, rollout
    all_reduce_sum(test_loss)
    all_reduce_sum(test_energy)
    all_reduce_sum(test_raw_num)
    all_reduce_sum(test_raw_den)
    test_rel = float(torch.sqrt(test_loss / test_energy).detach().cpu())
    test_raw_rel = float(torch.sqrt(test_raw_num / test_raw_den).detach().cpu())
    return {
        "seconds": time.perf_counter() - start,
        "train_data_loss": float(train_eval.data_loss.detach().cpu()),
        "train_relative_error": train_rel,
        "train_qoi_relative_error_normalized": train_rel,
        "train_qoi_relative_error_raw": train_raw_rel,
        "test_data_loss": float(test_loss.detach().cpu()),
        "test_relative_error": test_rel,
        "test_qoi_relative_error_normalized": test_rel,
        "test_qoi_relative_error_raw": test_raw_rel,
        "test_samples": total,
        "normal_relative_residual": float(train_eval.normal_solve.relative_residual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cascadia packed reduced GOATTM model.")
    parser.add_argument("--train-packed", required=True)
    parser.add_argument("--test-packed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--fixed-nn-checkpoint")
    parser.add_argument("--fixed-nn-decoder-hidden", default="128,128")
    parser.add_argument("--fixed-nn-activation", default="silu")
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--linear-a", choices=("dense", "dissipative_skew"), default="dense")
    parser.add_argument("--a-rank", type=int, default=20)
    parser.add_argument("--a-damping-init", type=float, default=0.1)
    parser.add_argument("--a-damping-shift", type=float, default=0.0)
    parser.add_argument("--quadratic", choices=("energy_dense", "energy_tucker"), default="energy_dense")
    parser.add_argument("--h-reduced-rank", type=int, default=50)
    parser.add_argument("--h-tt-rank", type=int, default=16)
    parser.add_argument("--decoder-quadratic-mode", choices=("full", "masked_cross", "none"), default="full")
    parser.add_argument("--decoder-cross-terms", type=int, default=2000)
    parser.add_argument("--decoder-mask-seed", type=int, default=20260705)
    parser.add_argument("--optimizer", choices=("lbfgs", "lbfgs_armijo", "adam", "adam_armijo"), default="lbfgs")
    parser.add_argument("--optimizer-steps", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--line-search", choices=("none", "strong_wolfe", "armijo"), default="none")
    parser.add_argument("--armijo-c1", type=float, default=1.0e-4)
    parser.add_argument("--armijo-shrink", type=float, default=0.5)
    parser.add_argument("--armijo-grow", type=float, default=1.0)
    parser.add_argument("--armijo-max-alpha", type=float, default=0.0)
    parser.add_argument("--armijo-max-trials", type=int, default=10)
    parser.add_argument("--normalize-search-direction", action="store_true")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--reset-lbfgs-interval", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--validation-chunk-samples", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=512)
    parser.add_argument("--micro-cache-mode", choices=("none", "full"), default="none")
    parser.add_argument("--force-microbatch", action="store_true")
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--test-sample-limit", type=int, default=0)
    parser.add_argument("--stepper", choices=("lagged", "rk4", "rk4_substep"), default="lagged")
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--time-mode", choices=("normalized", "step_index"), default="normalized")
    parser.add_argument("--picard-iters", type=int, default=2)
    parser.add_argument("--decoder-ridge", type=float, default=1.0e-5)
    parser.add_argument("--dynamics-ridge", type=float, default=1.0e-7)
    parser.add_argument("--a-symmetric-spectral-penalty", type=float, default=0.0)
    parser.add_argument("--a-symmetric-spectral-temperature", type=float, default=1.0e-2)
    parser.add_argument("--a-symmetric-spectral-tolerance", type=float, default=0.0)
    parser.add_argument("--dense-a-symmetric-penalty", type=float, default=0.0)
    parser.add_argument("--dense-a-symmetric-temperature", type=float, default=1.0e-2)
    parser.add_argument("--project-initial-dense-a-symmetric-positive", action="store_true")
    parser.add_argument("--project-initial-dense-a-symmetric-tolerance", type=float, default=0.0)
    parser.add_argument("--normal-chunk-size", type=int, default=4096)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()
    use_armijo_lbfgs = args.optimizer == "lbfgs_armijo" or (
        args.optimizer == "lbfgs" and args.line_search == "armijo"
    )
    use_adam_armijo = args.optimizer == "adam_armijo"

    torch.set_default_dtype(torch.float64)
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    train_payload, train_meta = load_packed_payload(Path(args.train_packed))
    train_payload = limit_packed_payload_samples(train_payload, int(args.train_sample_limit))
    train_meta = dict(train_payload.get("metadata", train_meta))
    global_train_count = int(train_payload["qoi"].shape[1])
    local_train_indices = shard_indices(global_train_count, rank, world_size)
    if world_size > 1:
        train_batch = batch_from_payload(
            train_payload,
            device,
            sample_indices=local_train_indices,
            time_mode=args.time_mode,
        )
    else:
        train_batch = batch_from_payload(train_payload, torch.device("cpu"), time_mode=args.time_mode)
    test_payload, _ = load_packed_payload(Path(args.test_packed))
    test_payload = limit_packed_payload_samples(test_payload, int(args.test_sample_limit))
    input_dim = int(train_batch.input_values.shape[-1])
    output_dim = int(train_batch.qoi.shape[-1])
    dynamics = make_dynamics(
        int(args.latent_dim),
        input_dim,
        float(args.init_scale),
        device,
        linear_a=args.linear_a,
        a_rank=int(args.a_rank),
        damping_init=float(args.a_damping_init),
        damping_shift=float(args.a_damping_shift),
        quadratic=args.quadratic,
        h_reduced_rank=int(args.h_reduced_rank),
        h_tt_rank=int(args.h_tt_rank),
    )
    if args.fixed_nn_checkpoint:
        decoder = FixedCorrectionReadoutDecoder(
            int(args.latent_dim),
            output_dim,
            parse_hidden_dims(args.fixed_nn_decoder_hidden),
            activation=args.fixed_nn_activation,
            bias=True,
        ).double().to(device)
    elif args.decoder_quadratic_mode == "masked_cross":
        decoder = MaskedCrossQuadraticReadoutDecoder(
            int(args.latent_dim),
            output_dim,
            cross_terms=int(args.decoder_cross_terms),
            mask_seed=int(args.decoder_mask_seed),
            bias=True,
        ).double().to(device)
    else:
        decoder = QuadraticReadoutDecoder(
            int(args.latent_dim),
            output_dim,
            include_quadratic=args.decoder_quadratic_mode == "full",
            bias=True,
        ).double().to(device)
    init_report = apply_pod_initializer(dynamics, decoder, Path(args.initializer))
    fixed_nn_model_state = None
    if args.fixed_nn_checkpoint and args.load_checkpoint:
        raise ValueError("--fixed-nn-checkpoint and --load-checkpoint are mutually exclusive")
    if args.fixed_nn_checkpoint:
        checkpoint_path = Path(args.fixed_nn_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        fixed_nn_model_state = dict(checkpoint.get("model", checkpoint))
        dynamics_state = {
            key[len("dynamics.") :]: value
            for key, value in fixed_nn_model_state.items()
            if key.startswith("dynamics.")
        }
        correction_state = {
            key[len("decoder.correction.") :]: value
            for key, value in fixed_nn_model_state.items()
            if key.startswith("decoder.correction.")
        }
        if not dynamics_state:
            raise ValueError(f"{checkpoint_path} does not contain LDNet dynamics.* state")
        if not correction_state:
            raise ValueError(f"{checkpoint_path} does not contain LDNet decoder.correction.* state")
        dynamics.load_state_dict(dynamics_state)
        decoder.readout.weight.data.copy_(fixed_nn_model_state["decoder.linear_readout.weight"])
        decoder.readout.bias.data.copy_(fixed_nn_model_state["decoder.linear_readout.bias"])
        decoder.fixed_correction.load_state_dict(correction_state)
        for param in decoder.fixed_correction.parameters():
            param.requires_grad_(False)
        init_report["fixed_nn_checkpoint"] = str(checkpoint_path)
        init_report["fixed_nn_mode"] = "varpro_on_q_minus_frozen_nn_correction"
    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        init_report["loaded_checkpoint"] = str(checkpoint_path)
    if args.project_initial_dense_a_symmetric_positive:
        init_report["dense_a_symmetric_projection"] = project_dense_a_symmetric_positive_part(
            dynamics,
            tolerance=float(args.project_initial_dense_a_symmetric_tolerance),
        )
    if args.stepper == "lagged":
        stepper = DenseLaggedMidpointStepper(picard_iters=int(args.picard_iters))
        gradient_mode = "lagged_adjoint"
    elif args.stepper == "rk4":
        stepper = RungeKutta4Stepper()
        gradient_mode = "rk4_adjoint"
    elif args.stepper == "rk4_substep":
        stepper = SubstepRungeKutta4Stepper(substeps=int(args.rk4_substeps))
        gradient_mode = "rk4_adjoint"
    else:
        raise ValueError(f"unknown stepper {args.stepper!r}")

    base_objective = ReducedObjective(
        dynamics,
        decoder,
        stepper,
        decoder_ridge=float(args.decoder_ridge),
        dynamics_ridge=float(args.dynamics_ridge),
        normal_chunk_size=int(args.normal_chunk_size),
        gradient_mode=gradient_mode,
    )
    install_dense_a_symmetric_spectral_penalty(
        base_objective,
        weight=float(args.dense_a_symmetric_penalty),
        temperature=float(args.dense_a_symmetric_temperature),
    )
    install_a_symmetric_spectral_penalty(
        base_objective,
        weight=float(args.a_symmetric_spectral_penalty),
        temperature=float(args.a_symmetric_spectral_temperature),
        tolerance=float(args.a_symmetric_spectral_tolerance),
    )
    if world_size > 1 and not args.force_microbatch:
        train_objective = DistributedReducedObjective(base_objective)
        objective_mode = "distributed_exact_full_objective"
    elif args.force_microbatch or train_batch.batch_size > int(args.micro_batch_size):
        train_objective = MicroBatchReducedObjective(
            base_objective,
            micro_batch_size=int(args.micro_batch_size),
            cache_mode=args.micro_cache_mode,
        )
        objective_mode = "single_process_exact_microbatch_full_objective"
    else:
        train_objective = base_objective
        objective_mode = "single_process_full_batch_objective"
    dynamics_parameters = list(dynamics.parameters())

    def make_optimizer():
        if args.optimizer == "adam":
            return torch.optim.Adam(
                dynamics_parameters,
                lr=float(args.lr),
                betas=(float(args.adam_beta1), float(args.adam_beta2)),
                eps=float(args.adam_eps),
                weight_decay=float(args.adam_weight_decay),
            )
        if use_armijo_lbfgs or use_adam_armijo:
            return None
        return torch.optim.LBFGS(
            dynamics_parameters,
            lr=float(args.lr),
            max_iter=int(args.max_iter),
            history_size=int(args.history_size),
            line_search_fn=None if args.line_search == "none" else args.line_search,
        )

    optimizer = make_optimizer()
    log_path = output_dir / "optimization.jsonl"
    log = log_path.open("w", encoding="utf-8") if rank == 0 else None
    metadata = {
        "train_packed": str(Path(args.train_packed)),
        "test_packed": str(Path(args.test_packed)),
        "train_samples": int(train_batch.batch_size),
        "global_train_samples": int(global_train_count),
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "objective_mode": objective_mode,
        "steps": int(train_batch.steps),
        "dt": float(train_batch.step_size),
        "time_mode": args.time_mode,
        "stepper": args.stepper,
        "rk4_substeps": int(args.rk4_substeps) if args.stepper == "rk4_substep" else None,
        "gradient_mode": gradient_mode,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "latent_dim": int(args.latent_dim),
        "linear_a": args.linear_a,
        "linear_a_rank": int(args.a_rank) if args.linear_a == "dissipative_skew" else None,
        "dense_a_symmetric_penalty": {
            "weight": float(args.dense_a_symmetric_penalty),
            "temperature": float(args.dense_a_symmetric_temperature),
            "active": bool(float(args.dense_a_symmetric_penalty) > 0.0),
            "form": "0.5 * weight * (temperature * softplus(lambda_max(sym(A)) / temperature))^2",
        },
        "a_symmetric_spectral_penalty": {
            "weight": float(args.a_symmetric_spectral_penalty),
            "temperature": float(args.a_symmetric_spectral_temperature),
            "tolerance": float(args.a_symmetric_spectral_tolerance),
            "active": bool(float(args.a_symmetric_spectral_penalty) > 0.0),
            "form": "0.5 * weight * smooth_positive(lambda_max(sym(A)) - tolerance)^2",
        },
        "quadratic": args.quadratic,
        "h_reduced_rank": int(args.h_reduced_rank) if args.quadratic == "energy_tucker" else None,
        "h_tt_rank": int(args.h_tt_rank) if args.quadratic == "energy_tucker" else None,
        "decoder_quadratic_mode": args.decoder_quadratic_mode,
        "decoder_feature_dim": int(decoder.feature_dim),
        "decoder_cross_terms": int(decoder.quadratic_i.numel()) if hasattr(decoder, "quadratic_i") else None,
        "decoder_mask_seed": int(args.decoder_mask_seed) if args.decoder_quadratic_mode == "masked_cross" else None,
        "fixed_nn_checkpoint": str(Path(args.fixed_nn_checkpoint)) if args.fixed_nn_checkpoint else None,
        "fixed_nn_decoder_hidden": args.fixed_nn_decoder_hidden if args.fixed_nn_checkpoint else None,
        "initializer": init_report,
        "train_metadata": train_meta,
        "optimizer": {
            "name": args.optimizer,
            "effective_name": "lbfgs_armijo" if use_armijo_lbfgs else ("adam_armijo" if use_adam_armijo else args.optimizer),
            "optimizer_steps": int(args.optimizer_steps),
            "max_iter": int(args.max_iter),
            "lr": float(args.lr),
            "line_search": args.line_search,
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "adam_eps": float(args.adam_eps),
            "adam_weight_decay": float(args.adam_weight_decay),
            "reset_lbfgs_interval": int(args.reset_lbfgs_interval),
            "validation_interval": int(args.validation_interval),
            "checkpoint_interval": int(args.checkpoint_interval),
            "micro_batch_size": int(args.micro_batch_size),
            "micro_cache_mode": args.micro_cache_mode,
            "force_microbatch": bool(args.force_microbatch),
            "train_sample_limit": int(args.train_sample_limit),
            "test_sample_limit": int(args.test_sample_limit),
            "time_mode": args.time_mode,
            "stepper": args.stepper,
            "rk4_substeps": int(args.rk4_substeps),
            "armijo_c1": float(args.armijo_c1),
            "armijo_shrink": float(args.armijo_shrink),
            "armijo_grow": float(args.armijo_grow),
            "armijo_max_alpha": float(args.armijo_max_alpha),
            "armijo_max_trials": int(args.armijo_max_trials),
            "normalize_search_direction": bool(args.normalize_search_direction),
        },
    }
    if log is not None:
        log.write(json.dumps({"event": "metadata", **metadata}, sort_keys=True) + "\n")
        log.flush()

    validations = []

    def make_ldnet_compatible_state() -> dict[str, torch.Tensor] | None:
        if fixed_nn_model_state is None:
            return None
        state = {key: value.detach().cpu().clone() for key, value in fixed_nn_model_state.items()}
        for key, value in dynamics.state_dict().items():
            state[f"dynamics.{key}"] = value.detach().cpu().clone()
        state["decoder.linear_readout.weight"] = decoder.readout.weight.detach().cpu().clone()
        if decoder.readout.bias is not None:
            state["decoder.linear_readout.bias"] = decoder.readout.bias.detach().cpu().clone()
        for key, value in decoder.fixed_correction.state_dict().items():
            state[f"decoder.correction.{key}"] = value.detach().cpu().clone()
        return state

    def save_step_checkpoint(step: int) -> None:
        if rank != 0:
            return
        payload = {
            "metadata": metadata,
            "dynamics_state_dict": dynamics.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_step": int(step),
        }
        torch.save(payload, output_dir / f"checkpoint_step{int(step):04d}.pt")
        torch.save(payload, output_dir / "checkpoint_latest.pt")
        ldnet_state = make_ldnet_compatible_state()
        if ldnet_state is not None:
            torch.save(
                {
                    "model": ldnet_state,
                    "metadata": metadata,
                    "optimizer_step": int(step),
                    "stage": "fixed_nn_varpro",
                },
                output_dir / f"ldnet_checkpoint_step{int(step):04d}.pt",
            )
            torch.save(
                {
                    "model": ldnet_state,
                    "metadata": metadata,
                    "optimizer_step": int(step),
                    "stage": "fixed_nn_varpro",
                },
                output_dir / "ldnet_checkpoint_latest.pt",
            )

    initial_validation = validate_fixed_decoder(
        train_objective,
        base_objective,
        train_batch,
        test_payload,
        device,
        validation_chunk_samples=int(args.validation_chunk_samples),
        rank=rank,
        world_size=world_size,
        time_mode=args.time_mode,
    )
    validations.append({"optimizer_step": 0, "stage": "initial", **initial_validation})
    if log is not None:
        log.write(json.dumps({"event": "validation", **validations[-1]}, sort_keys=True) + "\n")
        log.flush()
        print(json.dumps({"event": "validation", **validations[-1]}, sort_keys=True), flush=True)

    closure_calls = 0
    value_calls = 0
    train_start = time.perf_counter()
    last_loss = float("nan")
    lbfgs_history_s: list[torch.Tensor] = []
    lbfgs_history_y: list[torch.Tensor] = []
    lbfgs_history_rho: list[torch.Tensor] = []
    pending_lbfgs_x: torch.Tensor | None = None
    pending_lbfgs_grad: torch.Tensor | None = None
    armijo_next_alpha = float(args.lr)
    adam_armijo_m: torch.Tensor | None = None
    adam_armijo_v: torch.Tensor | None = None
    adam_armijo_step = 0
    try:
        for step in range(1, int(args.optimizer_steps) + 1):
            if (
                args.optimizer in {"lbfgs", "lbfgs_armijo"}
                and int(args.reset_lbfgs_interval) > 0
                and step > 1
                and (step - 1) % int(args.reset_lbfgs_interval) == 0
            ):
                optimizer = make_optimizer()
                lbfgs_history_s.clear()
                lbfgs_history_y.clear()
                lbfgs_history_rho.clear()
                pending_lbfgs_x = None
                pending_lbfgs_grad = None
                if log is not None:
                    log.write(json.dumps({"event": "optimizer_reset", "optimizer_step": step}, sort_keys=True) + "\n")
                    log.flush()

            def closure() -> torch.Tensor:
                nonlocal closure_calls, last_loss
                t0 = time.perf_counter()
                result = train_objective.value_and_grad(train_batch)
                closure_calls += 1
                last_loss = float(result.loss.detach().cpu())
                record = {
                    "event": "closure",
                    "optimizer_step": step,
                    "closure_call": closure_calls,
                    "closure_seconds": time.perf_counter() - t0,
                    "elapsed_seconds": time.perf_counter() - train_start,
                    "loss": last_loss,
                    "data_loss": float(result.data_loss.detach().cpu()),
                    "decoder_regularization_loss": float(result.decoder_regularization_loss.detach().cpu()),
                    "dynamics_regularization_loss": float(result.dynamics_regularization_loss.detach().cpu()),
                    "normal_relative_residual": float(result.normal_solve.relative_residual),
                    "grad_norm": grad_norm(dynamics),
                }
                record.update(
                    a_symmetric_spectral_report(
                        dynamics,
                        weight=float(args.a_symmetric_spectral_penalty),
                        temperature=float(args.a_symmetric_spectral_temperature),
                        tolerance=float(args.a_symmetric_spectral_tolerance),
                    )
                )
                record.update(
                    dense_a_symmetric_spectral_report(
                        dynamics,
                        weight=float(args.dense_a_symmetric_penalty),
                        temperature=float(args.dense_a_symmetric_temperature),
                    )
                )
                if log is not None:
                    log.write(json.dumps(record, sort_keys=True) + "\n")
                    log.flush()
                return result.loss.detach()

            if args.optimizer == "adam":
                optimizer.zero_grad(set_to_none=True)
                closure()
                optimizer.step()
            elif use_adam_armijo:
                for param in dynamics_parameters:
                    param.grad = None
                result = closure()
                current_loss = float(result.detach().cpu())
                current_flat = flatten_parameters(dynamics_parameters)
                current_grad = flatten_gradients(dynamics_parameters)
                if float(args.adam_weight_decay) != 0.0:
                    current_grad = current_grad + float(args.adam_weight_decay) * current_flat

                if adam_armijo_m is None or adam_armijo_v is None:
                    adam_armijo_m = torch.zeros_like(current_grad)
                    adam_armijo_v = torch.zeros_like(current_grad)
                adam_armijo_step += 1
                beta1 = float(args.adam_beta1)
                beta2 = float(args.adam_beta2)
                adam_armijo_m = beta1 * adam_armijo_m + (1.0 - beta1) * current_grad
                adam_armijo_v = beta2 * adam_armijo_v + (1.0 - beta2) * current_grad.square()
                m_hat = adam_armijo_m / (1.0 - beta1**adam_armijo_step)
                v_hat = adam_armijo_v / (1.0 - beta2**adam_armijo_step)
                direction = -m_hat / (torch.sqrt(v_hat) + float(args.adam_eps))
                directional_derivative = torch.dot(current_grad, direction)
                if float(directional_derivative.detach().cpu()) >= 0.0:
                    direction = -current_grad
                    directional_derivative = torch.dot(current_grad, direction)
                    if log is not None:
                        log.write(
                            json.dumps(
                                {
                                    "event": "adam_armijo_direction_reset",
                                    "optimizer_step": step,
                                    "reason": "non_descent_direction",
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        log.flush()

                direction_norm = torch.linalg.vector_norm(direction)
                direction_norm_value = float(direction_norm.detach().cpu())
                if bool(args.normalize_search_direction) and direction_norm_value > 0.0:
                    direction = direction / direction_norm
                    directional_derivative = torch.dot(current_grad, direction)

                accepted = False
                accepted_alpha = 0.0
                accepted_loss = current_loss
                initial_alpha = float(armijo_next_alpha)
                alpha = initial_alpha
                c1 = float(args.armijo_c1)
                shrink = float(args.armijo_shrink)
                if not (0.0 < shrink < 1.0):
                    raise ValueError("--armijo-shrink must be between 0 and 1")
                grow = float(args.armijo_grow)
                if grow < 1.0:
                    raise ValueError("--armijo-grow must be at least 1")
                max_alpha = float(args.armijo_max_alpha)
                if max_alpha <= 0.0:
                    max_alpha = float("inf")
                gtd_value = float(directional_derivative.detach().cpu())
                trial_records = []
                for trial in range(1, int(args.armijo_max_trials) + 1):
                    trial_flat = current_flat + alpha * direction
                    assign_flat_parameters(dynamics_parameters, trial_flat)
                    t0 = time.perf_counter()
                    trial_result = train_objective.evaluate(train_batch, return_prediction=False)
                    value_calls += 1
                    trial_loss = float(trial_result.loss.detach().cpu())
                    threshold = current_loss + c1 * alpha * gtd_value
                    trial_accepted = bool(trial_loss <= threshold)
                    trial_record = {
                        "event": "adam_armijo_trial",
                        "optimizer_step": step,
                        "trial": trial,
                        "value_call": value_calls,
                        "alpha": alpha,
                        "accepted": trial_accepted,
                        "value_seconds": time.perf_counter() - t0,
                        "elapsed_seconds": time.perf_counter() - train_start,
                        "loss": trial_loss,
                        "data_loss": float(trial_result.data_loss.detach().cpu()),
                        "decoder_regularization_loss": float(trial_result.decoder_regularization_loss.detach().cpu()),
                        "dynamics_regularization_loss": float(trial_result.dynamics_regularization_loss.detach().cpu()),
                        "normal_relative_residual": float(trial_result.normal_solve.relative_residual),
                        "armijo_threshold": threshold,
                        "directional_derivative": gtd_value,
                        "direction_norm_before_normalization": direction_norm_value,
                        "search_direction_normalized": bool(args.normalize_search_direction),
                        "adam_armijo_m_norm": float(torch.linalg.vector_norm(adam_armijo_m).detach().cpu()),
                        "adam_armijo_v_norm": float(torch.linalg.vector_norm(adam_armijo_v).detach().cpu()),
                    }
                    trial_record.update(
                        a_symmetric_spectral_report(
                            dynamics,
                            weight=float(args.a_symmetric_spectral_penalty),
                            temperature=float(args.a_symmetric_spectral_temperature),
                            tolerance=float(args.a_symmetric_spectral_tolerance),
                        )
                    )
                    trial_record.update(
                        dense_a_symmetric_spectral_report(
                            dynamics,
                            weight=float(args.dense_a_symmetric_penalty),
                            temperature=float(args.dense_a_symmetric_temperature),
                        )
                    )
                    trial_records.append(trial_record)
                    if log is not None:
                        log.write(json.dumps(trial_record, sort_keys=True) + "\n")
                        log.flush()
                    if trial_accepted:
                        accepted = True
                        accepted_alpha = alpha
                        accepted_loss = trial_loss
                        break
                    alpha *= shrink

                if not accepted:
                    assign_flat_parameters(dynamics_parameters, current_flat)
                    armijo_next_alpha = alpha
                else:
                    armijo_next_alpha = min(grow * accepted_alpha, max_alpha)
                last_loss = accepted_loss
                if log is not None:
                    best_trial_loss = min((record["loss"] for record in trial_records), default=current_loss)
                    log.write(
                        json.dumps(
                            {
                                "event": "adam_armijo_step",
                                "optimizer_step": step,
                                "accepted": accepted,
                                "initial_alpha": initial_alpha,
                                "accepted_alpha": accepted_alpha,
                                "next_initial_alpha": armijo_next_alpha,
                                "grow": grow,
                                "max_alpha": None if max_alpha == float("inf") else max_alpha,
                                "start_loss": current_loss,
                                "accepted_loss": accepted_loss,
                                "best_trial_loss": best_trial_loss,
                                "direction_norm_before_normalization": direction_norm_value,
                                "search_direction_normalized": bool(args.normalize_search_direction),
                                "adam_armijo_inner_step": adam_armijo_step,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    log.flush()
            elif use_armijo_lbfgs:
                for param in dynamics_parameters:
                    param.grad = None
                result = closure()
                current_loss = float(result.detach().cpu())
                current_flat = flatten_parameters(dynamics_parameters)
                current_grad = flatten_gradients(dynamics_parameters)

                if pending_lbfgs_x is not None and pending_lbfgs_grad is not None:
                    s_vec = current_flat - pending_lbfgs_x
                    y_vec = current_grad - pending_lbfgs_grad
                    ys = torch.dot(y_vec, s_vec)
                    ys_value = float(ys.detach().cpu())
                    if ys_value > 1.0e-12:
                        lbfgs_history_s.append(s_vec.detach().clone())
                        lbfgs_history_y.append(y_vec.detach().clone())
                        lbfgs_history_rho.append((1.0 / ys).detach().clone())
                        if len(lbfgs_history_s) > int(args.history_size):
                            lbfgs_history_s.pop(0)
                            lbfgs_history_y.pop(0)
                            lbfgs_history_rho.pop(0)
                    elif log is not None:
                        log.write(
                            json.dumps(
                                {
                                    "event": "lbfgs_history_skip",
                                    "optimizer_step": step,
                                    "reason": "nonpositive_curvature",
                                    "s_dot_y": ys_value,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        log.flush()

                direction = lbfgs_direction_from_history(
                    current_grad,
                    lbfgs_history_s,
                    lbfgs_history_y,
                    lbfgs_history_rho,
                )
                directional_derivative = torch.dot(current_grad, direction)
                if float(directional_derivative.detach().cpu()) >= 0.0:
                    direction = -current_grad
                    directional_derivative = torch.dot(current_grad, direction)
                    lbfgs_history_s.clear()
                    lbfgs_history_y.clear()
                    lbfgs_history_rho.clear()
                    if log is not None:
                        log.write(
                            json.dumps(
                                {
                                    "event": "lbfgs_direction_reset",
                                    "optimizer_step": step,
                                    "reason": "non_descent_direction",
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        log.flush()

                direction_norm = torch.linalg.vector_norm(direction)
                direction_norm_value = float(direction_norm.detach().cpu())
                if bool(args.normalize_search_direction) and direction_norm_value > 0.0:
                    direction = direction / direction_norm
                    directional_derivative = torch.dot(current_grad, direction)

                accepted = False
                accepted_alpha = 0.0
                accepted_loss = current_loss
                initial_alpha = float(armijo_next_alpha)
                alpha = initial_alpha
                c1 = float(args.armijo_c1)
                shrink = float(args.armijo_shrink)
                if not (0.0 < shrink < 1.0):
                    raise ValueError("--armijo-shrink must be between 0 and 1")
                grow = float(args.armijo_grow)
                if grow < 1.0:
                    raise ValueError("--armijo-grow must be at least 1")
                max_alpha = float(args.armijo_max_alpha)
                if max_alpha <= 0.0:
                    max_alpha = float("inf")
                gtd_value = float(directional_derivative.detach().cpu())
                trial_records = []
                for trial in range(1, int(args.armijo_max_trials) + 1):
                    trial_flat = current_flat + alpha * direction
                    assign_flat_parameters(dynamics_parameters, trial_flat)
                    t0 = time.perf_counter()
                    trial_result = train_objective.evaluate(train_batch, return_prediction=False)
                    value_calls += 1
                    trial_loss = float(trial_result.loss.detach().cpu())
                    threshold = current_loss + c1 * alpha * gtd_value
                    trial_accepted = bool(trial_loss <= threshold)
                    trial_record = {
                        "event": "armijo_trial",
                        "optimizer_step": step,
                        "trial": trial,
                        "value_call": value_calls,
                        "alpha": alpha,
                        "accepted": trial_accepted,
                        "value_seconds": time.perf_counter() - t0,
                        "elapsed_seconds": time.perf_counter() - train_start,
                        "loss": trial_loss,
                        "data_loss": float(trial_result.data_loss.detach().cpu()),
                        "decoder_regularization_loss": float(trial_result.decoder_regularization_loss.detach().cpu()),
                        "dynamics_regularization_loss": float(trial_result.dynamics_regularization_loss.detach().cpu()),
                        "normal_relative_residual": float(trial_result.normal_solve.relative_residual),
                        "armijo_threshold": threshold,
                        "directional_derivative": gtd_value,
                        "direction_norm_before_normalization": direction_norm_value,
                        "search_direction_normalized": bool(args.normalize_search_direction),
                    }
                    trial_record.update(
                        a_symmetric_spectral_report(
                            dynamics,
                            weight=float(args.a_symmetric_spectral_penalty),
                            temperature=float(args.a_symmetric_spectral_temperature),
                            tolerance=float(args.a_symmetric_spectral_tolerance),
                        )
                    )
                    trial_record.update(
                        dense_a_symmetric_spectral_report(
                            dynamics,
                            weight=float(args.dense_a_symmetric_penalty),
                            temperature=float(args.dense_a_symmetric_temperature),
                        )
                    )
                    trial_records.append(trial_record)
                    if log is not None:
                        log.write(json.dumps(trial_record, sort_keys=True) + "\n")
                        log.flush()
                    if trial_accepted:
                        accepted = True
                        accepted_alpha = alpha
                        accepted_loss = trial_loss
                        break
                    alpha *= shrink

                if not accepted:
                    assign_flat_parameters(dynamics_parameters, current_flat)
                    pending_lbfgs_x = None
                    pending_lbfgs_grad = None
                    armijo_next_alpha = alpha
                else:
                    pending_lbfgs_x = current_flat.detach().clone()
                    pending_lbfgs_grad = current_grad.detach().clone()
                    armijo_next_alpha = min(grow * accepted_alpha, max_alpha)
                last_loss = accepted_loss
                if log is not None:
                    best_trial_loss = min((record["loss"] for record in trial_records), default=current_loss)
                    log.write(
                        json.dumps(
                            {
                                "event": "armijo_step",
                                "optimizer_step": step,
                                "accepted": accepted,
                                "initial_alpha": initial_alpha,
                                "accepted_alpha": accepted_alpha,
                                "next_initial_alpha": armijo_next_alpha,
                                "grow": grow,
                                "max_alpha": None if max_alpha == float("inf") else max_alpha,
                                "start_loss": current_loss,
                                "accepted_loss": accepted_loss,
                                "best_trial_loss": best_trial_loss,
                                "history_size": len(lbfgs_history_s),
                                "direction_norm_before_normalization": direction_norm_value,
                                "search_direction_normalized": bool(args.normalize_search_direction),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    log.flush()
            else:
                optimizer.step(closure)
            checkpoint_interval = int(args.checkpoint_interval)
            if checkpoint_interval > 0 and (step % checkpoint_interval == 0 or step == int(args.optimizer_steps)):
                save_step_checkpoint(step)
            if step % int(args.validation_interval) == 0 or step == int(args.optimizer_steps):
                validation = validate_fixed_decoder(
                    train_objective,
                    base_objective,
                    train_batch,
                    test_payload,
                    device,
                    validation_chunk_samples=int(args.validation_chunk_samples),
                    rank=rank,
                    world_size=world_size,
                    time_mode=args.time_mode,
                )
                validations.append({"optimizer_step": step, "stage": "periodic", **validation})
                if log is not None:
                    log.write(json.dumps({"event": "validation", **validations[-1]}, sort_keys=True) + "\n")
                    log.flush()
                    print(json.dumps({"event": "validation", **validations[-1]}, sort_keys=True), flush=True)
                if checkpoint_interval <= 0:
                    save_step_checkpoint(step)
    finally:
        if log is not None:
            log.close()

    checkpoint_path = output_dir / "checkpoint.pt"
    peak_memory = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0],
        device=device,
        dtype=torch.float64,
    )
    all_reduce_max(peak_memory)
    if rank == 0:
        final_payload = {
            "metadata": metadata,
            "dynamics_state_dict": dynamics.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        }
        torch.save(final_payload, checkpoint_path)
        ldnet_state = make_ldnet_compatible_state()
        if ldnet_state is not None:
            torch.save(
                {
                    "model": ldnet_state,
                    "metadata": metadata,
                    "stage": "fixed_nn_varpro",
                },
                output_dir / "ldnet_checkpoint.pt",
            )
        summary = {
            **metadata,
            "closure_calls": closure_calls,
            "last_loss": last_loss,
            "elapsed_seconds": time.perf_counter() - train_start,
            "peak_memory_mib": float(peak_memory.item()),
            "validations": validations,
            "checkpoint_path": str(checkpoint_path),
            "log_path": str(log_path),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({
            "closure_calls": closure_calls,
            "last_loss": last_loss,
            "peak_memory_mib": float(peak_memory.item()),
            "validations": validations,
            "checkpoint_path": str(checkpoint_path),
        }, indent=2))
    cleanup_distributed()


if __name__ == "__main__":
    main()
