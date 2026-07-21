from __future__ import annotations

from dataclasses import dataclass

import torch

from .decoders import QuadraticReadoutDecoder


@dataclass
class DecoderNormalSolveResult:
    normal_matrix: torch.Tensor
    rhs: torch.Tensor
    coefficients: torch.Tensor
    relative_residual: float
    feature_dim: int
    observation_count: int


@dataclass
class DecoderNormalTerms:
    normal_matrix: torch.Tensor
    rhs: torch.Tensor
    feature_dim: int
    observation_count: int


def assemble_decoder_normal_terms(
    decoder: QuadraticReadoutDecoder,
    states: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    chunk_size: int = 8192,
) -> DecoderNormalTerms:
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
            features = decoder.features(flat_states[start:end])
            if decoder.readout.bias is not None:
                ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
                features = torch.cat((features, ones), dim=1)
            if flat_weights is not None:
                weighted = features * flat_weights[start:end, None]
            else:
                weighted = features
            normal.add_(features.T @ weighted)
            rhs.add_(flat_targets[start:end].T @ weighted)
    return DecoderNormalTerms(
        normal_matrix=normal,
        rhs=rhs,
        feature_dim=feature_dim,
        observation_count=leading,
    )


def solve_decoder_normal_terms(
    decoder: QuadraticReadoutDecoder,
    terms: DecoderNormalTerms,
    ridge: float = 1.0e-8,
) -> DecoderNormalSolveResult:
    with torch.no_grad():
        normal = terms.normal_matrix + float(ridge) * torch.eye(
            terms.normal_matrix.shape[0],
            device=terms.normal_matrix.device,
            dtype=terms.normal_matrix.dtype,
        )
        rhs = terms.rhs
        try:
            coeff = torch.linalg.solve(normal, rhs.T).T
        except torch.linalg.LinAlgError:
            jitter = max(float(ridge), 1.0e-10)
            eye = torch.eye(normal.shape[0], device=normal.device, dtype=normal.dtype)
            coeff = torch.linalg.solve(normal + jitter * eye, rhs.T).T
        residual = normal @ coeff.T - rhs.T
        relative_residual = float((torch.linalg.norm(residual) / (1.0 + torch.linalg.norm(rhs.T))).detach().cpu())
        if decoder.readout.bias is not None:
            decoder.readout.weight.copy_(coeff[:, :-1])
            decoder.readout.bias.copy_(coeff[:, -1])
        else:
            decoder.readout.weight.copy_(coeff)
    return DecoderNormalSolveResult(
        normal_matrix=normal,
        rhs=rhs,
        coefficients=coeff,
        relative_residual=relative_residual,
        feature_dim=terms.feature_dim,
        observation_count=terms.observation_count,
    )


def solve_decoder_normal_equation(
    decoder: QuadraticReadoutDecoder,
    states: torch.Tensor,
    targets: torch.Tensor,
    ridge: float = 1.0e-8,
    weights: torch.Tensor | None = None,
    chunk_size: int = 8192,
) -> DecoderNormalSolveResult:
    terms = assemble_decoder_normal_terms(
        decoder,
        states,
        targets,
        weights=weights,
        chunk_size=chunk_size,
    )
    return solve_decoder_normal_terms(decoder, terms, ridge=ridge)
