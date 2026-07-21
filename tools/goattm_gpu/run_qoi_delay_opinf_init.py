from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from quadrode_gpu_goattm import (  # noqa: E402
    ContinuousBatch,
    ContinuousDataset,
    DenseLaggedMidpointStepper,
    DissipativeSkewA,
    EnergyDenseQuadratic,
    LinearSource,
    QuadraticDynamics,
    QuadraticReadoutDecoder,
    ReducedObjective,
    load_manifest_npz,
)
from quadrode_gpu_goattm.decoders import symmetric_quadratic_features  # noqa: E402
from quadrode_gpu_goattm.varpro import solve_decoder_normal_equation  # noqa: E402
from tests.run_dataset_training import (  # noqa: E402
    load_batch,
    normalize_batch_channels,
    normalize_time,
    zero_latent_initial_state,
)


def make_delay_state(qoi: torch.Tensor, delay_count: int, delay_stride: int) -> torch.Tensor:
    if delay_count <= 0:
        raise ValueError("delay_count must be positive")
    if delay_stride <= 0:
        raise ValueError("delay_stride must be positive")
    pieces = []
    for lag in range(delay_count):
        shift = lag * delay_stride
        if lag == 0:
            pieces.append(qoi)
        else:
            pad = qoi[:1].expand(shift, *qoi.shape[1:])
            pieces.append(torch.cat((pad, qoi[:-shift]), dim=0))
    return torch.cat(pieces, dim=-1)


def central_difference(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    if values.shape[0] != times.numel():
        raise ValueError("time dimension mismatch")
    deriv = torch.empty_like(values)
    deriv[1:-1] = (values[2:] - values[:-2]) / (times[2:, None, None] - times[:-2, None, None])
    deriv[0] = (values[1] - values[0]) / (times[1] - times[0])
    deriv[-1] = (values[-1] - values[-2]) / (times[-1] - times[-2])
    return deriv


def fit_delay_pod(
    batch: ContinuousBatch,
    *,
    latent_dim: int,
    delay_count: int,
    delay_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = make_delay_state(batch.qoi, delay_count, delay_stride)
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(dim=0)
    centered = flat - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:latent_dim].contiguous()
    u_raw = (x - mean) @ basis.T
    u = u_raw - u_raw[:1]
    udot = central_difference(u, batch.observation_times)
    return u, udot, basis, mean


def assemble_opinf_normal(
    u: torch.Tensor,
    udot: torch.Tensor,
    inputs: torch.Tensor,
    *,
    ridge: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    leading = int(u.shape[0] * u.shape[1])
    r = int(u.shape[-1])
    m = int(inputs.shape[-1])
    feature_dim = r + r * (r + 1) // 2 + m + 1
    normal = u.new_zeros(feature_dim, feature_dim)
    rhs = u.new_zeros(r, feature_dim)
    flat_u = u.reshape(leading, r)
    flat_udot = udot.reshape(leading, r)
    flat_input = inputs.reshape(leading, m)
    for start in range(0, leading, int(chunk_size)):
        end = min(start + int(chunk_size), leading)
        uu = flat_u[start:end]
        features = torch.cat(
            (
                uu,
                symmetric_quadratic_features(uu),
                flat_input[start:end],
                torch.ones(end - start, 1, device=u.device, dtype=u.dtype),
            ),
            dim=1,
        )
        normal.add_(features.T @ features)
        rhs.add_(flat_udot[start:end].T @ features)
    normal.add_(float(ridge) * torch.eye(feature_dim, device=u.device, dtype=u.dtype))
    return normal, rhs


def symmetric_feature_pairs(latent_dim: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(latent_dim) for j in range(i, latent_dim)]


def coeff_to_tensor(coeff_quad: torch.Tensor, latent_dim: int) -> torch.Tensor:
    tensor = coeff_quad.new_zeros(latent_dim, latent_dim, latent_dim)
    pairs = symmetric_feature_pairs(latent_dim)
    for k, (i, j) in enumerate(pairs):
        if i == j:
            tensor[:, i, i] = coeff_quad[:, k]
        else:
            value = 0.5 * coeff_quad[:, k]
            tensor[:, i, j] = value
            tensor[:, j, i] = value
    return tensor


def set_dissipative_a(linear: DissipativeSkewA, dense_a: torch.Tensor) -> dict[str, float]:
    a = dense_a.detach().cpu().numpy()
    sym = 0.5 * (a + a.T)
    skew = 0.5 * (a - a.T)
    damping = np.maximum(-np.diag(sym), 1.0e-10)
    p = np.zeros((linear.latent_dim, linear.skew_rank), dtype=np.float64)
    q = np.zeros((linear.latent_dim, linear.skew_rank), dtype=np.float64)
    if linear.skew_rank > 0:
        try:
            from scipy.linalg import schur

            t, z = schur(skew, output="real")
            blocks: list[tuple[float, int]] = []
            i = 0
            while i < t.shape[0] - 1:
                if abs(t[i + 1, i]) > 1.0e-12 or abs(t[i, i + 1]) > 1.0e-12:
                    strength = 0.5 * (abs(t[i, i + 1]) + abs(t[i + 1, i]))
                    blocks.append((strength, i))
                    i += 2
                else:
                    i += 1
            blocks.sort(reverse=True, key=lambda item: item[0])
            for col, (_, i) in enumerate(blocks[: linear.skew_rank]):
                value = t[i, i + 1]
                scale = math.sqrt(abs(value))
                sign = 1.0 if value >= 0.0 else -1.0
                p[:, col] = scale * z[:, i]
                q[:, col] = sign * scale * z[:, i + 1]
        except Exception:
            u, s, _ = np.linalg.svd(skew, full_matrices=False)
            for col in range(min(linear.skew_rank, s.size // 2)):
                j = 2 * col
                scale = math.sqrt(max(s[j], 0.0))
                p[:, col] = scale * u[:, j]
                q[:, col] = scale * u[:, j + 1]
    with torch.no_grad():
        linear.raw_damping.copy_(torch.as_tensor(np.sqrt(damping), device=linear.raw_damping.device, dtype=linear.raw_damping.dtype))
        linear.P.copy_(torch.as_tensor(p, device=linear.P.device, dtype=linear.P.dtype))
        linear.Q.copy_(torch.as_tensor(q, device=linear.Q.device, dtype=linear.Q.dtype))
    projected = linear.dense_matrix().detach().cpu().numpy()
    return {
        "dense_to_dissipative_relative_error": float(np.linalg.norm(projected - a) / max(np.linalg.norm(a), 1.0e-30)),
        "dense_A_norm": float(np.linalg.norm(a)),
        "projected_A_norm": float(np.linalg.norm(projected)),
    }


def set_energy_dense_h(quadratic: EnergyDenseQuadratic, tensor: torch.Tensor) -> None:
    with torch.no_grad():
        values = tensor[
            quadratic.free_a.to(device=tensor.device),
            quadratic.free_b.to(device=tensor.device),
            quadratic.free_c.to(device=tensor.device),
        ]
        quadratic.free_values.copy_(values.to(device=quadratic.free_values.device, dtype=quadratic.free_values.dtype))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a QoI-delay OpInf initializer checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--delay-count", type=int, required=True)
    parser.add_argument("--delay-stride", type=int, default=1)
    parser.add_argument("--a-skew-rank", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--decoder-ridge", type=float, default=1.0e-5)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--picard-iters", type=int, default=2)
    parser.add_argument("--normalization-target-max-abs", type=float, default=0.9)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = load_batch(Path(args.manifest), args.train_count, device=device)
    batch = zero_latent_initial_state(batch)
    batch = normalize_time(batch)
    batch, stats = normalize_batch_channels(batch, target_max_abs=args.normalization_target_max_abs)
    if batch.input_values is None:
        raise ValueError("dataset has no inputs")

    u, udot, delay_basis, delay_mean = fit_delay_pod(
        batch,
        latent_dim=args.latent_dim,
        delay_count=args.delay_count,
        delay_stride=args.delay_stride,
    )
    normal, rhs = assemble_opinf_normal(
        u,
        udot,
        batch.input_values,
        ridge=args.ridge,
        chunk_size=args.chunk_size,
    )
    coeff = torch.linalg.solve(normal, rhs.T).T

    r = int(args.latent_dim)
    input_dim = int(batch.input_values.shape[-1])
    quad_dim = r * (r + 1) // 2
    dense_a = coeff[:, :r]
    coeff_quad = coeff[:, r : r + quad_dim]
    source_b = coeff[:, r + quad_dim : r + quad_dim + input_dim]
    source_c = coeff[:, -1]
    tensor = coeff_to_tensor(coeff_quad, r)

    a_rank = args.a_skew_rank if args.a_skew_rank is not None else r // 2
    dynamics = QuadraticDynamics(
        DissipativeSkewA(r, skew_rank=a_rank, damping_init=0.1, factor_scale=0.0),
        EnergyDenseQuadratic(r, scale=0.0),
        LinearSource(r, input_dim, init_scale=0.0),
    ).double().to(device)
    report_a = set_dissipative_a(dynamics.linear, dense_a)
    set_energy_dense_h(dynamics.quadratic, tensor)
    with torch.no_grad():
        dynamics.source.B.copy_(source_b.to(device=dynamics.source.B.device, dtype=dynamics.source.B.dtype))
        if dynamics.source.c is not None:
            dynamics.source.c.copy_(source_c.to(device=dynamics.source.c.device, dtype=dynamics.source.c.dtype))

    decoder = QuadraticReadoutDecoder(r, int(batch.qoi.shape[-1]), include_quadratic=True, bias=True).double().to(device)
    objective = ReducedObjective(
        dynamics,
        decoder,
        DenseLaggedMidpointStepper(picard_iters=args.picard_iters),
        decoder_ridge=args.decoder_ridge,
        gradient_mode="lagged_adjoint",
    )
    with torch.no_grad():
        rollout = objective.stepper.rollout_with_picard_history(
            objective.dynamics,
            objective._initial_state(batch),
            batch.step_size,
            batch.steps,
            p_mid=batch.midpoint_inputs(),
        )
        normal_decoder = solve_decoder_normal_equation(
            decoder,
            rollout.states,
            batch.qoi,
            ridge=args.decoder_ridge,
            weights=objective._loss_weights(batch),
            chunk_size=args.chunk_size,
        )
        eval_result = objective.evaluate(batch, return_prediction=False)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "initializer": "qoi_delay_opinf",
        "manifest": str(Path(args.manifest)),
        "train_count": int(args.train_count),
        "latent_dim": r,
        "delay_count": int(args.delay_count),
        "delay_stride": int(args.delay_stride),
        "delay_window_steps": int((args.delay_count - 1) * args.delay_stride),
        "pseudo_state_dim": int(delay_basis.shape[1]),
        "a_skew_rank": int(a_rank),
        "ridge": float(args.ridge),
        "decoder_ridge": float(args.decoder_ridge),
        "device": str(device),
        "normalization": stats.to_json(),
        "opinf_normal_relative_residual": float((torch.linalg.norm(normal @ coeff.T - rhs.T) / (1.0 + torch.linalg.norm(rhs.T))).detach().cpu()),
        "decoder_normal_relative_residual": float(normal_decoder.relative_residual),
        "initial_reduced_loss": float(eval_result.loss.detach().cpu()),
        "initial_data_loss": float(eval_result.data_loss.detach().cpu()),
        **report_a,
    }
    torch.save(
        {
            "metadata": metadata,
            "dynamics_state_dict": dynamics.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "delay_basis": delay_basis.detach().cpu(),
            "delay_mean": delay_mean.detach().cpu(),
        },
        outdir / "checkpoint.pt",
    )
    (outdir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("wrote", outdir / "checkpoint.pt")


if __name__ == "__main__":
    main()
