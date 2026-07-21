from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import time

import torch

from .data import ContinuousBatch
from .dynamics import QuadraticDynamics
from .incremental import (
    Direction,
    direction_from_flat,
    flatten_direction,
    lagged_midpoint_gauss_newton_hvp,
    lagged_midpoint_rollout_incremental,
    projected_decoder_residual_tangent,
)
from .linear import DenseLinearA, DissipativeSkewA
from .quadratic import EnergyTuckerTTQuadratic
from .reduced import ReducedObjective, trapezoidal_weights
from .source import LinearSource, ZeroSource
from .steppers import exact_frozen_solve
from .steppers import DensePicardRolloutResult
from .varpro import DecoderNormalSolveResult, solve_decoder_normal_equation


def _sync_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _masked_cross_feature_chunk_size(
    feature_count: int,
    row_count: int,
    direction_count: int,
    *,
    element_size: int,
    device: torch.device,
) -> int:
    """Choose the masked-cross feature chunk without creating GiB-scale temporaries."""
    feature_count = max(1, int(feature_count))
    requested = os.environ.get("GOATTM_MASKED_CROSS_FEATURE_CHUNK", "auto").strip().lower()
    if requested not in {"", "auto"}:
        return min(max(1, int(requested)), feature_count)

    preferred = min(512, feature_count)
    row_count = max(1, int(row_count))
    direction_count = max(1, int(direction_count))
    element_size = max(1, int(element_size))
    # The direct path materializes several m x k x feature_chunk tensors at once
    # through index_select/products/adds. Capping this estimate keeps high-k
    # probing from failing only because one masked-cross block is too large.
    budget_mib = float(os.environ.get("GOATTM_MASKED_CROSS_TEMP_BUDGET_MIB", "2048"))
    budget_bytes = max(64.0, budget_mib) * 1024.0 * 1024.0
    bytes_per_feature = 4.0 * float(row_count) * float(direction_count) * float(element_size)
    auto_chunk = int(budget_bytes // max(1.0, bytes_per_feature))
    return min(preferred, max(1, auto_chunk))


def _solve_decoder_normal(
    normal_matrix: torch.Tensor,
    rhs: torch.Tensor,
    normal_cholesky: torch.Tensor | None = None,
) -> torch.Tensor:
    if normal_cholesky is None:
        return torch.linalg.solve(normal_matrix, rhs)
    factor = normal_cholesky.to(device=rhs.device, dtype=rhs.dtype)
    return torch.cholesky_solve(rhs, factor, upper=False)


@dataclass
class ReducedGNCache:
    batch: ContinuousBatch
    u0: torch.Tensor
    p_mid: torch.Tensor | None
    weights: torch.Tensor | None
    rollout: DensePicardRolloutResult
    normal: DecoderNormalSolveResult
    normal_cholesky: torch.Tensor | None = None


@dataclass
class ReducedJVPResult:
    residual_dot: torch.Tensor
    decoder_coeff_dot: torch.Tensor
    residual_vector: torch.Tensor
    quadratic_form: torch.Tensor
    states_dot: torch.Tensor


@dataclass
class BatchedReducedJVPResult:
    residual_dot: torch.Tensor
    decoder_coeff_dot: torch.Tensor
    residual_vectors: torch.Tensor
    quadratic_form: torch.Tensor
    states_dot: torch.Tensor


@dataclass
class StreamingSketchJVPResult:
    sketch_matrix: torch.Tensor
    decoder_coeff_dot: torch.Tensor
    quadratic_form: torch.Tensor
    residual_dim: int
    sketch_dim: int
    direction_count: int


@dataclass
class BatchedEnergyTuckerDirectionCache:
    basis: torch.Tensor
    basis_dot: torch.Tensor
    reduced_tensor: torch.Tensor
    reduced_tensor_dot: torch.Tensor
    reduced_tensor_free_dot: torch.Tensor | None = None
    reconstruct_out_b: torch.Tensor | None = None
    reconstruct_out_c: torch.Tensor | None = None
    reconstruct_source_padded: torch.Tensor | None = None
    reconstruct_coeff_padded: torch.Tensor | None = None
    reconstruct_lengths: torch.Tensor | None = None
    solve_tau: float | None = None
    solve_basis_t: torch.Tensor | None = None
    solve_gram: torch.Tensor | None = None
    solve_eye: torch.Tensor | None = None
    linear_solve_tau: float | None = None
    linear_solve_scale: torch.Tensor | None = None
    linear_solve_v: torch.Tensor | None = None
    linear_solve_k_t: torch.Tensor | None = None
    linear_solve_small_inv_t: torch.Tensor | None = None
    direction_stacks: dict[str, torch.Tensor] | None = None


def _quadratic_cache_context(dynamics: QuadraticDynamics):
    cached = getattr(dynamics.quadratic, "cached_reduced_tensor", None)
    if cached is None:
        return nullcontext()
    return cached()


def _energy_reconstruction_padded_maps(
    quad: EnergyTuckerTTQuadratic,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = int(quad.reduced_rank)
    target = quad.reconstruct_target.to(device=device)
    source = quad.reconstruct_source.to(device=device)
    coeff = quad.reconstruct_coeff.to(device=device, dtype=dtype)
    out_a = target // (q * q)
    rem = target - out_a * q * q
    out_b = rem // q
    out_c = rem - out_b * q
    lengths = torch.stack([(out_a == a).sum() for a in range(q)]).to(device=device, dtype=torch.long)
    max_len = int(lengths.max().detach().cpu()) if lengths.numel() else 0
    b_pad = torch.zeros(q, max_len, device=device, dtype=torch.long)
    c_pad = torch.zeros(q, max_len, device=device, dtype=torch.long)
    src_pad = torch.zeros(q, max_len, device=device, dtype=torch.long)
    coeff_pad = torch.zeros(q, max_len, device=device, dtype=dtype)
    for a in range(q):
        idx = (out_a == a).nonzero(as_tuple=False).flatten()
        count = int(idx.numel())
        if count <= 0:
            continue
        b_pad[a, :count] = out_b.index_select(0, idx)
        c_pad[a, :count] = out_c.index_select(0, idx)
        src_pad[a, :count] = source.index_select(0, idx)
        coeff_pad[a, :count] = coeff.index_select(0, idx)
    return b_pad, c_pad, src_pad, coeff_pad, lengths


def _make_batched_energy_tucker_direction_cache(
    dynamics: QuadraticDynamics,
    directions: list[Direction],
    *,
    device: torch.device,
    dtype: torch.dtype,
    tau: float | None = None,
) -> BatchedEnergyTuckerDirectionCache | None:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyTuckerTTQuadratic):
        return None
    direction_stacks = {
        name: _direction_stack(directions, name, param).to(device=device, dtype=dtype)
        for name, param in dynamics.named_parameters()
        if param.requires_grad
    }
    basis = quad.basis.to(device=device, dtype=dtype)
    solve_basis_t = None
    solve_gram = None
    solve_eye = None
    linear_solve_scale = None
    linear_solve_v = None
    linear_solve_k_t = None
    linear_solve_small_inv_t = None
    if tau is not None:
        solve_basis_t = dynamics.linear.solve_base(basis, float(tau), transpose=False).T
        solve_gram = basis @ solve_basis_t
        solve_eye = torch.eye(solve_gram.shape[0], device=device, dtype=dtype)
        if isinstance(dynamics.linear, DissipativeSkewA):
            diagonal = dynamics.linear.diagonal.to(device=device, dtype=dtype)
            linear_solve_scale = 1.0 - float(tau) * diagonal
            if dynamics.linear.skew_rank > 0:
                p = dynamics.linear.P.to(device=device, dtype=dtype)
                q = dynamics.linear.Q.to(device=device, dtype=dtype)
                u = torch.cat((p, -q), dim=1)
                linear_solve_v = torch.cat((q, p), dim=1)
                k_base = u / linear_solve_scale[:, None]
                gram = linear_solve_v.T @ k_base
                small = torch.eye(gram.shape[0], device=device, dtype=dtype) - float(tau) * gram
                linear_solve_k_t = k_base.T
                linear_solve_small_inv_t = torch.linalg.inv(small).T
    reduced_tensor_dot = _reconstruct_energy_tucker_tensor_dot_batched(
        quad,
        directions,
        device=device,
        dtype=dtype,
    )
    reduced_tensor_free_dot = None
    b_pad = c_pad = src_pad = coeff_pad = lengths = None
    if os.environ.get("GOATTM_TRITON_PARAM_CORE", "0") == "1":
        b_pad, c_pad, src_pad, coeff_pad, lengths = _energy_reconstruction_padded_maps(quad, device=device, dtype=dtype)
        reduced_tensor_free_dot = reduced_tensor_dot[
            :,
            quad.free_a.to(device=device),
            quad.free_b.to(device=device),
            quad.free_c.to(device=device),
        ].contiguous()
    return BatchedEnergyTuckerDirectionCache(
        basis=basis,
        basis_dot=direction_stacks["quadratic.basis"],
        reduced_tensor=quad.reduced_tensor().to(device=device, dtype=dtype),
        reduced_tensor_dot=reduced_tensor_dot,
        reduced_tensor_free_dot=reduced_tensor_free_dot,
        reconstruct_out_b=b_pad,
        reconstruct_out_c=c_pad,
        reconstruct_source_padded=src_pad,
        reconstruct_coeff_padded=coeff_pad,
        reconstruct_lengths=lengths,
        solve_tau=None if tau is None else float(tau),
        solve_basis_t=solve_basis_t,
        solve_gram=solve_gram,
        solve_eye=solve_eye,
        linear_solve_tau=None if tau is None else float(tau),
        linear_solve_scale=linear_solve_scale,
        linear_solve_v=linear_solve_v,
        linear_solve_k_t=linear_solve_k_t,
        linear_solve_small_inv_t=linear_solve_small_inv_t,
        direction_stacks=direction_stacks,
    )


def direction_dot(a: Direction, b: Direction, dynamics: QuadraticDynamics) -> torch.Tensor:
    total = None
    for name, param in dynamics.named_parameters():
        if not param.requires_grad:
            continue
        av = a.get(name, torch.zeros_like(param)).to(device=param.device, dtype=param.dtype)
        bv = b.get(name, torch.zeros_like(param)).to(device=param.device, dtype=param.dtype)
        piece = (av * bv).sum()
        total = piece if total is None else total + piece
    if total is None:
        return next(dynamics.parameters()).new_zeros(())
    return total


def direction_norm(direction: Direction, dynamics: QuadraticDynamics) -> torch.Tensor:
    return torch.sqrt(direction_dot(direction, direction, dynamics))


def weighted_residual_tangent_vector(
    residual_dot: torch.Tensor,
    decoder_coeff_dot: torch.Tensor,
    weights: torch.Tensor | None,
    decoder_ridge: float,
) -> torch.Tensor:
    pieces = []
    if weights is None:
        pieces.append(residual_dot.reshape(-1))
    else:
        sqrt_weights = torch.sqrt(weights).to(device=residual_dot.device, dtype=residual_dot.dtype)
        pieces.append((residual_dot * sqrt_weights[..., None]).reshape(-1))
    if float(decoder_ridge) > 0.0:
        pieces.append(float(decoder_ridge) ** 0.5 * decoder_coeff_dot.reshape(-1))
    return torch.cat(pieces)


def weighted_residual_tangent_matrix(
    residual_dot: torch.Tensor,
    decoder_coeff_dot: torch.Tensor,
    weights: torch.Tensor | None,
    decoder_ridge: float,
) -> torch.Tensor:
    pieces = []
    direction_count = int(residual_dot.shape[-2])
    if weights is None:
        pieces.append(residual_dot.movedim(-2, -1).reshape(-1, direction_count))
    else:
        sqrt_weights = torch.sqrt(weights).to(device=residual_dot.device, dtype=residual_dot.dtype)
        weighted = residual_dot * sqrt_weights.reshape(*sqrt_weights.shape, 1, 1)
        pieces.append(weighted.movedim(-2, -1).reshape(-1, direction_count))
    if float(decoder_ridge) > 0.0:
        ridge_rows = float(decoder_ridge) ** 0.5 * decoder_coeff_dot.reshape(decoder_coeff_dot.shape[0], -1).T
        pieces.append(ridge_rows)
    return torch.cat(pieces, dim=0)


def _direction_stack(directions: list[Direction], name: str, param: torch.Tensor) -> torch.Tensor:
    values = []
    for direction in directions:
        if name in direction:
            values.append(direction[name].to(device=param.device, dtype=param.dtype))
        else:
            values.append(torch.zeros_like(param))
    return torch.stack(values, dim=0)


def _direction_stack_cached(
    directions: list[Direction],
    name: str,
    param: torch.Tensor,
    stack_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if stack_cache is not None and name in stack_cache:
        return stack_cache[name].to(device=param.device, dtype=param.dtype)
    return _direction_stack(directions, name, param)


def _linear_param_action_batched(
    dynamics: QuadraticDynamics,
    x: torch.Tensor,
    directions: list[Direction],
    stack_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    linear = dynamics.linear
    if isinstance(linear, DenseLinearA):
        d_a = _direction_stack_cached(directions, "linear.A", linear.A, stack_cache)
        k, r, _ = d_a.shape
        return x.matmul(d_a.reshape(k * r, r).T).reshape(x.shape[0], k, r)
    if isinstance(linear, DissipativeSkewA):
        raw_dot = _direction_stack_cached(directions, "linear.raw_damping", linear.raw_damping, stack_cache)
        damping = linear.raw_damping.to(device=x.device, dtype=x.dtype)
        out = (-2.0 * damping[None, :] * raw_dot.to(device=x.device, dtype=x.dtype))[None, :, :] * x[:, None, :]
        if linear.skew_rank > 0:
            p = linear.P.to(device=x.device, dtype=x.dtype)
            q = linear.Q.to(device=x.device, dtype=x.dtype)
            dp = _direction_stack_cached(directions, "linear.P", linear.P, stack_cache).to(device=x.device, dtype=x.dtype)
            dq = _direction_stack_cached(directions, "linear.Q", linear.Q, stack_cache).to(device=x.device, dtype=x.dtype)
            k, r, s = dp.shape
            x_dq = torch.matmul(x.unsqueeze(0), dq).permute(1, 0, 2)
            x_dp = torch.matmul(x.unsqueeze(0), dp).permute(1, 0, 2)
            xq = x @ q
            xp = x @ p
            out = (
                out
                + x_dq.matmul(p.T)
                + xq.matmul(dp.reshape(k * r, s).T).reshape(x.shape[0], k, r)
                - x_dp.matmul(q.T)
                - xp.matmul(dq.reshape(k * r, s).T).reshape(x.shape[0], k, r)
            )
        return out
    raise TypeError(f"unsupported linear type {type(linear)!r}")


def _linear_tangent_batched(
    dynamics: QuadraticDynamics,
    x: torch.Tensor,
    x_dot: torch.Tensor,
    directions: list[Direction],
    stack_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    n, k, r = x_dot.shape
    state_part = dynamics.linear(x_dot.reshape(n * k, r)).reshape(n, k, r)
    return state_part + _linear_param_action_batched(dynamics, x, directions, stack_cache=stack_cache)


def _source_param_action_batched(
    dynamics: QuadraticDynamics,
    p_mid: torch.Tensor | None,
    like: torch.Tensor,
    directions: list[Direction],
    stack_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    source = dynamics.source
    if isinstance(source, ZeroSource):
        return torch.zeros_like(like)
    if not isinstance(source, LinearSource):
        raise TypeError(f"unsupported source type {type(source)!r}")
    out = torch.zeros_like(like)
    if p_mid is not None:
        d_b = _direction_stack_cached(directions, "source.B", source.B, stack_cache).to(device=like.device, dtype=like.dtype)
        k, r, input_dim = d_b.shape
        out = out + p_mid.matmul(d_b.reshape(k * r, input_dim).T).reshape(p_mid.shape[0], k, r)
    if source.c is not None:
        d_c = _direction_stack_cached(directions, "source.c", source.c, stack_cache).to(device=like.device, dtype=like.dtype)
        out = out + d_c[None, :, :]
    return out


def _reconstruct_energy_tucker_tensor_dot_batched(
    quad: EnergyTuckerTTQuadratic,
    directions: list[Direction],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    c0 = quad.core0.to(device=device, dtype=dtype)
    c1 = quad.core1.to(device=device, dtype=dtype)
    c2 = quad.core2.to(device=device, dtype=dtype)
    dc0 = _direction_stack(directions, "quadratic.core0", quad.core0).to(device=device, dtype=dtype)
    dc1 = _direction_stack(directions, "quadratic.core1", quad.core1).to(device=device, dtype=dtype)
    dc2 = _direction_stack(directions, "quadratic.core2", quad.core2).to(device=device, dtype=dtype)
    direction_count = int(len(directions))
    q = int(quad.reduced_rank)
    left_rank = int(c0.shape[1])
    right_rank = int(c2.shape[0])

    c1c2 = c1.reshape(left_rank * q, right_rank).matmul(c2).reshape(left_rank, q * q)
    term0 = dc0.reshape(direction_count * q, left_rank).matmul(c1c2).reshape(direction_count, q, q, q)

    dc1c2 = dc1.reshape(direction_count * left_rank * q, right_rank).matmul(c2).reshape(direction_count, left_rank, q * q)
    term1 = c0.unsqueeze(0).matmul(dc1c2).reshape(direction_count, q, q, q)

    c0c1 = c0.matmul(c1.reshape(left_rank, q * right_rank)).reshape(q * q, right_rank)
    term2 = c0c1.unsqueeze(0).matmul(dc2).reshape(direction_count, q, q, q)

    free_full_dot = term0 + term1 + term2
    free_values_dot = free_full_dot[:, quad.free_a, quad.free_b, quad.free_c]
    flat = free_values_dot.new_zeros(len(directions), quad.reduced_rank**3)
    coeff = quad.reconstruct_coeff.to(device=device, dtype=dtype)
    target = quad.reconstruct_target.to(device=device).expand(len(directions), -1)
    values = coeff[None, :] * free_values_dot[:, quad.reconstruct_source]
    flat.scatter_add_(1, target, values)
    return flat.reshape(len(directions), quad.reduced_rank, quad.reduced_rank, quad.reduced_rank)


def _project_state_tangent_to_energy_basis(x_dot: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    n, k, r = x_dot.shape
    q = p.shape[0]
    return (x_dot.reshape(n * k, r) @ p.T).reshape(n, k, q)


def _project_param_tangent_to_energy_basis(x: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
    n, r = x.shape
    k, q, _ = dp.shape
    return (x @ dp.reshape(k * q, r).T).reshape(n, k, q)


def _energy_tucker_left_apply(c: torch.Tensor, dz_ell: torch.Tensor, z_x: torch.Tensor) -> torch.Tensor:
    q = int(c.shape[0])
    sample_mats = z_x.matmul(c.reshape(q * q, q).T).reshape(z_x.shape[0], q, q)
    return torch.bmm(dz_ell, sample_mats.transpose(1, 2))


def _energy_tucker_reduced_matrix(c: torch.Tensor, z_ell: torch.Tensor) -> torch.Tensor:
    q = int(c.shape[0])
    return z_ell.matmul(c.permute(0, 2, 1).reshape(q * q, q).T).reshape(z_ell.shape[0], q, q)


def _energy_tucker_right_apply(c: torch.Tensor, z_ell: torch.Tensor, dz_x: torch.Tensor) -> torch.Tensor:
    sample_mats = _energy_tucker_reduced_matrix(c, z_ell)
    return torch.bmm(dz_x, sample_mats.transpose(1, 2))


_TRITON_PARAM_CORE_KERNEL = None


def _triton_param_core_kernel():
    global _TRITON_PARAM_CORE_KERNEL
    if _TRITON_PARAM_CORE_KERNEL is not None:
        return _TRITON_PARAM_CORE_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        free_dot,
        z_lag,
        z_x,
        out_b,
        out_c,
        src,
        coeff,
        lengths,
        out,
        n_total: tl.constexpr,
        k_total: tl.constexpr,
        q: tl.constexpr,
        free_dim: tl.constexpr,
        max_len: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        a = tl.program_id(2)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        l_offsets = tl.arange(0, BLOCK_L)
        valid_n = n_offsets < n_total
        valid_k = k_offsets < k_total
        length = tl.load(lengths + a)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float64)
        for start in range(0, max_len, BLOCK_L):
            ls = start + l_offsets
            valid_l = ls < length
            b = tl.load(out_b + a * max_len + ls, mask=valid_l, other=0)
            c = tl.load(out_c + a * max_len + ls, mask=valid_l, other=0)
            s = tl.load(src + a * max_len + ls, mask=valid_l, other=0)
            cf = tl.load(coeff + a * max_len + ls, mask=valid_l, other=0.0)
            zl = tl.load(z_lag + n_offsets[:, None] * q + b[None, :], mask=valid_n[:, None] & valid_l[None, :], other=0.0)
            zx = tl.load(z_x + n_offsets[:, None] * q + c[None, :], mask=valid_n[:, None] & valid_l[None, :], other=0.0)
            w = zl * zx * cf[None, :]
            d = tl.load(free_dot + k_offsets[:, None] * free_dim + s[None, :], mask=valid_k[:, None] & valid_l[None, :], other=0.0)
            acc += tl.dot(w, tl.trans(d), out_dtype=tl.float64)
        tl.store(out + (n_offsets[:, None] * k_total + k_offsets[None, :]) * q + a, acc, mask=valid_n[:, None] & valid_k[None, :])

    _TRITON_PARAM_CORE_KERNEL = _kernel
    return _TRITON_PARAM_CORE_KERNEL


def _energy_tucker_param_core_apply_triton(
    free_dot: torch.Tensor,
    z_ell: torch.Tensor,
    z_x: torch.Tensor,
    out_b: torch.Tensor,
    out_c: torch.Tensor,
    src: torch.Tensor,
    coeff: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if free_dot.device.type != "cuda":
        raise RuntimeError("Triton parameter-core apply requires CUDA tensors")
    import triton

    n = int(z_ell.shape[0])
    k = int(free_dot.shape[0])
    q = int(z_ell.shape[1])
    free_dim = int(free_dot.shape[1])
    max_len = int(out_b.shape[1])
    out = z_ell.new_empty(n, k, q)
    block_n = 16
    block_k = 16
    block_l = 32
    grid = (triton.cdiv(n, block_n), triton.cdiv(k, block_k), q)
    _triton_param_core_kernel()[grid](
        free_dot,
        z_ell,
        z_x,
        out_b,
        out_c,
        src,
        coeff,
        lengths,
        out,
        n,
        k,
        q,
        free_dim,
        max_len,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_L=block_l,
    )
    return out


def _energy_tucker_param_core_apply(dc: torch.Tensor, z_ell: torch.Tensor, z_x: torch.Tensor) -> torch.Tensor:
    n = z_ell.shape[0]
    k, q, _, _ = dc.shape
    z_outer = torch.bmm(z_ell.unsqueeze(2), z_x.unsqueeze(1)).reshape(n, q * q)
    dc_rows = dc.reshape(k * q, q * q)
    return z_outer.matmul(dc_rows.T).reshape(n, k, q)


def _energy_tucker_param_core_apply_cached(
    dc: torch.Tensor,
    z_ell: torch.Tensor,
    z_x: torch.Tensor,
    quad_cache: BatchedEnergyTuckerDirectionCache | None,
) -> torch.Tensor:
    if (
        os.environ.get("GOATTM_TRITON_PARAM_CORE", "0") == "1"
        and quad_cache is not None
        and quad_cache.reduced_tensor_free_dot is not None
        and quad_cache.reconstruct_out_b is not None
        and quad_cache.reconstruct_out_c is not None
        and quad_cache.reconstruct_source_padded is not None
        and quad_cache.reconstruct_coeff_padded is not None
        and quad_cache.reconstruct_lengths is not None
        and z_ell.device.type == "cuda"
    ):
        return _energy_tucker_param_core_apply_triton(
            quad_cache.reduced_tensor_free_dot,
            z_ell,
            z_x,
            quad_cache.reconstruct_out_b,
            quad_cache.reconstruct_out_c,
            quad_cache.reconstruct_source_padded,
            quad_cache.reconstruct_coeff_padded,
            quad_cache.reconstruct_lengths,
        )
    return _energy_tucker_param_core_apply(dc, z_ell, z_x)


def _back_project_energy_tangent(reduced_dot: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    n, k, q = reduced_dot.shape
    return (reduced_dot.reshape(n * k, q) @ p).reshape(n, k, p.shape[1])


def _back_project_energy_param_tangent(reduced: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
    return torch.matmul(reduced[None, :, :], dp).permute(1, 0, 2).contiguous()


def _quadratic_frozen_param_action_batched(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    directions: list[Direction],
    quad_cache: BatchedEnergyTuckerDirectionCache | None = None,
) -> torch.Tensor:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyTuckerTTQuadratic):
        raise TypeError("batched reduced-GN JVP currently supports EnergyTuckerTTQuadratic")
    if quad_cache is None:
        p = quad.basis.to(device=x.device, dtype=x.dtype)
        dp = _direction_stack(directions, "quadratic.basis", quad.basis).to(device=x.device, dtype=x.dtype)
        c = quad.reduced_tensor().to(device=x.device, dtype=x.dtype)
        dc = _reconstruct_energy_tucker_tensor_dot_batched(quad, directions, device=x.device, dtype=x.dtype)
    else:
        p = quad_cache.basis
        dp = quad_cache.basis_dot
        c = quad_cache.reduced_tensor
        dc = quad_cache.reduced_tensor_dot
    z_lag = ell @ p.T
    z = x @ p.T
    dz_lag = _project_param_tangent_to_energy_basis(ell, dp)
    dz = _project_param_tangent_to_energy_basis(x, dp)
    reduced = torch.bmm(_energy_tucker_reduced_matrix(c, z_lag), z.unsqueeze(-1)).squeeze(-1)
    reduced_dot = (
        _energy_tucker_param_core_apply_cached(dc, z_lag, z, quad_cache)
        + _energy_tucker_left_apply(c, dz_lag, z)
        + _energy_tucker_right_apply(c, z_lag, dz)
    )
    return _back_project_energy_tangent(reduced_dot, p) + _back_project_energy_param_tangent(reduced, dp)


def _frozen_action_ell_dot(dynamics: QuadraticDynamics, ell_dot: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    n, k, r = ell_dot.shape
    x_repeat = x[:, None, :].expand(n, k, r).reshape(n * k, r)
    return dynamics.quadratic.frozen_action(ell_dot.reshape(n * k, r), x_repeat).reshape(n, k, r)


def _frozen_action_x_dot(dynamics: QuadraticDynamics, ell: torch.Tensor, x_dot: torch.Tensor) -> torch.Tensor:
    n, k, r = x_dot.shape
    ell_repeat = ell[:, None, :].expand(n, k, r).reshape(n * k, r)
    return dynamics.quadratic.frozen_action(ell_repeat, x_dot.reshape(n * k, r)).reshape(n, k, r)


def _energy_tucker_frozen_tangent_batched(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    ell_dot: torch.Tensor,
    x_dot: torch.Tensor | None,
    directions: list[Direction],
    *,
    quad_cache: BatchedEnergyTuckerDirectionCache | None,
    z_ell: torch.Tensor | None = None,
    reduced_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyTuckerTTQuadratic):
        raise TypeError("expected EnergyTuckerTTQuadratic")
    if quad_cache is None:
        p = quad.basis.to(device=x.device, dtype=x.dtype)
        dp = _direction_stack(directions, "quadratic.basis", quad.basis).to(device=x.device, dtype=x.dtype)
        c = quad.reduced_tensor().to(device=x.device, dtype=x.dtype)
        dc = _reconstruct_energy_tucker_tensor_dot_batched(quad, directions, device=x.device, dtype=x.dtype)
    else:
        p = quad_cache.basis
        dp = quad_cache.basis_dot
        c = quad_cache.reduced_tensor
        dc = quad_cache.reduced_tensor_dot
    if z_ell is None:
        z_ell = ell @ p.T
    z_x = x @ p.T
    dz_ell = _project_state_tangent_to_energy_basis(ell_dot, p)
    dz_ell = dz_ell + _project_param_tangent_to_energy_basis(ell, dp)
    reduced_dot = _energy_tucker_left_apply(c, dz_ell, z_x)
    if x_dot is not None:
        dz_x = _project_state_tangent_to_energy_basis(x_dot, p)
        dz_x = dz_x + _project_param_tangent_to_energy_basis(x, dp)
    else:
        dz_x = _project_param_tangent_to_energy_basis(x, dp)
    reduced_dot = reduced_dot + _energy_tucker_right_apply(c, z_ell, dz_x)
    reduced_dot = reduced_dot + _energy_tucker_param_core_apply_cached(dc, z_ell, z_x, quad_cache)
    if reduced_matrix is None:
        reduced = torch.bmm(_energy_tucker_reduced_matrix(c, z_ell), z_x.unsqueeze(-1)).squeeze(-1)
    else:
        reduced = torch.bmm(reduced_matrix, z_x.unsqueeze(-1)).squeeze(-1)
    return _back_project_energy_tangent(reduced_dot, p) + _back_project_energy_param_tangent(reduced, dp)


def _quadratic_frozen_tangent_batched(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    ell_dot: torch.Tensor,
    x_dot: torch.Tensor,
    directions: list[Direction],
    quad_cache: BatchedEnergyTuckerDirectionCache | None = None,
    z_ell: torch.Tensor | None = None,
    reduced_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic):
        return _energy_tucker_frozen_tangent_batched(
            dynamics,
            ell,
            x,
            ell_dot,
            x_dot,
            directions,
            quad_cache=quad_cache,
            z_ell=z_ell,
            reduced_matrix=reduced_matrix,
        )
    return (
        _frozen_action_ell_dot(dynamics, ell_dot, x)
        + _frozen_action_x_dot(dynamics, ell, x_dot)
        + _quadratic_frozen_param_action_batched(dynamics, ell, x, directions, quad_cache=quad_cache)
    )


def _rhs_tangent_batched(
    dynamics: QuadraticDynamics,
    x: torch.Tensor,
    x_dot: torch.Tensor,
    p_mid: torch.Tensor | None,
    directions: list[Direction],
    quad_cache: BatchedEnergyTuckerDirectionCache | None = None,
) -> torch.Tensor:
    stack_cache = None if quad_cache is None else quad_cache.direction_stacks
    return (
        _linear_tangent_batched(dynamics, x, x_dot, directions, stack_cache=stack_cache)
        + _quadratic_frozen_tangent_batched(dynamics, x, x, x_dot, x_dot, directions, quad_cache=quad_cache)
        + _source_param_action_batched(dynamics, p_mid, x_dot, directions, stack_cache=stack_cache)
    )


def _exact_energy_tucker_frozen_solve_multi_rhs(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    rhs: torch.Tensor,
    tau: float,
    quad_cache: BatchedEnergyTuckerDirectionCache | None = None,
    reduced_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    quad = dynamics.quadratic
    if not isinstance(quad, EnergyTuckerTTQuadratic):
        raise TypeError("expected EnergyTuckerTTQuadratic")
    n, k, r = rhs.shape
    p = quad.basis.to(device=rhs.device, dtype=rhs.dtype) if quad_cache is None else quad_cache.basis
    if reduced_matrix is None:
        reduced = quad.reduced_frozen_matrix(ell).to(device=rhs.device, dtype=rhs.dtype)
    else:
        reduced = reduced_matrix.to(device=rhs.device, dtype=rhs.dtype)
    y = _linear_solve_base_cached(dynamics, rhs.reshape(n * k, r), tau, quad_cache).reshape(n, k, r)
    if (
        quad_cache is not None
        and quad_cache.solve_basis_t is not None
        and quad_cache.solve_gram is not None
        and quad_cache.solve_eye is not None
        and quad_cache.solve_tau == float(tau)
    ):
        k_base = quad_cache.solve_basis_t
        gram = quad_cache.solve_gram
        small_eye = quad_cache.solve_eye.expand(n, -1, -1)
    else:
        k_base = dynamics.linear.solve_base(p, tau, transpose=False).T
        gram = p @ k_base
        small_eye = torch.eye(gram.shape[0], device=rhs.device, dtype=rhs.dtype).expand(n, -1, -1)
    q = int(gram.shape[0])
    small = small_eye - float(tau) * torch.bmm(reduced, gram.expand(n, q, q))
    py = (y.reshape(n * k, r) @ p.T).reshape(n, k, q)
    reduced_rhs = torch.bmm(reduced, py.transpose(1, 2))
    eta = torch.linalg.solve(small, reduced_rhs)
    correction = (eta.transpose(1, 2).reshape(n * k, q) @ k_base.T).reshape(n, k, r)
    return y + float(tau) * correction


def _linear_solve_base_cached(
    dynamics: QuadraticDynamics,
    rhs: torch.Tensor,
    tau: float,
    quad_cache: BatchedEnergyTuckerDirectionCache | None,
) -> torch.Tensor:
    if (
        quad_cache is not None
        and quad_cache.linear_solve_tau == float(tau)
        and quad_cache.linear_solve_scale is not None
    ):
        y = rhs / quad_cache.linear_solve_scale
        if (
            quad_cache.linear_solve_v is None
            or quad_cache.linear_solve_k_t is None
            or quad_cache.linear_solve_small_inv_t is None
        ):
            return y
        rhs_small = y @ quad_cache.linear_solve_v
        coeff = rhs_small @ quad_cache.linear_solve_small_inv_t
        return y + float(tau) * (coeff @ quad_cache.linear_solve_k_t)
    return dynamics.linear.solve_base(rhs, tau, transpose=False)


def _exact_frozen_solve_batched(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    rhs: torch.Tensor,
    tau: float,
    quad_cache: BatchedEnergyTuckerDirectionCache | None = None,
    reduced_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    n, k, r = rhs.shape
    if isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic) and int(dynamics.quadratic.reduced_rank) < int(dynamics.latent_dim):
        return _exact_energy_tucker_frozen_solve_multi_rhs(
            dynamics,
            ell,
            rhs,
            tau,
            quad_cache=quad_cache,
            reduced_matrix=reduced_matrix,
        )
    ell_flat = ell[:, None, :].expand(n, k, r).reshape(n * k, r)
    solved = exact_frozen_solve(dynamics, ell_flat, rhs.reshape(n * k, r), tau, transpose=False)
    return solved.reshape(n, k, r)


def lagged_midpoint_rollout_incremental_batched(
    dynamics: QuadraticDynamics,
    u0: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None,
    directions: list[Direction],
    *,
    rollout: DensePicardRolloutResult,
) -> torch.Tensor:
    if len(directions) <= 0:
        raise ValueError("directions must be nonempty")
    steps = int(rollout.states.shape[0] - 1)
    k = int(len(directions))
    tau = 0.5 * float(h)
    u_dot = u0.new_zeros(u0.shape[0], k, u0.shape[1])
    states_dot = u0.new_empty(steps + 1, u0.shape[0], k, u0.shape[1])
    states_dot[0].copy_(u_dot)
    quad_cache = _make_batched_energy_tucker_direction_cache(
        dynamics,
        directions,
        device=u0.device,
        dtype=u0.dtype,
        tau=tau,
    )
    with torch.no_grad(), _quadratic_cache_context(dynamics):
        for n in range(steps):
            p_n = None if p_mid is None else p_mid[n]
            u_dot = _batched_tangent_next(
                dynamics,
                rollout.states[n],
                u_dot,
                h,
                p_n,
                directions,
                picard_iterates=rollout.picard_iterates[n],
                quad_cache=quad_cache,
            )
            states_dot[n + 1].copy_(u_dot)
    return states_dot


def decoder_feature_tangent_batched(
    decoder,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    leading = int(states.shape[:-1].numel())
    flat = states.reshape(leading, states.shape[-1])
    flat_dot = states_dot.reshape(leading, states_dot.shape[-2], states_dot.shape[-1])
    pieces = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, leading, chunk_size):
        end = min(start + chunk_size, leading)
        u = flat[start:end]
        du = flat_dot[start:end]
        linear = du
        if not getattr(decoder, "include_quadratic", False):
            pieces.append(linear)
            continue
        if not hasattr(decoder, "quadratic_i"):
            raise TypeError("batched decoder tangent currently supports masked_cross quadratic features")
        idx_i = decoder.quadratic_i.to(device=u.device)
        idx_j = decoder.quadratic_j.to(device=u.device)
        quad_dot = du.index_select(2, idx_i) * u[:, None, :].index_select(2, idx_j) + u[:, None, :].index_select(2, idx_i) * du.index_select(2, idx_j)
        pieces.append(torch.cat((linear, quad_dot), dim=2))
    return torch.cat(pieces, dim=0)


def decoder_feature_tangent_chunk_batched(decoder, u: torch.Tensor, du: torch.Tensor) -> torch.Tensor:
    linear = du
    if not getattr(decoder, "include_quadratic", False):
        return linear
    if not hasattr(decoder, "quadratic_i"):
        raise TypeError("batched decoder tangent currently supports masked_cross quadratic features")
    idx_i = decoder.quadratic_i.to(device=u.device)
    idx_j = decoder.quadratic_j.to(device=u.device)
    u_i = u[:, None, :].index_select(2, idx_i)
    u_j = u[:, None, :].index_select(2, idx_j)
    quad_dot = du.index_select(2, idx_i) * u_j + u_i * du.index_select(2, idx_j)
    return torch.cat((linear, quad_dot), dim=2)


def _feature_normal_cross_batched(dfeat: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return torch.matmul(dfeat.permute(1, 2, 0), features)


def _target_feature_cross_batched(targets: torch.Tensor, dfeat: torch.Tensor) -> torch.Tensor:
    return torch.matmul(dfeat.permute(1, 2, 0), targets).transpose(1, 2)


def _decoder_prediction_tangent_dense(
    dfeat: torch.Tensor,
    features: torch.Tensor,
    coeff_no_bias: torch.Tensor,
    dcoeff_no_bias: torch.Tensor,
) -> torch.Tensor:
    m, k, feature_dim = dfeat.shape
    output_dim = int(coeff_no_bias.shape[0])
    return (
        dfeat.reshape(m * k, feature_dim).matmul(coeff_no_bias.T).reshape(m, k, output_dim)
        + features.matmul(dcoeff_no_bias.reshape(k * output_dim, feature_dim).T).reshape(m, k, output_dim)
    )


def projected_decoder_residual_tangent_batched(
    decoder,
    normal,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    ridge: float,
    chunk_size: int = 8192,
    normal_cholesky: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leading = int(states.shape[:-1].numel())
    flat_states = states.reshape(leading, states.shape[-1])
    flat_targets = targets.reshape(leading, targets.shape[-1])
    flat_dot = states_dot.reshape(leading, states_dot.shape[-2], states_dot.shape[-1])
    k = int(states_dot.shape[-2])
    feature_dim = int(decoder.feature_dim) + int(decoder.readout.bias is not None)
    d_normal = states.new_zeros(k, feature_dim, feature_dim)
    d_rhs = states.new_zeros(k, flat_targets.shape[1], feature_dim)
    flat_weights = None if weights is None else weights.reshape(leading).to(device=states.device, dtype=states.dtype)
    for start in range(0, leading, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), leading)
        u = flat_states[start:end]
        features = decoder.features(u)
        dfeat = decoder_feature_tangent_chunk_batched(decoder, u, flat_dot[start:end])
        if decoder.readout.bias is not None:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            zeros = torch.zeros(dfeat.shape[0], dfeat.shape[1], 1, device=dfeat.device, dtype=dfeat.dtype)
            features = torch.cat((features, ones), dim=1)
            dfeat = torch.cat((dfeat, zeros), dim=2)
        if flat_weights is None:
            weighted_features = features
            weighted_dfeat = dfeat
        else:
            w = flat_weights[start:end]
            weighted_features = features * w[:, None]
            weighted_dfeat = dfeat * w[:, None, None]
        d_normal.add_(_feature_normal_cross_batched(dfeat, weighted_features))
        d_normal.add_(_feature_normal_cross_batched(weighted_dfeat, features).transpose(1, 2))
        d_rhs.add_(_target_feature_cross_batched(flat_targets[start:end], weighted_dfeat))
    coeff = normal.coefficients.to(device=states.device, dtype=states.dtype)
    normal_matrix = normal.normal_matrix.to(device=states.device, dtype=states.dtype)
    rhs_for_dcoeff_t = d_rhs.transpose(1, 2) - d_normal.matmul(coeff.T)
    solved = _solve_decoder_normal(
        normal_matrix,
        rhs_for_dcoeff_t.permute(1, 0, 2).reshape(feature_dim, k * flat_targets.shape[1]),
        normal_cholesky,
    )
    dcoeff = solved.reshape(feature_dim, k, flat_targets.shape[1]).permute(1, 2, 0).contiguous()
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    dcoeff_no_bias = dcoeff[:, :, : decoder.feature_dim]
    d_bias = None if decoder.readout.bias is None else dcoeff[:, :, -1]
    residual_dot = states.new_empty(leading, k, flat_targets.shape[1])
    qform = states.new_zeros(k)
    for start in range(0, leading, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), leading)
        u = flat_states[start:end]
        features = decoder.features(u)
        dfeat = decoder_feature_tangent_chunk_batched(decoder, u, flat_dot[start:end])
        dres = _decoder_prediction_tangent_dense(dfeat, features, coeff_no_bias, dcoeff_no_bias)
        if d_bias is not None:
            dres = dres + d_bias[None, :, :]
        residual_dot[start:end] = dres
        if flat_weights is None:
            qform = qform + dres.square().sum(dim=(0, 2))
        else:
            qform = qform + (dres.square().sum(dim=2) * flat_weights[start:end, None]).sum(dim=0)
    ridge_qform = float(ridge) * dcoeff.square().sum(dim=(1, 2))
    return residual_dot.reshape(*targets.shape[:-1], k, targets.shape[-1]), dcoeff, qform + ridge_qform


def _decoder_feature_state_grad_batched(decoder, u: torch.Tensor, feature_cotangent: torch.Tensor) -> torch.Tensor:
    grad = feature_cotangent[:, :, : decoder.latent_dim].clone()
    if not getattr(decoder, "include_quadratic", False):
        return grad
    if not hasattr(decoder, "quadratic_i"):
        raise TypeError("batched GN state cotangents currently support masked_cross quadratic features")
    idx_i = decoder.quadratic_i.to(device=u.device)
    idx_j = decoder.quadratic_j.to(device=u.device)
    quad_cot = feature_cotangent[:, :, decoder.latent_dim :]
    scatter_i = idx_i.reshape(1, 1, -1).expand(u.shape[0], feature_cotangent.shape[1], -1)
    scatter_j = idx_j.reshape(1, 1, -1).expand(u.shape[0], feature_cotangent.shape[1], -1)
    grad.scatter_add_(2, scatter_i, quad_cot * u[:, None, :].index_select(2, idx_j))
    grad.scatter_add_(2, scatter_j, quad_cot * u[:, None, :].index_select(2, idx_i))
    return grad


def _masked_cross_output_cotangent_state_grad_batched(
    decoder,
    u: torch.Tensor,
    output_cotangent: torch.Tensor,
    coeff_no_bias: torch.Tensor,
    *,
    feature_chunk_size: int = 512,
) -> torch.Tensor:
    """Apply D(features(u) @ coeff.T)^T without materializing feature cotangents."""
    m, k, output_dim = output_cotangent.shape
    grad = (output_cotangent.reshape(m * k, output_dim) @ coeff_no_bias[:, : decoder.latent_dim]).reshape(m, k, -1)
    if not getattr(decoder, "include_quadratic", False):
        return grad
    if not hasattr(decoder, "quadratic_i"):
        feature_cotangent = (output_cotangent.reshape(m * k, output_dim) @ coeff_no_bias).reshape(m, k, -1)
        return _decoder_feature_state_grad_batched(decoder, u, feature_cotangent)

    idx_i_all = decoder.quadratic_i.to(device=u.device)
    idx_j_all = decoder.quadratic_j.to(device=u.device)
    quad_coeff = coeff_no_bias[:, decoder.latent_dim :]
    chunk = max(1, int(feature_chunk_size))
    for start in range(0, idx_i_all.numel(), chunk):
        end = min(start + chunk, idx_i_all.numel())
        idx_i = idx_i_all[start:end]
        idx_j = idx_j_all[start:end]
        quad_cot = (output_cotangent.reshape(m * k, output_dim) @ quad_coeff[:, start:end]).reshape(m, k, end - start)
        scatter_i = idx_i.reshape(1, 1, -1).expand(u.shape[0], output_cotangent.shape[1], -1)
        scatter_j = idx_j.reshape(1, 1, -1).expand(u.shape[0], output_cotangent.shape[1], -1)
        grad.scatter_add_(2, scatter_i, quad_cot * u.index_select(1, idx_j)[:, None, :])
        grad.scatter_add_(2, scatter_j, quad_cot * u.index_select(1, idx_i)[:, None, :])
    return grad


def _gauss_newton_state_cotangents_batched_masked_cross_direct(
    decoder,
    normal,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    ridge: float,
    chunk_size: int = 8192,
    normal_cholesky: torch.Tensor | None = None,
    reuse_states_dot_buffer: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked-cross fast path that avoids forming per-probe dnormal matrices."""
    leading = int(states.shape[:-1].numel())
    flat_states = states.reshape(leading, states.shape[-1])
    flat_targets = targets.reshape(leading, targets.shape[-1])
    flat_dot = states_dot.reshape(leading, states_dot.shape[-2], states_dot.shape[-1])
    k = int(states_dot.shape[-2])
    output_dim = int(flat_targets.shape[1])
    feature_dim = int(decoder.feature_dim) + int(decoder.readout.bias is not None)
    flat_weights = None if weights is None else weights.reshape(leading).to(device=states.device, dtype=states.dtype)
    coeff = normal.coefficients.to(device=states.device, dtype=states.dtype)
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    rhs_for_dcoeff_t = states.new_zeros(k, feature_dim, output_dim)
    row_chunk = max(1, int(chunk_size))
    feature_chunk = _masked_cross_feature_chunk_size(
        int(decoder.feature_dim),
        min(row_chunk, leading),
        k,
        element_size=states.element_size(),
        device=states.device,
    )
    cache_mode = os.environ.get("GOATTM_CACHE_D_PRED_FIXED", "auto").strip().lower()
    if cache_mode in {"1", "true", "yes", "on"}:
        cache_d_pred_fixed = True
    elif cache_mode in {"0", "false", "no", "off"}:
        cache_d_pred_fixed = False
    elif cache_mode == "auto":
        cache_bytes = int(leading) * int(k) * int(output_dim) * int(states.element_size())
        if states.device.type == "cuda":
            try:
                with torch.cuda.device(states.device):
                    free_bytes, _total_bytes = torch.cuda.mem_get_info()
            except RuntimeError:
                free_bytes = 0
            cache_d_pred_fixed = cache_bytes <= 0.5 * int(free_bytes)
        else:
            cache_d_pred_fixed = False
    else:
        raise ValueError("GOATTM_CACHE_D_PRED_FIXED must be one of auto, 0/1, false/true, no/yes, off/on")
    d_pred_fixed_chunks: list[torch.Tensor | None] | None = [] if cache_d_pred_fixed else None

    for start in range(0, leading, row_chunk):
        end = min(start + row_chunk, leading)
        u = flat_states[start:end]
        du = flat_dot[start:end]
        features = decoder.features(u)
        if decoder.readout.bias is not None:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            features_aug = torch.cat((features, ones), dim=1)
        else:
            features_aug = features
        pred = features_aug @ coeff.T
        residual = pred - flat_targets[start:end]
        if flat_weights is None:
            weighted_residual = residual
        else:
            weighted_residual = residual * flat_weights[start:end, None]

        d_pred_fixed = _masked_cross_prediction_and_residual_terms_batched(
            rhs_for_dcoeff_t,
            decoder,
            u,
            du,
            weighted_residual,
            coeff_no_bias,
            feature_chunk_size=feature_chunk,
        )
        if flat_weights is None:
            weighted_d_pred = d_pred_fixed
        else:
            weighted_d_pred = d_pred_fixed * flat_weights[start:end, None, None]
        coeff_cross = torch.matmul(weighted_d_pred.permute(1, 2, 0), features_aug)
        rhs_for_dcoeff_t.add_(-coeff_cross.transpose(1, 2))
        if d_pred_fixed_chunks is not None:
            d_pred_fixed_chunks.append(d_pred_fixed)

    normal_matrix = normal.normal_matrix.to(device=states.device, dtype=states.dtype)
    solved = _solve_decoder_normal(
        normal_matrix,
        rhs_for_dcoeff_t.permute(1, 0, 2).reshape(feature_dim, k * output_dim),
        normal_cholesky,
    )
    dcoeff = solved.reshape(feature_dim, k, output_dim).permute(1, 2, 0).contiguous()

    state_cot = flat_dot if bool(reuse_states_dot_buffer) else states.new_empty(leading, k, states.shape[-1])
    qform = states.new_zeros(k)
    d_pred_chunk_index = 0
    for start in range(0, leading, row_chunk):
        end = min(start + row_chunk, leading)
        u = flat_states[start:end]
        du = flat_dot[start:end]
        features = decoder.features(u)
        if decoder.readout.bias is not None:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            features_aug = torch.cat((features, ones), dim=1)
        else:
            features_aug = features
        if d_pred_fixed_chunks is None:
            d_pred_fixed = _masked_cross_prediction_tangent_batched(
                decoder,
                u,
                du,
                coeff_no_bias,
                output_chunk_size=16,
            )
        else:
            d_pred_fixed = d_pred_fixed_chunks[d_pred_chunk_index]
            d_pred_fixed_chunks[d_pred_chunk_index] = None
            d_pred_chunk_index += 1
            if d_pred_fixed is None:
                raise RuntimeError("cached d_pred_fixed chunk was unexpectedly released")
        dres = d_pred_fixed + (features_aug @ dcoeff.reshape(k * output_dim, feature_dim).T).reshape(end - start, k, output_dim)
        if flat_weights is None:
            weighted_output = dres
            qform = qform + dres.square().sum(dim=(0, 2))
        else:
            w = flat_weights[start:end]
            weighted_output = dres * w[:, None, None]
            qform = qform + (dres.square().sum(dim=2) * w[:, None]).sum(dim=0)
        state_cot[start:end].copy_(
            _masked_cross_output_cotangent_state_grad_batched(
                decoder,
                u,
                weighted_output,
                coeff_no_bias,
                feature_chunk_size=feature_chunk,
            )
        )
    ridge_qform = float(ridge) * dcoeff.square().sum(dim=(1, 2))
    return state_cot.reshape(*states.shape[:-1], k, states.shape[-1]), qform + ridge_qform


def gauss_newton_state_cotangents_batched(
    decoder,
    normal,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    ridge: float,
    chunk_size: int = 8192,
    normal_cholesky: torch.Tensor | None = None,
    reuse_states_dot_buffer: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(decoder, "include_quadratic", False) and hasattr(decoder, "quadratic_i"):
        return _gauss_newton_state_cotangents_batched_masked_cross_direct(
            decoder,
            normal,
            states,
            states_dot,
            targets,
            weights=weights,
            ridge=ridge,
            chunk_size=chunk_size,
            normal_cholesky=normal_cholesky,
            reuse_states_dot_buffer=reuse_states_dot_buffer,
        )
    leading = int(states.shape[:-1].numel())
    flat_states = states.reshape(leading, states.shape[-1])
    flat_targets = targets.reshape(leading, targets.shape[-1])
    flat_dot = states_dot.reshape(leading, states_dot.shape[-2], states_dot.shape[-1])
    k = int(states_dot.shape[-2])
    feature_dim = int(decoder.feature_dim) + int(decoder.readout.bias is not None)
    d_normal = states.new_zeros(k, feature_dim, feature_dim)
    d_rhs = states.new_zeros(k, flat_targets.shape[1], feature_dim)
    flat_weights = None if weights is None else weights.reshape(leading).to(device=states.device, dtype=states.dtype)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, leading, chunk_size):
        end = min(start + chunk_size, leading)
        u = flat_states[start:end]
        features = decoder.features(u)
        dfeat = decoder_feature_tangent_chunk_batched(decoder, u, flat_dot[start:end])
        if decoder.readout.bias is not None:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            zeros = torch.zeros(dfeat.shape[0], dfeat.shape[1], 1, device=dfeat.device, dtype=dfeat.dtype)
            features = torch.cat((features, ones), dim=1)
            dfeat = torch.cat((dfeat, zeros), dim=2)
        if flat_weights is None:
            weighted_features = features
            weighted_dfeat = dfeat
        else:
            w = flat_weights[start:end]
            weighted_features = features * w[:, None]
            weighted_dfeat = dfeat * w[:, None, None]
        d_normal.add_(_feature_normal_cross_batched(dfeat, weighted_features))
        d_normal.add_(_feature_normal_cross_batched(weighted_dfeat, features).transpose(1, 2))
        d_rhs.add_(_target_feature_cross_batched(flat_targets[start:end], weighted_dfeat))

    coeff = normal.coefficients.to(device=states.device, dtype=states.dtype)
    normal_matrix = normal.normal_matrix.to(device=states.device, dtype=states.dtype)
    rhs_for_dcoeff_t = d_rhs.transpose(1, 2) - d_normal.matmul(coeff.T)
    solved = _solve_decoder_normal(
        normal_matrix,
        rhs_for_dcoeff_t.permute(1, 0, 2).reshape(feature_dim, k * flat_targets.shape[1]),
        normal_cholesky,
    )
    dcoeff = solved.reshape(feature_dim, k, flat_targets.shape[1]).permute(1, 2, 0).contiguous()
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    dcoeff_no_bias = dcoeff[:, :, : decoder.feature_dim]
    d_bias = None if decoder.readout.bias is None else dcoeff[:, :, -1]
    state_cot = states.new_empty(leading, k, states.shape[-1])
    qform = states.new_zeros(k)
    for start in range(0, leading, chunk_size):
        end = min(start + chunk_size, leading)
        u = flat_states[start:end]
        features = decoder.features(u)
        dfeat = decoder_feature_tangent_chunk_batched(decoder, u, flat_dot[start:end])
        dres = _decoder_prediction_tangent_dense(dfeat, features, coeff_no_bias, dcoeff_no_bias)
        if d_bias is not None:
            dres = dres + d_bias[None, :, :]
        if flat_weights is None:
            weighted = dres
            qform = qform + dres.square().sum(dim=(0, 2))
        else:
            weighted = dres * flat_weights[start:end, None, None]
            qform = qform + (dres.square().sum(dim=2) * flat_weights[start:end, None]).sum(dim=0)
        feature_cot = weighted.reshape((end - start) * k, weighted.shape[-1]).matmul(coeff_no_bias).reshape(end - start, k, -1)
        state_cot[start:end] = _decoder_feature_state_grad_batched(decoder, u, feature_cot)
    ridge_qform = float(ridge) * dcoeff.square().sum(dim=(1, 2))
    return state_cot.reshape(*states.shape[:-1], k, states.shape[-1]), qform + ridge_qform


def _splitmix64_signed(ids: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic 64-bit mixer for CountSketch row ids."""
    x = ids.to(dtype=torch.long) + int(seed) * 1000003
    x = x + -7046029254386353131
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 30)) * -4658895280553007687
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 27)) * -7723592293110705685
    return torch.bitwise_xor(x, torch.bitwise_right_shift(x, 31))


def _countsketch_hash(
    row_ids: torch.Tensor,
    sketch_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = row_ids.to(dtype=torch.long)
    bucket_hash = _splitmix64_signed(ids, int(seed))
    sign_hash = _splitmix64_signed(ids, int(seed) + 104729)
    buckets = torch.remainder(bucket_hash, int(sketch_dim)).to(dtype=torch.long)
    signs = torch.where(
        torch.remainder(sign_hash, 2) == 0,
        torch.ones((), device=ids.device, dtype=torch.float64),
        -torch.ones((), device=ids.device, dtype=torch.float64),
    )
    return buckets, signs

def _countsketch_add_rows_(
    sketch: torch.Tensor,
    rows: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    seed: int,
) -> None:
    buckets, signs = _countsketch_hash(row_ids.to(device=rows.device), sketch.shape[0], seed=seed)
    sketch.index_add_(0, buckets, signs.to(device=rows.device, dtype=rows.dtype)[:, None] * rows)


def _decoder_tangent_normal_terms_for_state(
    decoder,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = decoder.features(states)
    dfeat = decoder_feature_tangent_batched(decoder, states, states_dot, chunk_size=states.shape[0])
    if decoder.readout.bias is not None:
        ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
        zeros = torch.zeros(dfeat.shape[0], dfeat.shape[1], 1, device=dfeat.device, dtype=dfeat.dtype)
        features = torch.cat((features, ones), dim=1)
        dfeat = torch.cat((dfeat, zeros), dim=2)
    if features.shape[1] != int(feature_dim):
        raise ValueError("feature_dim does not match decoder feature shape")
    if weights is None:
        weighted_features = features
        weighted_dfeat = dfeat
    else:
        w = weights.to(device=states.device, dtype=states.dtype)
        weighted_features = features * w[:, None]
        weighted_dfeat = dfeat * w[:, None, None]
    d_normal = _feature_normal_cross_batched(dfeat, weighted_features)
    d_normal = d_normal + _feature_normal_cross_batched(weighted_dfeat, features).transpose(1, 2)
    d_rhs = _target_feature_cross_batched(targets, weighted_dfeat)
    return d_normal, d_rhs


def _decoder_residual_sketch_for_state_(
    sketch: torch.Tensor,
    decoder,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    coeff: torch.Tensor,
    dcoeff: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    global_sample_start: int,
    sample_count: int,
    state_index: int,
    data_row_seed: int,
) -> torch.Tensor:
    features = decoder.features(states)
    dfeat = decoder_feature_tangent_batched(decoder, states, states_dot, chunk_size=states.shape[0])
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    dcoeff_no_bias = dcoeff[:, :, : decoder.feature_dim]
    d_bias = None if decoder.readout.bias is None else dcoeff[:, :, -1]
    dres = _decoder_prediction_tangent_dense(dfeat, features, coeff_no_bias, dcoeff_no_bias)
    if d_bias is not None:
        dres = dres + d_bias[None, :, :]
    if weights is None:
        weighted_dres = dres
        qform = dres.square().sum(dim=(0, 2))
    else:
        w = weights.to(device=states.device, dtype=states.dtype)
        weighted_dres = dres * torch.sqrt(w)[:, None, None]
        qform = (dres.square().sum(dim=2) * w[:, None]).sum(dim=0)
    output_dim = int(dres.shape[-1])
    sample_ids = torch.arange(
        int(global_sample_start),
        int(global_sample_start) + int(states.shape[0]),
        device=states.device,
        dtype=torch.long,
    )
    output_ids = torch.arange(output_dim, device=states.device, dtype=torch.long)
    row_ids = ((int(state_index) * int(sample_count) + sample_ids[:, None]) * output_dim + output_ids[None, :]).reshape(-1)
    rows = weighted_dres.movedim(1, 2).reshape(-1, weighted_dres.shape[1])
    _countsketch_add_rows_(sketch, rows, row_ids, seed=data_row_seed)
    return qform


def _batched_tangent_next(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    u_dot: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None,
    directions: list[Direction],
    *,
    picard_iterates: torch.Tensor,
    quad_cache: BatchedEnergyTuckerDirectionCache | None = None,
) -> torch.Tensor:
    tau = 0.5 * float(h)
    stack_cache = None if quad_cache is None else quad_cache.direction_stacks
    linear_tangent = _linear_tangent_batched(dynamics, u, u_dot, directions, stack_cache=stack_cache)
    source_tangent = _source_param_action_batched(dynamics, p_mid, u_dot, directions, stack_cache=stack_cache)
    y_dot = u_dot + float(h) * (
        linear_tangent
        + _quadratic_frozen_tangent_batched(dynamics, u, u, u_dot, u_dot, directions, quad_cache=quad_cache)
        + source_tangent
    )
    iterates_dot = [y_dot]
    for j in range(picard_iterates.shape[0] - 1):
        y_prev = picard_iterates[j]
        y_next = picard_iterates[j + 1]
        y_prev_dot = iterates_dot[-1]
        ell = 0.5 * (u + y_prev)
        ell_dot = 0.5 * (u_dot + y_prev_dot)
        z_ell = None
        reduced_matrix = None
        if isinstance(dynamics.quadratic, EnergyTuckerTTQuadratic):
            if quad_cache is None:
                p = dynamics.quadratic.basis.to(device=u.device, dtype=u.dtype)
                c = dynamics.quadratic.reduced_tensor().to(device=u.device, dtype=u.dtype)
            else:
                p = quad_cache.basis
                c = quad_cache.reduced_tensor
            z_ell = ell @ p.T
            reduced_matrix = _energy_tucker_reduced_matrix(c, z_ell)
        rhs_dot = (
            u_dot
            + tau * linear_tangent
            + tau
            * _quadratic_frozen_tangent_batched(
                dynamics,
                ell,
                u,
                ell_dot,
                u_dot,
                directions,
                quad_cache=quad_cache,
                z_ell=z_ell,
                reduced_matrix=reduced_matrix,
            )
            + float(h) * source_tangent
            + tau * _linear_param_action_batched(dynamics, y_next, directions, stack_cache=stack_cache)
            + tau
            * _energy_tucker_frozen_tangent_batched(
                dynamics,
                ell,
                y_next,
                ell_dot,
                None,
                directions,
                quad_cache=quad_cache,
                z_ell=z_ell,
                reduced_matrix=reduced_matrix,
            )
        )
        y_dot = _exact_frozen_solve_batched(dynamics, ell, rhs_dot, tau, quad_cache=quad_cache, reduced_matrix=reduced_matrix)
        iterates_dot.append(y_dot)
    return y_dot


def _row_countsketch_add_(
    target: torch.Tensor,
    rows: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    seed: int,
) -> None:
    buckets, signs = _countsketch_hash(row_ids.to(device=rows.device), target.shape[0], seed=seed)
    view_shape = (rows.shape[0],) + (1,) * (rows.ndim - 1)
    target.index_add_(0, buckets, signs.to(device=rows.device, dtype=rows.dtype).reshape(view_shape) * rows)


def _masked_cross_prediction_tangent_batched(
    decoder,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    coeff_no_bias: torch.Tensor,
    *,
    output_chunk_size: int = 10,
) -> torch.Tensor:
    """Compute d(features(states)) @ coeff_no_bias.T without materializing dfeatures."""
    m, k, r = states_dot.shape
    d_pred = (states_dot.reshape(m * k, r) @ coeff_no_bias[:, : decoder.latent_dim].T).reshape(m, k, -1)
    if not getattr(decoder, "include_quadratic", False):
        return d_pred
    if not hasattr(decoder, "quadratic_i"):
        dfeat = decoder_feature_tangent_batched(decoder, states, states_dot, chunk_size=states.shape[0])
        return (dfeat.reshape(m * k, dfeat.shape[-1]) @ coeff_no_bias.T).reshape(m, k, -1)

    idx_i = decoder.quadratic_i.to(device=states.device)
    idx_j = decoder.quadratic_j.to(device=states.device)
    quad_coeff = coeff_no_bias[:, decoder.latent_dim :]
    feature_chunk = _masked_cross_feature_chunk_size(
        int(idx_i.numel()),
        m,
        k,
        element_size=states.element_size(),
        device=states.device,
    )
    for start in range(0, idx_i.numel(), feature_chunk):
        end = min(start + feature_chunk, idx_i.numel())
        idx_i_chunk = idx_i[start:end]
        idx_j_chunk = idx_j[start:end]
        quad_dot = (
            states_dot.index_select(2, idx_i_chunk) * states.index_select(1, idx_j_chunk)[:, None, :]
            + states.index_select(1, idx_i_chunk)[:, None, :] * states_dot.index_select(2, idx_j_chunk)
        )
        d_pred.add_(
            quad_dot.reshape(m * k, end - start).matmul(quad_coeff[:, start:end].T).reshape(m, k, -1)
        )
    return d_pred


def _add_masked_cross_tangent_residual_terms_(
    rhs_for_dcoeff_t: torch.Tensor,
    decoder,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    weighted_residual: torch.Tensor,
    *,
    feature_chunk_size: int = 512,
) -> None:
    """Accumulate -dF^T weighted_residual exactly, chunking masked-cross features."""
    rhs_for_dcoeff_t[:, : decoder.latent_dim, :].add_(-torch.matmul(states_dot.permute(1, 2, 0), weighted_residual))
    if not getattr(decoder, "include_quadratic", False):
        return
    if not hasattr(decoder, "quadratic_i"):
        dfeat = decoder_feature_tangent_batched(decoder, states, states_dot, chunk_size=states.shape[0])
        rhs_for_dcoeff_t[:, decoder.latent_dim :, :].add_(
            -torch.matmul(dfeat[:, :, decoder.latent_dim :].permute(1, 2, 0), weighted_residual)
        )
        return

    idx_i_all = decoder.quadratic_i.to(device=states.device)
    idx_j_all = decoder.quadratic_j.to(device=states.device)
    feature_offset = int(decoder.latent_dim)
    chunk = max(1, int(feature_chunk_size))
    for start in range(0, idx_i_all.numel(), chunk):
        end = min(start + chunk, idx_i_all.numel())
        idx_i = idx_i_all[start:end]
        idx_j = idx_j_all[start:end]
        quad_dot = (
            states_dot.index_select(2, idx_i) * states.index_select(1, idx_j)[:, None, :]
            + states.index_select(1, idx_i)[:, None, :] * states_dot.index_select(2, idx_j)
        )
        rhs_for_dcoeff_t[:, feature_offset + start : feature_offset + end, :].add_(
            -torch.matmul(quad_dot.permute(1, 2, 0), weighted_residual)
        )


def _masked_cross_prediction_and_residual_terms_batched(
    rhs_for_dcoeff_t: torch.Tensor,
    decoder,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    weighted_residual: torch.Tensor,
    coeff_no_bias: torch.Tensor,
    *,
    feature_chunk_size: int = 512,
) -> torch.Tensor:
    """Compute fixed decoder tangents while accumulating ``-dF.T @ residual``."""
    m, k, r = states_dot.shape
    output_dim = int(coeff_no_bias.shape[0])
    d_pred = states_dot.reshape(m * k, r).matmul(coeff_no_bias[:, : decoder.latent_dim].T).reshape(m, k, output_dim)
    rhs_for_dcoeff_t[:, : decoder.latent_dim, :].add_(-torch.matmul(states_dot.permute(1, 2, 0), weighted_residual))
    if not getattr(decoder, "include_quadratic", False):
        return d_pred
    if not hasattr(decoder, "quadratic_i"):
        dfeat = decoder_feature_tangent_batched(decoder, states, states_dot, chunk_size=states.shape[0])
        dfeat_quad = dfeat[:, :, decoder.latent_dim :]
        d_pred.add_(
            dfeat_quad.reshape(m * k, dfeat_quad.shape[-1]).matmul(coeff_no_bias[:, decoder.latent_dim :].T).reshape(m, k, output_dim)
        )
        rhs_for_dcoeff_t[:, decoder.latent_dim :, :].add_(
            -torch.matmul(dfeat_quad.permute(1, 2, 0), weighted_residual)
        )
        return d_pred

    idx_i_all = decoder.quadratic_i.to(device=states.device)
    idx_j_all = decoder.quadratic_j.to(device=states.device)
    quad_coeff = coeff_no_bias[:, decoder.latent_dim :]
    feature_offset = int(decoder.latent_dim)
    chunk = max(1, int(feature_chunk_size))
    for start in range(0, idx_i_all.numel(), chunk):
        end = min(start + chunk, idx_i_all.numel())
        idx_i = idx_i_all[start:end]
        idx_j = idx_j_all[start:end]
        quad_dot = (
            states_dot.index_select(2, idx_i) * states.index_select(1, idx_j)[:, None, :]
            + states.index_select(1, idx_i)[:, None, :] * states_dot.index_select(2, idx_j)
        )
        d_pred.add_(quad_dot.reshape(m * k, end - start).matmul(quad_coeff[:, start:end].T).reshape(m, k, output_dim))
        rhs_for_dcoeff_t[:, feature_offset + start : feature_offset + end, :].add_(
            -torch.matmul(quad_dot.permute(1, 2, 0), weighted_residual)
        )
    return d_pred


def _single_pass_schur_sketch_state_(
    rhs_for_dcoeff_t: torch.Tensor,
    fixed_coeff_cross: torch.Tensor,
    fixed_fixed_gram: torch.Tensor | None,
    feature_sketch: torch.Tensor,
    fixed_residual_sketch: torch.Tensor,
    decoder,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    coeff: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    global_sample_start: int,
    sample_count: int,
    state_index: int,
    row_seed: int,
    feature_chunk_size: int = 512,
) -> torch.Tensor:
    features = decoder.features(states)
    if decoder.readout.bias is not None:
        ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
        features = torch.cat((features, ones), dim=1)
    pred = features @ coeff.T
    residual = pred - targets
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    m, k, r = states_dot.shape
    output_dim = int(coeff.shape[0])
    d_pred_fixed = states_dot.reshape(m * k, r).matmul(coeff_no_bias[:, : decoder.latent_dim].T).reshape(m, k, output_dim)
    if weights is None:
        weighted_residual = residual
        sqrt_weight = None
    else:
        w = weights.to(device=states.device, dtype=states.dtype)
        weighted_residual = residual * w[:, None]
        sqrt_weight = torch.sqrt(w)
    rhs_for_dcoeff_t[:, : decoder.latent_dim, :].add_(-torch.matmul(states_dot.permute(1, 2, 0), weighted_residual))
    if getattr(decoder, "include_quadratic", False):
        if not hasattr(decoder, "quadratic_i"):
            dfeat = decoder_feature_tangent_batched(decoder, states, states_dot, chunk_size=states.shape[0])
            dfeat_quad = dfeat[:, :, decoder.latent_dim :]
            d_pred_fixed.add_(
                dfeat_quad.reshape(m * k, dfeat_quad.shape[-1]).matmul(coeff_no_bias[:, decoder.latent_dim :].T).reshape(m, k, output_dim)
            )
            rhs_for_dcoeff_t[:, decoder.latent_dim :, :].add_(
                -torch.matmul(dfeat_quad.permute(1, 2, 0), weighted_residual)
            )
        else:
            idx_i_all = decoder.quadratic_i.to(device=states.device)
            idx_j_all = decoder.quadratic_j.to(device=states.device)
            feature_offset = int(decoder.latent_dim)
            chunk = idx_i_all.numel() if int(feature_chunk_size) <= 0 else max(1, int(feature_chunk_size))
            for start in range(0, idx_i_all.numel(), chunk):
                end = min(start + chunk, idx_i_all.numel())
                idx_i = idx_i_all[start:end]
                idx_j = idx_j_all[start:end]
                quad_dot = (
                    states_dot.index_select(2, idx_i) * states.index_select(1, idx_j)[:, None, :]
                    + states.index_select(1, idx_i)[:, None, :] * states_dot.index_select(2, idx_j)
                )
                coeff_chunk = coeff_no_bias[:, feature_offset + start : feature_offset + end]
                d_pred_fixed.add_(quad_dot.reshape(m * k, end - start).matmul(coeff_chunk.T).reshape(m, k, output_dim))
                rhs_for_dcoeff_t[:, feature_offset + start : feature_offset + end, :].add_(
                    -torch.matmul(quad_dot.permute(1, 2, 0), weighted_residual)
                )
    if weights is None:
        weighted_d_pred = d_pred_fixed
        qform = d_pred_fixed.square().sum(dim=(0, 2))
        if fixed_fixed_gram is not None:
            rows = d_pred_fixed.permute(0, 2, 1).reshape(m * output_dim, k)
            fixed_fixed_gram.add_(rows.T.matmul(rows))
    else:
        weighted_d_pred = d_pred_fixed * w[:, None, None]
        qform = (d_pred_fixed.square().sum(dim=2) * w[:, None]).sum(dim=0)
        if fixed_fixed_gram is not None:
            rows = (d_pred_fixed * torch.sqrt(w)[:, None, None]).permute(0, 2, 1).reshape(m * output_dim, k)
            fixed_fixed_gram.add_(rows.T.matmul(rows))
    coeff_cross = torch.matmul(weighted_d_pred.permute(1, 2, 0), features)
    rhs_for_dcoeff_t.add_(-coeff_cross.transpose(1, 2))
    fixed_coeff_cross.add_(coeff_cross)
    sample_ids = torch.arange(
        int(global_sample_start),
        int(global_sample_start) + int(states.shape[0]),
        device=states.device,
        dtype=torch.long,
    )
    row_ids = int(state_index) * int(sample_count) + sample_ids
    if sqrt_weight is None:
        feature_rows = features
        fixed_rows = d_pred_fixed
    else:
        feature_rows = features * sqrt_weight[:, None]
        fixed_rows = d_pred_fixed * sqrt_weight[:, None, None]
    if feature_sketch.numel() > 0 and fixed_residual_sketch.numel() > 0:
        _row_countsketch_add_(feature_sketch, feature_rows, row_ids, seed=row_seed)
        _row_countsketch_add_(fixed_residual_sketch, fixed_rows, row_ids, seed=row_seed)
    return qform


class ReducedGNWorkspace:
    """Cached analytic actions for the reduced VarPro Gauss-Newton operator.

    The residual tangent returned by :meth:`jvp` is the Schur-projected
    variable-projection residual tangent, including optional decoder-ridge rows.
    Its squared norm equals the decoder/data part of the reduced GN quadratic
    form. Dynamics ridge terms are kept as parameter-space diagonal terms and
    are included by :meth:`hvp`.
    """

    def __init__(self, objective: ReducedObjective, batch: ContinuousBatch) -> None:
        if objective.gradient_mode not in {"lagged_adjoint", "frozen_adjoint", "autograd"}:
            raise ValueError("ReducedGNWorkspace currently supports lagged-midpoint objectives")
        if not hasattr(objective.stepper, "rollout_with_picard_history"):
            raise TypeError("ReducedGNWorkspace requires a lagged-midpoint stepper with Picard history")
        self.objective = objective
        self.batch = batch
        self.cache = self._build_cache()

    @property
    def dynamics(self) -> QuadraticDynamics:
        return self.objective.dynamics

    @property
    def parameter_dim(self) -> int:
        return int(sum(param.numel() for param in self.dynamics.parameters() if param.requires_grad))

    def _decoder_normal_cholesky(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache = self.cache
        factor = cache.normal_cholesky
        if factor is not None and factor.device == device and factor.dtype == dtype:
            return factor
        normal_matrix = cache.normal.normal_matrix.to(device=device, dtype=dtype)
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
        factor, info = torch.linalg.cholesky_ex(normal_matrix)
        if int(info.detach().cpu()) != 0:
            diag_scale = float(torch.diagonal(normal_matrix).abs().mean().detach().cpu())
            if not torch.isfinite(torch.tensor(diag_scale)) or diag_scale <= 0.0:
                diag_scale = 1.0
            eye = torch.eye(normal_matrix.shape[0], device=device, dtype=dtype)
            jitter = max(1.0e-12 * diag_scale, 1.0e-12)
            last_info = int(info.detach().cpu())
            for _ in range(12):
                factor, info = torch.linalg.cholesky_ex(normal_matrix + jitter * eye)
                last_info = int(info.detach().cpu())
                if last_info == 0:
                    break
                jitter *= 10.0
            if last_info != 0:
                raise RuntimeError(
                    "decoder normal Cholesky failed after adaptive jitter; "
                    f"last failing leading minor={last_info}, final jitter={jitter:.3e}"
                )
        cache.normal_cholesky = factor
        return factor

    def _build_cache(self) -> ReducedGNCache:
        objective = self.objective
        batch = self.batch
        with torch.no_grad():
            p_mid = objective._stepper_inputs(batch)
            u0 = objective._initial_state(batch)
            rollout = objective.stepper.rollout_with_picard_history(
                self.dynamics,
                u0,
                batch.step_size,
                batch.steps,
                p_mid=p_mid,
            )
            weights = objective._loss_weights(batch)
            normal = solve_decoder_normal_equation(
                objective.decoder,
                rollout.states,
                batch.qoi,
                ridge=objective.decoder_ridge,
                weights=weights,
                chunk_size=objective.normal_chunk_size,
            )
        return ReducedGNCache(
            batch=batch,
            u0=u0.detach(),
            p_mid=None if p_mid is None else p_mid.detach(),
            weights=None if weights is None else weights.detach(),
            rollout=rollout,
            normal=normal,
        )

    def refresh(self) -> None:
        self.cache = self._build_cache()

    def flatten(self, direction: Direction) -> torch.Tensor:
        return flatten_direction(direction, self.dynamics)

    def direction_from_flat(self, flat: torch.Tensor) -> Direction:
        return direction_from_flat(self.dynamics, flat)

    def gradient_flat(self) -> tuple[torch.Tensor, object]:
        result = self.objective.value_and_grad(self.batch)
        pieces = []
        for name, param in self.dynamics.named_parameters():
            if not param.requires_grad:
                continue
            grad = result.parameter_grads.get(name) if result.parameter_grads is not None else param.grad
            if grad is None:
                grad = torch.zeros_like(param)
            pieces.append(grad.detach().reshape(-1))
        return torch.cat(pieces), result

    def jvp(self, direction: Direction, *, keep_states_dot: bool = False) -> ReducedJVPResult:
        cache = self.cache
        with torch.no_grad():
            inc = lagged_midpoint_rollout_incremental(
                self.dynamics,
                cache.u0,
                None,
                cache.batch.step_size,
                cache.p_mid,
                direction,
                rollout=cache.rollout,
            )
            residual_dot, coeff_dot, qform = projected_decoder_residual_tangent(
                self.objective.decoder,
                cache.normal,
                cache.rollout.states,
                inc.states_dot,
                cache.batch.qoi,
                weights=cache.weights,
                ridge=self.objective.decoder_ridge,
                chunk_size=self.objective.normal_chunk_size,
            )
            vector = weighted_residual_tangent_vector(
                residual_dot,
                coeff_dot,
                cache.weights,
                self.objective.decoder_ridge,
            )
            states_dot = inc.states_dot if keep_states_dot else residual_dot.new_empty(0)
        return ReducedJVPResult(
            residual_dot=residual_dot,
            decoder_coeff_dot=coeff_dot,
            residual_vector=vector,
            quadratic_form=qform,
            states_dot=states_dot,
        )

    def jvp_batch(self, directions: list[Direction], *, keep_states_dot: bool = False) -> BatchedReducedJVPResult:
        if len(directions) <= 0:
            raise ValueError("directions must be nonempty")
        cache = self.cache
        with torch.no_grad():
            states_dot = lagged_midpoint_rollout_incremental_batched(
                self.dynamics,
                cache.u0,
                cache.batch.step_size,
                cache.p_mid,
                directions,
                rollout=cache.rollout,
            )
            residual_dot, coeff_dot, qform = projected_decoder_residual_tangent_batched(
                self.objective.decoder,
                cache.normal,
                cache.rollout.states,
                states_dot,
                cache.batch.qoi,
                weights=cache.weights,
                ridge=self.objective.decoder_ridge,
                chunk_size=self.objective.normal_chunk_size,
            )
            vectors = weighted_residual_tangent_matrix(
                residual_dot,
                coeff_dot,
                cache.weights,
                self.objective.decoder_ridge,
            )
            kept_states_dot = states_dot if keep_states_dot else residual_dot.new_empty(0)
        return BatchedReducedJVPResult(
            residual_dot=residual_dot,
            decoder_coeff_dot=coeff_dot,
            residual_vectors=vectors,
            quadratic_form=qform,
            states_dot=kept_states_dot,
        )

    def jvp_batch_from_flat(self, flat_directions: torch.Tensor, *, keep_states_dot: bool = False) -> BatchedReducedJVPResult:
        if flat_directions.ndim != 2:
            raise ValueError("flat_directions must have shape parameter_dim x direction_count")
        directions = [self.direction_from_flat(flat_directions[:, idx]) for idx in range(flat_directions.shape[1])]
        return self.jvp_batch(directions, keep_states_dot=keep_states_dot)

    def sketched_jvp_batch_from_flat_streaming(
        self,
        flat_directions: torch.Tensor,
        *,
        sketch_dim: int,
        seed: int = 0,
        sample_chunk_size: int = 64,
    ) -> StreamingSketchJVPResult:
        if flat_directions.ndim != 2:
            raise ValueError("flat_directions must have shape parameter_dim x direction_count")
        if int(sketch_dim) <= 0:
            raise ValueError("sketch_dim must be positive for streaming sketch")
        directions = [self.direction_from_flat(flat_directions[:, idx]) for idx in range(flat_directions.shape[1])]
        cache = self.cache
        states = cache.rollout.states
        targets = cache.batch.qoi
        if states.shape[:-1] != targets.shape[:-1]:
            raise ValueError("states and targets leading shapes must match")
        steps = int(states.shape[0] - 1)
        sample_count = int(states.shape[1])
        direction_count = int(len(directions))
        output_dim = int(targets.shape[-1])
        feature_dim = int(self.objective.decoder.feature_dim) + int(self.objective.decoder.readout.bias is not None)
        chunk_size = max(1, int(sample_chunk_size))
        d_normal = states.new_zeros(direction_count, feature_dim, feature_dim)
        d_rhs = states.new_zeros(direction_count, output_dim, feature_dim)
        quad_cache = _make_batched_energy_tucker_direction_cache(
            self.dynamics,
            directions,
            device=states.device,
            dtype=states.dtype,
            tau=0.5 * float(cache.batch.step_size),
        )
        with torch.no_grad(), _quadratic_cache_context(self.dynamics):
            for start in range(0, sample_count, chunk_size):
                end = min(start + chunk_size, sample_count)
                sample_slice = slice(start, end)
                u_dot = states.new_zeros(end - start, direction_count, states.shape[-1])
                weights_0 = None if cache.weights is None else cache.weights[0, sample_slice]
                dn, dr = _decoder_tangent_normal_terms_for_state(
                    self.objective.decoder,
                    states[0, sample_slice],
                    u_dot,
                    targets[0, sample_slice],
                    weights=weights_0,
                    feature_dim=feature_dim,
                )
                d_normal.add_(dn)
                d_rhs.add_(dr)
                for n in range(steps):
                    p_n = None if cache.p_mid is None else cache.p_mid[n, sample_slice]
                    u_dot = _batched_tangent_next(
                        self.dynamics,
                        states[n, sample_slice],
                        u_dot,
                        cache.batch.step_size,
                        p_n,
                        directions,
                        picard_iterates=cache.rollout.picard_iterates[n, :, sample_slice],
                        quad_cache=quad_cache,
                    )
                    weights_n = None if cache.weights is None else cache.weights[n + 1, sample_slice]
                    dn, dr = _decoder_tangent_normal_terms_for_state(
                        self.objective.decoder,
                        states[n + 1, sample_slice],
                        u_dot,
                        targets[n + 1, sample_slice],
                        weights=weights_n,
                        feature_dim=feature_dim,
                    )
                    d_normal.add_(dn)
                    d_rhs.add_(dr)
            coeff = cache.normal.coefficients.to(device=states.device, dtype=states.dtype)
            normal_matrix = cache.normal.normal_matrix.to(device=states.device, dtype=states.dtype)
            normal_cholesky = self._decoder_normal_cholesky(device=states.device, dtype=states.dtype)
            rhs_for_dcoeff_t = d_rhs.transpose(1, 2) - d_normal.matmul(coeff.T)
            solved = _solve_decoder_normal(
                normal_matrix,
                rhs_for_dcoeff_t.permute(1, 0, 2).reshape(feature_dim, direction_count * output_dim),
                normal_cholesky,
            )
            dcoeff = solved.reshape(feature_dim, direction_count, output_dim).permute(1, 2, 0).contiguous()
            sketch = states.new_zeros(int(sketch_dim), direction_count)
            qform = states.new_zeros(direction_count)
            for start in range(0, sample_count, chunk_size):
                end = min(start + chunk_size, sample_count)
                sample_slice = slice(start, end)
                u_dot = states.new_zeros(end - start, direction_count, states.shape[-1])
                weights_0 = None if cache.weights is None else cache.weights[0, sample_slice]
                qform.add_(
                    _decoder_residual_sketch_for_state_(
                        sketch,
                        self.objective.decoder,
                        states[0, sample_slice],
                        u_dot,
                        coeff,
                        dcoeff,
                        weights=weights_0,
                        global_sample_start=start,
                        sample_count=sample_count,
                        state_index=0,
                        data_row_seed=int(seed),
                    )
                )
                for n in range(steps):
                    p_n = None if cache.p_mid is None else cache.p_mid[n, sample_slice]
                    u_dot = _batched_tangent_next(
                        self.dynamics,
                        states[n, sample_slice],
                        u_dot,
                        cache.batch.step_size,
                        p_n,
                        directions,
                        picard_iterates=cache.rollout.picard_iterates[n, :, sample_slice],
                        quad_cache=quad_cache,
                    )
                    weights_n = None if cache.weights is None else cache.weights[n + 1, sample_slice]
                    qform.add_(
                        _decoder_residual_sketch_for_state_(
                            sketch,
                            self.objective.decoder,
                            states[n + 1, sample_slice],
                            u_dot,
                            coeff,
                            dcoeff,
                            weights=weights_n,
                            global_sample_start=start,
                            sample_count=sample_count,
                            state_index=n + 1,
                            data_row_seed=int(seed),
                        )
                    )
            if float(self.objective.decoder_ridge) > 0.0:
                ridge_rows = float(self.objective.decoder_ridge) ** 0.5 * dcoeff.reshape(direction_count, -1).T
                ridge_row_count = int(ridge_rows.shape[0])
                data_row_count = int(states.shape[0]) * sample_count * output_dim
                ridge_ids = data_row_count + torch.arange(ridge_row_count, device=states.device, dtype=torch.long)
                _countsketch_add_rows_(sketch, ridge_rows, ridge_ids, seed=int(seed))
                qform = qform + float(self.objective.decoder_ridge) * dcoeff.square().sum(dim=(1, 2))
        residual_dim = int(states.shape[0]) * sample_count * output_dim
        if float(self.objective.decoder_ridge) > 0.0:
            residual_dim += int(dcoeff.numel() // direction_count)
        return StreamingSketchJVPResult(
            sketch_matrix=sketch,
            decoder_coeff_dot=dcoeff,
            quadratic_form=qform,
            residual_dim=residual_dim,
            sketch_dim=int(sketch_dim),
            direction_count=direction_count,
        )

    def sketched_jvp_batch_from_flat_single_pass(
        self,
        flat_directions: torch.Tensor,
        *,
        sketch_dim: int,
        seed: int = 0,
        sample_chunk_size: int = 1024,
        global_sample_offset: int = 0,
        global_sample_count: int | None = None,
    ) -> StreamingSketchJVPResult:
        if flat_directions.ndim != 2:
            raise ValueError("flat_directions must have shape parameter_dim x direction_count")
        if int(sketch_dim) <= 0:
            raise ValueError("sketch_dim must be positive for single-pass streaming sketch")
        directions = [self.direction_from_flat(flat_directions[:, idx]) for idx in range(flat_directions.shape[1])]
        cache = self.cache
        states = cache.rollout.states
        targets = cache.batch.qoi
        if states.shape[:-1] != targets.shape[:-1]:
            raise ValueError("states and targets leading shapes must match")
        steps = int(states.shape[0] - 1)
        sample_count = int(states.shape[1])
        global_count = sample_count if global_sample_count is None else int(global_sample_count)
        global_offset = int(global_sample_offset)
        if global_offset < 0 or global_count < sample_count or global_offset + sample_count > global_count:
            raise ValueError("global sample offset/count are inconsistent with local sample count")
        direction_count = int(len(directions))
        output_dim = int(targets.shape[-1])
        feature_dim = int(self.objective.decoder.feature_dim) + int(self.objective.decoder.readout.bias is not None)
        row_sketch_dim = 0 if int(sketch_dim) <= 0 else max(1, (int(sketch_dim) + output_dim - 1) // output_dim)
        chunk_size = max(1, int(sample_chunk_size))
        coeff = cache.normal.coefficients.to(device=states.device, dtype=states.dtype)
        normal_matrix = cache.normal.normal_matrix.to(device=states.device, dtype=states.dtype)
        rhs_for_dcoeff_t = states.new_zeros(direction_count, feature_dim, output_dim)
        fixed_coeff_cross = states.new_zeros(direction_count, output_dim, feature_dim)
        fixed_fixed_gram = states.new_zeros(direction_count, direction_count)
        feature_sketch = states.new_zeros(row_sketch_dim, feature_dim)
        fixed_residual_sketch = states.new_zeros(row_sketch_dim, direction_count, output_dim)
        fixed_qform = states.new_zeros(direction_count)
        quad_cache = _make_batched_energy_tucker_direction_cache(
            self.dynamics,
            directions,
            device=states.device,
            dtype=states.dtype,
            tau=0.5 * float(cache.batch.step_size),
        )
        with torch.no_grad(), _quadratic_cache_context(self.dynamics):
            for start in range(0, sample_count, chunk_size):
                end = min(start + chunk_size, sample_count)
                sample_slice = slice(start, end)
                u_dot = states.new_zeros(end - start, direction_count, states.shape[-1])
                weights_0 = None if cache.weights is None else cache.weights[0, sample_slice]
                fixed_qform.add_(
                    _single_pass_schur_sketch_state_(
                        rhs_for_dcoeff_t,
                        fixed_coeff_cross,
                        fixed_fixed_gram,
                        feature_sketch,
                        fixed_residual_sketch,
                        self.objective.decoder,
                        states[0, sample_slice],
                        u_dot,
                        targets[0, sample_slice],
                        coeff,
                        weights=weights_0,
                        global_sample_start=global_offset + start,
                        sample_count=global_count,
                        state_index=0,
                        row_seed=int(seed),
                    )
                )
                for n in range(steps):
                    p_n = None if cache.p_mid is None else cache.p_mid[n, sample_slice]
                    u_dot = _batched_tangent_next(
                        self.dynamics,
                        states[n, sample_slice],
                        u_dot,
                        cache.batch.step_size,
                        p_n,
                        directions,
                        picard_iterates=cache.rollout.picard_iterates[n, :, sample_slice],
                        quad_cache=quad_cache,
                    )
                    weights_n = None if cache.weights is None else cache.weights[n + 1, sample_slice]
                    fixed_qform.add_(
                        _single_pass_schur_sketch_state_(
                            rhs_for_dcoeff_t,
                            fixed_coeff_cross,
                            fixed_fixed_gram,
                            feature_sketch,
                            fixed_residual_sketch,
                            self.objective.decoder,
                            states[n + 1, sample_slice],
                            u_dot,
                            targets[n + 1, sample_slice],
                            coeff,
                            weights=weights_n,
                            global_sample_start=global_offset + start,
                            sample_count=global_count,
                            state_index=n + 1,
                            row_seed=int(seed),
                        )
                    )
            normal_cholesky = self._decoder_normal_cholesky(device=states.device, dtype=states.dtype)
            solved = _solve_decoder_normal(
                normal_matrix,
                rhs_for_dcoeff_t.permute(1, 0, 2).reshape(feature_dim, direction_count * output_dim),
                normal_cholesky,
            )
            dcoeff = solved.reshape(feature_dim, direction_count, output_dim).permute(1, 2, 0).contiguous()
            coeff_residual_sketch = feature_sketch.matmul(dcoeff.reshape(direction_count * output_dim, feature_dim).T).reshape(
                row_sketch_dim,
                direction_count,
                output_dim,
            )
            residual_sketch = fixed_residual_sketch + coeff_residual_sketch
            sketch_matrix = residual_sketch.movedim(1, 2).reshape(row_sketch_dim * output_dim, direction_count)
            normal_no_ridge = normal_matrix - float(self.objective.decoder_ridge) * torch.eye(
                normal_matrix.shape[0],
                device=normal_matrix.device,
                dtype=normal_matrix.dtype,
            )
            fixed_coeff_rows = fixed_coeff_cross.reshape(direction_count, output_dim * feature_dim)
            dcoeff_rows = dcoeff.reshape(direction_count, output_dim * feature_dim)
            fixed_coeff_gram = fixed_coeff_rows.matmul(dcoeff_rows.T)
            normal_dcoeff = dcoeff.matmul(normal_no_ridge.T)
            coeff_gram = normal_dcoeff.reshape(direction_count, output_dim * feature_dim).matmul(dcoeff_rows.T)
            exact_projected_gram = fixed_fixed_gram + fixed_coeff_gram + fixed_coeff_gram.T + coeff_gram
            dcoeff_qform = 2.0 * (fixed_coeff_rows * dcoeff_rows).sum(dim=1)
            dcoeff_qform = dcoeff_qform + (normal_dcoeff * dcoeff).sum(dim=(1, 2))
            qform = fixed_qform + dcoeff_qform
            if float(self.objective.decoder_ridge) > 0.0:
                ridge_rows = float(self.objective.decoder_ridge) ** 0.5 * dcoeff.reshape(direction_count, -1).T
                sketch_matrix = torch.cat((sketch_matrix, ridge_rows), dim=0)
                ridge_gram = float(self.objective.decoder_ridge) * dcoeff_rows.matmul(dcoeff_rows.T)
                exact_projected_gram = exact_projected_gram + ridge_gram
                qform = qform + float(self.objective.decoder_ridge) * dcoeff.square().sum(dim=(1, 2))
            sketch_matrix = torch.linalg.cholesky(
                0.5 * (exact_projected_gram + exact_projected_gram.T)
                + 1.0e-12 * torch.eye(direction_count, device=states.device, dtype=states.dtype),
                upper=False,
            ).T
        residual_dim = int(states.shape[0]) * global_count * output_dim
        if float(self.objective.decoder_ridge) > 0.0:
            residual_dim += int(dcoeff.numel() // direction_count)
        return StreamingSketchJVPResult(
            sketch_matrix=sketch_matrix,
            decoder_coeff_dot=dcoeff,
            quadratic_form=qform,
            residual_dim=residual_dim,
            sketch_dim=int(sketch_matrix.shape[0]),
            direction_count=direction_count,
        )

    def hvp(self, direction: Direction) -> tuple[Direction, torch.Tensor]:
        cache = self.cache
        with torch.no_grad():
            return lagged_midpoint_gauss_newton_hvp(
                self.dynamics,
                self.objective.decoder,
                cache.normal,
                cache.batch,
                direction,
                rollout=cache.rollout,
                weights=cache.weights,
                picard_iters=self.objective.stepper.picard_iters,
                decoder_ridge=self.objective.decoder_ridge,
                linear_ridge=self.objective.linear_ridge,
                quadratic_ridge=self.objective.quadratic_ridge,
                source_ridge=self.objective.source_ridge,
                chunk_size=self.objective.normal_chunk_size,
            )

    def hessian_flat(self, flat_direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hvp, qform = self.hvp(self.direction_from_flat(flat_direction))
        return self.flatten(hvp), qform

    def _flatten_batched_parameter_grads(self, grads: dict[str, torch.Tensor], direction_count: int) -> torch.Tensor:
        pieces = []
        for name, param in self.dynamics.named_parameters():
            if not param.requires_grad:
                continue
            value = grads.get(name)
            if value is None:
                value = torch.zeros(int(direction_count), *param.shape, device=param.device, dtype=param.dtype)
            pieces.append(value.reshape(int(direction_count), -1).T)
        return torch.cat(pieces, dim=0)

    def hessian_batch_flat(
        self,
        flat_directions: torch.Tensor,
        *,
        sample_chunk_size: int = 64,
        decoder_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the GNVP Hessian to several directions with a batched adjoint.

        This computes ``J.T @ (J @ flat_directions)`` and the corresponding
        directional quadratic forms.  It streams over sample chunks to avoid
        materializing ``states_dot`` for the full local shard.
        """
        from .adjoint import lagged_midpoint_rollout_adjoint_batched

        if flat_directions.ndim != 2:
            raise ValueError("flat_directions must have shape parameter_dim x direction_count")
        if int(flat_directions.shape[0]) != self.parameter_dim:
            raise ValueError("flat_directions first dimension must equal parameter_dim")
        param = next(self.dynamics.parameters())
        work = flat_directions.to(device=param.device, dtype=param.dtype)
        direction_count = int(work.shape[1])
        directions = [self.direction_from_flat(work[:, idx]) for idx in range(direction_count)]
        cache = self.cache
        sample_count = int(cache.rollout.states.shape[1])
        chunk_size = max(1, int(sample_chunk_size))
        decoder_chunk = int(self.objective.normal_chunk_size if decoder_chunk_size is None else decoder_chunk_size)
        decoder_chunk = max(1, decoder_chunk)
        hvp_flat = work.new_zeros(work.shape)
        qforms = work.new_zeros(direction_count)
        timings = {
            "chunks": 0,
            "decoder_normal_factor": 0.0,
            "incremental_tangent": 0.0,
            "decoder_gn_cotangent": 0.0,
            "batched_adjoint": 0.0,
            "flatten_accumulate": 0.0,
            "regularization": 0.0,
        }
        t0 = time.perf_counter()
        normal_cholesky = self._decoder_normal_cholesky(device=param.device, dtype=param.dtype)
        _sync_if_cuda(normal_cholesky)
        timings["decoder_normal_factor"] = time.perf_counter() - t0

        for start in range(0, sample_count, chunk_size):
            end = min(start + chunk_size, sample_count)
            timings["chunks"] += 1
            sample_slice = slice(start, end)
            p_mid = None if cache.p_mid is None else cache.p_mid[:, sample_slice]
            weights = None if cache.weights is None else cache.weights[:, sample_slice]
            rollout_chunk = DensePicardRolloutResult(
                states=cache.rollout.states[:, sample_slice],
                lags=cache.rollout.lags[:, sample_slice],
                picard_iterates=cache.rollout.picard_iterates[:, :, sample_slice],
            )
            u0 = cache.u0[sample_slice]
            t0 = time.perf_counter()
            states_dot = lagged_midpoint_rollout_incremental_batched(
                self.dynamics,
                u0,
                cache.batch.step_size,
                p_mid,
                directions,
                rollout=rollout_chunk,
            )
            _sync_if_cuda(states_dot)
            timings["incremental_tangent"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            state_cot, qform = gauss_newton_state_cotangents_batched(
                self.objective.decoder,
                cache.normal,
                rollout_chunk.states,
                states_dot,
                cache.batch.qoi[:, sample_slice],
                weights=weights,
                ridge=self.objective.decoder_ridge,
                chunk_size=decoder_chunk,
                normal_cholesky=normal_cholesky,
                reuse_states_dot_buffer=os.environ.get("GOATTM_REUSE_STATE_DOT_BUFFER", "1") != "0",
            )
            _sync_if_cuda(state_cot)
            timings["decoder_gn_cotangent"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            adj = lagged_midpoint_rollout_adjoint_batched(
                self.dynamics,
                u0,
                cache.batch.step_size,
                state_cot,
                p_mid,
                picard_iters=self.objective.stepper.picard_iters,
                states=rollout_chunk.states,
                picard_iterates=rollout_chunk.picard_iterates,
                return_input_adjoint=False,
            )
            first_grad = next(iter(adj.parameter_grads.values())) if adj.parameter_grads else qform
            _sync_if_cuda(first_grad)
            timings["batched_adjoint"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            hvp_flat.add_(self._flatten_batched_parameter_grads(adj.parameter_grads, direction_count))
            qforms.add_(qform)
            _sync_if_cuda(hvp_flat)
            timings["flatten_accumulate"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        offset = 0
        for name, param in self.dynamics.named_parameters():
            if not param.requires_grad:
                continue
            count = int(param.numel())
            if name.startswith("linear."):
                ridge = float(self.objective.linear_ridge)
            elif name.startswith("quadratic."):
                ridge = float(self.objective.quadratic_ridge)
            elif name.startswith("source."):
                ridge = float(self.objective.source_ridge)
            else:
                ridge = 0.0
            if ridge > 0.0:
                block = work[offset : offset + count]
                hvp_flat[offset : offset + count].add_(ridge * block)
                qforms.add_(ridge * block.square().sum(dim=0))
            offset += count
        _sync_if_cuda(hvp_flat)
        timings["regularization"] += time.perf_counter() - t0
        self.last_hessian_batch_timings = timings
        return hvp_flat, qforms

    def assemble_direction_hessian_flat(
        self,
        flat_directions: torch.Tensor,
        *,
        output_device: torch.device | str | None = None,
        symmetrize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the exact dense Hessian projected to a supplied direction basis.

        If ``flat_directions`` has shape ``(parameter_dim, k)``, this returns
        ``flat_directions.T @ H @ flat_directions`` and the directional
        quadratic forms from the HVP calls.  This is the most practical exact
        comparator for sketch ranks k=16/32/64 because it avoids materializing
        the full parameter-space Hessian.
        """
        if flat_directions.ndim != 2:
            raise ValueError("flat_directions must have shape parameter_dim x direction_count")
        if int(flat_directions.shape[0]) != self.parameter_dim:
            raise ValueError("flat_directions first dimension must equal parameter_dim")
        param = next(self.dynamics.parameters())
        work = flat_directions.to(device=param.device, dtype=param.dtype)
        out_device = param.device if output_device is None else torch.device(output_device)
        direction_count = int(work.shape[1])
        projected = torch.empty(direction_count, direction_count, device=out_device, dtype=param.dtype)
        qforms = torch.empty(direction_count, device=out_device, dtype=param.dtype)
        for j in range(direction_count):
            hvp_flat, qform = self.hessian_flat(work[:, j])
            projected[:, j] = (work.T @ hvp_flat).to(device=out_device)
            qforms[j] = qform.to(device=out_device)
        if symmetrize:
            projected = 0.5 * (projected + projected.T)
        return projected, qforms

    def assemble_hessian_flat(
        self,
        *,
        column_start: int = 0,
        column_count: int | None = None,
        output_device: torch.device | str | None = None,
        symmetrize: bool = True,
        max_columns_without_confirmation: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble explicit HVP columns of the full parameter-space Hessian.

        This routine is intentionally simple and memory-aware rather than fast:
        it applies the existing exact HVP to standard-basis directions one
        column at a time.  Use it for small research probes; for sketch-quality
        checks, prefer :meth:`assemble_direction_hessian_flat`.
        """
        parameter_dim = self.parameter_dim
        start = int(column_start)
        if start < 0 or start > parameter_dim:
            raise ValueError("column_start is out of range")
        count = parameter_dim - start if column_count is None else int(column_count)
        if count < 0 or start + count > parameter_dim:
            raise ValueError("column range is out of bounds")
        if count > int(max_columns_without_confirmation):
            raise ValueError(
                "refusing to assemble more than "
                f"{int(max_columns_without_confirmation)} columns; pass a larger "
                "max_columns_without_confirmation if this is intentional"
            )
        param = next(self.dynamics.parameters())
        out_device = param.device if output_device is None else torch.device(output_device)
        hessian_columns = torch.empty(parameter_dim, count, device=out_device, dtype=param.dtype)
        qforms = torch.empty(count, device=out_device, dtype=param.dtype)
        basis_vector = torch.zeros(parameter_dim, device=param.device, dtype=param.dtype)
        for local_col, global_col in enumerate(range(start, start + count)):
            basis_vector.zero_()
            basis_vector[global_col] = 1.0
            hvp_flat, qform = self.hessian_flat(basis_vector)
            hessian_columns[:, local_col] = hvp_flat.to(device=out_device)
            qforms[local_col] = qform.to(device=out_device)
        if symmetrize and start == 0 and count == parameter_dim:
            hessian_columns = 0.5 * (hessian_columns + hessian_columns.T)
        return hessian_columns, qforms


def countsketch_apply(
    vector: torch.Tensor,
    indices: torch.Tensor,
    signs: torch.Tensor,
    sketch_dim: int,
) -> torch.Tensor:
    out = vector.new_zeros(int(sketch_dim))
    out.scatter_add_(0, indices, signs.to(device=vector.device, dtype=vector.dtype) * vector)
    return out


def countsketch_apply_matrix(
    matrix: torch.Tensor,
    indices: torch.Tensor,
    signs: torch.Tensor,
    sketch_dim: int,
) -> torch.Tensor:
    out = matrix.new_zeros(int(sketch_dim), matrix.shape[1])
    out.index_add_(0, indices, signs.to(device=matrix.device, dtype=matrix.dtype)[:, None] * matrix)
    return out


def make_countsketch(
    residual_dim: int,
    sketch_dim: int,
    *,
    device: torch.device,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    indices = torch.randint(
        int(sketch_dim),
        (int(residual_dim),),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    signs = torch.randint(
        2,
        (int(residual_dim),),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    signs = signs.to(dtype=dtype).mul_(2.0).sub_(1.0)
    return indices, signs
