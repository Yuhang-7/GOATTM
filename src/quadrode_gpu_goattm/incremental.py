from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .data import ContinuousBatch
from .dynamics import QuadraticDynamics
from .linear import DenseLinearA, DissipativeSkewA
from .quadratic import DenseQuadratic, EnergyDenseQuadratic, EnergyTuckerTTQuadratic, SkewCPQuadratic
from .source import LinearSource, ZeroSource
from .steppers import DensePicardRolloutResult, exact_frozen_solve, predictor


Direction = dict[str, torch.Tensor]


def _quadratic_cache_context(dynamics: QuadraticDynamics):
    cached = getattr(dynamics.quadratic, "cached_reduced_tensor", None)
    if cached is None:
        return nullcontext()
    return cached()


@dataclass(frozen=True)
class IncrementalRolloutResult:
    states_dot: torch.Tensor
    lags_dot: torch.Tensor
    picard_iterates_dot: torch.Tensor


def zero_direction_like(dynamics: QuadraticDynamics) -> Direction:
    return {name: torch.zeros_like(param) for name, param in dynamics.named_parameters() if param.requires_grad}


def random_direction_like(dynamics: QuadraticDynamics, *, seed: int = 0, scale: float = 1.0) -> Direction:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    out: Direction = {}
    for name, param in dynamics.named_parameters():
        if not param.requires_grad:
            continue
        value = torch.randn(param.shape, generator=generator, dtype=param.dtype)
        out[name] = float(scale) * value.to(device=param.device)
    return out


def flatten_direction(direction: Direction, dynamics: QuadraticDynamics) -> torch.Tensor:
    pieces = []
    for name, param in dynamics.named_parameters():
        if param.requires_grad:
            pieces.append(direction.get(name, torch.zeros_like(param)).to(device=param.device, dtype=param.dtype).reshape(-1))
    if not pieces:
        return next(dynamics.parameters()).new_empty(0)
    return torch.cat(pieces)


def direction_from_flat(dynamics: QuadraticDynamics, flat: torch.Tensor) -> Direction:
    out: Direction = {}
    offset = 0
    for name, param in dynamics.named_parameters():
        if not param.requires_grad:
            continue
        count = param.numel()
        out[name] = flat[offset : offset + count].view_as(param).to(device=param.device, dtype=param.dtype)
        offset += count
    if offset != flat.numel():
        raise ValueError("flat direction has the wrong length")
    return out


def add_scaled_parameters(dynamics: QuadraticDynamics, direction: Direction, scale: float) -> None:
    with torch.no_grad():
        for name, param in dynamics.named_parameters():
            if name in direction:
                param.add_(float(scale) * direction[name].to(device=param.device, dtype=param.dtype))


def _direction(direction: Direction | None, name: str, param: torch.Tensor) -> torch.Tensor:
    if direction is None or name not in direction:
        return torch.zeros_like(param)
    return direction[name].to(device=param.device, dtype=param.dtype)


def linear_param_action(dynamics: QuadraticDynamics, x: torch.Tensor, direction: Direction | None) -> torch.Tensor:
    linear = dynamics.linear
    if isinstance(linear, DenseLinearA):
        dA = _direction(direction, "linear.A", linear.A)
        return x @ dA.T
    if isinstance(linear, DissipativeSkewA):
        raw_dot = _direction(direction, "linear.raw_damping", linear.raw_damping)
        out = (-2.0 * linear.raw_damping.to(device=x.device, dtype=x.dtype) * raw_dot.to(device=x.device, dtype=x.dtype)) * x
        if linear.skew_rank > 0:
            p = linear.P.to(device=x.device, dtype=x.dtype)
            q = linear.Q.to(device=x.device, dtype=x.dtype)
            dp = _direction(direction, "linear.P", linear.P).to(device=x.device, dtype=x.dtype)
            dq = _direction(direction, "linear.Q", linear.Q).to(device=x.device, dtype=x.dtype)
            out = out + (x @ dq) @ p.T + (x @ q) @ dp.T - (x @ dp) @ q.T - (x @ p) @ dq.T
        return out
    raise TypeError(f"unsupported linear type {type(linear)!r}")


def linear_tangent(dynamics: QuadraticDynamics, x: torch.Tensor, x_dot: torch.Tensor, direction: Direction | None) -> torch.Tensor:
    return dynamics.linear(x_dot) + linear_param_action(dynamics, x, direction)


def source_param_action(dynamics: QuadraticDynamics, p_mid: torch.Tensor | None, like: torch.Tensor, direction: Direction | None) -> torch.Tensor:
    source = dynamics.source
    if isinstance(source, ZeroSource):
        return torch.zeros_like(like)
    if not isinstance(source, LinearSource):
        raise TypeError(f"unsupported source type {type(source)!r}")
    out = torch.zeros_like(like)
    if p_mid is not None:
        dB = _direction(direction, "source.B", source.B).to(device=like.device, dtype=like.dtype)
        out = out + p_mid @ dB.T
    if source.c is not None:
        dc = _direction(direction, "source.c", source.c).to(device=like.device, dtype=like.dtype)
        out = out + dc
    return out


def _reconstruct_energy_dense_tensor_dot(quad: EnergyDenseQuadratic, direction: Direction | None, *, device, dtype) -> torch.Tensor:
    free_dot = _direction(direction, "quadratic.free_values", quad.free_values).to(device=device, dtype=dtype)
    flat = free_dot.new_zeros(quad.latent_dim**3)
    coeff = quad.reconstruct_coeff.to(device=device, dtype=dtype)
    flat.index_add_(0, quad.reconstruct_target, coeff * free_dot[quad.reconstruct_source])
    return flat.reshape(quad.latent_dim, quad.latent_dim, quad.latent_dim)


def _reconstruct_energy_tucker_tensor_dot(quad: EnergyTuckerTTQuadratic, direction: Direction | None, *, device, dtype) -> torch.Tensor:
    c0 = quad.core0.to(device=device, dtype=dtype)
    c1 = quad.core1.to(device=device, dtype=dtype)
    c2 = quad.core2.to(device=device, dtype=dtype)
    dc0 = _direction(direction, "quadratic.core0", quad.core0).to(device=device, dtype=dtype)
    dc1 = _direction(direction, "quadratic.core1", quad.core1).to(device=device, dtype=dtype)
    dc2 = _direction(direction, "quadratic.core2", quad.core2).to(device=device, dtype=dtype)
    free_full_dot = (
        torch.einsum("ar,rbs,sc->abc", dc0, c1, c2)
        + torch.einsum("ar,rbs,sc->abc", c0, dc1, c2)
        + torch.einsum("ar,rbs,sc->abc", c0, c1, dc2)
    )
    free_values_dot = free_full_dot[quad.free_a, quad.free_b, quad.free_c]
    flat = free_values_dot.new_zeros(quad.reduced_rank**3)
    coeff = quad.reconstruct_coeff.to(device=device, dtype=dtype)
    flat.index_add_(0, quad.reconstruct_target, coeff * free_values_dot[quad.reconstruct_source])
    return flat.reshape(quad.reduced_rank, quad.reduced_rank, quad.reduced_rank)


def quadratic_frozen_param_action(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    direction: Direction | None,
) -> torch.Tensor:
    quad = dynamics.quadratic
    if isinstance(quad, SkewCPQuadratic):
        u = quad.U.to(device=x.device, dtype=x.dtype)
        v = quad.V.to(device=x.device, dtype=x.dtype)
        w = quad.W.to(device=x.device, dtype=x.dtype)
        du = _direction(direction, "quadratic.U", quad.U).to(device=x.device, dtype=x.dtype)
        dv = _direction(direction, "quadratic.V", quad.V).to(device=x.device, dtype=x.dtype)
        dw = _direction(direction, "quadratic.W", quad.W).to(device=x.device, dtype=x.dtype)
        gamma = ell @ w
        alpha = x @ u
        beta = x @ v
        dgamma = ell @ dw
        dalpha = x @ du
        dbeta = x @ dv
        return (
            (dgamma * beta + gamma * dbeta) @ u.T
            + (gamma * beta) @ du.T
            - (dgamma * alpha + gamma * dalpha) @ v.T
            - (gamma * alpha) @ dv.T
        )
    if isinstance(quad, DenseQuadratic):
        dc = _direction(direction, "quadratic.tensor", quad.tensor).to(device=x.device, dtype=x.dtype)
        return torch.einsum("abc,nb,nc->na", dc, ell, x)
    if isinstance(quad, EnergyDenseQuadratic):
        dc = _reconstruct_energy_dense_tensor_dot(quad, direction, device=x.device, dtype=x.dtype)
        return torch.einsum("abc,nb,nc->na", dc, ell, x)
    if isinstance(quad, EnergyTuckerTTQuadratic):
        p = quad.basis.to(device=x.device, dtype=x.dtype)
        dp = _direction(direction, "quadratic.basis", quad.basis).to(device=x.device, dtype=x.dtype)
        c = quad.reduced_tensor().to(device=x.device, dtype=x.dtype)
        dc = _reconstruct_energy_tucker_tensor_dot(quad, direction, device=x.device, dtype=x.dtype)
        z_lag = ell @ p.T
        z = x @ p.T
        dz_lag = ell @ dp.T
        dz = x @ dp.T
        reduced = torch.einsum("abc,nb,nc->na", c, z_lag, z)
        reduced_dot = (
            torch.einsum("abc,nb,nc->na", dc, z_lag, z)
            + torch.einsum("abc,nb,nc->na", c, dz_lag, z)
            + torch.einsum("abc,nb,nc->na", c, z_lag, dz)
        )
        return reduced_dot @ p + reduced @ dp
    raise TypeError(f"unsupported quadratic type {type(quad)!r}")


def quadratic_frozen_tangent(
    dynamics: QuadraticDynamics,
    ell: torch.Tensor,
    x: torch.Tensor,
    ell_dot: torch.Tensor,
    x_dot: torch.Tensor,
    direction: Direction | None,
) -> torch.Tensor:
    return (
        dynamics.quadratic.frozen_action(ell_dot, x)
        + dynamics.quadratic.frozen_action(ell, x_dot)
        + quadratic_frozen_param_action(dynamics, ell, x, direction)
    )


def rhs_tangent(
    dynamics: QuadraticDynamics,
    x: torch.Tensor,
    x_dot: torch.Tensor,
    p_mid: torch.Tensor | None,
    direction: Direction | None,
) -> torch.Tensor:
    return (
        linear_tangent(dynamics, x, x_dot, direction)
        + quadratic_frozen_tangent(dynamics, x, x, x_dot, x_dot, direction)
        + source_param_action(dynamics, p_mid, x, direction)
    )


def lagged_midpoint_step_incremental(
    dynamics: QuadraticDynamics,
    u: torch.Tensor,
    u_dot: torch.Tensor,
    h: float,
    p_mid: torch.Tensor | None,
    direction: Direction | None,
    *,
    picard_iterates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = 0.5 * float(h)
    y_dot = u_dot + float(h) * rhs_tangent(dynamics, u, u_dot, p_mid, direction)
    iterates_dot = [y_dot]
    for k in range(picard_iterates.shape[0] - 1):
        y_prev = picard_iterates[k]
        y_next = picard_iterates[k + 1]
        y_prev_dot = iterates_dot[-1]
        ell = 0.5 * (u + y_prev)
        ell_dot = 0.5 * (u_dot + y_prev_dot)
        rhs_dot = (
            u_dot
            + tau * linear_tangent(dynamics, u, u_dot, direction)
            + tau * quadratic_frozen_tangent(dynamics, ell, u, ell_dot, u_dot, direction)
            + float(h) * source_param_action(dynamics, p_mid, u, direction)
            + tau * linear_param_action(dynamics, y_next, direction)
            + tau
            * (
                dynamics.quadratic.frozen_action(ell_dot, y_next)
                + quadratic_frozen_param_action(dynamics, ell, y_next, direction)
            )
        )
        y_dot = exact_frozen_solve(dynamics, ell, rhs_dot, tau, transpose=False)
        iterates_dot.append(y_dot)
    return y_dot, 0.5 * (u_dot + iterates_dot[-2]), torch.stack(iterates_dot, dim=0)


def lagged_midpoint_rollout_incremental(
    dynamics: QuadraticDynamics,
    u0: torch.Tensor,
    u0_dot: torch.Tensor | None,
    h: float,
    p_mid: torch.Tensor | None,
    direction: Direction | None,
    *,
    rollout: DensePicardRolloutResult,
) -> IncrementalRolloutResult:
    steps = int(rollout.states.shape[0] - 1)
    u_dot = torch.zeros_like(u0) if u0_dot is None else u0_dot
    states_dot = torch.empty_like(rollout.states)
    lags_dot = torch.empty_like(rollout.lags)
    picard_dot = torch.empty_like(rollout.picard_iterates)
    states_dot[0].copy_(u_dot)
    with _quadratic_cache_context(dynamics):
        for n in range(steps):
            p_n = None if p_mid is None else p_mid[n]
            y_dot, lag_dot, it_dot = lagged_midpoint_step_incremental(
                dynamics,
                rollout.states[n],
                u_dot,
                h,
                p_n,
                direction,
                picard_iterates=rollout.picard_iterates[n],
            )
            lags_dot[n].copy_(lag_dot)
            picard_dot[n].copy_(it_dot)
            u_dot = y_dot
            states_dot[n + 1].copy_(u_dot)
    return IncrementalRolloutResult(states_dot=states_dot, lags_dot=lags_dot, picard_iterates_dot=picard_dot)


def decoder_feature_tangent(decoder, states: torch.Tensor, states_dot: torch.Tensor, *, chunk_size: int = 8192) -> torch.Tensor:
    leading = int(states.shape[:-1].numel())
    flat = states.reshape(leading, states.shape[-1])
    flat_dot = states_dot.reshape(leading, states_dot.shape[-1])
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
        if hasattr(decoder, "quadratic_i"):
            idx_i = decoder.quadratic_i.to(device=u.device)
            idx_j = decoder.quadratic_j.to(device=u.device)
            quad_dot = du.index_select(1, idx_i) * u.index_select(1, idx_j) + u.index_select(1, idx_i) * du.index_select(1, idx_j)
            pieces.append(torch.cat((linear, quad_dot), dim=1))
        else:
            quad_pieces = []
            r = u.shape[1]
            for i in range(r):
                quad_pieces.append(du[:, i:] * u[:, i : i + 1] + u[:, i:] * du[:, i : i + 1])
            pieces.append(torch.cat((linear, torch.cat(quad_pieces, dim=1)), dim=1))
    return torch.cat(pieces, dim=0)


def variable_projection_decoder_tangent(
    decoder,
    normal,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    ridge: float,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leading = int(states.shape[:-1].numel())
    flat_states = states.reshape(leading, states.shape[-1])
    flat_targets = targets.reshape(leading, targets.shape[-1])
    d_features = decoder_feature_tangent(decoder, states, states_dot, chunk_size=chunk_size)
    feature_dim = int(decoder.feature_dim) + int(decoder.readout.bias is not None)
    d_normal = states.new_zeros(feature_dim, feature_dim)
    d_rhs = states.new_zeros(flat_targets.shape[1], feature_dim)
    flat_weights = None if weights is None else weights.reshape(leading).to(device=states.device, dtype=states.dtype)
    for start in range(0, leading, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), leading)
        features = decoder.features(flat_states[start:end])
        dfeat = d_features[start:end]
        if decoder.readout.bias is not None:
            ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
            zeros = torch.zeros(dfeat.shape[0], 1, device=dfeat.device, dtype=dfeat.dtype)
            features = torch.cat((features, ones), dim=1)
            dfeat = torch.cat((dfeat, zeros), dim=1)
        if flat_weights is None:
            weighted_features = features
            weighted_dfeat = dfeat
        else:
            w = flat_weights[start:end, None]
            weighted_features = features * w
            weighted_dfeat = dfeat * w
        d_normal.add_(dfeat.T @ weighted_features + features.T @ weighted_dfeat)
        d_rhs.add_(flat_targets[start:end].T @ weighted_dfeat)
    coeff = normal.coefficients.to(device=states.device, dtype=states.dtype)
    normal_matrix = normal.normal_matrix.to(device=states.device, dtype=states.dtype)
    # The stored normal matrix already contains ridge * I, while d_normal does not.
    rhs_for_dcoeff_t = d_rhs.T - d_normal @ coeff.T
    dcoeff_t = torch.linalg.solve(normal_matrix, rhs_for_dcoeff_t)
    dcoeff = dcoeff_t.T
    return d_features, dcoeff, d_normal


def gauss_newton_state_cotangents(
    decoder,
    normal,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    ridge: float,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_features, dcoeff, _ = variable_projection_decoder_tangent(
        decoder,
        normal,
        states,
        states_dot,
        targets,
        weights=weights,
        ridge=ridge,
        chunk_size=chunk_size,
    )
    leading = int(states.shape[:-1].numel())
    flat_states = states.reshape(leading, states.shape[-1])
    coeff = normal.coefficients.to(device=states.device, dtype=states.dtype)
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    dcoeff_no_bias = dcoeff[:, : decoder.feature_dim]
    d_bias = None if decoder.readout.bias is None else dcoeff[:, -1]
    state_cot = torch.empty_like(flat_states)
    gn_loss = states.new_zeros(())
    flat_weights = None if weights is None else weights.reshape(leading).to(device=states.device, dtype=states.dtype)
    for start in range(0, leading, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), leading)
        u = flat_states[start:end]
        features = decoder.features(u)
        dfeat = d_features[start:end]
        d_pred = dfeat @ coeff_no_bias.T + features @ dcoeff_no_bias.T
        if d_bias is not None:
            d_pred = d_pred + d_bias
        if flat_weights is None:
            weighted = d_pred
            gn_loss = gn_loss + 0.5 * d_pred.square().sum()
        else:
            w = flat_weights[start:end]
            weighted = d_pred * w[:, None]
            gn_loss = gn_loss + 0.5 * (d_pred.square().sum(dim=1) * w).sum()
        if hasattr(decoder, "feature_state_grad"):
            state_cot[start:end] = decoder.feature_state_grad(u, weighted @ coeff_no_bias)
        else:
            from .decoders import quadratic_feature_state_grad

            feature_cot = weighted @ coeff_no_bias
            grad = feature_cot[:, : decoder.latent_dim].clone()
            if getattr(decoder, "include_quadratic", False):
                grad = grad + quadratic_feature_state_grad(u, feature_cot[:, decoder.latent_dim :])
            state_cot[start:end] = grad
    reg_loss = 0.5 * float(ridge) * dcoeff.square().sum()
    return state_cot.reshape_as(states), gn_loss + reg_loss


def decoder_feature_state_grad_tangent(decoder, u: torch.Tensor, du: torch.Tensor, feature_cotangent: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(u)
    if not getattr(decoder, "include_quadratic", False):
        return grad
    if hasattr(decoder, "quadratic_i"):
        idx_i = decoder.quadratic_i.to(device=u.device)
        idx_j = decoder.quadratic_j.to(device=u.device)
        quad_cot = feature_cotangent[:, decoder.latent_dim :]
        scatter_i = idx_i.unsqueeze(0).expand(u.shape[0], -1)
        scatter_j = idx_j.unsqueeze(0).expand(u.shape[0], -1)
        grad.scatter_add_(1, scatter_i, quad_cot * du.index_select(1, idx_j))
        grad.scatter_add_(1, scatter_j, quad_cot * du.index_select(1, idx_i))
        return grad
    offset = decoder.latent_dim
    r = int(decoder.latent_dim)
    for i in range(r):
        count = r - i
        cot = feature_cotangent[:, offset : offset + count]
        grad[:, i:] = grad[:, i:] + cot * du[:, i : i + 1]
        grad[:, i : i + 1] = grad[:, i : i + 1] + (cot * du[:, i:]).sum(dim=1, keepdim=True)
        offset += count
    return grad


def exact_decoder_state_gradient_tangent(
    decoder,
    normal,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    ridge: float,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_features, dcoeff, _ = variable_projection_decoder_tangent(
        decoder,
        normal,
        states,
        states_dot,
        targets,
        weights=weights,
        ridge=ridge,
        chunk_size=chunk_size,
    )
    leading = int(states.shape[:-1].numel())
    flat_states = states.reshape(leading, states.shape[-1])
    flat_states_dot = states_dot.reshape(leading, states_dot.shape[-1])
    flat_targets = targets.reshape(leading, targets.shape[-1])
    coeff = normal.coefficients.to(device=states.device, dtype=states.dtype)
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    dcoeff_no_bias = dcoeff[:, : decoder.feature_dim]
    bias = None if decoder.readout.bias is None else coeff[:, -1]
    d_bias = None if decoder.readout.bias is None else dcoeff[:, -1]
    state_cot_dot = torch.empty_like(flat_states)
    flat_weights = None if weights is None else weights.reshape(leading).to(device=states.device, dtype=states.dtype)
    for start in range(0, leading, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), leading)
        u = flat_states[start:end]
        du = flat_states_dot[start:end]
        features = decoder.features(u)
        dfeat = d_features[start:end]
        residual = features @ coeff_no_bias.T - flat_targets[start:end]
        if bias is not None:
            residual = residual + bias
        dres = dfeat @ coeff_no_bias.T + features @ dcoeff_no_bias.T
        if d_bias is not None:
            dres = dres + d_bias
        if flat_weights is None:
            weighted_residual = residual
            weighted_dres = dres
        else:
            w = flat_weights[start:end]
            weighted_residual = residual * w[:, None]
            weighted_dres = dres * w[:, None]
        feature_cot = weighted_residual @ coeff_no_bias
        d_feature_cot = weighted_dres @ coeff_no_bias + weighted_residual @ dcoeff_no_bias
        if hasattr(decoder, "feature_state_grad"):
            state_cot_dot[start:end] = decoder.feature_state_grad(u, d_feature_cot) + decoder_feature_state_grad_tangent(
                decoder,
                u,
                du,
                feature_cot,
            )
        else:
            from .decoders import quadratic_feature_state_grad

            grad = d_feature_cot[:, : decoder.latent_dim].clone()
            if getattr(decoder, "include_quadratic", False):
                grad = grad + quadratic_feature_state_grad(u, d_feature_cot[:, decoder.latent_dim :])
                grad = grad + decoder_feature_state_grad_tangent(decoder, u, du, feature_cot)
            state_cot_dot[start:end] = grad
    return state_cot_dot.reshape_as(states), dcoeff


def projected_decoder_residual_tangent(
    decoder,
    normal,
    states: torch.Tensor,
    states_dot: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    ridge: float,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the varpro Schur projection to the decoder residual tangent.

    For fixed state tangent ``du``, this returns

        dr = dPhi[u;du] c + Phi[u] dc,

    where ``dc`` solves the differentiated ridge-normal equation.  The
    weighted norm of ``dr`` plus ``ridge * ||dc||^2`` is the decoder block
    Gauss-Newton/Schur quadratic form in the state direction ``du``.

    This is not the full exact reduced Hessian when the variable-projected
    residual is nonzero.  The exact reduced Hessian also has residual-weighted
    second-variation terms from the decoder features and from the state map.
    """

    d_features, dcoeff, _ = variable_projection_decoder_tangent(
        decoder,
        normal,
        states,
        states_dot,
        targets,
        weights=weights,
        ridge=ridge,
        chunk_size=chunk_size,
    )
    leading = int(states.shape[:-1].numel())
    flat_states = states.reshape(leading, states.shape[-1])
    output_dim = int(targets.shape[-1])
    coeff = normal.coefficients.to(device=states.device, dtype=states.dtype)
    coeff_no_bias = coeff[:, : decoder.feature_dim]
    dcoeff_no_bias = dcoeff[:, : decoder.feature_dim]
    d_bias = None if decoder.readout.bias is None else dcoeff[:, -1]
    residual_dot = states.new_empty(leading, output_dim)
    qform = states.new_zeros(())
    flat_weights = None if weights is None else weights.reshape(leading).to(device=states.device, dtype=states.dtype)
    for start in range(0, leading, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), leading)
        u = flat_states[start:end]
        features = decoder.features(u)
        dfeat = d_features[start:end]
        dres = dfeat @ coeff_no_bias.T + features @ dcoeff_no_bias.T
        if d_bias is not None:
            dres = dres + d_bias
        residual_dot[start:end] = dres
        if flat_weights is None:
            qform = qform + dres.square().sum()
        else:
            qform = qform + (dres.square().sum(dim=1) * flat_weights[start:end]).sum()
    ridge_qform = float(ridge) * dcoeff.square().sum()
    return residual_dot.reshape_as(targets), dcoeff, qform + ridge_qform


def lagged_midpoint_gauss_newton_hvp(
    dynamics: QuadraticDynamics,
    decoder,
    normal,
    batch: ContinuousBatch,
    direction: Direction,
    *,
    rollout: DensePicardRolloutResult,
    weights: torch.Tensor | None,
    picard_iters: int,
    decoder_ridge: float,
    linear_ridge: float = 0.0,
    quadratic_ridge: float = 0.0,
    source_ridge: float = 0.0,
    chunk_size: int = 8192,
) -> tuple[Direction, torch.Tensor]:
    from .adjoint import lagged_midpoint_rollout_adjoint

    p_mid = batch.midpoint_inputs()
    u0 = batch.u0 if batch.u0 is not None else batch.qoi.new_zeros(batch.batch_size, dynamics.latent_dim)
    inc = lagged_midpoint_rollout_incremental(
        dynamics,
        u0,
        None,
        batch.step_size,
        p_mid,
        direction,
        rollout=rollout,
    )
    state_cot, quadratic_form = gauss_newton_state_cotangents(
        decoder,
        normal,
        rollout.states,
        inc.states_dot,
        batch.qoi,
        weights=weights,
        ridge=decoder_ridge,
        chunk_size=chunk_size,
    )
    adj = lagged_midpoint_rollout_adjoint(
        dynamics,
        u0,
        batch.step_size,
        state_cot,
        p_mid,
        picard_iters=picard_iters,
        states=rollout.states,
        picard_iterates=rollout.picard_iterates,
        return_input_adjoint=False,
    )
    hvp: Direction = {name: value.clone() for name, value in adj.parameter_grads.items()}
    for name, param in dynamics.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("linear."):
            ridge = float(linear_ridge)
        elif name.startswith("quadratic."):
            ridge = float(quadratic_ridge)
        elif name.startswith("source."):
            ridge = float(source_ridge)
        else:
            ridge = 0.0
        if ridge > 0.0:
            hvp[name] = hvp.get(name, torch.zeros_like(param)) + ridge * direction.get(name, torch.zeros_like(param))
            quadratic_form = quadratic_form + 0.5 * ridge * direction.get(name, torch.zeros_like(param)).square().sum()
    return hvp, quadratic_form


def lagged_midpoint_schur_decoder_hvp(
    dynamics: QuadraticDynamics,
    decoder,
    normal,
    batch: ContinuousBatch,
    direction: Direction,
    *,
    rollout: DensePicardRolloutResult,
    weights: torch.Tensor | None,
    picard_iters: int,
    decoder_ridge: float,
    linear_ridge: float = 0.0,
    quadratic_ridge: float = 0.0,
    source_ridge: float = 0.0,
    chunk_size: int = 8192,
) -> Direction:
    """Schur-complement decoder part of the reduced Hessian action.

    This uses the exact tangent of the variable-projected decoder state
    gradient.  It includes the ``dc/dtheta`` feedback from the normal equation,
    unlike the pure Gauss-Newton state cotangent.  It still omits residual
    second variations of the dynamics map; those belong to the full exact
    reduced Hessian/incremental-adjoint layer.
    """

    from .adjoint import lagged_midpoint_rollout_adjoint

    p_mid = batch.midpoint_inputs()
    u0 = batch.u0 if batch.u0 is not None else batch.qoi.new_zeros(batch.batch_size, dynamics.latent_dim)
    inc = lagged_midpoint_rollout_incremental(
        dynamics,
        u0,
        None,
        batch.step_size,
        p_mid,
        direction,
        rollout=rollout,
    )
    state_cot_dot, _ = exact_decoder_state_gradient_tangent(
        decoder,
        normal,
        rollout.states,
        inc.states_dot,
        batch.qoi,
        weights=weights,
        ridge=decoder_ridge,
        chunk_size=chunk_size,
    )
    adj = lagged_midpoint_rollout_adjoint(
        dynamics,
        u0,
        batch.step_size,
        state_cot_dot,
        p_mid,
        picard_iters=picard_iters,
        states=rollout.states,
        picard_iterates=rollout.picard_iterates,
        return_input_adjoint=False,
    )
    hvp: Direction = {name: value.clone() for name, value in adj.parameter_grads.items()}
    for name, param in dynamics.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("linear."):
            ridge = float(linear_ridge)
        elif name.startswith("quadratic."):
            ridge = float(quadratic_ridge)
        elif name.startswith("source."):
            ridge = float(source_ridge)
        else:
            ridge = 0.0
        if ridge > 0.0:
            hvp[name] = hvp.get(name, torch.zeros_like(param)) + ridge * direction.get(name, torch.zeros_like(param))
    return hvp
