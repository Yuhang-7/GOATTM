from __future__ import annotations

import json

import torch

from quadrode_gpu_goattm.adjoint import lagged_midpoint_rollout_adjoint
from quadrode_gpu_goattm.dynamics import QuadraticDynamics
from quadrode_gpu_goattm.incremental import lagged_midpoint_rollout_incremental, random_direction_like
from quadrode_gpu_goattm.linear import DissipativeSkewA
from quadrode_gpu_goattm.quadratic import EnergyTuckerTTQuadratic
from quadrode_gpu_goattm.source import LinearSource
from quadrode_gpu_goattm.steppers import DenseLaggedMidpointStepper


def direction_dot(direction, grads, dynamics) -> torch.Tensor:
    total = None
    for name, param in dynamics.named_parameters():
        if not param.requires_grad:
            continue
        piece = (direction.get(name, torch.zeros_like(param)) * grads.get(name, torch.zeros_like(param))).sum()
        total = piece if total is None else total + piece
    return next(dynamics.parameters()).new_zeros(()) if total is None else total


def main() -> None:
    torch.manual_seed(20260705)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    batch = 32
    steps = 7
    r = 10
    input_dim = 5
    h = 0.04
    dynamics = QuadraticDynamics(
        DissipativeSkewA(r, skew_rank=4, damping_init=0.12, factor_scale=0.04, damping_shift=0.08),
        EnergyTuckerTTQuadratic(r, reduced_rank=6, tt_rank=5, scale=0.03, basis_trainable=True),
        LinearSource(r, input_dim, init_scale=0.04),
    ).to(device=device, dtype=dtype)
    stepper = DenseLaggedMidpointStepper(picard_iters=2)
    u0 = torch.zeros(batch, r, device=device, dtype=dtype)
    p_mid = 0.1 * torch.randn(steps, batch, input_dim, device=device, dtype=dtype)
    rollout = stepper.rollout_with_picard_history(dynamics, u0, h, steps, p_mid=p_mid)
    records = []
    for seed in range(8):
        direction = random_direction_like(dynamics, seed=1000 + seed, scale=1.0)
        state_cot = torch.randn_like(rollout.states)
        inc = lagged_midpoint_rollout_incremental(
            dynamics,
            u0,
            None,
            h,
            p_mid,
            direction,
            rollout=rollout,
        )
        adj = lagged_midpoint_rollout_adjoint(
            dynamics,
            u0,
            h,
            state_cot,
            p_mid,
            picard_iters=stepper.picard_iters,
            states=rollout.states,
            picard_iterates=rollout.picard_iterates,
            return_input_adjoint=False,
        )
        lhs = (state_cot * inc.states_dot).sum()
        rhs = direction_dot(direction, adj.parameter_grads, dynamics)
        rel = torch.abs(lhs - rhs) / (1.0 + torch.abs(lhs) + torch.abs(rhs))
        records.append(
            {
                "seed": seed,
                "lhs_state_dot": float(lhs.detach().cpu()),
                "rhs_param_adjoint": float(rhs.detach().cpu()),
                "relative_gap": float(rel.detach().cpu()),
            }
        )
    print(json.dumps({"device": str(device), "records": records}, indent=2))


if __name__ == "__main__":
    main()
