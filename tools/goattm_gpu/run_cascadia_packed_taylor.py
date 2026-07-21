from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_cascadia_packed import (  # noqa: E402
    MaskedCrossQuadraticReadoutDecoder,
    apply_pod_initializer,
    dense_a_symmetric_spectral_report,
    install_dense_a_symmetric_spectral_penalty,
    load_packed_batch,
    make_dynamics,
)
from quadrode_gpu_goattm import (  # noqa: E402
    DenseLaggedMidpointStepper,
    QuadraticReadoutDecoder,
    ReducedObjective,
    RungeKutta4Stepper,
    SubstepRungeKutta4Stepper,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clone_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def restore_params(module: torch.nn.Module, values: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in values:
                param.copy_(values[name])


def random_unit_direction(module: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    device = next(module.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    direction: dict[str, torch.Tensor] = {}
    norm2 = None
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        value = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=generator)
        direction[name] = value
        piece = value.square().sum()
        norm2 = piece if norm2 is None else norm2 + piece
    if norm2 is None:
        raise ValueError("no trainable parameters")
    scale = torch.sqrt(norm2)
    return {name: value / scale for name, value in direction.items()}


def add_direction(module: torch.nn.Module, direction: dict[str, torch.Tensor], alpha: float) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in direction:
                param.add_(direction[name], alpha=float(alpha))


def direction_dot_grad(module: torch.nn.Module, direction: dict[str, torch.Tensor]) -> torch.Tensor:
    out = None
    for name, param in module.named_parameters():
        if name not in direction:
            continue
        piece = (
            torch.zeros((), device=param.device, dtype=param.dtype)
            if param.grad is None
            else (param.grad * direction[name]).sum()
        )
        out = piece if out is None else out + piece
    if out is None:
        raise ValueError("empty direction")
    return out


def make_decoder(args: argparse.Namespace, output_dim: int, device: torch.device) -> torch.nn.Module:
    if args.decoder_quadratic_mode == "masked_cross":
        decoder = MaskedCrossQuadraticReadoutDecoder(
            int(args.latent_dim),
            output_dim,
            cross_terms=int(args.decoder_cross_terms),
            mask_seed=int(args.decoder_mask_seed),
            bias=True,
        )
    else:
        decoder = QuadraticReadoutDecoder(
            int(args.latent_dim),
            output_dim,
            include_quadratic=args.decoder_quadratic_mode == "full",
            bias=True,
        )
    return decoder.double().to(device)


def write_plot(points: list[dict[str, float]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eps = [p["eps"] for p in points]
        first = [p["first_remainder"] for p in points]
        directional = [p["directional_error"] for p in points]
        plt.figure(figsize=(7.5, 4.8))
        plt.loglog(eps, first, "-o", label="|J(m+eps p)-J(m)-eps <g,p>|")
        plt.loglog(eps, directional, "-s", label="first remainder / eps")
        plt.gca().invert_xaxis()
        plt.xlabel("epsilon")
        plt.ylabel("Taylor error")
        plt.title("Cascadia packed Taylor test")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
    except Exception as exc:
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascadia packed-data Taylor test.")
    parser.add_argument("--train-packed", required=True)
    parser.add_argument("--initializer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--linear-a", choices=("dense", "dissipative_skew"), default="dissipative_skew")
    parser.add_argument("--a-rank", type=int, default=20)
    parser.add_argument("--a-damping-init", type=float, default=0.1)
    parser.add_argument("--a-damping-shift", type=float, default=0.0)
    parser.add_argument("--quadratic", choices=("energy_dense", "energy_tucker"), default="energy_tucker")
    parser.add_argument("--h-reduced-rank", type=int, default=50)
    parser.add_argument("--h-tt-rank", type=int, default=16)
    parser.add_argument("--decoder-quadratic-mode", choices=("full", "masked_cross", "none"), default="masked_cross")
    parser.add_argument("--decoder-cross-terms", type=int, default=2000)
    parser.add_argument("--decoder-mask-seed", type=int, default=20260705)
    parser.add_argument("--stepper", choices=("lagged", "rk4", "rk4_substep"), default="lagged")
    parser.add_argument("--rk4-substeps", type=int, default=1)
    parser.add_argument("--time-mode", choices=("normalized", "step_index"), default="normalized")
    parser.add_argument("--picard-iters", type=int, default=2)
    parser.add_argument("--decoder-ridge", type=float, default=1.0e-5)
    parser.add_argument("--dynamics-ridge", type=float, default=1.0e-7)
    parser.add_argument("--dense-a-symmetric-penalty", type=float, default=0.0)
    parser.add_argument("--dense-a-symmetric-temperature", type=float, default=1.0e-2)
    parser.add_argument("--normal-chunk-size", type=int, default=4096)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--eps-min", type=float, default=1.0e-7)
    parser.add_argument("--eps-max", type=float, default=1.0e-2)
    parser.add_argument("--eps-count", type=int, default=7)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sample_slice = slice(int(args.sample_offset), int(args.sample_offset) + int(args.sample_count))
    batch, data_metadata = load_packed_batch(
        Path(args.train_packed),
        device,
        sample_slice=sample_slice,
        time_mode=args.time_mode,
    )
    input_dim = int(batch.input_values.shape[-1]) if batch.input_values is not None else 0
    output_dim = int(batch.qoi.shape[-1])
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
    decoder = make_decoder(args, output_dim, device)
    initializer_metadata = None
    if args.initializer:
        initializer_metadata = apply_pod_initializer(dynamics, decoder, Path(args.initializer))

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

    objective = ReducedObjective(
        dynamics,
        decoder,
        stepper,
        decoder_ridge=float(args.decoder_ridge),
        dynamics_ridge=float(args.dynamics_ridge),
        normal_chunk_size=int(args.normal_chunk_size),
        gradient_mode=gradient_mode,
    )
    install_dense_a_symmetric_spectral_penalty(
        objective,
        weight=float(args.dense_a_symmetric_penalty),
        temperature=float(args.dense_a_symmetric_temperature),
    )

    base = clone_params(objective.dynamics)
    direction = random_unit_direction(objective.dynamics, int(args.seed) + 17)
    sync(device)
    start = time.perf_counter()
    result = objective.value_and_grad(batch)
    sync(device)
    gradient_seconds = time.perf_counter() - start
    base_loss = float(result.loss.detach().cpu())
    grad_dot = float(direction_dot_grad(objective.dynamics, direction).detach().cpu())

    eps_values = torch.logspace(
        math.log10(float(args.eps_max)),
        math.log10(float(args.eps_min)),
        int(args.eps_count),
        dtype=torch.float64,
    ).tolist()
    points: list[dict[str, float]] = []
    for eps in eps_values:
        restore_params(objective.dynamics, base)
        add_direction(objective.dynamics, direction, float(eps))
        sync(device)
        trial_start = time.perf_counter()
        trial = objective.evaluate(batch)
        sync(device)
        trial_seconds = time.perf_counter() - trial_start
        trial_loss = float(trial.loss.detach().cpu())
        first = abs(trial_loss - base_loss - float(eps) * grad_dot)
        points.append(
            {
                "eps": float(eps),
                "trial_loss": trial_loss,
                "first_remainder": first,
                "directional_error": first / float(eps),
                "eval_seconds": trial_seconds,
            }
        )
    restore_params(objective.dynamics, base)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_packed": str(Path(args.train_packed)),
        "device": str(device),
        "torch_version": torch.__version__,
        "sample_offset": int(args.sample_offset),
        "sample_count": int(args.sample_count),
        "latent_dim": int(args.latent_dim),
        "linear_a": args.linear_a,
        "linear_a_rank": int(args.a_rank),
        "stepper": args.stepper,
        "rk4_substeps": int(args.rk4_substeps) if args.stepper == "rk4_substep" else None,
        "time_mode": args.time_mode,
        "gradient_mode": gradient_mode,
        "quadratic": args.quadratic,
        "h_reduced_rank": int(args.h_reduced_rank),
        "h_tt_rank": int(args.h_tt_rank),
        "decoder_quadratic_mode": args.decoder_quadratic_mode,
        "decoder_feature_dim": int(getattr(decoder, "feature_dim", decoder.readout.in_features)),
        "decoder_cross_terms": int(getattr(decoder, "quadratic_i", torch.empty(0)).numel()),
        "decoder_mask_seed": int(args.decoder_mask_seed),
        "picard_iters": int(args.picard_iters),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "steps": int(batch.steps),
        "dt": float(batch.step_size),
        "base_loss": base_loss,
        "data_loss": float(result.data_loss.detach().cpu()),
        "dynamics_regularization_loss": float(result.dynamics_regularization_loss.detach().cpu()),
        "decoder_regularization_loss": float(result.decoder_regularization_loss.detach().cpu()),
        "grad_dot": grad_dot,
        "gradient_seconds": gradient_seconds,
        "normal_relative_residual": float(result.normal_solve.relative_residual),
        "dense_a_symmetric_penalty": {
            "weight": float(args.dense_a_symmetric_penalty),
            "temperature": float(args.dense_a_symmetric_temperature),
            "active": bool(float(args.dense_a_symmetric_penalty) > 0.0),
            **dense_a_symmetric_spectral_report(
                dynamics,
                weight=float(args.dense_a_symmetric_penalty),
                temperature=float(args.dense_a_symmetric_temperature),
            ),
        },
        "peak_memory_mib": (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "data_metadata": data_metadata,
        "initializer": initializer_metadata,
        "points": points,
    }
    (outdir / "taylor.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_plot(points, outdir / "taylor.png")

    summary_keys = [
        "device",
        "sample_count",
        "latent_dim",
        "linear_a",
        "stepper",
        "time_mode",
        "quadratic",
        "decoder_feature_dim",
        "decoder_cross_terms",
        "input_dim",
        "output_dim",
        "steps",
        "dt",
        "base_loss",
        "gradient_seconds",
        "normal_relative_residual",
        "peak_memory_mib",
    ]
    print(json.dumps({key: payload[key] for key in summary_keys}, indent=2), flush=True)
    print("wrote", outdir / "taylor.json", flush=True)
    print("wrote", outdir / "taylor.png", flush=True)


if __name__ == "__main__":
    main()
