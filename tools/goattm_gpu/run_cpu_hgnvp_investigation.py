from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


def add_repo_to_path(repo: Path) -> None:
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def flatten_parameters(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters() if p.requires_grad])


def assign_flat_parameters(module: torch.nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in module.parameters():
            if not param.requires_grad:
                continue
            count = int(param.numel())
            param.copy_(flat[offset : offset + count].reshape_as(param))
            offset += count
    if offset != int(flat.numel()):
        raise ValueError(f"flat vector has {flat.numel()} entries, but assigned {offset}")


def clone_trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters() if p.requires_grad}


def restore_trainable_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in state:
                param.copy_(state[name])


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_float_list(text: str) -> list[float]:
    values = [float(piece) for piece in str(text).split(",") if piece.strip()]
    if not values:
        raise ValueError("alpha list must contain at least one value")
    return values


def safe_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def cholesky_qr_subspace(
    omega: torch.Tensor,
    h_omega: torch.Tensor,
    gram: torch.Tensor,
    *,
    rank: int,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sym = 0.5 * (gram + gram.T)
    eye = torch.eye(sym.shape[0], device=sym.device, dtype=sym.dtype)
    scale = torch.diagonal(sym).abs().max().clamp_min(1.0)
    trial_jitter = float(jitter) * scale
    info = torch.tensor(1, device=sym.device)
    factor = None
    for _ in range(12):
        factor, info = torch.linalg.cholesky_ex(sym + trial_jitter * eye, upper=False)
        if int(info.detach().cpu()) == 0:
            break
        trial_jitter *= 10.0
    if factor is None or int(info.detach().cpu()) != 0:
        raise RuntimeError("failed to Cholesky-factor Omega.T @ H @ Omega")
    r_upper = factor.T
    # K = H Omega R^{-1}; compute without explicitly forming inv(R).
    k_matrix = torch.linalg.solve_triangular(r_upper, h_omega.T, upper=True).T
    left, singular_values, _right = torch.linalg.svd(k_matrix, full_matrices=False)
    basis = left[:, : int(rank)].contiguous()
    return basis, singular_values[: int(rank)].contiguous(), trial_jitter.detach().clone()


def choose_gamma(
    *,
    requested: float,
    mode: str,
    omega: torch.Tensor,
    h_omega: torch.Tensor,
    basis: torch.Tensor,
    singular_values: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    tail_rayleigh = float(requested)
    if basis.numel() > 0:
        omega_res = omega - basis @ (basis.T @ omega)
        h_res = h_omega - basis @ (basis.T @ h_omega)
        numerator = torch.sum(omega_res * h_res)
        denominator = torch.sum(omega_res * omega_res).clamp_min(1.0e-30)
        tail_rayleigh = max(safe_float(numerator / denominator), 1.0e-30)
    min_sigma2 = safe_float(singular_values[-1].square()) if singular_values.numel() else float(requested)
    if mode == "constant":
        gamma = float(requested)
    elif mode == "tail-rayleigh":
        gamma = tail_rayleigh
    elif mode == "max-tail-rayleigh":
        gamma = max(float(requested), tail_rayleigh)
    elif mode == "min-sigma2":
        gamma = min_sigma2
    elif mode == "max-min-sigma2":
        gamma = max(float(requested), min_sigma2)
    else:
        raise ValueError(f"unknown gamma mode {mode!r}")
    return max(float(gamma), 1.0e-30), {
        "tail_rayleigh_gamma": float(tail_rayleigh),
        "min_sketch_sigma2": float(min_sigma2),
    }


def solve_redesigned_newton(
    gradient: torch.Tensor,
    basis: torch.Tensor,
    projected_gram: torch.Tensor,
    *,
    damping: float,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    projected_gradient = basis.T @ gradient
    system = projected_gram + float(damping) * torch.eye(
        projected_gram.shape[0],
        device=projected_gram.device,
        dtype=projected_gram.dtype,
    )
    projected_step = -torch.linalg.solve(system, projected_gradient)
    orthogonal_gradient = gradient - basis @ projected_gradient
    direction = basis @ projected_step - orthogonal_gradient / float(gamma)
    predicted_linear = torch.dot(gradient, direction)
    predicted_quadratic = 0.5 * torch.dot(projected_step, projected_gram @ projected_step)
    complement = direction - basis @ projected_step
    predicted_quadratic = predicted_quadratic + 0.5 * float(gamma) * torch.dot(complement, complement)
    return direction, {
        "projected_gradient_norm": safe_float(torch.linalg.vector_norm(projected_gradient)),
        "projected_step_norm": safe_float(torch.linalg.vector_norm(projected_step)),
        "direction_norm": safe_float(torch.linalg.vector_norm(direction)),
        "gradient_dot_direction": safe_float(torch.dot(gradient, direction)),
        "predicted_linear": safe_float(predicted_linear),
        "predicted_quadratic": safe_float(predicted_quadratic),
        "predicted_model_decrease": safe_float(-(predicted_linear + predicted_quadratic)),
    }


def load_pipeline(repo: Path):
    add_repo_to_path(repo)
    from outputs import run_sketched_gnvp_pipeline as pipeline  # noqa: PLC0415
    from quadrode_gpu_goattm.reduced_gn import ReducedGNWorkspace  # noqa: PLC0415

    return pipeline, ReducedGNWorkspace


def build_workspace(objective, batch, ReducedGNWorkspace, pipeline, device: torch.device):
    workspace = ReducedGNWorkspace(objective, batch)
    pipeline.apply_decoder_coefficients(
        objective.decoder,
        workspace.cache.normal.coefficients.to(device=device, dtype=torch.float64),
    )
    return workspace


def full_loss_from_workspace(objective, workspace, data_loss: torch.Tensor | None = None) -> torch.Tensor:
    if data_loss is None:
        from outputs.run_sketched_gnvp_pipeline import decoder_loss_and_state_grad_for_current_coefficients  # noqa: PLC0415

        data_loss, _ = decoder_loss_and_state_grad_for_current_coefficients(
            objective.decoder,
            workspace.cache.rollout.states.detach(),
            workspace.batch.qoi,
            weights=workspace.cache.weights,
            chunk_size=min(objective.normal_chunk_size, 2048),
        )
    return data_loss + objective._decoder_regularization(workspace.cache.normal) + objective._dynamics_regularization()


def evaluate_loss_current(objective, batch, ReducedGNWorkspace, pipeline, device: torch.device) -> tuple[float, float]:
    workspace = build_workspace(objective, batch, ReducedGNWorkspace, pipeline, device)
    loss = full_loss_from_workspace(objective, workspace)
    sync(device)
    return safe_float(loss), float(workspace.cache.normal.relative_residual)


def compute_gradient_current(objective, batch, workspace, pipeline, device: torch.device) -> tuple[torch.Tensor, float, float, dict[str, float]]:
    p_mid = workspace.cache.p_mid
    u0 = workspace.cache.u0
    weights = workspace.cache.weights
    grad_flat, data_loss, grad_timings = pipeline.gradient_from_rollout(
        objective,
        batch,
        workspace.cache.rollout,
        u0,
        p_mid,
        weights,
        device,
    )
    grad = grad_flat + pipeline.dynamics_regularization_gradient_flat(objective).to(device=device, dtype=torch.float64)
    loss = full_loss_from_workspace(objective, workspace, data_loss=data_loss)
    return grad, safe_float(loss), safe_float(data_loss), dict(grad_timings)


def armijo_search(
    *,
    objective,
    batch,
    ReducedGNWorkspace,
    pipeline,
    device: torch.device,
    base_flat: torch.Tensor,
    base_decoder_state: dict[str, torch.Tensor],
    base_loss: float,
    gradient: torch.Tensor,
    direction: torch.Tensor,
    alphas: list[float],
    c1: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.perf_counter()
    gtd = safe_float(torch.dot(gradient, direction))
    records: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "accepted": False,
        "alpha": 0.0,
        "loss": float(base_loss),
        "decrease": 0.0,
        "normal_relative_residual": math.nan,
    }
    for trial, alpha in enumerate(alphas, start=1):
        assign_flat_parameters(objective.dynamics, base_flat + float(alpha) * direction)
        t0 = time.perf_counter()
        loss, normal_rel = evaluate_loss_current(objective, batch, ReducedGNWorkspace, pipeline, device)
        elapsed = time.perf_counter() - t0
        threshold = float(base_loss) + float(c1) * float(alpha) * gtd
        accepted = bool(loss <= threshold)
        record = {
            "trial": trial,
            "alpha": float(alpha),
            "loss": float(loss),
            "decrease": float(base_loss - loss),
            "threshold": float(threshold),
            "accepted": accepted,
            "seconds": elapsed,
            "normal_relative_residual": float(normal_rel),
        }
        records.append(record)
        if accepted:
            best = {
                "accepted": True,
                "alpha": float(alpha),
                "loss": float(loss),
                "decrease": float(base_loss - loss),
                "normal_relative_residual": float(normal_rel),
            }
            break
        assign_flat_parameters(objective.dynamics, base_flat)
        restore_trainable_state(objective.decoder, base_decoder_state)
    best["line_search_seconds"] = time.perf_counter() - start
    assign_flat_parameters(objective.dynamics, base_flat)
    restore_trainable_state(objective.decoder, base_decoder_state)
    return best, records


def run_step(args, objective, batch, ReducedGNWorkspace, pipeline, device: torch.device, step: int) -> dict[str, Any]:
    timings: dict[str, float] = {}
    step_start = time.perf_counter()
    base_flat = flatten_parameters(objective.dynamics)

    t0 = time.perf_counter()
    workspace = build_workspace(objective, batch, ReducedGNWorkspace, pipeline, device)
    sync(device)
    timings["base_workspace_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gradient, base_loss, data_loss, grad_timings = compute_gradient_current(objective, batch, workspace, pipeline, device)
    sync(device)
    timings["gradient_seconds"] = time.perf_counter() - t0
    for key, value in grad_timings.items():
        timings[f"gradient_{key}"] = float(value)

    grad_norm = safe_float(torch.linalg.vector_norm(gradient))
    base_decoder_state = clone_trainable_state(objective.decoder)

    t0 = time.perf_counter()
    omega = pipeline.make_flat_directions(objective, int(args.probe_rank), int(args.seed) + 104729 * step, device)
    h_omega, qforms = workspace.hessian_batch_flat(
        omega,
        sample_chunk_size=int(args.sample_chunk_size),
        decoder_chunk_size=int(args.decoder_feature_chunk_size) if int(args.decoder_feature_chunk_size) > 0 else None,
    )
    sync(device)
    timings["hgnvp_sketch_seconds"] = time.perf_counter() - t0
    for key, value in getattr(workspace, "last_hessian_batch_timings", {}).items():
        timings[f"hgnvp_sketch_{key}"] = float(value)

    gram = omega.T @ h_omega
    diag_relerr = torch.linalg.vector_norm(torch.diagonal(gram) - qforms) / torch.clamp(torch.linalg.vector_norm(qforms), min=1.0e-30)

    t0 = time.perf_counter()
    basis, sketch_singular_values, used_jitter = cholesky_qr_subspace(
        omega,
        h_omega,
        gram,
        rank=int(args.basis_rank),
        jitter=float(args.cholesky_jitter),
    )
    sync(device)
    timings["subspace_svd_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    h_basis, basis_qforms = workspace.hessian_batch_flat(
        basis,
        sample_chunk_size=int(args.sample_chunk_size),
        decoder_chunk_size=int(args.decoder_feature_chunk_size) if int(args.decoder_feature_chunk_size) > 0 else None,
    )
    projected_gram = 0.5 * (basis.T @ h_basis + h_basis.T @ basis)
    sync(device)
    timings["exact_projected_gram_seconds"] = time.perf_counter() - t0
    exact_diag_relerr = torch.linalg.vector_norm(torch.diagonal(projected_gram) - basis_qforms) / torch.clamp(
        torch.linalg.vector_norm(basis_qforms),
        min=1.0e-30,
    )

    gamma, gamma_record = choose_gamma(
        requested=float(args.gamma),
        mode=str(args.gamma_mode),
        omega=omega,
        h_omega=h_omega,
        basis=basis,
        singular_values=sketch_singular_values,
    )
    direction, direction_record = solve_redesigned_newton(
        gradient,
        basis,
        projected_gram,
        damping=float(args.direction_damping),
        gamma=gamma,
    )
    if direction_record["gradient_dot_direction"] >= 0.0:
        direction = -gradient
        direction_record["direction_kind"] = "negative_gradient_fallback"
        direction_record["direction_norm"] = safe_float(torch.linalg.vector_norm(direction))
        direction_record["gradient_dot_direction"] = safe_float(torch.dot(gradient, direction))
    else:
        direction_record["direction_kind"] = "cpu_hgnvp_redesigned_newton"

    hgnvp_direction_seconds = (
        timings["gradient_seconds"]
        + timings["hgnvp_sketch_seconds"]
        + timings["subspace_svd_seconds"]
        + timings["exact_projected_gram_seconds"]
    )

    hgnvp_best, hgnvp_trials = armijo_search(
        objective=objective,
        batch=batch,
        ReducedGNWorkspace=ReducedGNWorkspace,
        pipeline=pipeline,
        device=device,
        base_flat=base_flat,
        base_decoder_state=base_decoder_state,
        base_loss=base_loss,
        gradient=gradient,
        direction=direction,
        alphas=parse_float_list(args.hgnvp_alphas),
        c1=float(args.armijo_c1),
    )
    hgnvp_total_seconds = hgnvp_direction_seconds + float(hgnvp_best["line_search_seconds"])

    gradient_direction = -gradient
    gradient_best, gradient_trials = armijo_search(
        objective=objective,
        batch=batch,
        ReducedGNWorkspace=ReducedGNWorkspace,
        pipeline=pipeline,
        device=device,
        base_flat=base_flat,
        base_decoder_state=base_decoder_state,
        base_loss=base_loss,
        gradient=gradient,
        direction=gradient_direction,
        alphas=parse_float_list(args.gradient_alphas),
        c1=float(args.armijo_c1),
    )
    gradient_total_seconds = timings["gradient_seconds"] + float(gradient_best["line_search_seconds"])

    update_kind = str(args.update_kind)
    if update_kind == "hgnvp" and hgnvp_best["accepted"]:
        assign_flat_parameters(objective.dynamics, base_flat + float(hgnvp_best["alpha"]) * direction)
        build_workspace(objective, batch, ReducedGNWorkspace, pipeline, device)
    elif update_kind == "gradient" and gradient_best["accepted"]:
        assign_flat_parameters(objective.dynamics, base_flat + float(gradient_best["alpha"]) * gradient_direction)
        build_workspace(objective, batch, ReducedGNWorkspace, pipeline, device)
    else:
        assign_flat_parameters(objective.dynamics, base_flat)
        restore_trainable_state(objective.decoder, base_decoder_state)

    sketch_eigs = torch.linalg.eigvalsh(0.5 * (gram + gram.T))
    projected_eigs = torch.linalg.eigvalsh(projected_gram)
    record = {
        "event": "cpu_hgnvp_step",
        "step": int(step),
        "base_loss": float(base_loss),
        "base_data_loss": float(data_loss),
        "grad_norm": float(grad_norm),
        "probe_rank": int(args.probe_rank),
        "basis_rank": int(basis.shape[1]),
        "parameter_dim": int(gradient.numel()),
        "sample_count": int(args.sample_count),
        "diag_qform_relerr": safe_float(diag_relerr),
        "exact_gram_diag_qform_relerr": safe_float(exact_diag_relerr),
        "sketch_cholesky_jitter": safe_float(used_jitter),
        "sketch_singular_values": [safe_float(x) for x in sketch_singular_values.detach().cpu()],
        "sketch_gram_min_eig": safe_float(sketch_eigs.min()),
        "sketch_gram_max_eig": safe_float(sketch_eigs.max()),
        "projected_gram_min_eig": safe_float(projected_eigs.min()),
        "projected_gram_max_eig": safe_float(projected_eigs.max()),
        "direction_damping": float(args.direction_damping),
        "gamma": float(gamma),
        "gamma_mode": str(args.gamma_mode),
        **gamma_record,
        **direction_record,
        "hgnvp_best": hgnvp_best,
        "gradient_best": gradient_best,
        "hgnvp_trials": hgnvp_trials,
        "gradient_trials": gradient_trials,
        "hgnvp_direction_seconds": float(hgnvp_direction_seconds),
        "hgnvp_total_seconds": float(hgnvp_total_seconds),
        "gradient_total_seconds": float(gradient_total_seconds),
        "hgnvp_decrease_per_second": float(hgnvp_best["decrease"]) / max(hgnvp_total_seconds, 1.0e-30),
        "gradient_decrease_per_second": float(gradient_best["decrease"]) / max(gradient_total_seconds, 1.0e-30),
        "update_kind": update_kind,
        "breakdown_seconds": timings,
        "step_seconds": time.perf_counter() - step_start,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-process CPU investigation driver for sketched HGNVP redesigned Newton steps."
    )
    parser.add_argument("--repo", default="/global/homes/y/yuuuhang/quad_goattm")
    parser.add_argument("--train-packed", required=True)
    parser.add_argument("--initializer", default="")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=120)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--probe-rank", type=int, default=16)
    parser.add_argument("--basis-rank", type=int, default=8)
    parser.add_argument("--sample-chunk-size", type=int, default=16)
    parser.add_argument("--decoder-feature-chunk-size", type=int, default=1024)
    parser.add_argument("--direction-damping", type=float, default=1.0e-8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--gamma-mode",
        choices=("constant", "tail-rayleigh", "max-tail-rayleigh", "min-sigma2", "max-min-sigma2"),
        default="max-tail-rayleigh",
    )
    parser.add_argument("--cholesky-jitter", type=float, default=1.0e-12)
    parser.add_argument("--hgnvp-alphas", default="1,0.5,0.25,0.125,0.0625,0.03125")
    parser.add_argument("--gradient-alphas", default="1e-8,3e-9,1e-9,3e-10,1e-10,3e-11,1e-11")
    parser.add_argument("--armijo-c1", type=float, default=1.0e-4)
    parser.add_argument("--update-kind", choices=("hgnvp", "gradient", "none"), default="hgnvp")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    device = torch.device("cuda" if bool(args.allow_cuda) and torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))

    repo = Path(args.repo)
    pipeline, ReducedGNWorkspace = load_pipeline(repo)
    pipeline.DATA = Path(args.train_packed)

    objective, batch, model_record = pipeline.load_objective(
        int(args.sample_start),
        int(args.sample_count),
        device,
        str(args.initializer),
        int(args.latent_dim),
        str(args.load_checkpoint),
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "cpu_hgnvp_investigation.jsonl"
    metadata = {
        "event": "metadata",
        "device": str(device),
        "torch_version": torch.__version__,
        "repo": str(repo),
        "train_packed": str(args.train_packed),
        "initializer": str(args.initializer),
        "load_checkpoint": str(args.load_checkpoint),
        "sample_start": int(args.sample_start),
        "sample_count": int(args.sample_count),
        "latent_dim": int(args.latent_dim),
        "probe_rank": int(args.probe_rank),
        "basis_rank": int(args.basis_rank),
        "model": model_record,
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(metadata, sort_keys=True) + "\n")
        log.flush()
        for step in range(1, int(args.steps) + 1):
            record = run_step(args, objective, batch, ReducedGNWorkspace, pipeline, device, step)
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            print(
                json.dumps(
                    {
                        "event": record["event"],
                        "step": record["step"],
                        "base_loss": record["base_loss"],
                        "hgnvp_accepted": record["hgnvp_best"]["accepted"],
                        "hgnvp_alpha": record["hgnvp_best"]["alpha"],
                        "hgnvp_decrease": record["hgnvp_best"]["decrease"],
                        "hgnvp_dps": record["hgnvp_decrease_per_second"],
                        "gradient_accepted": record["gradient_best"]["accepted"],
                        "gradient_alpha": record["gradient_best"]["alpha"],
                        "gradient_decrease": record["gradient_best"]["decrease"],
                        "gradient_dps": record["gradient_decrease_per_second"],
                        "gamma": record["gamma"],
                        "step_seconds": record["step_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {"log": str(log_path), "output_dir": str(outdir)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
