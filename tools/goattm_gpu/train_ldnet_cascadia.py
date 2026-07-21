from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

GOATTM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOATTM_ROOT))

from ldnet_cascadia import (  # noqa: E402
    LDNetLossConfig,
    apply_pod_initializer,
    armijo_step,
    batch_from_payload,
    dense_a_symmetric_energy_penalty,
    finite_difference_check,
    ldnet_loss,
    load_packed_payload,
    make_denseA_lowrankH_neural_decoder_model,
    relative_error,
)
from ldnet_cascadia.losses import trajectory_data_loss  # noqa: E402
from quadrode_gpu_goattm import runge_kutta4_rollout_adjoint, trapezoidal_weights  # noqa: E402


class NullLog:
    def write(self, _: str) -> None:
        return None

    def flush(self) -> None:
        return None


def parse_hidden(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part)


def select_indices(total: int, count: int, rng: random.Random) -> list[int]:
    if count <= 0 or count >= total:
        return list(range(total))
    return rng.sample(range(total), count)


def tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def setup_distributed(requested_device: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if requested_device.startswith("cuda") and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if world_size > 1 and requested_device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
    return rank, world_size, local_rank, device


def distributed_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        out = value.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out = out / dist.get_world_size()
        return out
    return value.detach()


def all_reduce_parameter_grads(model: torch.nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def snapshot_params(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def restore_params(params: list[torch.nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(params, values):
            param.copy_(value)


def global_steepest_direction(
    params: list[torch.nn.Parameter],
    *,
    trust: float | None,
    min_norm: float,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    grad_sq = None
    param_sq = None
    for param in params:
        if param.grad is None:
            continue
        gterm = param.grad.detach().square().sum()
        pterm = param.detach().square().sum()
        grad_sq = gterm if grad_sq is None else grad_sq + gterm
        param_sq = pterm if param_sq is None else param_sq + pterm
    if grad_sq is None:
        zero = params[0].new_zeros(()) if params else torch.zeros(())
        return [torch.zeros_like(param) for param in params], zero, zero
    grad_norm_value = torch.sqrt(grad_sq)
    scale = 1.0
    if trust is not None and float(trust) > 0.0:
        param_norm = 0.0 if param_sq is None else float(torch.sqrt(param_sq).detach().cpu())
        radius = float(trust) * max(param_norm, float(min_norm))
        scale = min(1.0, radius / max(float(grad_norm_value.detach().cpu()), torch.finfo(grad_norm_value.dtype).tiny))
    directions = [
        torch.zeros_like(param) if param.grad is None else -scale * param.grad.detach().clone()
        for param in params
    ]
    directional = None
    for param, direction in zip(params, directions):
        if param.grad is None:
            continue
        term = (param.grad.detach() * direction).sum()
        directional = term if directional is None else directional + term
    if directional is None:
        directional = grad_norm_value.new_zeros(())
    return directions, grad_norm_value, directional


def ddp_armijo_step(
    model: torch.nn.Module,
    closure,
    *,
    initial_alpha: float,
    max_alpha: float,
    shrink: float,
    grow: float,
    c1: float,
    max_trials: int,
    trust: float | None,
    min_norm: float,
) -> dict[str, float | bool]:
    model.zero_grad(set_to_none=True)
    local_base = closure()
    local_base.backward()
    params = [param for param in model.parameters() if param.requires_grad]
    directions, grad_norm_value, directional = global_steepest_direction(params, trust=trust, min_norm=min_norm)
    base_global = distributed_mean(local_base)
    if not params or float(grad_norm_value.detach().cpu()) == 0.0:
        value = float(base_global.detach().cpu())
        return {
            "accepted": False,
            "alpha": 0.0,
            "next_initial_alpha": float(initial_alpha),
            "initial_loss": value,
            "trial_loss": value,
            "grad_norm": 0.0,
            "directional_derivative": 0.0,
        }
    base_values = snapshot_params(params)
    alpha = min(float(initial_alpha), float(max_alpha))
    accepted = False
    trial_value = float("nan")
    for _ in range(int(max_trials)):
        restore_params(params, base_values)
        with torch.no_grad():
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=alpha)
            local_trial = closure()
            trial_global = distributed_mean(local_trial)
        trial_value = float(trial_global.detach().cpu())
        sufficient = base_global + float(c1) * alpha * directional
        if torch.isfinite(trial_global) and bool((trial_global <= sufficient).detach().cpu()):
            accepted = True
            break
        alpha *= float(shrink)
    if not accepted:
        restore_params(params, base_values)
    next_alpha = min(float(max_alpha), alpha * float(grow)) if accepted else max(alpha, float(initial_alpha) * float(shrink))
    model.zero_grad(set_to_none=True)
    return {
        "accepted": accepted,
        "alpha": float(alpha if accepted else 0.0),
        "next_initial_alpha": float(next_alpha),
        "initial_loss": float(base_global.detach().cpu()),
        "trial_loss": float(trial_value),
        "grad_norm": float(grad_norm_value.detach().cpu()),
        "directional_derivative": float(directional.detach().cpu()),
    }


def evaluate_microbatches(microbatches: list[object], loss_for_batch) -> dict[str, torch.Tensor]:
    if not microbatches:
        raise ValueError("microbatches must not be empty")
    total: dict[str, torch.Tensor] = {}
    count = float(len(microbatches))
    with torch.no_grad():
        for batch in microbatches:
            _, terms = loss_for_batch(batch)
            for key, value in terms.items():
                if key not in {"loss", "data_loss", "relative_error", "a_symmetric_lambda_max", "a_symmetric_penalty"}:
                    continue
                piece = value.detach() / count
                total[key] = piece if key not in total else total[key] + piece
    return total


def rk4_adjoint_value_terms(model, batch, loss_config, *, decode_chunk_size: int) -> dict[str, torch.Tensor]:
    z0 = model.encode(batch)
    p_mid = batch.midpoint_inputs() if model.use_time_dependent_source else None
    with torch.no_grad():
        states = model.stepper.rollout(
            model.dynamics,
            z0.detach(),
            batch.step_size,
            batch.steps,
            p_mid=p_mid,
        )
        prediction = model.decode_states(states, chunk_size=decode_chunk_size)
        data = trajectory_data_loss(
            prediction,
            batch.qoi,
            batch.observation_times,
            time_weighted=loss_config.time_weighted,
            normalize_by_batch=loss_config.normalize_by_batch,
        )
        rel = relative_error(prediction, batch.qoi, batch.observation_times)
        if loss_config.dense_a_symmetric_penalty > 0.0:
            energy = dense_a_symmetric_energy_penalty(
                model.dynamics.linear,
                weight=loss_config.dense_a_symmetric_penalty,
                temperature=loss_config.dense_a_symmetric_temperature,
            )
        else:
            energy = {
                "loss": data.new_zeros(()),
                "lambda_max": data.new_zeros(()),
                "smooth_positive": data.new_zeros(()),
            }
        total = data + energy["loss"]
    return {
        "loss": total.detach(),
        "data_loss": data.detach(),
        "relative_error": rel.detach(),
        "a_symmetric_lambda_max": energy["lambda_max"].detach(),
        "a_symmetric_penalty": energy["loss"].detach(),
    }


def rk4_adjoint_batch_backward(
    model,
    batch,
    loss_config,
    *,
    decode_chunk_size: int,
    scale: float,
) -> dict[str, torch.Tensor]:
    z0 = model.encode(batch)
    p_mid = batch.midpoint_inputs() if model.use_time_dependent_source else None
    needs_dynamics_grad = any(param.requires_grad for param in model.dynamics.parameters())
    with torch.no_grad():
        states = model.stepper.rollout(
            model.dynamics,
            z0.detach(),
            batch.step_size,
            batch.steps,
            p_mid=p_mid,
        )

    states_leaf = states.detach().requires_grad_(True)
    prediction = model.decode_states(states_leaf, chunk_size=decode_chunk_size)
    data = trajectory_data_loss(
        prediction,
        batch.qoi,
        batch.observation_times,
        time_weighted=loss_config.time_weighted,
        normalize_by_batch=loss_config.normalize_by_batch,
    )
    rel = relative_error(prediction.detach(), batch.qoi, batch.observation_times)
    if needs_dynamics_grad and loss_config.dense_a_symmetric_penalty > 0.0:
        energy = dense_a_symmetric_energy_penalty(
            model.dynamics.linear,
            weight=loss_config.dense_a_symmetric_penalty,
            temperature=loss_config.dense_a_symmetric_temperature,
        )
    else:
        energy = {
            "loss": data.new_zeros(()),
            "lambda_max": data.new_zeros(()),
            "smooth_positive": data.new_zeros(()),
        }
    loss = data + energy["loss"]
    (float(scale) * loss).backward()
    if needs_dynamics_grad or z0.requires_grad:
        state_cotangents = states_leaf.grad.detach()
        adjoint = runge_kutta4_rollout_adjoint(
            model.dynamics,
            z0.detach(),
            batch.step_size,
            float(scale) * state_cotangents,
            p_mid=p_mid,
            states=states.detach(),
            return_input_adjoint=False,
        )
        if needs_dynamics_grad:
            named = dict(model.dynamics.named_parameters())
            for name, grad in adjoint.parameter_grads.items():
                param = named.get(name)
                if param is None or not param.requires_grad:
                    continue
                if param.grad is None:
                    param.grad = grad.detach().clone()
                else:
                    param.grad.add_(grad.detach())
        if z0.requires_grad:
            z0.backward(adjoint.lambda_u0)
    return {
        "loss": loss.detach(),
        "data_loss": data.detach(),
        "relative_error": rel.detach(),
        "a_symmetric_lambda_max": energy["lambda_max"].detach(),
        "a_symmetric_penalty": energy["loss"].detach(),
    }


def evaluate_microbatches_adjoint(model, microbatches: list[object], loss_config, *, decode_chunk_size: int) -> dict[str, torch.Tensor]:
    total: dict[str, torch.Tensor] = {}
    count = float(len(microbatches))
    for batch in microbatches:
        terms = rk4_adjoint_value_terms(model, batch, loss_config, decode_chunk_size=decode_chunk_size)
        for key, value in terms.items():
            piece = value.detach() / count
            total[key] = piece if key not in total else total[key] + piece
    return total


def microbatch_loss_backward_adjoint(model, microbatches: list[object], loss_config, *, decode_chunk_size: int) -> torch.Tensor:
    scale = 1.0 / float(len(microbatches))
    total: torch.Tensor | None = None
    for batch in microbatches:
        terms = rk4_adjoint_batch_backward(
            model,
            batch,
            loss_config,
            decode_chunk_size=decode_chunk_size,
            scale=scale,
        )
        piece = terms["loss"].detach() * scale
        total = piece if total is None else total + piece
    if total is None:
        raise ValueError("microbatches must not be empty")
    all_reduce_parameter_grads(model)
    return total


def microbatch_loss_backward(microbatches: list[object], loss_for_batch) -> torch.Tensor:
    total: torch.Tensor | None = None
    scale = 1.0 / float(len(microbatches))
    for batch in microbatches:
        loss, _ = loss_for_batch(batch)
        scaled = loss * scale
        scaled.backward()
        piece = scaled.detach()
        total = piece if total is None else total + piece
    if total is None:
        raise ValueError("microbatches must not be empty")
    return total


def microbatch_loss_value(microbatches: list[object], loss_for_batch) -> torch.Tensor:
    total: torch.Tensor | None = None
    scale = 1.0 / float(len(microbatches))
    with torch.no_grad():
        for batch in microbatches:
            loss, _ = loss_for_batch(batch)
            piece = loss.detach() * scale
            total = piece if total is None else total + piece
    if total is None:
        raise ValueError("microbatches must not be empty")
    return total


def armijo_step_microbatched(
    model: torch.nn.Module,
    microbatches: list[object],
    loss_for_batch=None,
    *,
    backward_fn=None,
    value_fn=None,
    initial_alpha: float,
    max_alpha: float,
    shrink: float,
    grow: float,
    c1: float,
    max_trials: int,
    trust: float | None,
    min_norm: float,
) -> dict[str, float | bool]:
    model.zero_grad(set_to_none=True)
    if backward_fn is None:
        local_base = microbatch_loss_backward(microbatches, loss_for_batch)
        all_reduce_parameter_grads(model)
    else:
        local_base = backward_fn(microbatches)
    params = [param for param in model.parameters() if param.requires_grad]
    directions, grad_norm_value, directional = global_steepest_direction(params, trust=trust, min_norm=min_norm)
    base_global = distributed_mean(local_base)
    if not params or float(grad_norm_value.detach().cpu()) == 0.0:
        value = float(base_global.detach().cpu())
        return {
            "accepted": False,
            "alpha": 0.0,
            "next_initial_alpha": float(initial_alpha),
            "initial_loss": value,
            "trial_loss": value,
            "grad_norm": 0.0,
            "directional_derivative": 0.0,
        }
    base_values = snapshot_params(params)
    alpha = min(float(initial_alpha), float(max_alpha))
    accepted = False
    trial_value = float("nan")
    for _ in range(int(max_trials)):
        restore_params(params, base_values)
        with torch.no_grad():
            for param, direction in zip(params, directions):
                param.add_(direction, alpha=alpha)
        if value_fn is None:
            local_trial = microbatch_loss_value(microbatches, loss_for_batch)
        else:
            local_trial = value_fn(microbatches)["loss"]
        trial_global = distributed_mean(local_trial)
        trial_value = float(trial_global.detach().cpu())
        sufficient = base_global + float(c1) * alpha * directional
        if torch.isfinite(trial_global) and bool((trial_global <= sufficient).detach().cpu()):
            accepted = True
            break
        alpha *= float(shrink)
    if not accepted:
        restore_params(params, base_values)
    next_alpha = min(float(max_alpha), alpha * float(grow)) if accepted else max(alpha, float(initial_alpha) * float(shrink))
    model.zero_grad(set_to_none=True)
    return {
        "accepted": accepted,
        "alpha": float(alpha if accepted else 0.0),
        "next_initial_alpha": float(next_alpha),
        "initial_loss": float(base_global.detach().cpu()),
        "trial_loss": float(trial_value),
        "grad_norm": float(grad_norm_value.detach().cpu()),
        "directional_derivative": float(directional.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural-decoder LDNet on Cascadia packed data.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--initializer", type=Path, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=120)
    parser.add_argument("--h-rank", type=int, default=50)
    parser.add_argument("--h-tt-rank", type=int, default=50)
    parser.add_argument("--encoder-hidden", type=parse_hidden, default=(256, 256))
    parser.add_argument("--decoder-hidden", type=parse_hidden, default=(256, 256))
    parser.add_argument("--freeze-dynamics", action="store_true")
    parser.add_argument("--freeze-linear-readout", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--input-context-mode", choices=["initial", "final", "mean", "maxabs", "flatten"], default="final")
    parser.add_argument("--time-mode", choices=["normalized", "step_index"], default="step_index")
    parser.add_argument("--max-time-steps", type=int, default=0)
    parser.add_argument("--train-sample-limit", type=int, default=0)
    parser.add_argument("--test-sample-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--full-batch-distributed-shards",
        action="store_true",
        help="Shard full-training-set closures across distributed ranks instead of duplicating all samples on every rank.",
    )
    parser.add_argument("--gradient-mode", choices=["autograd", "rk4_adjoint"], default="autograd")
    parser.add_argument("--optimizer", choices=["adam", "armijo", "adam_armijo"], default="adam")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--adam-steps", type=int, default=-1)
    parser.add_argument("--armijo-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--dynamics-lr", type=float, default=-1.0)
    parser.add_argument("--linear-readout-lr", type=float, default=-1.0)
    parser.add_argument("--decoder-correction-lr", type=float, default=-1.0)
    parser.add_argument("--encoder-lr", type=float, default=-1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--armijo-initial-alpha", type=float, default=1.0)
    parser.add_argument("--armijo-max-alpha", type=float, default=1.0)
    parser.add_argument("--armijo-shrink", type=float, default=0.5)
    parser.add_argument("--armijo-grow", type=float, default=2.0)
    parser.add_argument("--armijo-c1", type=float, default=1.0e-4)
    parser.add_argument("--armijo-max-trials", type=int, default=20)
    parser.add_argument("--armijo-trust", type=float, default=1.0e-4)
    parser.add_argument("--armijo-min-norm", type=float, default=1.0)
    parser.add_argument("--a-energy-penalty", type=float, default=1.0e-2)
    parser.add_argument("--a-energy-temperature", type=float, default=1.0e-2)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-sample-limit", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--decode-chunk-size", type=int, default=0)
    parser.add_argument("--fd-check", action="store_true")
    parser.add_argument("--fd-sample-limit", type=int, default=2)
    parser.add_argument("--fd-max-time-steps", type=int, default=8)
    parser.add_argument("--fd-epsilons", default="1e-3,3e-4,1e-4,3e-5,1e-5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    rank, world_size, local_rank, device = setup_distributed(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1009 * rank)
    if is_rank0(rank):
        args.output.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    log_path = args.output / "optimization.jsonl"

    train_payload, train_metadata = load_packed_payload(args.train)
    test_payload = None
    if args.test is not None:
        test_payload, _ = load_packed_payload(args.test)
    max_time_steps = args.max_time_steps if args.max_time_steps > 0 else None
    total_train = int(train_payload["qoi"].shape[1])
    train_total_limited = min(total_train, args.train_sample_limit) if args.train_sample_limit > 0 else total_train
    probe = batch_from_payload(
        train_payload,
        device=device,
        dtype=dtype,
        sample_limit=1,
        time_mode=args.time_mode,
        max_time_steps=max_time_steps,
    )
    source_input_dim = probe.input_dim
    encoder_input_dim = probe.input_context(args.input_context_mode).shape[-1]
    output_dim = probe.output_dim
    model = make_denseA_lowrankH_neural_decoder_model(
        input_dim=source_input_dim,
        encoder_input_dim=encoder_input_dim,
        output_dim=output_dim,
        latent_dim=args.latent_dim,
        h_rank=args.h_rank,
        h_tt_rank=args.h_tt_rank,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        input_context_mode=args.input_context_mode,
        dtype=dtype,
    ).to(device)
    init_report = None
    if args.initializer is not None:
        init_report = apply_pod_initializer(model, args.initializer)
    loaded_checkpoint = None
    if args.load_checkpoint is not None:
        loaded_checkpoint = torch.load(args.load_checkpoint, map_location=device)
        state = loaded_checkpoint.get("model", loaded_checkpoint)
        model.load_state_dict(state)
    if args.freeze_dynamics:
        for param in model.dynamics.parameters():
            param.requires_grad_(False)
    if args.freeze_linear_readout:
        for param in model.decoder.linear_readout.parameters():
            param.requires_grad_(False)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
    forward_model = (
        DDP(model, device_ids=[local_rank], output_device=local_rank)
        if args.gradient_mode == "autograd" and world_size > 1 and device.type == "cuda"
        else model
    )
    def parameter_groups() -> list[dict]:
        groups = [
            ("encoder", model.encoder.parameters(), args.encoder_lr),
            ("dynamics", model.dynamics.parameters(), args.dynamics_lr),
            ("linear_readout", model.decoder.linear_readout.parameters(), args.linear_readout_lr),
            ("decoder_correction", model.decoder.correction.parameters(), args.decoder_correction_lr),
        ]
        result = []
        for name, params_iter, group_lr in groups:
            params = [param for param in params_iter if param.requires_grad]
            if not params:
                continue
            result.append(
                {
                    "params": params,
                    "lr": args.lr if group_lr < 0.0 else group_lr,
                    "name": name,
                }
            )
        return result

    optimizer = torch.optim.AdamW(parameter_groups(), lr=args.lr, weight_decay=args.weight_decay)
    if loaded_checkpoint is not None and isinstance(loaded_checkpoint, dict) and "optimizer" in loaded_checkpoint:
        try:
            optimizer.load_state_dict(loaded_checkpoint["optimizer"])
        except ValueError:
            if is_rank0(rank):
                print("warning: optimizer state in checkpoint did not match current model; continuing without it", flush=True)
    loss_config = LDNetLossConfig(
        dense_a_symmetric_penalty=args.a_energy_penalty,
        dense_a_symmetric_temperature=args.a_energy_temperature,
        normalize_by_batch=True,
    )

    metadata = {
        "train": str(args.train),
        "test": None if args.test is None else str(args.test),
        "train_metadata": train_metadata,
        "source_input_dim": source_input_dim,
        "encoder_input_dim": encoder_input_dim,
        "output_dim": output_dim,
        "initializer": init_report,
        "load_checkpoint": None if args.load_checkpoint is None else str(args.load_checkpoint),
        "trainable_parameter_count": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "total_parameter_count": int(sum(param.numel() for param in model.parameters())),
        "frozen": {
            "dynamics": bool(args.freeze_dynamics),
            "linear_readout": bool(args.freeze_linear_readout),
            "encoder": bool(args.freeze_encoder),
        },
        "args": vars(args) | {"train": str(args.train), "test": None if args.test is None else str(args.test), "output": str(args.output)},
    }
    if args.initializer is not None:
        metadata["args"]["initializer"] = str(args.initializer)
    if args.load_checkpoint is not None:
        metadata["args"]["load_checkpoint"] = str(args.load_checkpoint)
    metadata.update({"distributed": {"rank": rank, "world_size": world_size, "local_rank": local_rank}})
    if is_rank0(rank):
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def make_batch(indices: list[int]) -> object:
        return batch_from_payload(
            train_payload,
            device=device,
            dtype=dtype,
            sample_indices=indices,
            time_mode=args.time_mode,
            max_time_steps=max_time_steps,
        )

    def make_microbatches(indices: list[int]) -> list[object]:
        chunks = []
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start : start + args.batch_size]
            if chunk:
                chunks.append(make_batch(chunk))
        return chunks

    def select_training_indices() -> list[int]:
        requested = args.batch_size * max(1, args.accumulation_steps)
        if args.full_batch_distributed_shards and requested >= train_total_limited:
            return list(range(rank, train_total_limited, max(1, world_size)))
        return select_indices(train_total_limited, requested, rng)

    def loss_for_batch(batch: object) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        result = forward_model(batch, decode_chunk_size=args.decode_chunk_size)
        terms = ldnet_loss(model, batch, result, loss_config)
        return terms["loss"], terms

    def value_terms_for_microbatches(microbatches: list[object]) -> dict[str, torch.Tensor]:
        if args.gradient_mode == "rk4_adjoint":
            return evaluate_microbatches_adjoint(
                model,
                microbatches,
                loss_config,
                decode_chunk_size=args.decode_chunk_size,
            )
        return evaluate_microbatches(microbatches, loss_for_batch)

    def backward_microbatches(microbatches: list[object]) -> torch.Tensor:
        if args.gradient_mode == "rk4_adjoint":
            return microbatch_loss_backward_adjoint(
                model,
                microbatches,
                loss_config,
                decode_chunk_size=args.decode_chunk_size,
            )
        local = microbatch_loss_backward(microbatches, loss_for_batch)
        all_reduce_parameter_grads(model)
        return local

    def validate_chunk(batch: object) -> dict[str, torch.Tensor]:
        if args.gradient_mode == "rk4_adjoint":
            z0 = model.encode(batch)
            p_mid = batch.midpoint_inputs() if model.use_time_dependent_source else None
            states = model.stepper.rollout(
                model.dynamics,
                z0.detach(),
                batch.step_size,
                batch.steps,
                p_mid=p_mid,
            )
            prediction = model.decode_states(states, chunk_size=args.decode_chunk_size)
        else:
            result = model(batch, decode_chunk_size=args.decode_chunk_size)
            prediction = result.prediction
        residual_sq = (prediction - batch.qoi).square().sum(dim=-1)
        target_sq = batch.qoi.square().sum(dim=-1)
        if loss_config.time_weighted:
            weights = trapezoidal_weights(batch.observation_times).to(device=residual_sq.device, dtype=residual_sq.dtype)
            residual_sq = residual_sq * weights[:, None]
            target_sq = target_sq * weights[:, None]
        return {
            "residual_sum": residual_sq.sum().detach(),
            "target_sum": target_sq.sum().detach(),
            "sample_count": residual_sq.new_tensor(float(batch.batch_size)),
        }

    def validate(payload: dict, limit: int) -> dict[str, float]:
        model.eval()
        total_samples = int(payload["qoi"].shape[1])
        sample_count = min(total_samples, int(limit)) if limit > 0 else total_samples
        chunk_size = max(1, int(args.validation_batch_size))
        residual_total: torch.Tensor | None = None
        target_total: torch.Tensor | None = None
        count_total: torch.Tensor | None = None
        with torch.no_grad():
            for start in range(0, sample_count, chunk_size):
                end = min(sample_count, start + chunk_size)
                batch = batch_from_payload(
                    payload,
                    device=device,
                    dtype=dtype,
                    sample_indices=list(range(start, end)),
                    time_mode=args.time_mode,
                    max_time_steps=max_time_steps,
                )
                terms = validate_chunk(batch)
                residual_total = terms["residual_sum"] if residual_total is None else residual_total + terms["residual_sum"]
                target_total = terms["target_sum"] if target_total is None else target_total + terms["target_sum"]
                count_total = terms["sample_count"] if count_total is None else count_total + terms["sample_count"]
        model.train()
        if residual_total is None or target_total is None or count_total is None:
            raise ValueError("validation sample count must be positive")
        tiny = torch.finfo(residual_total.dtype).tiny
        relative = torch.sqrt(residual_total / target_total.clamp_min(tiny))
        data_loss = 0.5 * residual_total
        if loss_config.normalize_by_batch:
            data_loss = data_loss / count_total.clamp_min(1.0)
        return {
            "relative_error": tensor_float(relative),
            "data_loss": tensor_float(data_loss),
        }

    def write_validation(log, step: int, start_time: float) -> None:
        train_val_limit = args.validation_sample_limit if args.validation_sample_limit > 0 else args.train_sample_limit
        val = validate(train_payload, train_val_limit)
        record = {
            "event": "validation",
            "optimizer_step": step,
            "train_relative_error": val["relative_error"],
            "train_data_loss": val["data_loss"],
            "validation_sample_limit": train_val_limit,
            "elapsed_sec": time.time() - start_time,
        }
        if test_payload is not None:
            test_limit = args.test_sample_limit if args.test_sample_limit > 0 else args.validation_sample_limit
            test_val = validate(test_payload, test_limit)
            record.update(
                test_relative_error=test_val["relative_error"],
                test_data_loss=test_val["data_loss"],
                test_sample_limit=test_limit,
            )
        log.write(json.dumps(record, sort_keys=True) + "\n")
        log.flush()

    if args.optimizer == "adam":
        adam_steps = args.steps if args.adam_steps < 0 else args.adam_steps
        armijo_steps = 0 if args.armijo_steps < 0 else args.armijo_steps
    elif args.optimizer == "armijo":
        adam_steps = 0 if args.adam_steps < 0 else args.adam_steps
        armijo_steps = args.steps if args.armijo_steps < 0 else args.armijo_steps
    else:
        adam_steps = 300 if args.adam_steps < 0 else args.adam_steps
        armijo_steps = max(0, args.steps - adam_steps) if args.armijo_steps < 0 else args.armijo_steps

    start_time = time.time()
    log_context = log_path.open("a", encoding="utf-8") if is_rank0(rank) else nullcontext(NullLog())
    with log_context as log:
        if args.fd_check:
            if is_rank0(rank):
                fd_steps = args.fd_max_time_steps if args.fd_max_time_steps > 0 else args.max_time_steps
                fd_batch = batch_from_payload(
                    train_payload,
                    device=device,
                    dtype=dtype,
                    sample_limit=max(1, args.fd_sample_limit),
                    time_mode=args.time_mode,
                    max_time_steps=fd_steps,
                )

                def fd_closure() -> torch.Tensor:
                    result = model(fd_batch, decode_chunk_size=args.decode_chunk_size)
                    terms = ldnet_loss(model, fd_batch, result, loss_config)
                    return terms["loss"]

                epsilons = tuple(float(part) for part in args.fd_epsilons.split(",") if part)
                for row in finite_difference_check(model, fd_closure, epsilons=epsilons, seed=args.seed):
                    log.write(json.dumps({"event": "finite_difference", **row.__dict__}, sort_keys=True) + "\n")
                log.flush()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        global_step = int(args.start_step)
        for _ in range(1, adam_steps + 1):
            global_step += 1
            model.train()
            indices = select_training_indices()
            microbatches = make_microbatches(indices)
            optimizer.zero_grad(set_to_none=True)
            local_loss = backward_microbatches(microbatches)
            terms = value_terms_for_microbatches(microbatches)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e6)
            optimizer.step()
            global_loss = distributed_mean(local_loss)
            global_data_loss = distributed_mean(terms["data_loss"])
            if is_rank0(rank):
                record = {
                    "event": "adam_step",
                    "optimizer_step": global_step,
                    "loss": tensor_float(global_loss),
                    "train_batch_relative_error": tensor_float(terms["relative_error"]),
                    "train_batch_data_loss": tensor_float(global_data_loss),
                    "a_symmetric_lambda_max": tensor_float(terms["a_symmetric_lambda_max"]),
                    "a_symmetric_penalty": tensor_float(terms["a_symmetric_penalty"]),
                    "grad_norm": float(grad_norm),
                    "elapsed_sec": time.time() - start_time,
                    "world_size": world_size,
                    "per_rank_batch_size": args.batch_size,
                    "accumulation_steps": args.accumulation_steps,
                    "effective_global_batch_size": args.batch_size * max(1, args.accumulation_steps) * world_size,
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
            if args.validation_interval > 0 and global_step % args.validation_interval == 0:
                if is_rank0(rank):
                    write_validation(log, global_step, start_time)
                    log.flush()
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

        alpha = float(args.armijo_initial_alpha)
        for _ in range(1, armijo_steps + 1):
            global_step += 1
            model.train()
            indices = select_training_indices()
            microbatches = make_microbatches(indices)
            result_dict = armijo_step_microbatched(
                model,
                microbatches,
                loss_for_batch if args.gradient_mode == "autograd" else None,
                backward_fn=backward_microbatches if args.gradient_mode == "rk4_adjoint" else None,
                value_fn=value_terms_for_microbatches if args.gradient_mode == "rk4_adjoint" else None,
                initial_alpha=alpha,
                max_alpha=args.armijo_max_alpha,
                shrink=args.armijo_shrink,
                grow=args.armijo_grow,
                c1=args.armijo_c1,
                max_trials=args.armijo_max_trials,
                trust=None if args.armijo_trust <= 0.0 else args.armijo_trust,
                min_norm=args.armijo_min_norm,
            )
            alpha = float(result_dict["next_initial_alpha"])
            terms = value_terms_for_microbatches(microbatches)
            global_loss = distributed_mean(terms["loss"])
            global_data_loss = distributed_mean(terms["data_loss"])
            if is_rank0(rank):
                record = {
                    "event": "armijo_step",
                    "optimizer_step": global_step,
                    "accepted": bool(result_dict["accepted"]),
                    "accepted_alpha": float(result_dict["alpha"]),
                    "next_initial_alpha": float(result_dict["next_initial_alpha"]),
                    "initial_loss": float(result_dict["initial_loss"]),
                    "trial_loss": float(result_dict["trial_loss"]),
                    "loss": tensor_float(global_loss),
                    "train_batch_relative_error": tensor_float(terms["relative_error"]),
                    "train_batch_data_loss": tensor_float(global_data_loss),
                    "a_symmetric_lambda_max": tensor_float(terms["a_symmetric_lambda_max"]),
                    "a_symmetric_penalty": tensor_float(terms["a_symmetric_penalty"]),
                    "grad_norm": float(result_dict["grad_norm"]),
                    "directional_derivative": float(result_dict["directional_derivative"]),
                    "elapsed_sec": time.time() - start_time,
                    "world_size": world_size,
                    "per_rank_batch_size": args.batch_size,
                    "accumulation_steps": args.accumulation_steps,
                    "effective_global_batch_size": args.batch_size * max(1, args.accumulation_steps) * world_size,
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
            if args.validation_interval > 0 and global_step % args.validation_interval == 0:
                if is_rank0(rank):
                    write_validation(log, global_step, start_time)
                    log.flush()
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
    if is_rank0(rank):
        torch.save(
            {
                "model": model.state_dict(),
                "metadata": metadata,
                "optimizer": optimizer.state_dict(),
                "loss_config": loss_config,
                "step": global_step,
            },
            args.output / "checkpoint.pt",
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
