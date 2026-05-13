from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .dynamics import TorchQuadraticLatentDynamics, latent_rollout_euler
from .mpi import TorchMPIContext
from .selective_ssm import SelectiveSSMDecoder


@dataclass(frozen=True)
class LDNetBatch:
    times: torch.Tensor
    u0: torch.Tensor
    qoi: torch.Tensor
    inputs: torch.Tensor | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "LDNetBatch":
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return LDNetBatch(
            times=self.times.to(**kwargs),
            u0=self.u0.to(**kwargs),
            qoi=self.qoi.to(**kwargs),
            inputs=None if self.inputs is None else self.inputs.to(**kwargs),
        )


class LDNetModel(nn.Module):
    def __init__(self, dynamics: TorchQuadraticLatentDynamics, decoder: SelectiveSSMDecoder) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.decoder = decoder

    def latent_states(self, batch: LDNetBatch) -> torch.Tensor:
        return latent_rollout_euler(self.dynamics, batch.u0, batch.times, inputs=batch.inputs)

    def forward(self, batch: LDNetBatch) -> torch.Tensor:
        if batch.times.ndim != 1:
            raise ValueError(f"batch.times must have shape (T,), got {batch.times.shape}")
        steps = batch.times.shape[0]
        current_u = batch.u0
        decoder_state = self.decoder.initial_state(
            batch_size=current_u.shape[0],
            dtype=current_u.dtype,
            device=current_u.device,
        )
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            outputs.append(self.decoder.output_from_state(current_u, decoder_state))
            if step < steps - 1:
                dt = batch.times[step + 1] - batch.times[step]
                p = None if batch.inputs is None else batch.inputs[:, step, :]
                decoder_state = self.decoder.advance_state(current_u, decoder_state, dt)
                current_u = current_u + dt * self.dynamics.rhs(current_u, p=p)
        return torch.stack(outputs, dim=1)


@dataclass(frozen=True)
class AlternatingLDNetConfig:
    outer_cycles: int = 5
    dynamics_steps_per_cycle: int = 20
    decoder_steps_per_cycle: int = 20
    dynamics_learning_rate: float = 1e-3
    decoder_learning_rate: float = 1e-3
    gradient_clip_norm: float | None = 1.0
    output_dir: str | Path | None = None
    echo_progress: bool = True


@dataclass
class LDNetTrainingHistory:
    records: list[dict[str, float | int | str]] = field(default_factory=list)

    def append(self, record: dict[str, float | int | str]) -> None:
        self.records.append(record)

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def _loss_sum_and_count(model: LDNetModel, batch: LDNetBatch) -> tuple[torch.Tensor, int]:
    prediction = model(batch)
    loss_sum = F.mse_loss(prediction, batch.qoi, reduction="sum")
    return loss_sum, int(batch.qoi.numel())


def _run_one_step(
    model: LDNetModel,
    batch: LDNetBatch,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    context: TorchMPIContext,
    gradient_clip_norm: float | None,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss_sum, local_count = _loss_sum_and_count(model, batch)
    global_count = max(context.allreduce_int_sum(local_count), 1)
    loss = loss_sum / float(global_count)
    loss.backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
    context.allreduce_gradients_sum(parameters)
    optimizer.step()
    global_loss_sum = context.allreduce_float_sum(float(loss_sum.detach().cpu()))
    return global_loss_sum / float(global_count), float(global_count)


def evaluate_mse(model: LDNetModel, batch: LDNetBatch, context: TorchMPIContext | None = None) -> float:
    context = TorchMPIContext() if context is None else context
    with torch.no_grad():
        loss_sum, local_count = _loss_sum_and_count(model, batch)
    global_count = max(context.allreduce_int_sum(local_count), 1)
    global_loss_sum = context.allreduce_float_sum(float(loss_sum.detach().cpu()))
    return global_loss_sum / float(global_count)


def alternating_train_ldnet(
    model: LDNetModel,
    train_batch: LDNetBatch,
    config: AlternatingLDNetConfig,
    test_batch: LDNetBatch | None = None,
    context: TorchMPIContext | None = None,
) -> LDNetTrainingHistory:
    context = TorchMPIContext.from_mpi4py() if context is None else context
    dynamics_params = list(model.dynamics.parameters())
    decoder_params = list(model.decoder.parameters())
    dynamics_optimizer = torch.optim.Adam(dynamics_params, lr=config.dynamics_learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder_params, lr=config.decoder_learning_rate)
    history = LDNetTrainingHistory()

    for cycle in range(config.outer_cycles):
        _set_requires_grad(model.dynamics, True)
        _set_requires_grad(model.decoder, False)
        for inner in range(config.dynamics_steps_per_cycle):
            loss, _ = _run_one_step(
                model,
                train_batch,
                dynamics_optimizer,
                dynamics_params,
                context,
                config.gradient_clip_norm,
            )
            history.append({"cycle": cycle, "inner_step": inner, "phase": "dynamics", "train_mse": loss})

        _set_requires_grad(model.dynamics, False)
        _set_requires_grad(model.decoder, True)
        for inner in range(config.decoder_steps_per_cycle):
            loss, _ = _run_one_step(
                model,
                train_batch,
                decoder_optimizer,
                decoder_params,
                context,
                config.gradient_clip_norm,
            )
            history.append({"cycle": cycle, "inner_step": inner, "phase": "decoder", "train_mse": loss})

        _set_requires_grad(model.dynamics, True)
        if config.echo_progress:
            train_mse = evaluate_mse(model, train_batch, context)
            test_mse = None if test_batch is None else evaluate_mse(model, test_batch, context)
        if config.echo_progress and context.is_root:
            norms = model.dynamics.component_norms()
            print(
                f"cycle={cycle:03d} train_mse={train_mse:.6e} "
                f"test_mse={test_mse if test_mse is not None else float('nan'):.6e} "
                f"A={norms['a_fro_norm']:.3e} H={norms['h_fro_norm']:.3e}",
                flush=True,
            )
            history.append(
                {
                    "cycle": cycle,
                    "inner_step": -1,
                    "phase": "eval",
                    "train_mse": train_mse,
                    "test_mse": float("nan") if test_mse is None else test_mse,
                    **norms,
                }
            )

    if config.output_dir is not None and context.is_root:
        history.save_jsonl(Path(config.output_dir) / "ldnet_training_history.jsonl")
    context.barrier()
    return history
