from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from goattm.data.npz_dataset import NpzQoiSample, NpzSampleManifest, load_npz_qoi_sample, load_npz_sample_manifest
from goattm.losses.qoi_loss import ObservationAlignedRolloutLossGradientResult, rollout_qoi_loss_and_gradients_from_observations
from goattm.models.linear_dynamics import LinearDynamics
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.runtime.distributed import DistributedContext, sum_array_mapping
from goattm.solvers import TimeIntegrator


DynamicsLike = LinearDynamics | QuadraticDynamics | StabilizedQuadraticDynamics


@dataclass(frozen=True)
class DatasetQoiLossGradientResult:
    total_loss: float
    local_loss: float
    local_sample_count: int
    global_sample_count: int
    decoder_gradients: dict[str, np.ndarray]
    dynamics_gradients: dict[str, np.ndarray]
    local_sample_ids: tuple[str, ...]
    local_sample_results: tuple[ObservationAlignedRolloutLossGradientResult, ...]


def evaluate_npz_qoi_dataset_loss_and_gradients(
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    manifest: str | Path | NpzSampleManifest,
    max_dt: float,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    context: DistributedContext | None = None,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> DatasetQoiLossGradientResult:
    if context is None:
        context = DistributedContext.from_comm()
    if isinstance(manifest, (str, Path)):
        manifest = load_npz_sample_manifest(manifest)

    local_entries = manifest.entries_for_rank(context.rank, context.size)
    decoder_gradients = _zero_decoder_gradients(decoder)
    dynamics_gradients = _zero_dynamics_gradients(dynamics)
    local_results: list[ObservationAlignedRolloutLossGradientResult] = []
    local_sample_ids: list[str] = []
    local_loss = 0.0

    for sample_id, sample_path in local_entries:
        sample = load_npz_qoi_sample(sample_path)
        result = _evaluate_single_sample(
            dynamics=dynamics,
            decoder=decoder,
            sample=sample,
            max_dt=max_dt,
            time_integrator=time_integrator,
            dt_shrink=dt_shrink,
            dt_min=dt_min,
            tol=tol,
            max_iter=max_iter,
        )
        local_results.append(result)
        local_sample_ids.append(sample_id)
        local_loss += result.loss
        decoder_gradients["v1"] += result.decoder_partials.v1_grad
        decoder_gradients["v2"] += result.decoder_partials.v2_grad
        decoder_gradients["v0"] += result.decoder_partials.v0_grad
        for key, value in result.dynamics_gradients.items():
            dynamics_gradients[key] += value

    return DatasetQoiLossGradientResult(
        total_loss=context.allreduce_scalar_sum(local_loss),
        local_loss=local_loss,
        local_sample_count=len(local_entries),
        global_sample_count=context.allreduce_int_sum(len(local_entries)),
        decoder_gradients=sum_array_mapping(decoder_gradients, context),
        dynamics_gradients=sum_array_mapping(dynamics_gradients, context),
        local_sample_ids=tuple(local_sample_ids),
        local_sample_results=tuple(local_results),
    )


def _zero_decoder_gradients(decoder: QuadraticDecoder) -> dict[str, np.ndarray]:
    return {
        "v1": np.zeros_like(decoder.v1, dtype=np.float64),
        "v2": np.zeros_like(decoder.v2, dtype=np.float64),
        "v0": np.zeros_like(decoder.v0, dtype=np.float64),
    }


def _zero_dynamics_gradients(dynamics: DynamicsLike) -> dict[str, np.ndarray]:
    gradients: dict[str, np.ndarray] = {"c": np.zeros_like(dynamics.c, dtype=np.float64)}
    if not isinstance(dynamics, LinearDynamics):
        gradients["mu_h"] = np.zeros_like(dynamics.mu_h, dtype=np.float64)
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        gradients["s_params"] = np.zeros_like(dynamics.s_params, dtype=np.float64)
        gradients["w_params"] = np.zeros_like(dynamics.w_params, dtype=np.float64)
    else:
        gradients["a"] = np.zeros_like(dynamics.a, dtype=np.float64)
    if getattr(dynamics, "b", None) is not None:
        gradients["b"] = np.zeros_like(dynamics.b, dtype=np.float64)
    return gradients


def _evaluate_single_sample(
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    sample: NpzQoiSample,
    max_dt: float,
    time_integrator: TimeIntegrator,
    dt_shrink: float,
    dt_min: float,
    tol: float,
    max_iter: int,
) -> ObservationAlignedRolloutLossGradientResult:
    if decoder.latent_dimension != dynamics.dimension:
        raise ValueError(
            f"Decoder latent dimension {decoder.latent_dimension} does not match dynamics dimension {dynamics.dimension}."
        )
    if sample.latent_dimension != dynamics.dimension:
        raise ValueError(
            f"Sample latent dimension {sample.latent_dimension} does not match dynamics dimension {dynamics.dimension}."
        )
    if sample.output_dimension != decoder.output_dimension:
        raise ValueError(
            f"Sample output dimension {sample.output_dimension} does not match decoder output dimension {decoder.output_dimension}."
        )
    if sample.input_dimension != 0 and sample.input_dimension != dynamics.input_dimension:
        raise ValueError(
            f"Sample input dimension {sample.input_dimension} does not match dynamics input dimension {dynamics.input_dimension}."
        )

    return rollout_qoi_loss_and_gradients_from_observations(
        dynamics=dynamics,
        decoder=decoder,
        u0=sample.u0,
        observation_times=sample.observation_times,
        max_dt=max_dt,
        time_integrator=time_integrator,
        qoi_observations=sample.qoi_observations,
        input_function=sample.build_input_function(),
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
