from __future__ import annotations

"""
MPI-aware decoder normal equations.

For observation states u_n and QoI targets q_n, define the decoder feature vector

    phi(u_n) = [u_n, quad(u_n), 1]  or  [u_n, 1].

With trapezoidal weights w_n, the decoder block solves the matrix least-squares problem

    min_X  0.5 * sum_n w_n ||X^T phi(u_n) - q_n||_2^2
           + c_V1 ||V1||_F^2 + c_V2 ||V2||_F^2 + c_v ||v0||_2^2,

where X stacks the selected decoder coefficients as columns:

    X = [V1^T; V2^T; v0^T]  or  [V1^T; v0^T].

The first-order condition is the block normal equation

    (sum_n w_n phi_n phi_n^T + R) X = sum_n w_n phi_n q_n^T,

with

    R = diag(2*c_V1, ..., 2*c_V2, ..., 2*c_v).

This module assembles the sample-local contributions, reduces them across MPI ranks,
solves the global linear system on a designated root rank, and broadcasts the same
decoder coefficients back to the full communicator.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from goattm.core.parametrization import compressed_quadratic_dimension, quadratic_features
from goattm.data.npz_dataset import NpzQoiSample, NpzSampleManifest, load_npz_qoi_sample, load_npz_sample_manifest
from goattm.losses.qoi_loss import trapezoidal_rule_weights_from_times
from goattm.models.linear_dynamics import LinearDynamics
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.runtime import timed
from goattm.runtime.distributed import DistributedContext
from goattm.solvers.implicit_midpoint import rollout_implicit_midpoint_to_observation_times


DynamicsLike = LinearDynamics | QuadraticDynamics | StabilizedQuadraticDynamics


@dataclass(frozen=True)
class DecoderTikhonovRegularization:
    coeff_v1: float = 0.0
    coeff_v2: float = 0.0
    coeff_v0: float = 0.0

    def __post_init__(self) -> None:
        if self.coeff_v1 < 0.0 or self.coeff_v2 < 0.0 or self.coeff_v0 < 0.0:
            raise ValueError("Decoder Tikhonov coefficients must be nonnegative.")

    def diagonal(self, latent_dimension: int, decoder_form: str = "V1V2v") -> np.ndarray:
        if decoder_form not in {"V1v", "V1V2v"}:
            raise ValueError(f"decoder_form must be 'V1v' or 'V1V2v', got {decoder_form!r}")
        quad_dim = 0 if decoder_form == "V1v" else compressed_quadratic_dimension(latent_dimension)
        diagonal = np.empty(latent_dimension + quad_dim + 1, dtype=np.float64)
        diagonal[:latent_dimension] = 2.0 * self.coeff_v1
        diagonal[latent_dimension : latent_dimension + quad_dim] = 2.0 * self.coeff_v2
        diagonal[-1] = 2.0 * self.coeff_v0
        return diagonal


@dataclass(frozen=True)
class DecoderNormalEquationSystem:
    latent_dimension: int
    output_dimension: int
    decoder_form: str
    local_normal_matrix: np.ndarray
    global_normal_matrix: np.ndarray
    local_rhs: np.ndarray
    global_rhs: np.ndarray
    regularization_diagonal: np.ndarray
    local_sample_count: int
    global_sample_count: int
    local_observation_count: int
    global_observation_count: int
    local_sample_ids: tuple[str, ...]

    @property
    def feature_dimension(self) -> int:
        return self.local_normal_matrix.shape[0]

    @property
    def regularized_global_normal_matrix(self) -> np.ndarray:
        return self.global_normal_matrix + np.diag(self.regularization_diagonal)


@dataclass(frozen=True)
class DecoderNormalEquationSolveResult:
    decoder: QuadraticDecoder
    system: DecoderNormalEquationSystem
    solution_matrix: np.ndarray


def decoder_feature_dimension(latent_dimension: int, decoder_form: str = "V1V2v") -> int:
    if decoder_form == "V1v":
        return latent_dimension + 1
    if decoder_form == "V1V2v":
        return latent_dimension + compressed_quadratic_dimension(latent_dimension) + 1
    raise ValueError(f"decoder_form must be 'V1v' or 'V1V2v', got {decoder_form!r}")


def decoder_parameter_matrix(decoder: QuadraticDecoder) -> np.ndarray:
    if decoder.form == "V1v":
        return np.vstack(
            [
                decoder.v1.T.astype(np.float64, copy=False),
                decoder.v0.reshape(1, -1).astype(np.float64, copy=False),
            ]
        )
    return np.vstack(
        [
            decoder.v1.T.astype(np.float64, copy=False),
            decoder.v2.T.astype(np.float64, copy=False),
            decoder.v0.reshape(1, -1).astype(np.float64, copy=False),
        ]
    )


def assemble_decoder_normal_equation_from_npz_dataset(
    dynamics: DynamicsLike,
    decoder_template: QuadraticDecoder,
    manifest: str | Path | NpzSampleManifest,
    max_dt: float,
    regularization: DecoderTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> DecoderNormalEquationSystem:
    if context is None:
        context = DistributedContext.from_comm()
    if regularization is None:
        regularization = DecoderTikhonovRegularization()
    if isinstance(manifest, (str, Path)):
        manifest = load_npz_sample_manifest(manifest)

    feature_dim = decoder_feature_dimension(decoder_template.latent_dimension, decoder_template.form)
    local_normal_matrix = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    local_rhs = np.zeros((feature_dim, decoder_template.output_dimension), dtype=np.float64)
    local_sample_ids: list[str] = []
    local_observation_count = 0

    for sample_id, sample_path in manifest.entries_for_rank(context.rank, context.size):
        sample = load_npz_qoi_sample(sample_path)
        sample_normal, sample_rhs, sample_observation_count = _assemble_sample_decoder_normal_equation(
            dynamics=dynamics,
            decoder_template=decoder_template,
            sample=sample,
            max_dt=max_dt,
            dt_shrink=dt_shrink,
            dt_min=dt_min,
            tol=tol,
            max_iter=max_iter,
        )
        local_normal_matrix += sample_normal
        local_rhs += sample_rhs
        local_sample_ids.append(sample_id)
        local_observation_count += sample_observation_count

    return DecoderNormalEquationSystem(
        latent_dimension=decoder_template.latent_dimension,
        output_dimension=decoder_template.output_dimension,
        decoder_form=decoder_template.form,
        local_normal_matrix=local_normal_matrix,
        global_normal_matrix=context.allreduce_array_sum(local_normal_matrix),
        local_rhs=local_rhs,
        global_rhs=context.allreduce_array_sum(local_rhs),
        regularization_diagonal=regularization.diagonal(decoder_template.latent_dimension, decoder_template.form),
        local_sample_count=len(local_sample_ids),
        global_sample_count=context.allreduce_int_sum(len(local_sample_ids)),
        local_observation_count=local_observation_count,
        global_observation_count=context.allreduce_int_sum(local_observation_count),
        local_sample_ids=tuple(local_sample_ids),
    )


@timed("goattm.problems.solve_decoder_normal_equation")
def solve_decoder_normal_equation(
    system: DecoderNormalEquationSystem,
    context: DistributedContext | None = None,
    solve_root: int = 0,
) -> DecoderNormalEquationSolveResult:
    if context is None:
        context = DistributedContext.from_comm()
    if solve_root < 0 or solve_root >= context.size:
        raise ValueError(f"solve_root must satisfy 0 <= solve_root < size, got root={solve_root}, size={context.size}")

    local_solution = np.zeros_like(system.global_rhs, dtype=np.float64)
    if context.rank == solve_root:
        local_solution = np.linalg.solve(system.regularized_global_normal_matrix, system.global_rhs)
    solution_matrix = context.bcast_array(local_solution, root=solve_root)
    decoder = _decoder_from_solution_matrix(
        latent_dimension=system.latent_dimension,
        output_dimension=system.output_dimension,
        decoder_form=system.decoder_form,
        solution_matrix=solution_matrix,
    )
    return DecoderNormalEquationSolveResult(
        decoder=decoder,
        system=system,
        solution_matrix=solution_matrix,
    )


def update_decoder_from_normal_equation(
    dynamics: DynamicsLike,
    decoder_template: QuadraticDecoder,
    manifest: str | Path | NpzSampleManifest,
    max_dt: float,
    regularization: DecoderTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
    solve_root: int = 0,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> DecoderNormalEquationSolveResult:
    if context is None:
        context = DistributedContext.from_comm()
    system = assemble_decoder_normal_equation_from_npz_dataset(
        dynamics=dynamics,
        decoder_template=decoder_template,
        manifest=manifest,
        max_dt=max_dt,
        regularization=regularization,
        context=context,
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
    return solve_decoder_normal_equation(system=system, context=context, solve_root=solve_root)


def _assemble_sample_decoder_normal_equation(
    dynamics: DynamicsLike,
    decoder_template: QuadraticDecoder,
    sample: NpzQoiSample,
    max_dt: float,
    dt_shrink: float,
    dt_min: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if decoder_template.latent_dimension != dynamics.dimension:
        raise ValueError(
            f"Decoder latent dimension {decoder_template.latent_dimension} does not match dynamics dimension {dynamics.dimension}."
        )
    if sample.latent_dimension != dynamics.dimension:
        raise ValueError(
            f"Sample latent dimension {sample.latent_dimension} does not match dynamics dimension {dynamics.dimension}."
        )
    if sample.output_dimension != decoder_template.output_dimension:
        raise ValueError(
            f"Sample output dimension {sample.output_dimension} does not match decoder output dimension {decoder_template.output_dimension}."
        )
    if sample.input_dimension != 0 and sample.input_dimension != dynamics.input_dimension:
        raise ValueError(
            f"Sample input dimension {sample.input_dimension} does not match dynamics input dimension {dynamics.input_dimension}."
        )

    rollout, observation_indices = rollout_implicit_midpoint_to_observation_times(
        dynamics=dynamics,
        u0=sample.u0,
        observation_times=sample.observation_times,
        max_dt=max_dt,
        input_function=sample.build_input_function(),
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
    observed_states = rollout.states[observation_indices]
    weights = trapezoidal_rule_weights_from_times(sample.observation_times)
    feature_dim = decoder_feature_dimension(dynamics.dimension, decoder_template.form)
    sample_normal = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    sample_rhs = np.zeros((feature_dim, decoder_template.output_dimension), dtype=np.float64)

    for state, target, weight in zip(observed_states, sample.qoi_observations, weights):
        feature_vector = decoder_feature_vector(state, decoder_template.form)
        sample_normal += weight * np.outer(feature_vector, feature_vector)
        sample_rhs += weight * np.outer(feature_vector, target)

    return sample_normal, sample_rhs, observed_states.shape[0]


def decoder_feature_vector(state: np.ndarray, decoder_form: str = "V1V2v") -> np.ndarray:
    if state.ndim != 1:
        raise ValueError(f"state must be rank-1, got shape {state.shape}")
    if decoder_form == "V1v":
        feature = np.empty(state.shape[0] + 1, dtype=np.float64)
        feature[: state.shape[0]] = state
        feature[-1] = 1.0
        return feature
    if decoder_form != "V1V2v":
        raise ValueError(f"decoder_form must be 'V1v' or 'V1V2v', got {decoder_form!r}")
    quad = quadratic_features(state)
    feature = np.empty(state.shape[0] + quad.shape[0] + 1, dtype=np.float64)
    feature[: state.shape[0]] = state
    feature[state.shape[0] : state.shape[0] + quad.shape[0]] = quad
    feature[-1] = 1.0
    return feature


def _decoder_from_solution_matrix(
    latent_dimension: int,
    output_dimension: int,
    decoder_form: str,
    solution_matrix: np.ndarray,
) -> QuadraticDecoder:
    quad_dim = 0 if decoder_form == "V1v" else compressed_quadratic_dimension(latent_dimension)
    expected_shape = (latent_dimension + quad_dim + 1, output_dimension)
    if solution_matrix.shape != expected_shape:
        raise ValueError(f"solution_matrix must have shape {expected_shape}, got {solution_matrix.shape}")

    v1 = solution_matrix[:latent_dimension].T.copy()
    if decoder_form == "V1v":
        v2 = np.zeros((output_dimension, compressed_quadratic_dimension(latent_dimension)), dtype=np.float64)
    else:
        v2 = solution_matrix[latent_dimension : latent_dimension + quad_dim].T.copy()
    v0 = solution_matrix[-1].copy()
    return QuadraticDecoder(v1=v1, v2=v2, v0=v0, form=decoder_form)
