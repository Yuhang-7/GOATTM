from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from goattm.core.parametrization import (
    compressed_h_gradient_to_mu_h,
    mu_h_to_compressed_h,
    quadratic_features,
    s_params_to_matrix,
    w_params_to_matrix,
)
from goattm.core.quadratic import quadratic_jacobian_matrix
from goattm.data.npz_dataset import NpzQoiSample, NpzSampleManifest, load_npz_qoi_sample, load_npz_sample_manifest
from goattm.losses import rollout_qoi_loss_and_gradients_from_cached_observation_rollout
from goattm.losses.qoi_loss import qoi_trajectory_loss, trapezoidal_rule_weights_from_times
from goattm.models.linear_dynamics import LinearDynamics
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems.decoder_normal_equation import (
    DecoderNormalEquationSolveResult,
    DecoderNormalEquationSystem,
    DecoderTikhonovRegularization,
    decoder_feature_dimension,
    decoder_feature_vector,
    decoder_parameter_matrix,
)
from goattm.problems.qoi_dataset_problem import DatasetQoiLossGradientResult
from goattm.runtime import timed
from goattm.runtime.distributed import DistributedContext, sum_array_mapping
from goattm.solvers import (
    RolloutResult,
    TimeIntegrator,
    accumulate_explicit_euler_parameter_hessian_action_terms,
    accumulate_rk4_parameter_hessian_action_terms,
    compute_explicit_euler_incremental_discrete_adjoint,
    compute_rk4_incremental_discrete_adjoint,
    rollout_tangent_from_base_rollout,
    rollout_to_observation_times,
    validate_time_integrator,
)


DynamicsLike = LinearDynamics | QuadraticDynamics | StabilizedQuadraticDynamics


@dataclass(frozen=True)
class DynamicsParameterDirection:
    mu_h: np.ndarray
    c: np.ndarray
    a: np.ndarray | None = None
    s_params: np.ndarray | None = None
    w_params: np.ndarray | None = None
    b: np.ndarray | None = None


@dataclass(frozen=True)
class CachedObservationRollout:
    sample_index: int
    sample_id: str
    sample_path: Path
    sample: NpzQoiSample
    input_function: Callable[[float], np.ndarray] | None
    rollout: RolloutResult
    observation_indices: np.ndarray
    observation_times: np.ndarray
    observation_weights: np.ndarray
    observed_states: np.ndarray


@dataclass(frozen=True)
class ForwardRolloutCacheEntry:
    dynamics_key: str
    local_rollouts: tuple[CachedObservationRollout, ...]
    local_sample_ids: tuple[str, ...]
    local_sample_count: int
    global_sample_count: int


@dataclass
class ForwardRolloutFailure(RuntimeError):
    sample_index: int
    sample_id: str
    sample_path: str
    final_time: float
    reason: str = ""

    def __str__(self) -> str:
        message = (
            f"Forward rollout failed for sample '{self.sample_id}' "
            f"(index={self.sample_index}, path='{self.sample_path}') at time {self.final_time}."
        )
        if self.reason:
            message += f" Reason: {self.reason}"
        return message


@dataclass(frozen=True)
class DecoderBestResponseContext:
    dynamics_key: str
    regularization: DecoderTikhonovRegularization
    decoder: QuadraticDecoder
    system: DecoderNormalEquationSystem
    forward_cache: ForwardRolloutCacheEntry


@dataclass(frozen=True)
class ReducedQoiBestResponseResult:
    data_loss: float
    decoder_regularization_loss: float
    dynamics_regularization_loss: float
    reduced_data_gradient: np.ndarray | None
    reduced_objective_gradient: np.ndarray
    direct_dynamics_gradient: np.ndarray
    dynamics_regularization_gradient: np.ndarray
    decoder_chain_gradient: np.ndarray
    decoder_data_gradient_matrix: np.ndarray
    decoder: QuadraticDecoder
    best_response_context: DecoderBestResponseContext
    dataset_result: DatasetQoiLossGradientResult

    @property
    def objective_value(self) -> float:
        return self.data_loss + self.decoder_regularization_loss + self.dynamics_regularization_loss

    @property
    def gradient(self) -> np.ndarray:
        return self.reduced_objective_gradient


@dataclass(frozen=True)
class ReducedObjectivePreparedState:
    dynamics: DynamicsLike
    result: ReducedQoiBestResponseResult

    @property
    def objective_value(self) -> float:
        return self.result.objective_value

    @property
    def gradient(self) -> np.ndarray:
        return self.result.gradient


@dataclass(frozen=True)
class ReducedHessianActionResult:
    base_state: ReducedObjectivePreparedState
    direction: np.ndarray
    action: np.ndarray
    decoder_direction: QuadraticDecoder


@dataclass(frozen=True)
class ReducedExplicitHessianResult:
    base_state: ReducedObjectivePreparedState
    hessian: np.ndarray


@dataclass(frozen=True)
class DynamicsTikhonovRegularization:
    coeff_a: float = 0.0
    coeff_s: float = 0.0
    coeff_w: float = 0.0
    coeff_mu_h: float = 0.0
    coeff_b: float = 0.0
    coeff_c: float = 0.0
    coeff_spectral_abscissa: float = 0.0
    spectral_abscissa_alpha: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.coeff_a,
            self.coeff_s,
            self.coeff_w,
            self.coeff_mu_h,
            self.coeff_b,
            self.coeff_c,
            self.coeff_spectral_abscissa,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("Dynamics Tikhonov coefficients must be nonnegative.")


@dataclass(frozen=True)
class ReducedObjectiveWorkflow:
    evaluator: "ObservationAlignedBestResponseEvaluator"
    decoder_template: QuadraticDecoder
    regularization: DecoderTikhonovRegularization
    dynamics_regularization: DynamicsTikhonovRegularization
    solve_root: int = 0

    def solve_decoder_best_response(self, dynamics: DynamicsLike) -> DecoderBestResponseContext:
        return self.evaluator.solve_decoder_best_response(
            dynamics=dynamics,
            decoder_template=self.decoder_template,
            regularization=self.regularization,
            dynamics_regularization=self.dynamics_regularization,
            solve_root=self.solve_root,
        )

    def find_decode_param(self, dynamics: DynamicsLike) -> QuadraticDecoder:
        return self.solve_decoder_best_response(dynamics).decoder

    def evaluate_objective_and_gradient(self, dynamics: DynamicsLike) -> ReducedQoiBestResponseResult:
        return self.evaluator.evaluate_reduced_objective_and_gradient(
            dynamics=dynamics,
            decoder_template=self.decoder_template,
            regularization=self.regularization,
            dynamics_regularization=self.dynamics_regularization,
            solve_root=self.solve_root,
        )

    @timed("goattm.problems.ReducedObjectiveWorkflow.prepare")
    def prepare(self, dynamics: DynamicsLike) -> ReducedObjectivePreparedState:
        return ReducedObjectivePreparedState(
            dynamics=dynamics,
            result=self.evaluate_objective_and_gradient(dynamics),
        )

    def evaluate_gradient(self, dynamics: DynamicsLike) -> np.ndarray:
        return self.evaluate_objective_and_gradient(dynamics).gradient

    def evaluate_objective(self, dynamics: DynamicsLike) -> float:
        return self.evaluate_objective_and_gradient(dynamics).objective_value

    def lossfunction(self, dynamics: DynamicsLike) -> float:
        return self.evaluate_objective(dynamics)

    def lossfunction_gradient_wrt_mug(self, dynamics: DynamicsLike) -> np.ndarray:
        return self.evaluate_gradient(dynamics)

    def lossfunction_and_gradient_wrt_mug(self, dynamics: DynamicsLike) -> ReducedQoiBestResponseResult:
        return self.evaluate_objective_and_gradient(dynamics)

    def lossfunc_jacobian(self, dynamics: DynamicsLike) -> np.ndarray:
        return self.lossfunction_gradient_wrt_mug(dynamics)

    def evaluate_data_gradient(self, dynamics: DynamicsLike) -> ReducedQoiBestResponseResult:
        return self.evaluator.evaluate_reduced_data_loss_and_gradient(
            dynamics=dynamics,
            decoder_template=self.decoder_template,
            regularization=self.regularization,
            dynamics_regularization=self.dynamics_regularization,
            solve_root=self.solve_root,
        )

    @timed("goattm.problems.ReducedObjectiveWorkflow.evaluate_hessian_action")
    def evaluate_hessian_action(
        self,
        dynamics: DynamicsLike,
        direction: np.ndarray,
    ) -> ReducedHessianActionResult:
        return self.evaluate_hessian_action_from_prepared_state(
            prepared_state=self.prepare(dynamics),
            direction=direction,
        )

    @timed("goattm.problems.ReducedObjectiveWorkflow.evaluate_hessian_action_from_prepared_state")
    def evaluate_hessian_action_from_prepared_state(
        self,
        prepared_state: ReducedObjectivePreparedState,
        direction: np.ndarray,
    ) -> ReducedHessianActionResult:
        return self.evaluator.evaluate_reduced_objective_hessian_action(
            prepared_state=prepared_state,
            decoder_template=self.decoder_template,
            regularization=self.regularization,
            dynamics_regularization=self.dynamics_regularization,
            direction=direction,
            solve_root=self.solve_root,
        )

    @timed("goattm.problems.ReducedObjectiveWorkflow.evaluate_explicit_hessian")
    def evaluate_explicit_hessian(self, dynamics: DynamicsLike) -> ReducedExplicitHessianResult:
        return self.evaluate_explicit_hessian_from_prepared_state(self.prepare(dynamics))

    @timed("goattm.problems.ReducedObjectiveWorkflow.evaluate_explicit_hessian_from_prepared_state")
    def evaluate_explicit_hessian_from_prepared_state(
        self,
        prepared_state: ReducedObjectivePreparedState,
    ) -> ReducedExplicitHessianResult:
        return self.evaluator.evaluate_reduced_objective_explicit_hessian(
            prepared_state=prepared_state,
            decoder_template=self.decoder_template,
            regularization=self.regularization,
            dynamics_regularization=self.dynamics_regularization,
            solve_root=self.solve_root,
        )

    def lossfunction_hessian_action_wrt_mug(
        self,
        dynamics: DynamicsLike,
        direction: np.ndarray,
    ) -> np.ndarray:
        return self.evaluate_hessian_action(
            dynamics=dynamics,
            direction=direction,
        ).action


class ObservationAlignedBestResponseEvaluator:
    def __init__(
        self,
        manifest: str | Path | NpzSampleManifest,
        max_dt: float,
        context: DistributedContext | None = None,
        time_integrator: TimeIntegrator = "implicit_midpoint",
        dt_shrink: float = 0.8,
        dt_min: float = 1e-10,
        tol: float = 1e-10,
        max_iter: int = 25,
    ) -> None:
        if context is None:
            context = DistributedContext.from_comm()
        if isinstance(manifest, (str, Path)):
            manifest = load_npz_sample_manifest(manifest)
        self.manifest = manifest
        self.max_dt = float(max_dt)
        self.context = context
        self.time_integrator = validate_time_integrator(time_integrator)
        self.dt_shrink = float(dt_shrink)
        self.dt_min = float(dt_min)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        local_samples = []
        path_lookup = dict(zip(self.manifest.sample_ids, self.manifest.absolute_paths()))
        index_lookup = {sample_id: idx for idx, sample_id in enumerate(self.manifest.sample_ids)}
        for sample_id, sample_path in self.manifest.entries_for_rank(self.context.rank, self.context.size):
            local_samples.append(
                (
                    index_lookup[sample_id],
                    sample_id,
                    path_lookup[sample_id],
                    load_npz_qoi_sample(sample_path),
                )
            )
        self._local_samples = tuple(local_samples)
        self._forward_cache: ForwardRolloutCacheEntry | None = None
        self._best_response_cache_key: tuple[str, str] | None = None
        self._best_response_cache: DecoderBestResponseContext | None = None
        self._dataset_eval_cache_key: tuple[str, str] | None = None
        self._dataset_eval_cache: DatasetQoiLossGradientResult | None = None

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.get_forward_rollouts")
    def get_forward_rollouts(self, dynamics: DynamicsLike) -> ForwardRolloutCacheEntry:
        dynamics_key = dynamics_parameter_key(dynamics)
        if self._forward_cache is not None and self._forward_cache.dynamics_key == dynamics_key:
            return self._forward_cache

        local_rollouts: list[CachedObservationRollout] = []
        local_sample_ids: list[str] = []
        local_failure: ForwardRolloutFailure | None = None
        for sample_index, sample_id, sample_path, sample in self._local_samples:
            try:
                input_function = sample.build_input_function()
                rollout, observation_indices = rollout_to_observation_times(
                    dynamics=dynamics,
                    u0=sample.u0,
                    observation_times=sample.observation_times,
                    max_dt=self.max_dt,
                    input_function=input_function,
                    time_integrator=self.time_integrator,
                    dt_shrink=self.dt_shrink,
                    dt_min=self.dt_min,
                    tol=self.tol,
                    max_iter=self.max_iter,
                )
            except Exception as exc:
                local_failure = ForwardRolloutFailure(
                    sample_index=sample_index,
                    sample_id=sample_id,
                    sample_path=str(sample_path),
                    final_time=float("nan"),
                    reason=f"{type(exc).__name__}: {exc}",
                )
                break
            if not rollout.success:
                local_failure = ForwardRolloutFailure(
                    sample_index=sample_index,
                    sample_id=sample_id,
                    sample_path=str(sample_path),
                    final_time=float(rollout.final_time),
                    reason="time integrator returned success=False",
                )
                break
            local_rollouts.append(
                CachedObservationRollout(
                    sample_index=sample_index,
                    sample_id=sample_id,
                    sample_path=sample_path,
                    sample=sample,
                    input_function=input_function,
                    rollout=rollout,
                    observation_indices=observation_indices,
                    observation_times=sample.observation_times.copy(),
                    observation_weights=trapezoidal_rule_weights_from_times(sample.observation_times),
                    observed_states=rollout.states[observation_indices].copy(),
                )
            )
            local_sample_ids.append(sample_id)

        if self.context.allreduce_bool_any(local_failure is not None):
            failures = [
                str(item)
                for item in self.context.allgather_object(None if local_failure is None else str(local_failure))
                if item is not None
            ]
            if local_failure is not None:
                raise ForwardRolloutFailure(
                    sample_index=local_failure.sample_index,
                    sample_id=local_failure.sample_id,
                    sample_path=local_failure.sample_path,
                    final_time=local_failure.final_time,
                    reason=local_failure.reason,
                )
            raise RuntimeError("Forward rollout failed on another MPI rank: " + " | ".join(failures))

        self._forward_cache = ForwardRolloutCacheEntry(
            dynamics_key=dynamics_key,
            local_rollouts=tuple(local_rollouts),
            local_sample_ids=tuple(local_sample_ids),
            local_sample_count=len(local_rollouts),
            global_sample_count=self.context.allreduce_int_sum(len(local_rollouts)),
        )
        self._best_response_cache_key = None
        self._best_response_cache = None
        self._dataset_eval_cache_key = None
        self._dataset_eval_cache = None
        return self._forward_cache

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.solve_decoder_best_response")
    def solve_decoder_best_response(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> DecoderBestResponseContext:
        forward_cache = self.get_forward_rollouts(dynamics)
        if regularization is None:
            regularization = DecoderTikhonovRegularization()
        cache_key = (forward_cache.dynamics_key, decoder_template.form, regularization_key(regularization))
        if self._best_response_cache is not None and self._best_response_cache_key == cache_key:
            return self._best_response_cache

        feature_dim = decoder_feature_dimension(decoder_template.latent_dimension, decoder_template.form)
        local_normal_matrix = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        local_rhs = np.zeros((feature_dim, decoder_template.output_dimension), dtype=np.float64)
        for rollout_entry in forward_cache.local_rollouts:
            sample_normal, sample_rhs = _assemble_decoder_normal_terms_from_cached_rollout(
                rollout_entry=rollout_entry,
                decoder_output_dimension=decoder_template.output_dimension,
                decoder_form=decoder_template.form,
            )
            local_normal_matrix += sample_normal
            local_rhs += sample_rhs

        system = DecoderNormalEquationSystem(
            latent_dimension=decoder_template.latent_dimension,
            output_dimension=decoder_template.output_dimension,
            decoder_form=decoder_template.form,
            local_normal_matrix=local_normal_matrix,
            global_normal_matrix=self.context.allreduce_array_sum(local_normal_matrix),
            local_rhs=local_rhs,
            global_rhs=self.context.allreduce_array_sum(local_rhs),
            regularization_diagonal=regularization.diagonal(decoder_template.latent_dimension, decoder_template.form),
            local_sample_count=forward_cache.local_sample_count,
            global_sample_count=forward_cache.global_sample_count,
            local_observation_count=sum(item.observation_times.shape[0] for item in forward_cache.local_rollouts),
            global_observation_count=self.context.allreduce_int_sum(
                sum(item.observation_times.shape[0] for item in forward_cache.local_rollouts)
            ),
            local_sample_ids=forward_cache.local_sample_ids,
        )
        solve_result = solve_decoder_linear_system(system=system, context=self.context, solve_root=solve_root)
        best_response = DecoderBestResponseContext(
            dynamics_key=forward_cache.dynamics_key,
            regularization=regularization,
            decoder=solve_result.decoder,
            system=system,
            forward_cache=forward_cache,
        )
        self._best_response_cache_key = cache_key
        self._best_response_cache = best_response
        return best_response

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_dataset_loss_and_gradients")
    def evaluate_dataset_loss_and_gradients(
        self,
        dynamics: DynamicsLike,
        decoder: QuadraticDecoder,
    ) -> DatasetQoiLossGradientResult:
        forward_cache = self.get_forward_rollouts(dynamics)
        decoder_key = decoder_parameter_key(decoder)
        cache_key = (forward_cache.dynamics_key, decoder_key)
        if self._dataset_eval_cache is not None and self._dataset_eval_cache_key == cache_key:
            return self._dataset_eval_cache

        decoder_gradients = _zero_decoder_gradients(decoder)
        dynamics_gradients = _zero_dynamics_gradients(dynamics)
        local_results = []
        local_loss = 0.0
        for rollout_entry in forward_cache.local_rollouts:
            result = rollout_qoi_loss_and_gradients_from_cached_observation_rollout(
                dynamics=dynamics,
                decoder=decoder,
                rollout=rollout_entry.rollout,
                observation_indices=rollout_entry.observation_indices,
                observation_times=rollout_entry.observation_times,
                qoi_observations=rollout_entry.sample.qoi_observations,
                input_function=rollout_entry.input_function,
                time_integrator=self.time_integrator,
            )
            local_results.append(result)
            local_loss += result.loss
            decoder_gradients["v1"] += result.decoder_partials.v1_grad
            decoder_gradients["v2"] += result.decoder_partials.v2_grad
            decoder_gradients["v0"] += result.decoder_partials.v0_grad
            for key, value in result.dynamics_gradients.items():
                dynamics_gradients[key] += value

        dataset_result = DatasetQoiLossGradientResult(
            total_loss=self.context.allreduce_scalar_sum(local_loss),
            local_loss=local_loss,
            local_sample_count=forward_cache.local_sample_count,
            global_sample_count=forward_cache.global_sample_count,
            decoder_gradients=sum_array_mapping(decoder_gradients, self.context),
            dynamics_gradients=sum_array_mapping(dynamics_gradients, self.context),
            local_sample_ids=forward_cache.local_sample_ids,
            local_sample_results=tuple(local_results),
        )
        self._dataset_eval_cache_key = cache_key
        self._dataset_eval_cache = dataset_result
        return dataset_result

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_dataset_data_loss")
    def evaluate_dataset_data_loss(
        self,
        dynamics: DynamicsLike,
        decoder: QuadraticDecoder,
    ) -> float:
        forward_cache = self.get_forward_rollouts(dynamics)
        local_loss = 0.0
        for rollout_entry in forward_cache.local_rollouts:
            local_loss += qoi_trajectory_loss(
                states=rollout_entry.observed_states,
                decoder=decoder,
                qoi_observations=rollout_entry.sample.qoi_observations,
                times=rollout_entry.observation_times,
            )
        return self.context.allreduce_scalar_sum(local_loss)

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_dataset_relative_error")
    def evaluate_dataset_relative_error(
        self,
        dynamics: DynamicsLike,
        decoder: QuadraticDecoder,
    ) -> float:
        forward_cache = self.get_forward_rollouts(dynamics)
        local_residual_sumsq = 0.0
        local_target_sumsq = 0.0
        for rollout_entry in forward_cache.local_rollouts:
            weights = rollout_entry.observation_weights
            for state, target, weight in zip(
                rollout_entry.observed_states,
                rollout_entry.sample.qoi_observations,
                weights,
                strict=True,
            ):
                residual = decoder.decode(state) - target
                local_residual_sumsq += float(weight) * float(np.dot(residual, residual))
                local_target_sumsq += float(weight) * float(np.dot(target, target))
        residual_sumsq = self.context.allreduce_scalar_sum(local_residual_sumsq)
        target_sumsq = self.context.allreduce_scalar_sum(local_target_sumsq)
        if target_sumsq <= 0.0:
            return 0.0 if residual_sumsq <= 0.0 else float("inf")
        return float(np.sqrt(residual_sumsq / target_sumsq))

    def build_reduced_objective_workflow(
        self,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> ReducedObjectiveWorkflow:
        if regularization is None:
            regularization = DecoderTikhonovRegularization()
        if dynamics_regularization is None:
            dynamics_regularization = DynamicsTikhonovRegularization()
        return ReducedObjectiveWorkflow(
            evaluator=self,
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=dynamics_regularization,
            solve_root=solve_root,
        )

    def find_decode_param(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> QuadraticDecoder:
        return self.build_reduced_objective_workflow(
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=dynamics_regularization,
            solve_root=solve_root,
        ).find_decode_param(dynamics)

    def lossfunction(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> float:
        return self.build_reduced_objective_workflow(
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=dynamics_regularization,
            solve_root=solve_root,
        ).lossfunction(dynamics)

    def lossfunction_gradient_wrt_mug(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> np.ndarray:
        return self.build_reduced_objective_workflow(
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=dynamics_regularization,
            solve_root=solve_root,
        ).lossfunction_gradient_wrt_mug(dynamics)

    def lossfunction_hessian_action_wrt_mug(
        self,
        dynamics: DynamicsLike,
        direction: np.ndarray,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
        step_size: float = 1e-6,
    ) -> np.ndarray:
        _ = step_size
        return self.build_reduced_objective_workflow(
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=dynamics_regularization,
            solve_root=solve_root,
        ).lossfunction_hessian_action_wrt_mug(
            dynamics=dynamics,
            direction=direction,
        )

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.compute_decoder_mixed_hessian_action")
    def compute_decoder_mixed_hessian_action(
        self,
        dynamics: DynamicsLike,
        decoder: QuadraticDecoder,
        direction: DynamicsParameterDirection,
    ) -> np.ndarray:
        forward_cache = self.get_forward_rollouts(dynamics)
        x_matrix = decoder_parameter_matrix(decoder)
        feature_dim = decoder_feature_dimension(dynamics.dimension, decoder.form)
        local_action = np.zeros((feature_dim, decoder.output_dimension), dtype=np.float64)

        for rollout_entry in forward_cache.local_rollouts:
            tangent_states = rollout_tangent_from_base_rollout(
                dynamics=dynamics,
                base_rollout=rollout_entry.rollout,
                parameter_action=lambda midpoint_state, midpoint_time, input_function=rollout_entry.input_function: rhs_parameter_action(  # noqa: E731
                    dynamics,
                    direction,
                    midpoint_state,
                    midpoint_time,
                    input_function=input_function,
                ),
                input_function=rollout_entry.input_function,
                time_integrator=self.time_integrator,
            )
            observed_tangents = tangent_states[rollout_entry.observation_indices]
            for state, state_tangent, q_target, weight in zip(
                rollout_entry.observed_states,
                observed_tangents,
                rollout_entry.sample.qoi_observations,
                rollout_entry.observation_weights,
            ):
                phi = decoder_feature_vector(state, decoder.form)
                dphi = decoder_feature_directional_derivative(state, state_tangent)
                if decoder.form == "V1v":
                    dphi = np.concatenate([state_tangent, np.zeros(1, dtype=np.float64)])
                q_pred = x_matrix.T @ phi
                local_action += weight * (
                    np.outer(dphi, q_pred)
                    + np.outer(phi, x_matrix.T @ dphi)
                    - np.outer(dphi, q_target)
                )

        return self.context.allreduce_array_sum(local_action)

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.compute_decoder_best_response_action")
    def compute_decoder_best_response_action(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization,
        direction: DynamicsParameterDirection,
        solve_root: int = 0,
    ) -> np.ndarray:
        best_response = self.solve_decoder_best_response(
            dynamics=dynamics,
            decoder_template=decoder_template,
            regularization=regularization,
            solve_root=solve_root,
        )
        mixed_action = self.compute_decoder_mixed_hessian_action(dynamics, best_response.decoder, direction)
        return solve_decoder_best_response_action_matrix(best_response.system, mixed_action, self.context, solve_root=solve_root)

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_reduced_data_loss_and_gradient")
    def evaluate_reduced_data_loss_and_gradient(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> ReducedQoiBestResponseResult:
        best_response = self.solve_decoder_best_response(
            dynamics=dynamics,
            decoder_template=decoder_template,
            regularization=regularization,
            solve_root=solve_root,
        )
        dataset_result = self.evaluate_dataset_loss_and_gradients(dynamics, best_response.decoder)
        direct_dynamics_gradient = pack_dynamics_gradient_vector(dynamics, dataset_result.dynamics_gradients)
        if dynamics_regularization is None:
            dynamics_regularization = DynamicsTikhonovRegularization()
        dynamics_reg_gradient = dynamics_regularization_gradient_vector(dynamics, dynamics_regularization)
        dynamics_reg_loss = dynamics_regularization_loss(dynamics, dynamics_regularization)
        decoder_data_gradient = stack_decoder_gradient_matrix(dataset_result.decoder_gradients, best_response.decoder.form)

        reduced_gradient = direct_dynamics_gradient.copy()
        chain_gradient = np.zeros_like(reduced_gradient)
        param_dimension = dynamics_parameter_dimension(dynamics)
        for basis_idx in range(param_dimension):
            basis = np.zeros(param_dimension, dtype=np.float64)
            basis[basis_idx] = 1.0
            direction = unpack_dynamics_parameter_vector(dynamics, basis)
            decoder_action = self.compute_decoder_best_response_action(
                dynamics=dynamics,
                decoder_template=decoder_template,
                regularization=best_response.regularization,
                direction=direction,
                solve_root=solve_root,
            )
            chain_gradient[basis_idx] = float(np.sum(decoder_data_gradient * decoder_action))
        reduced_gradient += chain_gradient

        return ReducedQoiBestResponseResult(
            data_loss=dataset_result.total_loss,
            decoder_regularization_loss=decoder_regularization_loss(best_response.decoder, best_response.regularization),
            dynamics_regularization_loss=dynamics_reg_loss,
            reduced_data_gradient=reduced_gradient,
            reduced_objective_gradient=direct_dynamics_gradient + dynamics_reg_gradient,
            direct_dynamics_gradient=direct_dynamics_gradient,
            dynamics_regularization_gradient=dynamics_reg_gradient,
            decoder_chain_gradient=chain_gradient,
            decoder_data_gradient_matrix=decoder_data_gradient,
            decoder=best_response.decoder,
            best_response_context=best_response,
            dataset_result=dataset_result,
        )

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_reduced_objective_and_gradient")
    def evaluate_reduced_objective_and_gradient(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> ReducedQoiBestResponseResult:
        best_response = self.solve_decoder_best_response(
            dynamics=dynamics,
            decoder_template=decoder_template,
            regularization=regularization,
            solve_root=solve_root,
        )
        dataset_result = self.evaluate_dataset_loss_and_gradients(dynamics, best_response.decoder)
        direct_dynamics_gradient = pack_dynamics_gradient_vector(dynamics, dataset_result.dynamics_gradients)
        if dynamics_regularization is None:
            dynamics_regularization = DynamicsTikhonovRegularization()
        dynamics_reg_gradient = dynamics_regularization_gradient_vector(dynamics, dynamics_regularization)
        dynamics_reg_loss = dynamics_regularization_loss(dynamics, dynamics_regularization)
        decoder_data_gradient = stack_decoder_gradient_matrix(dataset_result.decoder_gradients, best_response.decoder.form)

        # GOAM-style reduced optimization uses the outer objective
        # J(mu_g) = J_data(mu_f^*(mu_g), mu_g) + R_f(mu_f^*(mu_g)),
        # where mu_f^* is obtained from the regularized normal equation.
        # Since the inner decoder solve enforces stationarity with respect to mu_f,
        # the chain contribution cancels in the reduced first derivative of the full objective.
        objective_gradient = direct_dynamics_gradient + dynamics_reg_gradient

        return ReducedQoiBestResponseResult(
            data_loss=dataset_result.total_loss,
            decoder_regularization_loss=decoder_regularization_loss(best_response.decoder, best_response.regularization),
            dynamics_regularization_loss=dynamics_reg_loss,
            reduced_data_gradient=None,
            reduced_objective_gradient=objective_gradient,
            direct_dynamics_gradient=direct_dynamics_gradient,
            dynamics_regularization_gradient=dynamics_reg_gradient,
            decoder_chain_gradient=np.zeros_like(objective_gradient),
            decoder_data_gradient_matrix=decoder_data_gradient,
            decoder=best_response.decoder,
            best_response_context=best_response,
            dataset_result=dataset_result,
        )

    def evaluate_goam_reduced_objective_and_gradient(
        self,
        dynamics: DynamicsLike,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None = None,
        dynamics_regularization: DynamicsTikhonovRegularization | None = None,
        solve_root: int = 0,
    ) -> ReducedQoiBestResponseResult:
        return self.evaluate_reduced_objective_and_gradient(
            dynamics=dynamics,
            decoder_template=decoder_template,
            regularization=regularization,
            dynamics_regularization=dynamics_regularization,
            solve_root=solve_root,
        )

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_reduced_objective_hessian_action")
    def evaluate_reduced_objective_hessian_action(
        self,
        prepared_state: ReducedObjectivePreparedState,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None,
        dynamics_regularization: DynamicsTikhonovRegularization | None,
        direction: np.ndarray,
        solve_root: int = 0,
    ) -> ReducedHessianActionResult:
        if regularization is None:
            regularization = DecoderTikhonovRegularization()
        if dynamics_regularization is None:
            dynamics_regularization = DynamicsTikhonovRegularization()
        direction_vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        base_vector = dynamics_parameter_vector(prepared_state.dynamics)
        if direction_vector.shape != base_vector.shape:
            raise ValueError(f"direction must have shape {base_vector.shape}, got {direction_vector.shape}")
        if float(np.linalg.norm(direction_vector)) <= 0.0:
            raise ValueError("direction must be nonzero for a Hessian action.")

        base_result = prepared_state.result
        dynamics_direction = unpack_dynamics_parameter_vector(prepared_state.dynamics, direction_vector)
        decoder_action_matrix = self.compute_decoder_best_response_action(
            dynamics=prepared_state.dynamics,
            decoder_template=decoder_template,
            regularization=regularization,
            direction=dynamics_direction,
            solve_root=solve_root,
        )
        decoder_direction = matrix_to_decoder(
            prepared_state.dynamics.dimension,
            base_result.decoder.output_dimension,
            decoder_action_matrix,
        )

        total_a_grad = np.zeros((prepared_state.dynamics.dimension, prepared_state.dynamics.dimension), dtype=np.float64)
        total_delta_a_grad = np.zeros_like(total_a_grad)
        total_h_grad = np.zeros_like(prepared_state.dynamics.h_matrix)
        total_delta_h_grad = np.zeros_like(total_h_grad)
        total_c_grad = np.zeros_like(prepared_state.dynamics.c)
        total_delta_c_grad = np.zeros_like(total_c_grad)
        total_b_grad = None if getattr(prepared_state.dynamics, "b", None) is None else np.zeros_like(prepared_state.dynamics.b)
        total_delta_b_grad = None if getattr(prepared_state.dynamics, "b", None) is None else np.zeros_like(prepared_state.dynamics.b)

        local_rollouts = base_result.best_response_context.forward_cache.local_rollouts
        local_results = base_result.dataset_result.local_sample_results
        if len(local_rollouts) != len(local_results):
            raise RuntimeError("Forward rollout cache and local sample results are misaligned.")

        for rollout_entry, sample_result in zip(local_rollouts, local_results):
            tangent_states = rollout_tangent_from_base_rollout(
                dynamics=prepared_state.dynamics,
                base_rollout=rollout_entry.rollout,
                parameter_action=lambda midpoint_state, midpoint_time, input_function=rollout_entry.input_function: rhs_parameter_action(  # noqa: E731
                    prepared_state.dynamics,
                    dynamics_direction,
                    midpoint_state,
                    midpoint_time,
                    input_function=input_function,
                ),
                input_function=rollout_entry.input_function,
                time_integrator=self.time_integrator,
            )
            if self.time_integrator == "implicit_midpoint":
                adjoint_tangents = _compute_incremental_midpoint_discrete_adjoint(
                    dynamics=prepared_state.dynamics,
                    decoder=base_result.decoder,
                    decoder_direction=decoder_direction,
                    rollout=rollout_entry.rollout,
                    observation_indices=rollout_entry.observation_indices,
                    observation_times=rollout_entry.observation_times,
                    qoi_observations=rollout_entry.sample.qoi_observations,
                    tangent_states=tangent_states,
                    base_adjoints=sample_result.adjoints,
                    base_decoder_partials=sample_result.decoder_partials,
                    direction=dynamics_direction,
                )

                sample_a_grad, sample_delta_a_grad, sample_h_grad, sample_delta_h_grad, sample_c_grad, sample_delta_c_grad, sample_b_grad, sample_delta_b_grad = _assemble_dynamic_hessian_action_terms(  # noqa: E501
                    dynamics=prepared_state.dynamics,
                    direction=dynamics_direction,
                    rollout=rollout_entry.rollout,
                    tangent_states=tangent_states,
                    adjoints=sample_result.adjoints,
                    adjoint_tangents=adjoint_tangents,
                    input_function=rollout_entry.input_function,
                )
            elif self.time_integrator == "explicit_euler":
                state_loss_grad_direction = np.zeros_like(rollout_entry.rollout.states, dtype=np.float64)
                observed_tangents = tangent_states[rollout_entry.observation_indices]
                for local_idx, global_idx in enumerate(rollout_entry.observation_indices):
                    state_loss_grad_direction[global_idx] = _observation_state_loss_gradient_direction(
                        decoder=base_result.decoder,
                        decoder_direction=decoder_direction,
                        state=rollout_entry.rollout.states[global_idx],
                        state_tangent=observed_tangents[local_idx],
                        residual=sample_result.decoder_partials.residuals[local_idx],
                        weight=float(sample_result.decoder_partials.quadrature_weights[local_idx]),
                    )
                adjoint_tangents = compute_explicit_euler_incremental_discrete_adjoint(
                    dynamics=prepared_state.dynamics,
                    rollout=rollout_entry.rollout,
                    tangent_states=tangent_states,
                    base_adjoints=sample_result.adjoints,
                    state_loss_gradient_direction=state_loss_grad_direction,
                    jacobian_direction=lambda state, state_tangent, time: _rhs_jacobian_direction(
                        prepared_state.dynamics,
                        dynamics_direction,
                        state,
                        state_tangent,
                    ),
                )
                sample_a_grad, sample_delta_a_grad, sample_h_grad, sample_delta_h_grad, sample_b_grad, sample_delta_b_grad, sample_c_grad, sample_delta_c_grad = accumulate_explicit_euler_parameter_hessian_action_terms(
                    dynamics=prepared_state.dynamics,
                    rollout=rollout_entry.rollout,
                    tangent_states=tangent_states,
                    adjoints=sample_result.adjoints,
                    adjoint_tangents=adjoint_tangents,
                    input_function=rollout_entry.input_function,
                )
            else:
                state_loss_grad_direction = np.zeros_like(rollout_entry.rollout.states, dtype=np.float64)
                observed_tangents = tangent_states[rollout_entry.observation_indices]
                for local_idx, global_idx in enumerate(rollout_entry.observation_indices):
                    state_loss_grad_direction[global_idx] = _observation_state_loss_gradient_direction(
                        decoder=base_result.decoder,
                        decoder_direction=decoder_direction,
                        state=rollout_entry.rollout.states[global_idx],
                        state_tangent=observed_tangents[local_idx],
                        residual=sample_result.decoder_partials.residuals[local_idx],
                        weight=float(sample_result.decoder_partials.quadrature_weights[local_idx]),
                    )
                parameter_action = lambda stage_state, stage_time, input_function=rollout_entry.input_function: rhs_parameter_action(  # noqa: E731
                    prepared_state.dynamics,
                    dynamics_direction,
                    stage_state,
                    stage_time,
                    input_function=input_function,
                )
                jacobian_direction = lambda state, state_tangent, time: _rhs_jacobian_direction(  # noqa: E731
                    prepared_state.dynamics,
                    dynamics_direction,
                    state,
                    state_tangent,
                )
                adjoint_tangents = compute_rk4_incremental_discrete_adjoint(
                    dynamics=prepared_state.dynamics,
                    rollout=rollout_entry.rollout,
                    tangent_states=tangent_states,
                    base_adjoints=sample_result.adjoints,
                    state_loss_gradient_direction=state_loss_grad_direction,
                    jacobian_direction=jacobian_direction,
                    parameter_action=parameter_action,
                    input_function=rollout_entry.input_function,
                )
                sample_a_grad, sample_delta_a_grad, sample_h_grad, sample_delta_h_grad, sample_b_grad, sample_delta_b_grad, sample_c_grad, sample_delta_c_grad = accumulate_rk4_parameter_hessian_action_terms(
                    dynamics=prepared_state.dynamics,
                    rollout=rollout_entry.rollout,
                    tangent_states=tangent_states,
                    adjoints=sample_result.adjoints,
                    adjoint_tangents=adjoint_tangents,
                    input_function=rollout_entry.input_function,
                    parameter_action=parameter_action,
                    jacobian_direction=jacobian_direction,
                )
            total_a_grad += sample_a_grad
            total_delta_a_grad += sample_delta_a_grad
            total_h_grad += sample_h_grad
            total_delta_h_grad += sample_delta_h_grad
            total_c_grad += sample_c_grad
            total_delta_c_grad += sample_delta_c_grad
            if total_b_grad is not None and sample_b_grad is not None:
                total_b_grad += sample_b_grad
                total_delta_b_grad += sample_delta_b_grad

        total_a_grad = self.context.allreduce_array_sum(total_a_grad)
        total_delta_a_grad = self.context.allreduce_array_sum(total_delta_a_grad)
        total_h_grad = self.context.allreduce_array_sum(total_h_grad)
        total_delta_h_grad = self.context.allreduce_array_sum(total_delta_h_grad)
        total_c_grad = self.context.allreduce_array_sum(total_c_grad)
        total_delta_c_grad = self.context.allreduce_array_sum(total_delta_c_grad)
        if total_b_grad is not None and total_delta_b_grad is not None:
            total_b_grad = self.context.allreduce_array_sum(total_b_grad)
            total_delta_b_grad = self.context.allreduce_array_sum(total_delta_b_grad)

        action = _pack_dynamic_hessian_action_vector(
            dynamics=prepared_state.dynamics,
            direction=dynamics_direction,
            base_a_grad=total_a_grad,
            delta_a_grad=total_delta_a_grad,
            delta_h_grad=total_delta_h_grad,
            delta_b_grad=total_delta_b_grad,
            delta_c_grad=total_delta_c_grad,
        )
        action = action + dynamics_regularization_hessian_action(
            prepared_state.dynamics,
            dynamics_regularization,
            direction_vector,
        )
        return ReducedHessianActionResult(
            base_state=prepared_state,
            direction=direction_vector.copy(),
            action=action,
            decoder_direction=decoder_direction,
        )

    @timed("goattm.problems.ObservationAlignedBestResponseEvaluator.evaluate_reduced_objective_explicit_hessian")
    def evaluate_reduced_objective_explicit_hessian(
        self,
        prepared_state: ReducedObjectivePreparedState,
        decoder_template: QuadraticDecoder,
        regularization: DecoderTikhonovRegularization | None,
        dynamics_regularization: DynamicsTikhonovRegularization | None,
        solve_root: int = 0,
    ) -> ReducedExplicitHessianResult:
        gradient = np.asarray(prepared_state.gradient, dtype=np.float64)
        dimension = int(gradient.shape[0])
        basis = np.eye(dimension, dtype=np.float64)
        hessian = np.zeros((dimension, dimension), dtype=np.float64)
        for column_idx in range(dimension):
            action_result = self.evaluate_reduced_objective_hessian_action(
                prepared_state=prepared_state,
                decoder_template=decoder_template,
                regularization=regularization,
                dynamics_regularization=dynamics_regularization,
                direction=basis[:, column_idx],
                solve_root=solve_root,
            )
            hessian[:, column_idx] = action_result.action
        hessian = 0.5 * (hessian + hessian.T)
        return ReducedExplicitHessianResult(
            base_state=prepared_state,
            hessian=hessian,
        )


def evaluate_reduced_qoi_data_loss_and_gradient(
    dynamics: DynamicsLike,
    decoder_template: QuadraticDecoder,
    manifest: str | Path | NpzSampleManifest,
    max_dt: float,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    regularization: DecoderTikhonovRegularization | None = None,
    dynamics_regularization: DynamicsTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
    solve_root: int = 0,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> ReducedQoiBestResponseResult:
    evaluator = ObservationAlignedBestResponseEvaluator(
        manifest=manifest,
        max_dt=max_dt,
        context=context,
        time_integrator=time_integrator,
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
    return evaluator.evaluate_reduced_data_loss_and_gradient(
        dynamics=dynamics,
        decoder_template=decoder_template,
        regularization=regularization,
        dynamics_regularization=dynamics_regularization,
        solve_root=solve_root,
    )


def evaluate_goam_reduced_qoi_objective_and_gradient(
    dynamics: DynamicsLike,
    decoder_template: QuadraticDecoder,
    manifest: str | Path | NpzSampleManifest,
    max_dt: float,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    regularization: DecoderTikhonovRegularization | None = None,
    dynamics_regularization: DynamicsTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
    solve_root: int = 0,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> ReducedQoiBestResponseResult:
    evaluator = ObservationAlignedBestResponseEvaluator(
        manifest=manifest,
        max_dt=max_dt,
        context=context,
        time_integrator=time_integrator,
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
    return evaluator.evaluate_reduced_objective_and_gradient(
        dynamics=dynamics,
        decoder_template=decoder_template,
        regularization=regularization,
        dynamics_regularization=dynamics_regularization,
        solve_root=solve_root,
    )


def build_reduced_objective_workflow(
    manifest: str | Path | NpzSampleManifest,
    max_dt: float,
    decoder_template: QuadraticDecoder,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    regularization: DecoderTikhonovRegularization | None = None,
    dynamics_regularization: DynamicsTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
    solve_root: int = 0,
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> ReducedObjectiveWorkflow:
    evaluator = ObservationAlignedBestResponseEvaluator(
        manifest=manifest,
        max_dt=max_dt,
        context=context,
        time_integrator=time_integrator,
        dt_shrink=dt_shrink,
        dt_min=dt_min,
        tol=tol,
        max_iter=max_iter,
    )
    return evaluator.build_reduced_objective_workflow(
        decoder_template=decoder_template,
        regularization=regularization,
        dynamics_regularization=dynamics_regularization,
        solve_root=solve_root,
    )


def _assemble_decoder_normal_terms_from_cached_rollout(
    rollout_entry: CachedObservationRollout,
    decoder_output_dimension: int,
    decoder_form: str = "V1V2v",
) -> tuple[np.ndarray, np.ndarray]:
    feature_dim = decoder_feature_dimension(rollout_entry.observed_states.shape[1], decoder_form)
    sample_normal = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    sample_rhs = np.zeros((feature_dim, decoder_output_dimension), dtype=np.float64)
    for state, target, weight in zip(
        rollout_entry.observed_states,
        rollout_entry.sample.qoi_observations,
        rollout_entry.observation_weights,
    ):
        phi = decoder_feature_vector(state, decoder_form)
        sample_normal += weight * np.outer(phi, phi)
        sample_rhs += weight * np.outer(phi, target)
    return sample_normal, sample_rhs


@timed("goattm.problems.solve_decoder_linear_system")
def solve_decoder_linear_system(
    system: DecoderNormalEquationSystem,
    context: DistributedContext,
    solve_root: int = 0,
) -> DecoderNormalEquationSolveResult:
    solution_matrix = np.zeros_like(system.global_rhs, dtype=np.float64)
    if context.rank == solve_root:
        solution_matrix = np.linalg.solve(system.regularized_global_normal_matrix, system.global_rhs)
    solution_matrix = context.bcast_array(solution_matrix, root=solve_root)
    decoder = matrix_to_decoder(system.latent_dimension, system.output_dimension, solution_matrix)
    return DecoderNormalEquationSolveResult(decoder=decoder, system=system, solution_matrix=solution_matrix)


@timed("goattm.problems.solve_decoder_best_response_action_matrix")
def solve_decoder_best_response_action_matrix(
    system: DecoderNormalEquationSystem,
    mixed_action: np.ndarray,
    context: DistributedContext,
    solve_root: int = 0,
) -> np.ndarray:
    local_solution = np.zeros_like(mixed_action, dtype=np.float64)
    if context.rank == solve_root:
        local_solution = np.linalg.solve(system.regularized_global_normal_matrix, -mixed_action)
    return context.bcast_array(local_solution, root=solve_root)


def matrix_to_decoder(latent_dimension: int, output_dimension: int, x_matrix: np.ndarray) -> QuadraticDecoder:
    quad_dim = x_matrix.shape[0] - latent_dimension - 1
    expected_shape = (latent_dimension + quad_dim + 1, output_dimension)
    if x_matrix.shape != expected_shape:
        raise ValueError(f"x_matrix must have shape {expected_shape}, got {x_matrix.shape}")
    v1 = x_matrix[:latent_dimension].T.copy()
    if quad_dim == 0:
        v2 = np.zeros((output_dimension, quadratic_features(np.zeros(latent_dimension, dtype=np.float64)).shape[0]))
        decoder_form = "V1v"
    else:
        v2 = x_matrix[latent_dimension : latent_dimension + quad_dim].T.copy()
        decoder_form = "V1V2v"
    v0 = x_matrix[-1].copy()
    return QuadraticDecoder(v1=v1, v2=v2, v0=v0, form=decoder_form)


def dynamics_parameter_dimension(dynamics: DynamicsLike) -> int:
    return dynamics_parameter_vector(dynamics).shape[0]


def dynamics_parameter_vector(dynamics: DynamicsLike) -> np.ndarray:
    blocks = []
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        blocks.extend([dynamics.s_params, dynamics.w_params])
        blocks.append(dynamics.mu_h)
    elif isinstance(dynamics, LinearDynamics):
        blocks.append(dynamics.a.reshape(-1))
    else:
        blocks.append(dynamics.a.reshape(-1))
        blocks.append(dynamics.mu_h)
    if dynamics.b is not None:
        blocks.append(dynamics.b.reshape(-1))
    blocks.append(dynamics.c)
    return np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in blocks], axis=0)


def dynamics_parameter_key(dynamics: DynamicsLike) -> str:
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        tag = "stabilized"
    elif isinstance(dynamics, LinearDynamics):
        tag = "linear"
    else:
        tag = "general"
    return _hashed_key(tag, dynamics_parameter_vector(dynamics))


def dynamics_from_parameter_vector(dynamics: DynamicsLike, vector: np.ndarray) -> DynamicsLike:
    direction = unpack_dynamics_parameter_vector(dynamics, vector)
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        if direction.s_params is None or direction.w_params is None:
            raise ValueError("Stabilized dynamics vector unpacking did not provide s/w parameters.")
        return StabilizedQuadraticDynamics(
            s_params=direction.s_params,
            w_params=direction.w_params,
            mu_h=direction.mu_h,
            b=direction.b,
            c=direction.c,
        )
    if isinstance(dynamics, LinearDynamics):
        if direction.a is None:
            raise ValueError("Linear dynamics vector unpacking did not provide a matrix.")
        return LinearDynamics(
            a=direction.a,
            b=direction.b,
            c=direction.c,
        )
    if direction.a is None:
        raise ValueError("General dynamics vector unpacking did not provide a matrix.")
    return QuadraticDynamics(
        a=direction.a,
        mu_h=direction.mu_h,
        b=direction.b,
        c=direction.c,
    )


def decoder_parameter_key(decoder: QuadraticDecoder) -> str:
    return _hashed_key("decoder", decoder_parameter_matrix(decoder).reshape(-1))


def regularization_key(regularization: DecoderTikhonovRegularization) -> str:
    return _hashed_key(
        "decoder_reg",
        np.array(
            [regularization.coeff_v1, regularization.coeff_v2, regularization.coeff_v0],
            dtype=np.float64,
        ),
    )


def unpack_dynamics_parameter_vector(dynamics: DynamicsLike, vector: np.ndarray) -> DynamicsParameterDirection:
    flat = np.asarray(vector, dtype=np.float64).reshape(-1)
    expected = dynamics_parameter_dimension(dynamics)
    if flat.shape[0] != expected:
        raise ValueError(f"vector must have length {expected}, got {flat.shape[0]}")
    offset = 0
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        s_size = dynamics.s_params.size
        w_size = dynamics.w_params.size
        s_params = flat[offset : offset + s_size].reshape(dynamics.s_params.shape)
        offset += s_size
        w_params = flat[offset : offset + w_size].reshape(dynamics.w_params.shape)
        offset += w_size
        a = None
    else:
        a_size = dynamics.a.size
        a = flat[offset : offset + a_size].reshape(dynamics.a.shape)
        offset += a_size
        s_params = None
        w_params = None
    if isinstance(dynamics, LinearDynamics):
        mu_h = dynamics.mu_h.copy()
    else:
        mu_h = flat[offset : offset + dynamics.mu_h.size].reshape(dynamics.mu_h.shape)
        offset += dynamics.mu_h.size
    if dynamics.b is not None:
        b = flat[offset : offset + dynamics.b.size].reshape(dynamics.b.shape)
        offset += dynamics.b.size
    else:
        b = None
    c = flat[offset : offset + dynamics.c.size].reshape(dynamics.c.shape)
    return DynamicsParameterDirection(mu_h=mu_h, c=c, a=a, s_params=s_params, w_params=w_params, b=b)


def pack_dynamics_gradient_vector(dynamics: DynamicsLike, gradients: dict[str, np.ndarray]) -> np.ndarray:
    blocks = []
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        blocks.extend([gradients["s_params"], gradients["w_params"]])
        blocks.append(gradients["mu_h"])
    elif isinstance(dynamics, LinearDynamics):
        blocks.append(gradients["a"])
    else:
        blocks.append(gradients["a"])
        blocks.append(gradients["mu_h"])
    if "b" in gradients:
        blocks.append(gradients["b"])
    elif dynamics.b is not None:
        blocks.append(np.zeros_like(dynamics.b, dtype=np.float64))
    blocks.append(gradients["c"])
    return np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in blocks], axis=0)


def rhs_parameter_action(
    dynamics: DynamicsLike,
    direction: DynamicsParameterDirection,
    state: np.ndarray,
    time: float,
    input_function: Callable[[float], np.ndarray] | None = None,
) -> np.ndarray:
    quadratic_term = quadratic_features(state)
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        if direction.s_params is None or direction.w_params is None:
            raise ValueError("direction must provide s_params and w_params for stabilized dynamics.")
        s_matrix = s_params_to_matrix(dynamics.s_params.astype(np.float64), dynamics.dimension)
        ds_matrix = s_params_to_matrix(direction.s_params.astype(np.float64), dynamics.dimension)
        dw_matrix = w_params_to_matrix(direction.w_params.astype(np.float64), dynamics.dimension)
        delta_a = -(ds_matrix @ s_matrix.T + s_matrix @ ds_matrix.T) + dw_matrix
    else:
        if direction.a is None:
            raise ValueError("direction must provide a for general dynamics.")
        delta_a = direction.a
    delta_h = mu_h_to_compressed_h(direction.mu_h.astype(np.float64), dynamics.dimension)
    action = delta_a @ state + delta_h @ quadratic_term + direction.c
    if dynamics.b is not None and direction.b is not None and input_function is not None:
        action = action + direction.b @ np.asarray(input_function(time), dtype=np.float64)
    return action


def decoder_feature_directional_derivative(state: np.ndarray, state_tangent: np.ndarray) -> np.ndarray:
    if state.shape != state_tangent.shape:
        raise ValueError(f"state_tangent must have shape {state.shape}, got {state_tangent.shape}")
    quad_state = quadratic_features(state)
    quad_tangent = np.zeros_like(quad_state, dtype=np.float64)
    index = 0
    for i in range(state.shape[0]):
        for j in range(i + 1):
            quad_tangent[index] = state_tangent[i] * state[j] + state[i] * state_tangent[j]
            index += 1
    out = np.empty(state.shape[0] + quad_state.shape[0] + 1, dtype=np.float64)
    out[: state.shape[0]] = state_tangent
    out[state.shape[0] : state.shape[0] + quad_state.shape[0]] = quad_tangent
    out[-1] = 0.0
    return out


def _dynamics_direction_explicit_matrices(
    dynamics: DynamicsLike,
    direction: DynamicsParameterDirection,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        if direction.s_params is None or direction.w_params is None:
            raise ValueError("direction must provide s_params and w_params for stabilized dynamics.")
        s_matrix = s_params_to_matrix(dynamics.s_params.astype(np.float64), dynamics.dimension)
        ds_matrix = s_params_to_matrix(direction.s_params.astype(np.float64), dynamics.dimension)
        dw_matrix = w_params_to_matrix(direction.w_params.astype(np.float64), dynamics.dimension)
        delta_a = -(ds_matrix @ s_matrix.T + s_matrix @ ds_matrix.T) + dw_matrix
    else:
        if direction.a is None:
            raise ValueError("direction must provide a for general dynamics.")
        delta_a = direction.a
    delta_h = mu_h_to_compressed_h(direction.mu_h.astype(np.float64), dynamics.dimension)
    return delta_a, delta_h


def _rhs_jacobian_direction(
    dynamics: DynamicsLike,
    direction: DynamicsParameterDirection,
    state: np.ndarray,
    state_tangent: np.ndarray,
) -> np.ndarray:
    delta_a, delta_h = _dynamics_direction_explicit_matrices(dynamics, direction)
    return delta_a + quadratic_jacobian_matrix(dynamics.h_matrix, state_tangent) + quadratic_jacobian_matrix(delta_h, state)


def _decoder_jacobian_direction(
    decoder: QuadraticDecoder,
    decoder_direction: QuadraticDecoder,
    state: np.ndarray,
    state_tangent: np.ndarray,
) -> np.ndarray:
    return quadratic_jacobian_matrix(decoder.v2, state_tangent) + decoder_direction.v1 + quadratic_jacobian_matrix(decoder_direction.v2, state)


def _decoder_output_direction(
    decoder_direction: QuadraticDecoder,
    state: np.ndarray,
) -> np.ndarray:
    return decoder_direction.decode(state)


def _observation_state_loss_gradient_direction(
    decoder: QuadraticDecoder,
    decoder_direction: QuadraticDecoder,
    state: np.ndarray,
    state_tangent: np.ndarray,
    residual: np.ndarray,
    weight: float,
) -> np.ndarray:
    jacobian = decoder.jacobian(state)
    delta_jacobian = _decoder_jacobian_direction(decoder, decoder_direction, state, state_tangent)
    delta_residual = jacobian @ state_tangent + _decoder_output_direction(decoder_direction, state)
    return weight * (delta_jacobian.T @ residual + jacobian.T @ delta_residual)


def _compute_incremental_midpoint_discrete_adjoint(
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    decoder_direction: QuadraticDecoder,
    rollout: RolloutResult,
    observation_indices: np.ndarray,
    observation_times: np.ndarray,
    qoi_observations: np.ndarray,
    tangent_states: np.ndarray,
    base_adjoints: np.ndarray,
    base_decoder_partials,
    direction: DynamicsParameterDirection,
) -> np.ndarray:
    del observation_times, qoi_observations
    n_steps = rollout.dt_history.shape[0]
    r = rollout.states.shape[1]
    identity = np.eye(r, dtype=np.float64)
    state_loss_grad_direction = np.zeros_like(rollout.states, dtype=np.float64)
    observed_tangents = tangent_states[observation_indices]
    for local_idx, global_idx in enumerate(observation_indices):
        state_loss_grad_direction[global_idx] = _observation_state_loss_gradient_direction(
            decoder=decoder,
            decoder_direction=decoder_direction,
            state=rollout.states[global_idx],
            state_tangent=observed_tangents[local_idx],
            residual=base_decoder_partials.residuals[local_idx],
            weight=float(base_decoder_partials.quadrature_weights[local_idx]),
        )

    adjoint_tangents = np.zeros_like(base_adjoints, dtype=np.float64)
    if n_steps == 0:
        return adjoint_tangents

    delta_jacobians = []
    jacobians = []
    for step_idx in range(n_steps):
        midpoint_state = 0.5 * (rollout.states[step_idx] + rollout.states[step_idx + 1])
        midpoint_tangent = 0.5 * (tangent_states[step_idx] + tangent_states[step_idx + 1])
        jacobians.append(dynamics.rhs_jacobian(midpoint_state))
        delta_jacobians.append(_rhs_jacobian_direction(dynamics, direction, midpoint_state, midpoint_tangent))

    terminal_lhs = identity - 0.5 * rollout.dt_history[-1] * jacobians[-1].T
    terminal_rhs = -state_loss_grad_direction[-1] + 0.5 * rollout.dt_history[-1] * (delta_jacobians[-1].T @ base_adjoints[-1])
    adjoint_tangents[-1] = np.linalg.solve(terminal_lhs, terminal_rhs)

    for n in range(n_steps - 1, 0, -1):
        lhs = identity - 0.5 * rollout.dt_history[n - 1] * jacobians[n - 1].T
        rhs = (identity + 0.5 * rollout.dt_history[n] * jacobians[n].T) @ adjoint_tangents[n + 1]
        rhs += 0.5 * rollout.dt_history[n] * (delta_jacobians[n].T @ base_adjoints[n + 1])
        rhs -= state_loss_grad_direction[n]
        rhs += 0.5 * rollout.dt_history[n - 1] * (delta_jacobians[n - 1].T @ base_adjoints[n])
        adjoint_tangents[n] = np.linalg.solve(lhs, rhs)

    return adjoint_tangents


def _assemble_dynamic_hessian_action_terms(
    dynamics: DynamicsLike,
    direction: DynamicsParameterDirection,
    rollout: RolloutResult,
    tangent_states: np.ndarray,
    adjoints: np.ndarray,
    adjoint_tangents: np.ndarray,
    input_function: Callable[[float], np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    a_grad = np.zeros((dynamics.dimension, dynamics.dimension), dtype=np.float64)
    delta_a_grad = np.zeros_like(a_grad)
    h_grad = np.zeros_like(dynamics.h_matrix)
    delta_h_grad = np.zeros_like(h_grad)
    c_grad = np.zeros(dynamics.dimension, dtype=np.float64)
    delta_c_grad = np.zeros_like(c_grad)
    b_grad = None if getattr(dynamics, "b", None) is None else np.zeros_like(dynamics.b, dtype=np.float64)
    delta_b_grad = None if getattr(dynamics, "b", None) is None else np.zeros_like(dynamics.b, dtype=np.float64)

    for step_idx in range(rollout.accepted_steps):
        dt = float(rollout.dt_history[step_idx])
        midpoint_state = 0.5 * (rollout.states[step_idx] + rollout.states[step_idx + 1])
        midpoint_tangent = 0.5 * (tangent_states[step_idx] + tangent_states[step_idx + 1])
        midpoint_time = float(rollout.times[step_idx] + 0.5 * dt)
        lam = adjoints[step_idx + 1]
        lam_tangent = adjoint_tangents[step_idx + 1]
        zeta = quadratic_features(midpoint_state)
        zeta_tangent = decoder_feature_directional_derivative(midpoint_state, midpoint_tangent)[
            dynamics.dimension : dynamics.dimension + zeta.shape[0]
        ]

        a_grad += -dt * np.outer(lam, midpoint_state)
        delta_a_grad += -dt * (np.outer(lam_tangent, midpoint_state) + np.outer(lam, midpoint_tangent))
        h_grad += -dt * np.outer(lam, zeta)
        delta_h_grad += -dt * (np.outer(lam_tangent, zeta) + np.outer(lam, zeta_tangent))
        c_grad += -dt * lam
        delta_c_grad += -dt * lam_tangent

        if b_grad is not None and delta_b_grad is not None and input_function is not None:
            p_mid = np.asarray(input_function(midpoint_time), dtype=np.float64)
            b_grad += -dt * np.outer(lam, p_mid)
            delta_b_grad += -dt * np.outer(lam_tangent, p_mid)

    return a_grad, delta_a_grad, h_grad, delta_h_grad, c_grad, delta_c_grad, b_grad, delta_b_grad


def _pack_dynamic_hessian_action_vector(
    dynamics: DynamicsLike,
    direction: DynamicsParameterDirection,
    base_a_grad: np.ndarray,
    delta_a_grad: np.ndarray,
    delta_h_grad: np.ndarray,
    delta_b_grad: np.ndarray | None,
    delta_c_grad: np.ndarray,
) -> np.ndarray:
    gradients: dict[str, np.ndarray] = {"c": delta_c_grad}
    if not isinstance(dynamics, LinearDynamics):
        gradients["mu_h"] = compressed_h_gradient_to_mu_h(delta_h_grad.astype(np.float64), dynamics.dimension)
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        if direction.s_params is None:
            raise ValueError("direction must provide s_params for stabilized dynamics.")
        s_matrix = s_params_to_matrix(dynamics.s_params.astype(np.float64), dynamics.dimension)
        ds_matrix = s_params_to_matrix(direction.s_params.astype(np.float64), dynamics.dimension)
        delta_s_grad_matrix = -(delta_a_grad + delta_a_grad.T) @ s_matrix - (base_a_grad + base_a_grad.T) @ ds_matrix
        delta_s_grad = np.zeros_like(dynamics.s_params, dtype=np.float64)
        idx = 0
        for i in range(dynamics.dimension):
            for j in range(i, dynamics.dimension):
                delta_s_grad[idx] = delta_s_grad_matrix[i, j]
                idx += 1
        delta_w_grad = np.zeros_like(dynamics.w_params, dtype=np.float64)
        idx = 0
        for i in range(dynamics.dimension):
            for j in range(i + 1, dynamics.dimension):
                delta_w_grad[idx] = delta_a_grad[i, j] - delta_a_grad[j, i]
                idx += 1
        gradients["s_params"] = delta_s_grad
        gradients["w_params"] = delta_w_grad
    else:
        gradients["a"] = delta_a_grad
    if delta_b_grad is not None:
        gradients["b"] = delta_b_grad
    return pack_dynamics_gradient_vector(dynamics, gradients)


def _softplus(value: float) -> float:
    if value > 0.0:
        return float(value + np.log1p(np.exp(-value)))
    return float(np.log1p(np.exp(value)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = np.exp(-value)
        return float(1.0 / (1.0 + exp_neg))
    exp_pos = np.exp(value)
    return float(exp_pos / (1.0 + exp_pos))


def symmetric_part_largest_eigenvalue(a_matrix: np.ndarray) -> float:
    sym_a = 0.5 * (np.asarray(a_matrix, dtype=np.float64) + np.asarray(a_matrix, dtype=np.float64).T)
    eigenvalues = np.linalg.eigvalsh(sym_a)
    return float(eigenvalues[-1])


def spectral_abscissa_softplus_penalty(
    a_matrix: np.ndarray,
    coefficient: float,
    alpha: float = 0.0,
) -> float:
    if coefficient == 0.0:
        return 0.0
    z = symmetric_part_largest_eigenvalue(a_matrix) - float(alpha)
    smooth_positive_part = _softplus(z)
    return float(coefficient) * smooth_positive_part * smooth_positive_part


def spectral_abscissa_softplus_gradient_matrix(
    a_matrix: np.ndarray,
    coefficient: float,
    alpha: float = 0.0,
) -> np.ndarray:
    a_array = np.asarray(a_matrix, dtype=np.float64)
    if coefficient == 0.0:
        return np.zeros_like(a_array, dtype=np.float64)
    sym_a = 0.5 * (a_array + a_array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym_a)
    z = float(eigenvalues[-1]) - float(alpha)
    scale = float(coefficient) * 2.0 * _softplus(z) * _sigmoid(z)
    dominant = eigenvectors[:, -1]
    return scale * np.outer(dominant, dominant)


def _spectral_regularization_gradient_vector(
    dynamics: DynamicsLike,
    regularization: DynamicsTikhonovRegularization,
) -> np.ndarray:
    a_grad = spectral_abscissa_softplus_gradient_matrix(
        dynamics.a,
        coefficient=regularization.coeff_spectral_abscissa,
        alpha=regularization.spectral_abscissa_alpha,
    )
    gradients: dict[str, np.ndarray] = {"c": np.zeros_like(dynamics.c, dtype=np.float64)}
    if not isinstance(dynamics, LinearDynamics):
        gradients["mu_h"] = np.zeros_like(dynamics.mu_h, dtype=np.float64)
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        s_grad, w_grad = dynamics.pullback_a_gradient_to_stabilized_params(a_grad)
        gradients["s_params"] = s_grad
        gradients["w_params"] = w_grad
    else:
        gradients["a"] = a_grad
    if dynamics.b is not None:
        gradients["b"] = np.zeros_like(dynamics.b, dtype=np.float64)
    return pack_dynamics_gradient_vector(dynamics, gradients)


def dynamics_regularization_loss(
    dynamics: DynamicsLike,
    regularization: DynamicsTikhonovRegularization,
) -> float:
    loss = 0.0
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        loss += regularization.coeff_s * float(np.sum(dynamics.s_params**2))
        loss += regularization.coeff_w * float(np.sum(dynamics.w_params**2))
    else:
        loss += regularization.coeff_a * float(np.sum(dynamics.a**2))
    if not isinstance(dynamics, LinearDynamics):
        loss += regularization.coeff_mu_h * float(np.sum(dynamics.mu_h**2))
    if dynamics.b is not None:
        loss += regularization.coeff_b * float(np.sum(dynamics.b**2))
    loss += regularization.coeff_c * float(np.sum(dynamics.c**2))
    loss += spectral_abscissa_softplus_penalty(
        dynamics.a,
        coefficient=regularization.coeff_spectral_abscissa,
        alpha=regularization.spectral_abscissa_alpha,
    )
    return float(loss)


def dynamics_regularization_gradient_vector(
    dynamics: DynamicsLike,
    regularization: DynamicsTikhonovRegularization,
) -> np.ndarray:
    gradients: dict[str, np.ndarray] = {"c": 2.0 * regularization.coeff_c * np.asarray(dynamics.c, dtype=np.float64)}
    if not isinstance(dynamics, LinearDynamics):
        gradients["mu_h"] = 2.0 * regularization.coeff_mu_h * np.asarray(dynamics.mu_h, dtype=np.float64)
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        gradients["s_params"] = 2.0 * regularization.coeff_s * np.asarray(dynamics.s_params, dtype=np.float64)
        gradients["w_params"] = 2.0 * regularization.coeff_w * np.asarray(dynamics.w_params, dtype=np.float64)
    else:
        gradients["a"] = 2.0 * regularization.coeff_a * np.asarray(dynamics.a, dtype=np.float64)
    if regularization.coeff_spectral_abscissa > 0.0:
        spectral_a_grad = spectral_abscissa_softplus_gradient_matrix(
            dynamics.a,
            coefficient=regularization.coeff_spectral_abscissa,
            alpha=regularization.spectral_abscissa_alpha,
        )
        if isinstance(dynamics, StabilizedQuadraticDynamics):
            spectral_s_grad, spectral_w_grad = dynamics.pullback_a_gradient_to_stabilized_params(spectral_a_grad)
            gradients["s_params"] = gradients["s_params"] + spectral_s_grad
            gradients["w_params"] = gradients["w_params"] + spectral_w_grad
        else:
            gradients["a"] = gradients["a"] + spectral_a_grad
    if dynamics.b is not None:
        gradients["b"] = 2.0 * regularization.coeff_b * np.asarray(dynamics.b, dtype=np.float64)
    return pack_dynamics_gradient_vector(dynamics, gradients)


def dynamics_regularization_hessian_action(
    dynamics: DynamicsLike,
    regularization: DynamicsTikhonovRegularization,
    direction_vector: np.ndarray,
) -> np.ndarray:
    direction = unpack_dynamics_parameter_vector(dynamics, direction_vector)
    blocks = []
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        if direction.s_params is None or direction.w_params is None:
            raise ValueError("direction must provide s_params and w_params for stabilized dynamics.")
        blocks.append(2.0 * regularization.coeff_s * np.asarray(direction.s_params, dtype=np.float64).reshape(-1))
        blocks.append(2.0 * regularization.coeff_w * np.asarray(direction.w_params, dtype=np.float64).reshape(-1))
    else:
        if direction.a is None:
            raise ValueError("direction must provide a for general dynamics.")
        blocks.append(2.0 * regularization.coeff_a * np.asarray(direction.a, dtype=np.float64).reshape(-1))
    if not isinstance(dynamics, LinearDynamics):
        blocks.append(2.0 * regularization.coeff_mu_h * np.asarray(direction.mu_h, dtype=np.float64).reshape(-1))
    if dynamics.b is not None:
        if direction.b is None:
            raise ValueError("direction must provide b when dynamics has input matrix b.")
        blocks.append(2.0 * regularization.coeff_b * np.asarray(direction.b, dtype=np.float64).reshape(-1))
    blocks.append(2.0 * regularization.coeff_c * np.asarray(direction.c, dtype=np.float64).reshape(-1))
    action = np.concatenate(blocks, axis=0)
    if regularization.coeff_spectral_abscissa > 0.0:
        parameters = dynamics_parameter_vector(dynamics)
        direction_flat = np.asarray(direction_vector, dtype=np.float64).reshape(-1)
        direction_norm = float(np.linalg.norm(direction_flat))
        if direction_norm > 0.0:
            step = 1.0e-6 * (1.0 + float(np.linalg.norm(parameters))) / direction_norm
            plus = dynamics_from_parameter_vector(dynamics, parameters + step * direction_flat)
            minus = dynamics_from_parameter_vector(dynamics, parameters - step * direction_flat)
            spectral_action = (
                _spectral_regularization_gradient_vector(plus, regularization)
                - _spectral_regularization_gradient_vector(minus, regularization)
            ) / (2.0 * step)
            action = action + spectral_action
    return action


def decoder_regularization_loss(
    decoder: QuadraticDecoder,
    regularization: DecoderTikhonovRegularization,
) -> float:
    return float(
        regularization.coeff_v1 * np.sum(decoder.v1**2)
        + (0.0 if decoder.form == "V1v" else regularization.coeff_v2 * np.sum(decoder.v2**2))
        + regularization.coeff_v0 * np.sum(decoder.v0**2)
    )


def stack_decoder_gradient_matrix(decoder_gradients: dict[str, np.ndarray], decoder_form: str = "V1V2v") -> np.ndarray:
    if decoder_form == "V1v":
        return np.vstack(
            [
                decoder_gradients["v1"].T.astype(np.float64, copy=False),
                decoder_gradients["v0"].reshape(1, -1).astype(np.float64, copy=False),
            ]
        )
    return np.vstack(
        [
            decoder_gradients["v1"].T.astype(np.float64, copy=False),
            decoder_gradients["v2"].T.astype(np.float64, copy=False),
            decoder_gradients["v0"].reshape(1, -1).astype(np.float64, copy=False),
        ]
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


def _hashed_key(tag: str, vector: np.ndarray) -> str:
    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(tag.encode("utf-8"))
    contiguous = np.ascontiguousarray(vector, dtype=np.float64)
    hasher.update(np.array(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()
