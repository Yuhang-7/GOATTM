from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from goattm.data import NpzSampleManifest, load_npz_sample_manifest
from goattm.models.general_quadratic_dynamics import GeneralQuadraticDynamics
from goattm.models.linear_dynamics import LinearDynamics
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.models.skew_cp_quadratic_dynamics import SkewCPQuadraticDynamics
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems import (
    DecoderTikhonovRegularization,
    DynamicsTikhonovRegularization,
    ObservationAlignedBestResponseEvaluator,
    dynamics_parameter_dimension,
    joint_parameter_dimension,
)
from goattm.runtime.distributed import DistributedContext
from goattm.solvers import TimeIntegrator


DynamicsLike = LinearDynamics | GeneralQuadraticDynamics | QuadraticDynamics | SkewCPQuadraticDynamics | StabilizedQuadraticDynamics
HessianCaseName = Literal[
    "general_joint_gn",
    "general_varpro_gn",
    "energy_joint_gn",
    "energy_varpro_gn",
]
JointDecoderMode = Literal["provided", "best_response"]


@dataclass(frozen=True)
class HessianLandscapeConfig:
    """Numerical controls for the four-case GN Hessian landscape experiment."""

    max_dt: float
    time_integrator: TimeIntegrator = "lagged_midpoint"
    k: int = 8
    which_values: tuple[str, ...] = ("LA", "SA")
    eigsh_tol: float = 1.0e-8
    eigsh_maxiter: int | None = None
    random_seed: int = 20260710
    joint_decoder_mode: JointDecoderMode = "provided"
    cases: tuple[HessianCaseName, ...] = (
        "general_joint_gn",
        "general_varpro_gn",
        "energy_joint_gn",
        "energy_varpro_gn",
    )
    store_eigenvectors: bool = False
    solve_root: int = 0


@dataclass(frozen=True)
class HessianCaseOperator:
    name: HessianCaseName
    dynamics_form: Literal["general", "energy"]
    varpro: bool
    dimension: int
    operator: LinearOperator
    prepare_seconds: float
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class HessianEigensolveResult:
    which: str
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    elapsed_seconds: float
    matvec_count: int
    eigenvectors: np.ndarray | None = None


@dataclass(frozen=True)
class HessianCaseResult:
    case: HessianCaseName
    dimension: int
    prepare_seconds: float
    eigensolves: tuple[HessianEigensolveResult, ...]
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class FourHessianLandscapeResult:
    cases: tuple[HessianCaseResult, ...]
    metadata: dict[str, str | int | float | bool]


class _CountingLinearOperator(LinearOperator):
    def __init__(self, dimension: int, matvec: Callable[[np.ndarray], np.ndarray]):
        self._matvec_impl = matvec
        self.matvec_count = 0
        super().__init__(dtype=np.float64, shape=(dimension, dimension))

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        self.matvec_count += 1
        return np.asarray(self._matvec_impl(np.asarray(vector, dtype=np.float64)), dtype=np.float64)


def energy_to_general_dynamics(dynamics: QuadraticDynamics) -> GeneralQuadraticDynamics:
    return GeneralQuadraticDynamics(
        a=dynamics.a.astype(np.float64, copy=True),
        h_matrix=dynamics.h_matrix.astype(np.float64, copy=True),
        c=dynamics.c.astype(np.float64, copy=True),
        b=None if dynamics.b is None else dynamics.b.astype(np.float64, copy=True),
    )


def as_energy_quadratic_dynamics(dynamics: DynamicsLike) -> QuadraticDynamics:
    if isinstance(dynamics, QuadraticDynamics):
        return dynamics
    if isinstance(dynamics, StabilizedQuadraticDynamics):
        return dynamics.explicit_dynamics
    raise ValueError(
        "Energy-preserving cases require QuadraticDynamics or StabilizedQuadraticDynamics. "
        f"Got {type(dynamics).__name__}."
    )


def zero_decoder_template_like(decoder: QuadraticDecoder) -> QuadraticDecoder:
    return QuadraticDecoder(
        v1=np.zeros_like(decoder.v1, dtype=np.float64),
        v2=np.zeros_like(decoder.v2, dtype=np.float64),
        v0=np.zeros_like(decoder.v0, dtype=np.float64),
        form=decoder.form,
    )


def build_four_hessian_case_operators(
    manifest: str | Path | NpzSampleManifest,
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    config: HessianLandscapeConfig,
    decoder_regularization: DecoderTikhonovRegularization | None = None,
    dynamics_regularization: DynamicsTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
) -> tuple[HessianCaseOperator, ...]:
    if context is None:
        context = DistributedContext.from_comm()
    if decoder_regularization is None:
        decoder_regularization = DecoderTikhonovRegularization()
    if dynamics_regularization is None:
        dynamics_regularization = DynamicsTikhonovRegularization()
    if isinstance(manifest, (str, Path)):
        manifest = load_npz_sample_manifest(manifest)

    evaluator = ObservationAlignedBestResponseEvaluator(
        manifest=manifest,
        max_dt=config.max_dt,
        context=context,
        time_integrator=config.time_integrator,
    )
    decoder_template = zero_decoder_template_like(decoder)
    energy_dynamics = as_energy_quadratic_dynamics(dynamics)
    general_dynamics = energy_to_general_dynamics(energy_dynamics)

    operators: list[HessianCaseOperator] = []
    for case_name in config.cases:
        case_dynamics: DynamicsLike
        dynamics_form: Literal["general", "energy"]
        varpro = "varpro" in case_name
        if case_name.startswith("general"):
            case_dynamics = general_dynamics
            dynamics_form = "general"
        elif case_name.startswith("energy"):
            case_dynamics = energy_dynamics
            dynamics_form = "energy"
        else:
            raise ValueError(f"Unknown Hessian case {case_name!r}")

        t0 = time.perf_counter()
        if varpro:
            workflow = evaluator.build_reduced_objective_workflow(
                decoder_template=decoder_template,
                regularization=decoder_regularization,
                dynamics_regularization=dynamics_regularization,
                solve_root=config.solve_root,
            )
            prepared = workflow.prepare(case_dynamics)
            dimension = dynamics_parameter_dimension(case_dynamics)

            def matvec(vector: np.ndarray, workflow=workflow, prepared=prepared) -> np.ndarray:
                return workflow.evaluate_gn_varpro_hessian_action_from_prepared_state(
                    prepared, vector
                ).action

            decoder_source = "varpro_best_response"
        else:
            joint_decoder = decoder
            decoder_source = "provided"
            if config.joint_decoder_mode == "best_response":
                best_response = evaluator.solve_decoder_best_response(
                    dynamics=case_dynamics,
                    decoder_template=decoder_template,
                    regularization=decoder_regularization,
                    solve_root=config.solve_root,
                )
                joint_decoder = best_response.decoder
                decoder_source = "best_response"
            dimension = joint_parameter_dimension(case_dynamics, joint_decoder)

            def matvec(vector: np.ndarray, case_dynamics=case_dynamics, joint_decoder=joint_decoder) -> np.ndarray:
                return evaluator.evaluate_joint_gauss_newton_hessian_action(
                    dynamics=case_dynamics,
                    decoder=joint_decoder,
                    direction=vector,
                    regularization=decoder_regularization,
                    dynamics_regularization=dynamics_regularization,
                ).action

        prepare_seconds = time.perf_counter() - t0
        counting_operator = _CountingLinearOperator(dimension, matvec)
        operators.append(
            HessianCaseOperator(
                name=case_name,
                dynamics_form=dynamics_form,
                varpro=varpro,
                dimension=dimension,
                operator=counting_operator,
                prepare_seconds=prepare_seconds,
                metadata={
                    "case": case_name,
                    "dynamics_form": dynamics_form,
                    "varpro": varpro,
                    "decoder_source": decoder_source,
                    "dimension": int(dimension),
                    "prepare_seconds": float(prepare_seconds),
                },
            )
        )
    return tuple(operators)


def eigensolve_hessian_case(
    case_operator: HessianCaseOperator,
    which: str,
    k: int,
    tol: float,
    maxiter: int | None,
    random_seed: int,
    store_eigenvectors: bool,
) -> HessianEigensolveResult:
    if k <= 0:
        return HessianEigensolveResult(
            which=which,
            eigenvalues=np.zeros(0, dtype=np.float64),
            residual_norms=np.zeros(0, dtype=np.float64),
            elapsed_seconds=0.0,
            matvec_count=0,
            eigenvectors=None,
        )
    dimension = int(case_operator.dimension)
    if dimension <= 1:
        raise ValueError(f"Cannot run eigsh on dimension {dimension}; need dimension > 1.")
    effective_k = min(int(k), dimension - 1)
    rng = np.random.default_rng(int(random_seed))
    v0 = rng.standard_normal(dimension)
    v0_norm = float(np.linalg.norm(v0))
    if v0_norm == 0.0:
        v0[0] = 1.0
    else:
        v0 /= v0_norm

    operator = case_operator.operator
    before_count = getattr(operator, "matvec_count", 0)
    t0 = time.perf_counter()
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=effective_k,
        which=which,
        tol=float(tol),
        maxiter=maxiter,
        v0=v0,
    )
    elapsed = time.perf_counter() - t0
    after_count = getattr(operator, "matvec_count", before_count)
    order = np.argsort(eigenvalues)
    if which in {"LA", "LM"}:
        order = order[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    residual_norms = np.empty(effective_k, dtype=np.float64)
    for idx in range(effective_k):
        vector = eigenvectors[:, idx]
        residual = operator @ vector - eigenvalues[idx] * vector
        residual_norms[idx] = float(np.linalg.norm(residual))
    return HessianEigensolveResult(
        which=str(which),
        eigenvalues=eigenvalues,
        residual_norms=residual_norms,
        elapsed_seconds=float(elapsed),
        matvec_count=int(after_count - before_count),
        eigenvectors=eigenvectors if store_eigenvectors else None,
    )


def run_four_hessian_landscape(
    manifest: str | Path | NpzSampleManifest,
    dynamics: DynamicsLike,
    decoder: QuadraticDecoder,
    config: HessianLandscapeConfig,
    decoder_regularization: DecoderTikhonovRegularization | None = None,
    dynamics_regularization: DynamicsTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
    output_dir: str | Path | None = None,
) -> FourHessianLandscapeResult:
    if context is None:
        context = DistributedContext.from_comm()
    operators = build_four_hessian_case_operators(
        manifest=manifest,
        dynamics=dynamics,
        decoder=decoder,
        config=config,
        decoder_regularization=decoder_regularization,
        dynamics_regularization=dynamics_regularization,
        context=context,
    )
    case_results: list[HessianCaseResult] = []
    for case_operator in operators:
        eigensolves = []
        for which_idx, which in enumerate(config.which_values):
            eigensolves.append(
                eigensolve_hessian_case(
                    case_operator=case_operator,
                    which=which,
                    k=config.k,
                    tol=config.eigsh_tol,
                    maxiter=config.eigsh_maxiter,
                    random_seed=config.random_seed + 1009 * which_idx,
                    store_eigenvectors=config.store_eigenvectors,
                )
            )
        case_results.append(
            HessianCaseResult(
                case=case_operator.name,
                dimension=case_operator.dimension,
                prepare_seconds=case_operator.prepare_seconds,
                eigensolves=tuple(eigensolves),
                metadata=case_operator.metadata,
            )
        )
    result = FourHessianLandscapeResult(
        cases=tuple(case_results),
        metadata={
            "max_dt": float(config.max_dt),
            "time_integrator": str(config.time_integrator),
            "k": int(config.k),
            "which_values": ",".join(config.which_values),
            "joint_decoder_mode": str(config.joint_decoder_mode),
            "rank": int(context.rank),
            "size": int(context.size),
        },
    )
    if output_dir is not None and context.rank == config.solve_root:
        write_hessian_landscape_result(output_dir, result)
    return result


def write_hessian_landscape_result(output_dir: str | Path, result: FourHessianLandscapeResult) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": result.metadata,
        "cases": [
            {
                "case": case.case,
                "dimension": case.dimension,
                "prepare_seconds": case.prepare_seconds,
                "metadata": case.metadata,
                "eigensolves": [
                    {
                        "which": eig.which,
                        "eigenvalues": eig.eigenvalues.tolist(),
                        "residual_norms": eig.residual_norms.tolist(),
                        "elapsed_seconds": eig.elapsed_seconds,
                        "matvec_count": eig.matvec_count,
                    }
                    for eig in case.eigensolves
                ],
            }
            for case in result.cases
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    arrays: dict[str, np.ndarray] = {}
    for case in result.cases:
        for eig in case.eigensolves:
            prefix = f"{case.case}_{eig.which}"
            arrays[f"{prefix}_eigenvalues"] = eig.eigenvalues
            arrays[f"{prefix}_residual_norms"] = eig.residual_norms
            if eig.eigenvectors is not None:
                arrays[f"{prefix}_eigenvectors"] = eig.eigenvectors
    np.savez(root / "eigenvalues.npz", **arrays)


def load_checkpoint_models(path: str | Path) -> tuple[QuadraticDecoder, DynamicsLike]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        decoder_form = str(data["decoder_form"].item()) if "decoder_form" in data.files else "V1V2v"
        decoder = QuadraticDecoder(
            v1=np.asarray(data["decoder_v1"], dtype=np.float64),
            v2=np.asarray(data["decoder_v2"], dtype=np.float64),
            v0=np.asarray(data["decoder_v0"], dtype=np.float64),
            form=decoder_form,
        )
        dynamics_type = str(data["dynamics_type"].item())
        c = np.asarray(data["c_vector"], dtype=np.float64)
        b = None if "b_matrix" not in data.files else np.asarray(data["b_matrix"], dtype=np.float64)
        if dynamics_type == "stabilized":
            dynamics: DynamicsLike = StabilizedQuadraticDynamics(
                s_params=np.asarray(data["s_params"], dtype=np.float64),
                w_params=np.asarray(data["w_params"], dtype=np.float64),
                mu_h=np.asarray(data["mu_h"], dtype=np.float64),
                c=c,
                b=b,
            )
        elif dynamics_type == "linear":
            dynamics = LinearDynamics(a=np.asarray(data["a_matrix"], dtype=np.float64), c=c, b=b)
        elif dynamics_type in {"general", "energy_quadratic"}:
            dynamics = QuadraticDynamics(
                a=np.asarray(data["a_matrix"], dtype=np.float64),
                mu_h=np.asarray(data["mu_h"], dtype=np.float64),
                c=c,
                b=b,
            )
        elif dynamics_type == "general_quadratic":
            dynamics = GeneralQuadraticDynamics(
                a=np.asarray(data["a_matrix"], dtype=np.float64),
                h_matrix=np.asarray(data["h_matrix"], dtype=np.float64),
                c=c,
                b=b,
            )
        elif dynamics_type == "skew_cp":
            dynamics = SkewCPQuadraticDynamics(
                a=np.asarray(data["a_matrix"], dtype=np.float64),
                skew_u=np.asarray(data["skew_u"], dtype=np.float64),
                skew_v=np.asarray(data["skew_v"], dtype=np.float64),
                skew_z=np.asarray(data["skew_z"], dtype=np.float64),
                c=c,
                b=b,
            )
        else:
            raise ValueError(f"Unsupported dynamics_type {dynamics_type!r} in checkpoint {path}")
    return decoder, dynamics


def run_four_hessian_landscape_from_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: HessianLandscapeConfig,
    decoder_regularization: DecoderTikhonovRegularization | None = None,
    dynamics_regularization: DynamicsTikhonovRegularization | None = None,
    context: DistributedContext | None = None,
) -> FourHessianLandscapeResult:
    decoder, dynamics = load_checkpoint_models(checkpoint_path)
    return run_four_hessian_landscape(
        manifest=manifest_path,
        dynamics=dynamics,
        decoder=decoder,
        config=config,
        decoder_regularization=decoder_regularization,
        dynamics_regularization=dynamics_regularization,
        context=context,
        output_dir=output_dir,
    )


def _parse_which_values(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--which must contain at least one eigsh selector.")
    return values


def _parse_cases(raw: str) -> tuple[HessianCaseName, ...]:
    if raw.strip().lower() == "all":
        return (
            "general_joint_gn",
            "general_varpro_gn",
            "energy_joint_gn",
            "energy_varpro_gn",
        )
    allowed = {"general_joint_gn", "general_varpro_gn", "energy_joint_gn", "energy_varpro_gn"}
    cases = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(cases) - allowed)
    if unknown:
        raise ValueError(f"Unknown Hessian cases: {unknown}")
    return cases  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute four matrix-free GOATTM Gauss-Newton Hessian spectra.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-dt", type=float, required=True)
    parser.add_argument("--time-integrator", default="lagged_midpoint")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--which", default="LA,SA", help="Comma-separated eigsh selectors, e.g. LA,SA.")
    parser.add_argument("--cases", default="all", help="Comma-separated cases or 'all'.")
    parser.add_argument("--eigsh-tol", type=float, default=1.0e-8)
    parser.add_argument("--eigsh-maxiter", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=20260710)
    parser.add_argument("--joint-decoder-mode", choices=("provided", "best_response"), default="provided")
    parser.add_argument("--store-eigenvectors", action="store_true")
    parser.add_argument("--solve-root", type=int, default=0)
    parser.add_argument("--reg-v1", type=float, default=0.0)
    parser.add_argument("--reg-v2", type=float, default=0.0)
    parser.add_argument("--reg-v0", type=float, default=0.0)
    parser.add_argument("--reg-a", type=float, default=0.0)
    parser.add_argument("--reg-h", type=float, default=0.0)
    parser.add_argument("--reg-b", type=float, default=0.0)
    parser.add_argument("--reg-c", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> FourHessianLandscapeResult:
    args = build_arg_parser().parse_args(argv)
    config = HessianLandscapeConfig(
        max_dt=float(args.max_dt),
        time_integrator=args.time_integrator,
        k=int(args.k),
        which_values=_parse_which_values(str(args.which)),
        eigsh_tol=float(args.eigsh_tol),
        eigsh_maxiter=args.eigsh_maxiter,
        random_seed=int(args.random_seed),
        joint_decoder_mode=args.joint_decoder_mode,
        cases=_parse_cases(str(args.cases)),
        store_eigenvectors=bool(args.store_eigenvectors),
        solve_root=int(args.solve_root),
    )
    decoder_regularization = DecoderTikhonovRegularization(
        coeff_v1=float(args.reg_v1),
        coeff_v2=float(args.reg_v2),
        coeff_v0=float(args.reg_v0),
    )
    dynamics_regularization = DynamicsTikhonovRegularization(
        coeff_a=float(args.reg_a),
        coeff_mu_h=float(args.reg_h),
        coeff_b=float(args.reg_b),
        coeff_c=float(args.reg_c),
    )
    context = DistributedContext.from_comm()
    result = run_four_hessian_landscape_from_checkpoint(
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        config=config,
        decoder_regularization=decoder_regularization,
        dynamics_regularization=dynamics_regularization,
        context=context,
    )
    if context.rank == config.solve_root:
        print(json.dumps({"output_dir": str(args.output_dir), "case_count": len(result.cases)}, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
