from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension, skew_symmetric_dimension, upper_triangular_dimension
from goattm.data import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest
from goattm.models import LinearDynamics, QuadraticDecoder, StabilizedQuadraticDynamics
from goattm.problems import (
    DecoderTikhonovRegularization,
    DynamicsTikhonovRegularization,
    build_reduced_objective_workflow,
    dynamics_from_parameter_vector,
    dynamics_parameter_vector,
)
from goattm.runtime import DistributedContext
from goattm.solvers import rollout_to_observation_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taylor tests for GOATTM dynamic/decoder model forms.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "module_test" / "model_forms_taylor" / "outputs")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--first-order-min-slope", type=float, default=1.7)
    parser.add_argument("--second-order-min-slope", type=float, default=1.7)
    return parser.parse_args()


def make_dynamics(dynamic_form: str, rng: np.random.Generator, r: int, input_dim: int):
    if dynamic_form == "ABc":
        a = -0.35 * np.eye(r) + 0.04 * rng.normal(size=(r, r))
        b = 0.08 * rng.normal(size=(r, input_dim))
        c = 0.03 * rng.normal(size=r)
        return LinearDynamics(a=a, b=b, c=c)
    if dynamic_form == "AHBc":
        s_params = np.zeros(upper_triangular_dimension(r), dtype=np.float64)
        idx = 0
        for i in range(r):
            for _ in range(i, r):
                s_params[idx] = 0.45 if idx in {0, 2} else 0.0
                idx += 1
        w_params = 0.04 * rng.normal(size=skew_symmetric_dimension(r))
        mu_h = 0.015 * rng.normal(size=mu_h_dimension(r))
        b = 0.08 * rng.normal(size=(r, input_dim))
        c = 0.03 * rng.normal(size=r)
        return StabilizedQuadraticDynamics(s_params=s_params, w_params=w_params, mu_h=mu_h, b=b, c=c)
    raise ValueError(f"Unsupported dynamic_form {dynamic_form!r}")


def make_decoder(decoder_form: str, rng: np.random.Generator, r: int, output_dim: int) -> QuadraticDecoder:
    v1 = rng.normal(scale=0.7, size=(output_dim, r))
    v2 = rng.normal(scale=0.15, size=(output_dim, compressed_quadratic_dimension(r)))
    if decoder_form == "V1v":
        v2[:] = 0.0
    v0 = rng.normal(scale=0.05, size=output_dim)
    return QuadraticDecoder(v1=v1, v2=v2, v0=v0, form=decoder_form)


def input_function_factory(sample_idx: int):
    def input_function(t: float) -> np.ndarray:
        return np.array(
            [
                np.sin(0.7 * t + 0.2 * sample_idx),
                np.cos(0.5 * t - 0.1 * sample_idx),
            ],
            dtype=np.float64,
        )

    return input_function


def write_dataset(
    root: Path,
    dynamics,
    decoder: QuadraticDecoder,
    rng: np.random.Generator,
    sample_count: int = 3,
) -> NpzSampleManifest:
    root.mkdir(parents=True, exist_ok=True)
    observation_times = np.linspace(0.0, 0.4, 6)
    sample_paths = []
    sample_ids = []
    for sample_idx in range(sample_count):
        u0 = 0.15 * rng.normal(size=dynamics.dimension)
        input_function = input_function_factory(sample_idx)
        rollout, observation_indices = rollout_to_observation_times(
            dynamics=dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=0.02,
            input_function=input_function,
            time_integrator="rk4",
        )
        states = rollout.states[observation_indices]
        qoi = np.vstack([decoder.decode(state) for state in states])
        input_times = observation_times.copy()
        input_values = np.vstack([input_function(float(t)) for t in input_times])
        sample_id = f"sample_{sample_idx:03d}"
        path = root / f"{sample_id}.npz"
        save_npz_qoi_sample(
            path,
            NpzQoiSample(
                sample_id=sample_id,
                observation_times=observation_times,
                u0=u0,
                qoi_observations=qoi,
                input_times=input_times,
                input_values=input_values,
            ),
        )
        sample_paths.append(path.name)
        sample_ids.append(sample_id)
    manifest = NpzSampleManifest(root_dir=root, sample_paths=tuple(Path(path) for path in sample_paths), sample_ids=tuple(sample_ids))
    save_npz_sample_manifest(root / "manifest.npz", manifest)
    return manifest


def estimate_slope(values: list[float], eps_values: np.ndarray) -> float:
    clipped = np.maximum(np.asarray(values, dtype=np.float64), 1e-300)
    return float(np.polyfit(np.log(eps_values), np.log(clipped), deg=1)[0])


def run_case(dynamic_form: str, decoder_form: str, output_dir: Path, rng: np.random.Generator) -> dict[str, float | str]:
    r = 2
    input_dim = 2
    output_dim = 3
    truth_dynamics = make_dynamics(dynamic_form, rng, r, input_dim)
    truth_decoder = make_decoder(decoder_form, rng, r, output_dim)
    manifest = write_dataset(output_dir / f"data_{dynamic_form}_{decoder_form}", truth_dynamics, truth_decoder, rng)

    base_vector = dynamics_parameter_vector(truth_dynamics)
    perturb = rng.normal(size=base_vector.shape)
    perturb /= np.linalg.norm(perturb)
    initial_dynamics = dynamics_from_parameter_vector(truth_dynamics, base_vector + 0.08 * perturb)
    decoder_template = make_decoder(decoder_form, rng, r, output_dim)
    workflow = build_reduced_objective_workflow(
        manifest=manifest,
        max_dt=0.02,
        decoder_template=decoder_template,
        time_integrator="rk4",
        regularization=DecoderTikhonovRegularization(coeff_v1=1e-6, coeff_v2=1e-6, coeff_v0=1e-6),
        dynamics_regularization=DynamicsTikhonovRegularization(
            coeff_a=1e-6,
            coeff_s=1e-6,
            coeff_w=1e-6,
            coeff_mu_h=1e-6,
            coeff_b=1e-6,
            coeff_c=1e-6,
        ),
        context=DistributedContext(),
    )
    prepared = workflow.prepare(initial_dynamics)
    x0 = dynamics_parameter_vector(initial_dynamics)
    direction = rng.normal(size=x0.shape)
    direction /= np.linalg.norm(direction)
    h_action = workflow.evaluate_hessian_action_from_prepared_state(prepared, direction).action

    eps_values = np.array([2.0**(-k) for k in range(4, 10)], dtype=np.float64)
    value_first_remainders = []
    gradient_zero_remainders = []
    gradient_first_remainders = []
    for eps in eps_values:
        trial = dynamics_from_parameter_vector(initial_dynamics, x0 + eps * direction)
        value = workflow.evaluate_objective(trial)
        gradient = workflow.evaluate_gradient(trial)
        first_order_value_model = prepared.objective_value + eps * float(np.dot(prepared.gradient, direction))
        gradient_delta = gradient - prepared.gradient
        value_first_remainders.append(abs(value - first_order_value_model))
        gradient_zero_remainders.append(float(np.linalg.norm(gradient_delta)))
        gradient_first_remainders.append(float(np.linalg.norm(gradient_delta - eps * h_action)))

    value_first_slope = estimate_slope(value_first_remainders, eps_values)
    gradient_zero_slope = estimate_slope(gradient_zero_remainders, eps_values)
    hessian_action_slope = estimate_slope(gradient_first_remainders, eps_values)
    return {
        "dynamic_form": dynamic_form,
        "decoder_form": decoder_form,
        "value_first_order_slope": value_first_slope,
        "gradient_zero_order_slope": gradient_zero_slope,
        "hessian_action_slope": hessian_action_slope,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    for dynamic_form in ("ABc", "AHBc"):
        for decoder_form in ("V1v", "V1V2v"):
            record = run_case(dynamic_form, decoder_form, args.output_dir, rng)
            records.append(record)
            print(
                f"{dynamic_form}-{decoder_form}: "
                f"value_first_order_slope={record['value_first_order_slope']:.3f}, "
                f"gradient_zero_order_slope={record['gradient_zero_order_slope']:.3f}, "
                f"hessian_action_slope={record['hessian_action_slope']:.3f}",
                flush=True,
            )

    bad_first = [record for record in records if float(record["value_first_order_slope"]) < args.first_order_min_slope]
    bad_second = [record for record in records if float(record["hessian_action_slope"]) < args.second_order_min_slope]
    if bad_first or bad_second:
        raise SystemExit(f"Taylor test failed: bad_first={bad_first}, bad_second={bad_second}")


if __name__ == "__main__":
    main()
