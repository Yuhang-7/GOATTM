from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension
from goattm.models.quadratic_decoder import QuadraticDecoder
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics
from goattm.problems.reduced_qoi_best_response import DecoderTikhonovRegularization, DynamicsTikhonovRegularization
from goattm.runtime.distributed import DistributedContext
from goattm.train import QuotientTrustRegionConfig, ReducedQoiTrainer, ReducedQoiTrainerConfig
from run_quotient_vp_metric_pullback_test import build_dataset


def main() -> None:
    context = DistributedContext.from_comm()
    rng = np.random.default_rng(20260529)
    r = 2
    dq = 2
    truth_dynamics = StabilizedQuadraticDynamics(
        s_params=np.array([0.42, 0.03, 0.35], dtype=np.float64),
        w_params=np.array([0.07], dtype=np.float64),
        mu_h=0.02 * rng.standard_normal(mu_h_dimension(r)),
        b=np.array([[0.20], [-0.08]], dtype=np.float64),
        c=np.array([0.015, -0.012], dtype=np.float64),
    )
    initial_dynamics = StabilizedQuadraticDynamics(
        s_params=truth_dynamics.s_params + 0.08 * rng.standard_normal(truth_dynamics.s_params.shape),
        w_params=truth_dynamics.w_params + 0.08 * rng.standard_normal(truth_dynamics.w_params.shape),
        mu_h=truth_dynamics.mu_h + 0.08 * rng.standard_normal(truth_dynamics.mu_h.shape),
        b=truth_dynamics.b + 0.08 * rng.standard_normal(truth_dynamics.b.shape),
        c=truth_dynamics.c + 0.08 * rng.standard_normal(truth_dynamics.c.shape),
    )
    truth_decoder = QuadraticDecoder(
        v1=0.25 * rng.standard_normal((dq, r)),
        v2=0.10 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
        v0=0.04 * rng.standard_normal(dq),
    )
    decoder_template = QuadraticDecoder(
        v1=np.zeros_like(truth_decoder.v1),
        v2=np.zeros_like(truth_decoder.v2),
        v0=np.zeros_like(truth_decoder.v0),
    )
    observation_times = np.linspace(0.0, 0.16, 5)

    with tempfile.TemporaryDirectory(prefix="goattm_qtr_smoke_") as tmpdir:
        root = Path(tmpdir)
        manifest_path = build_dataset(root, truth_dynamics, truth_decoder, 3, observation_times, rng)
        config = ReducedQoiTrainerConfig(
            output_dir=root / "runs",
            time_integrator="implicit_midpoint",
            run_name_prefix="quotient_trust_region_smoke",
            optimizer="quotient_trust_region",
            max_iterations=2,
            checkpoint_every=1,
            log_every=1,
            test_every=1,
            quotient_trust_region=QuotientTrustRegionConfig(
                initial_radius=1.0,
                max_radius=10.0,
                metric_ridge=1e-6,
                hessian_damping=1e-5,
                max_dense_dimension=40,
            ),
        )
        trainer = ReducedQoiTrainer(
            train_manifest=manifest_path,
            test_manifest=None,
            decoder_template=decoder_template,
            regularization=DecoderTikhonovRegularization(coeff_v1=1e-5, coeff_v2=1e-5, coeff_v0=1e-5),
            dynamics_regularization=DynamicsTikhonovRegularization(
                coeff_s=1e-5,
                coeff_w=1e-5,
                coeff_mu_h=1e-5,
                coeff_b=1e-5,
                coeff_c=1e-5,
            ),
            max_dt=float(observation_times[1] - observation_times[0]),
            config=config,
            context=context,
            dt_shrink=0.5,
            dt_min=1e-12,
            tol=1e-12,
            max_iter_newton=30,
        )
        result = trainer.train(initial_dynamics)
        if context.rank != 0:
            return
        records = [json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()]

    initial_obj = float(records[0]["train_objective"])
    final_obj = float(records[-1]["train_objective"])
    summary = {
        "test": "quotient_trust_region_smoke",
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "iterations_recorded": len(records),
        "optimizer": "quotient_trust_region",
    }
    print(json.dumps(summary, indent=2))
    if not final_obj < initial_obj:
        raise AssertionError(f"quotient_trust_region did not decrease objective: {initial_obj} -> {final_obj}")


if __name__ == "__main__":
    main()
