from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT")
sys.path.insert(0, str(REPO / "src"))

from goattm.problems.qoi_dataset_problem import evaluate_npz_qoi_dataset_loss_and_gradients  # noqa: E402
from goattm.train.plot_qoi_predictions import PlotConfig, load_run_artifacts  # noqa: E402

RUN_DIR = Path(
    "/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/outputs/"
    "production_1344_v1v_init_v1v2_train_lbfgs/"
    "20260513_141424_swe_skewcp_n1344_lbfgs_v1vinit_v1v2_r50_R100_r50_R100/"
    "swe_skewcp_smoke_r50_R100_20260513_163947/runs/"
    "swe_skewcp_lbfgs_r50_R100_ntrain1344_ntest384_20260513_164044_e3d910ce"
)


def latest_iteration_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints").glob("iter_*.npz"))
    return candidates[-1]


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def main() -> None:
    checkpoint_path = latest_iteration_checkpoint(RUN_DIR)
    artifacts = load_run_artifacts(
        PlotConfig(
            run_dir=RUN_DIR.resolve(),
            checkpoint_mode="latest",
            checkpoint_path=checkpoint_path.resolve(),
            train_manifest_path=None,
            test_manifest_path=None,
            output_root=None,
            max_pages_per_split=1,
            use_raw_qoi_scale=False,
        )
    )
    # This is a quick probe, not the full 1344-sample gradient. It is enough to
    # verify whether each block has a live adjoint signal.
    train_subset = artifacts.train_manifest.subset_by_indices(tuple(range(8)))
    result = evaluate_npz_qoi_dataset_loss_and_gradients(
        dynamics=artifacts.dynamics,
        decoder=artifacts.decoder,
        manifest=train_subset,
        max_dt=artifacts.max_dt,
        time_integrator=artifacts.time_integrator,
        dt_shrink=0.5,
        dt_min=1e-5,
        tol=1e-12,
        max_iter=40,
    )
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "sample_count": result.global_sample_count,
        "total_loss": float(result.total_loss),
        "dynamics_gradient_norms": {key: norm(value) for key, value in result.dynamics_gradients.items()},
        "decoder_gradient_norms": {key: norm(value) for key, value in result.decoder_gradients.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
