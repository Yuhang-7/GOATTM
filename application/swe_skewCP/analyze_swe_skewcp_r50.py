from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUN_DIR = Path(
    "/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/outputs/"
    "production_1344_v1v_init_v1v2_train_lbfgs/"
    "20260513_141424_swe_skewcp_n1344_lbfgs_v1vinit_v1v2_r50_R100_r50_R100/"
    "swe_skewcp_smoke_r50_R100_20260513_163947/runs/"
    "swe_skewcp_lbfgs_r50_R100_ntrain1344_ntest384_20260513_164044_e3d910ce"
)


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def main() -> None:
    init = np.load(RUN_DIR / "initial_parameters.npz")
    ckpts = sorted((RUN_DIR / "checkpoints").glob("iter_*.npz"))
    latest_path = ckpts[-1]
    latest = np.load(latest_path)
    best = np.load(RUN_DIR / "checkpoints" / "best.npz")

    blocks = [
        ("a_matrix", "A"),
        ("skew_u", "skew_u"),
        ("skew_v", "skew_v"),
        ("skew_z", "skew_z"),
        ("b_matrix", "B"),
        ("c_vector", "c"),
        ("decoder_v1", "decoder_v1"),
        ("decoder_v2", "decoder_v2"),
        ("decoder_v0", "decoder_v0"),
    ]
    rows = []
    for key, label in blocks:
        if key not in init.files or key not in latest.files:
            continue
        x0 = np.asarray(init[key], dtype=np.float64)
        xl = np.asarray(latest[key], dtype=np.float64)
        xb = np.asarray(best[key], dtype=np.float64)
        rows.append(
            {
                "key": key,
                "label": label,
                "initial_norm": norm(x0),
                "latest_norm": norm(xl),
                "best_norm": norm(xb),
                "delta_latest_norm": norm(xl - x0),
                "delta_best_norm": norm(xb - x0),
                "relative_delta_latest": norm(xl - x0) / max(norm(x0), 1e-300),
                "max_abs_delta_latest": float(np.max(np.abs(xl - x0))),
            }
        )

    records = [json.loads(line) for line in (RUN_DIR / "metrics.jsonl").read_text().splitlines() if line.strip()]
    print(json.dumps({"latest_checkpoint": str(latest_path), "latest_metric": records[-1], "blocks": rows}, indent=2))


if __name__ == "__main__":
    main()
