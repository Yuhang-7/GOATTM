#!/usr/bin/env python3
"""Sweep coupled regularizer scales around the current NS100 Matérn52 OpInf choice."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_order_opinf_initialize_ns100_matern52 import (
    DEFAULT_GRID_STATE_ROOT,
    DEFAULT_MANIFEST,
    compressed_quadratic_dimension,
    compute_pod_basis,
    compute_state_mean_and_scale,
    fit_decoder,
    fit_dynamics,
    load_payloads,
    mu_h_dimension,
    quadratic_feature_matrix,
    save_oldgoam_parameters,
    transformed_states,
)


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--grid-state-root", type=Path, default=DEFAULT_GRID_STATE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=112)
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--scale-list", default="1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3")
    parser.add_argument("--base-reg-a", type=float, default=1e-8)
    parser.add_argument("--base-reg-h", type=float, default=1.0)
    parser.add_argument("--base-reg-b", type=float, default=10.0)
    parser.add_argument("--base-reg-c", type=float, default=1e-6)
    parser.add_argument("--base-reg-decoder", type=float, default=1e-4)
    parser.add_argument(
        "--scale-target",
        choices=("other", "all"),
        default="other",
        help="'other' keeps reg_b fixed and scales reg_a/reg_h/reg_c/reg_decoder; 'all' scales every block.",
    )
    parser.add_argument("--save-best", action="store_true")
    return parser.parse_args()


def dynamics_validation_residual(payloads, basis, mean, scale, a, h, b, c) -> float:
    target = 0.0
    residual = 0.0
    for payload in payloads:
        z = basis.T @ transformed_states(payload, mean, scale)
        dt = np.diff(payload.times)
        zdot = (z[:, 1:] - z[:, :-1]) / dt[None, :]
        zmid = 0.5 * (z[:, 1:] + z[:, :-1])
        bmid = 0.5 * (payload.inputs[1:] + payload.inputs[:-1]).T
        pred = a @ zmid + h @ quadratic_feature_matrix(zmid) + b @ bmid + c[:, None]
        target += float(np.sum(zdot * zdot))
        residual += float(np.sum((pred - zdot) ** 2))
    return float(math.sqrt(residual / target)) if target > 0.0 else float("nan")


def decoder_validation_residual(payloads, basis, mean, scale, v1, v2, v0) -> float:
    target = 0.0
    residual = 0.0
    for payload in payloads:
        z = basis.T @ transformed_states(payload, mean, scale)
        pred = v1 @ z + v2 @ quadratic_feature_matrix(z) + v0[:, None]
        q = payload.qoi.T
        target += float(np.sum(q * q))
        residual += float(np.sum((pred - q) ** 2))
    return float(math.sqrt(residual / target)) if target > 0.0 else float("nan")


def main() -> None:
    args = parse_args()
    payloads = load_payloads(
        manifest_path=args.manifest_path,
        max_samples=args.max_samples,
        time_stride=args.time_stride,
        state_dof_stride=1,
        grid_state_root=args.grid_state_root,
    )
    if args.train_samples <= 0 or args.train_samples >= len(payloads):
        raise ValueError("--train-samples must be positive and smaller than --max-samples")

    train = payloads[: args.train_samples]
    valid = payloads[args.train_samples :]
    mean, state_scale = compute_state_mean_and_scale(train, center=False, scale_maxabs=False)
    basis, eigvals, captured = compute_pod_basis(train, rank=args.rank, mean=mean, scale=state_scale)

    rows = []
    best = None
    best_params = None
    for multiplier in parse_float_list(args.scale_list):
        if args.scale_target == "all":
            reg_a = args.base_reg_a * multiplier
            reg_h = args.base_reg_h * multiplier
            reg_b = args.base_reg_b * multiplier
            reg_c = args.base_reg_c * multiplier
            reg_decoder = args.base_reg_decoder * multiplier
        else:
            reg_a = args.base_reg_a * multiplier
            reg_h = args.base_reg_h * multiplier
            reg_b = args.base_reg_b
            reg_c = args.base_reg_c * multiplier
            reg_decoder = args.base_reg_decoder * multiplier

        a, h, b, c, dyn_train = fit_dynamics(
            train,
            basis=basis,
            mean=mean,
            scale=state_scale,
            dynamic_form="AHBc",
            regs=(reg_a, reg_h, reg_b, reg_c),
        )
        v1, v2, v0, dec_train = fit_decoder(
            train,
            basis=basis,
            mean=mean,
            scale=state_scale,
            decoder_form="V1V2v",
            regs=(reg_decoder, reg_decoder, reg_decoder),
        )
        dyn_valid = dynamics_validation_residual(valid, basis, mean, state_scale, a, h, b, c)
        dec_valid = decoder_validation_residual(valid, basis, mean, state_scale, v1, v2, v0)
        score = dyn_valid + dec_valid
        row = {
            "multiplier": float(multiplier),
            "reg_a": float(reg_a),
            "reg_h": float(reg_h),
            "reg_b": float(reg_b),
            "reg_c": float(reg_c),
            "reg_decoder": float(reg_decoder),
            "dynamics_train_relative_residual": float(dyn_train),
            "decoder_train_relative_residual": float(dec_train),
            "dynamics_valid_relative_residual": float(dyn_valid),
            "decoder_valid_relative_residual": float(dec_valid),
            "score": float(score),
        }
        rows.append(row)
        if best is None or score < best["score"]:
            best = row
            best_params = (a, h, b, c, v1, v2, v0)
        print(json.dumps(row, sort_keys=True), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "pipeline": "grid_state_opinf_coupled_regularizer_scale_sweep",
        "rank": int(args.rank),
        "scale_target": args.scale_target,
        "sample_count": len(payloads),
        "train_samples": len(train),
        "validation_samples": len(valid),
        "time_stride": int(args.time_stride),
        "state_dimension": int(train[0].states.shape[0]),
        "input_dimension": int(train[0].inputs.shape[1]),
        "output_dimension": int(train[0].qoi.shape[1]),
        "pod_captured_energy": float(captured),
        "pod_eigenvalues_first": [float(v) for v in eigvals[: min(20, eigvals.shape[0])]],
        "base_regularization": {
            "reg_a": args.base_reg_a,
            "reg_h": args.base_reg_h,
            "reg_b": args.base_reg_b,
            "reg_c": args.base_reg_c,
            "reg_decoder": args.base_reg_decoder,
        },
        "best": best,
        "rows": rows,
    }
    summary_path = args.output_root / f"coupled_scale_summary_r{args.rank}_{args.scale_target}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if args.save_best and best_params is not None:
        a, h, b, c, v1, v2, v0 = best_params
        best_dir = args.output_root / "best_oldgoam_init"
        muf_path, mug_path = save_oldgoam_parameters(
            output_dir=best_dir,
            rank=args.rank,
            a=a,
            h=h,
            b=b,
            c=c,
            v1=v1,
            v2=v2,
            v0=v0,
            overwrite=True,
        )
        best["muf_path"] = str(muf_path)
        best["mug_path"] = str(mug_path)
        best["muf_length"] = int(train[0].qoi.shape[1] * args.rank + train[0].qoi.shape[1] * compressed_quadratic_dimension(args.rank) + train[0].qoi.shape[1])
        best["mug_length"] = int(args.rank * args.rank + mu_h_dimension(args.rank) + args.rank * train[0].inputs.shape[1] + args.rank)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    multipliers = np.asarray([row["multiplier"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    dyn = np.asarray([row["dynamics_valid_relative_residual"] for row in rows], dtype=float)
    dec = np.asarray([row["decoder_valid_relative_residual"] for row in rows], dtype=float)
    order = np.argsort(multipliers)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogx(multipliers[order], scores[order], marker="o", label="score")
    ax.semilogx(multipliers[order], dyn[order], marker="s", label="dynamics")
    ax.semilogx(multipliers[order], dec[order], marker="^", label="decoder")
    ax.set_xlabel("coupled multiplier")
    ax.set_ylabel("relative residual")
    ax.set_title(f"Rank {args.rank}: coupled regularizer scaling ({args.scale_target})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = args.output_root / f"coupled_scale_r{args.rank}_{args.scale_target}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(json.dumps({"best": best, "summary_path": str(summary_path), "plot_path": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
