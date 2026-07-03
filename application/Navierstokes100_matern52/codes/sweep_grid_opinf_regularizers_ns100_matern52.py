#!/usr/bin/env python3
"""Sweep OpInf regularizers for NS100 Matérn52 grid-state initialization."""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from full_order_opinf_initialize_ns100_matern52 import (
    DEFAULT_GRID_STATE_ROOT,
    DEFAULT_MANIFEST,
    add_ridge_blocks,
    compressed_h_to_mu_h,
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
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--dynamic-form", choices=("ABc", "AHBc"), default="AHBc")
    parser.add_argument("--decoder-form", choices=("V1v", "V1V2v"), default="V1V2v")
    parser.add_argument("--reg-a-list", default="1e-10,1e-8,1e-6")
    parser.add_argument("--reg-h-list", default="1e-10,1e-8,1e-6,1e-4")
    parser.add_argument("--reg-b-list", default="1e-10,1e-8,1e-6")
    parser.add_argument("--reg-c-list", default="1e-10,1e-8")
    parser.add_argument("--reg-decoder-list", default="1e-6,1e-4,1e-2")
    parser.add_argument("--center-state", action="store_true")
    parser.add_argument("--scale-state-maxabs", action="store_true")
    parser.add_argument(
        "--score-mode",
        choices=("one_step", "rollout"),
        default="one_step",
        help="Use validation one-step residuals or free-rollout QoI error to select the saved best regularizer.",
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


def rollout_validation_qoi_error(payloads, basis, mean, scale, a, h, b, c, v1, v2, v0) -> tuple[float, int]:
    target = 0.0
    residual = 0.0
    unstable = 0
    for payload in payloads:
        z_true = basis.T @ transformed_states(payload, mean, scale)
        z = z_true[:, 0].copy()
        pred_q = np.empty((payload.qoi.shape[0], payload.qoi.shape[1]), dtype=np.float64)
        for k in range(payload.qoi.shape[0]):
            pred_q[k, :] = v1 @ z + v2 @ quadratic_feature_matrix(z[:, None])[:, 0] + v0
            if k == payload.qoi.shape[0] - 1:
                break
            dt = float(payload.times[k + 1] - payload.times[k])
            inp = payload.inputs[k]
            dz = a @ z + h @ quadratic_feature_matrix(z[:, None])[:, 0] + b @ inp + c
            z = z + dt * dz
            if not np.all(np.isfinite(z)) or np.linalg.norm(z) > 1.0e6:
                unstable += 1
                pred_q[k + 1 :, :] = np.nan
                break
        q = payload.qoi
        good = np.isfinite(pred_q)
        target += float(np.sum(q[good] * q[good]))
        residual += float(np.sum((pred_q[good] - q[good]) ** 2))
    return (float(math.sqrt(residual / target)) if target > 0.0 else float("inf"), unstable)


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

    mean, scale = compute_state_mean_and_scale(train, center=args.center_state, scale_maxabs=args.scale_state_maxabs)
    basis, eigvals, captured = compute_pod_basis(train, rank=args.rank, mean=mean, scale=scale)

    reg_as = parse_float_list(args.reg_a_list)
    reg_hs = parse_float_list(args.reg_h_list)
    reg_bs = parse_float_list(args.reg_b_list)
    reg_cs = parse_float_list(args.reg_c_list)
    reg_ds = parse_float_list(args.reg_decoder_list)

    rows = []
    best = None
    best_params = None
    for reg_a, reg_h, reg_b, reg_c, reg_d in product(reg_as, reg_hs, reg_bs, reg_cs, reg_ds):
        a, h, b, c, dyn_train = fit_dynamics(
            train,
            basis=basis,
            mean=mean,
            scale=scale,
            dynamic_form=args.dynamic_form,
            regs=(reg_a, reg_h, reg_b, reg_c),
        )
        v1, v2, v0, dec_train = fit_decoder(
            train,
            basis=basis,
            mean=mean,
            scale=scale,
            decoder_form=args.decoder_form,
            regs=(reg_d, reg_d, reg_d),
        )
        dyn_valid = dynamics_validation_residual(valid, basis, mean, scale, a, h, b, c)
        dec_valid = decoder_validation_residual(valid, basis, mean, scale, v1, v2, v0)
        rollout_valid, unstable_count = rollout_validation_qoi_error(valid, basis, mean, scale, a, h, b, c, v1, v2, v0)
        if args.score_mode == "one_step":
            score = dyn_valid + dec_valid
        else:
            score = rollout_valid + 0.1 * dyn_valid + 0.1 * dec_valid + 10.0 * unstable_count
        row = {
            "reg_a": reg_a,
            "reg_h": reg_h,
            "reg_b": reg_b,
            "reg_c": reg_c,
            "reg_decoder": reg_d,
            "dynamics_train_relative_residual": dyn_train,
            "decoder_train_relative_residual": dec_train,
            "dynamics_valid_relative_residual": dyn_valid,
            "decoder_valid_relative_residual": dec_valid,
            "rollout_valid_qoi_relative_error": rollout_valid,
            "unstable_validation_rollouts": unstable_count,
            "score": score,
        }
        rows.append(row)
        if best is None or score < best["score"]:
            best = row
            best_params = (a, h, b, c, v1, v2, v0)
        print(json.dumps(row, sort_keys=True), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "pipeline": "grid_state_opinf_regularizer_sweep",
        "rank": args.rank,
        "sample_count": len(payloads),
        "train_samples": len(train),
        "validation_samples": len(valid),
        "time_stride": args.time_stride,
        "state_dimension": int(train[0].states.shape[0]),
        "input_dimension": int(train[0].inputs.shape[1]),
        "output_dimension": int(train[0].qoi.shape[1]),
        "pod_captured_energy": float(captured),
        "pod_eigenvalues_first": [float(v) for v in eigvals[: min(20, eigvals.shape[0])]],
        "center_state": bool(args.center_state),
        "scale_state_maxabs": bool(args.scale_state_maxabs),
        "state_scale": float(scale),
        "score_mode": args.score_mode,
        "best": best,
        "rows": rows,
    }
    (args.output_root / f"sweep_summary_r{args.rank}.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.save_best and best_params is not None and best is not None:
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
        (args.output_root / f"sweep_summary_r{args.rank}.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({"best": best, "summary_path": str(args.output_root / f"sweep_summary_r{args.rank}.json")}, indent=2))


if __name__ == "__main__":
    main()
