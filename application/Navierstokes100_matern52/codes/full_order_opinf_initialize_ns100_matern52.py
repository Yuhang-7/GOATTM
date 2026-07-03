from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.core.parametrization import (  # noqa: E402
    compressed_h_to_mu_h,
    compressed_quadratic_dimension,
    mu_h_dimension,
    quadratic_features,
)
from goattm.data import load_npz_qoi_sample, load_npz_sample_manifest  # noqa: E402


APP_ROOT = Path("/work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52")
DEFAULT_MANIFEST = APP_ROOT / "data" / "processed_data_qoi5_input40" / "manifest.npz"
DEFAULT_OUTPUT_DIR = APP_ROOT / "initial" / "AHBc_fullstate"
DEFAULT_GRID_STATE_ROOT = Path("/scratch/08667/yuuuhang/NS_matern_input/grid_state_re100_matern52_nx80_ny24_448")


@dataclass(frozen=True)
class SamplePayload:
    sample_id: str
    times: np.ndarray
    qoi: np.ndarray
    inputs: np.ndarray
    states: np.ndarray
    source_npz_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-order-state OpInf initialization for NS100 Matérn52. This follows the old "
            "GOAM_clean idea: POD full-order state snapshots, regress latent dynamics from "
            "projected full states, regress decoder from latent states to QoIs, and save "
            "oldGOAM-compatible muf_r.npy/mug_r.npy."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--grid-state-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--state-dof-stride", type=int, default=1)
    parser.add_argument("--dynamic-form", choices=("ABc", "AHBc"), default="AHBc")
    parser.add_argument("--decoder-form", choices=("V1v", "V1V2v"), default="V1V2v")
    parser.add_argument("--reg-a", type=float, default=1e-9)
    parser.add_argument("--reg-h", type=float, default=1e-9)
    parser.add_argument("--reg-b", type=float, default=1e-9)
    parser.add_argument("--reg-c", type=float, default=1e-9)
    parser.add_argument("--reg-decoder-v1", type=float, default=1e-3)
    parser.add_argument("--reg-decoder-v2", type=float, default=1e-3)
    parser.add_argument("--reg-decoder-v0", type=float, default=1e-3)
    parser.add_argument(
        "--center-state",
        action="store_true",
        help="Subtract the training full-state mean before POD/regression. Old GOAM active code did not do this by default.",
    )
    parser.add_argument(
        "--scale-state-maxabs",
        action="store_true",
        help="Divide centered/raw full states by their global max abs before POD/regression.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata_path(sample, key: str) -> Path:
    if sample.metadata is None or key not in sample.metadata:
        raise ValueError(f"Sample {sample.sample_id} is missing metadata key {key!r}")
    return Path(str(sample.metadata[key]))


def _case_id_from_source_path(source_path: Path) -> int:
    stem = source_path.stem
    if not stem.startswith("output_"):
        raise ValueError(f"Cannot parse case id from source path {source_path}")
    return int(stem.split("_", 1)[1])


def load_payloads(
    manifest_path: Path,
    max_samples: int,
    time_stride: int,
    state_dof_stride: int,
    grid_state_root: Path | None = None,
) -> list[SamplePayload]:
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if time_stride <= 0 or state_dof_stride <= 0:
        raise ValueError("--time-stride and --state-dof-stride must be positive")
    manifest = load_npz_sample_manifest(manifest_path)
    payloads: list[SamplePayload] = []
    for sample_path in manifest.absolute_paths():
        sample = load_npz_qoi_sample(sample_path)
        source_path = _metadata_path(sample, "source_npz_path")
        case_id = _case_id_from_source_path(source_path)
        if grid_state_root is None:
            if not source_path.exists():
                continue
            with np.load(source_path, allow_pickle=True) as source:
                if "state_saved" not in source.files or int(source["state_saved"]) != 1:
                    continue
                states = np.asarray(source["state_snapshots"], dtype=np.float64)
        else:
            grid_path = grid_state_root / f"grid_state_{case_id:06d}.npz"
            if not grid_path.exists():
                continue
            with np.load(grid_path, allow_pickle=True) as grid:
                grid_state = np.asarray(grid["grid_state"], dtype=np.float64)
            states = grid_state.reshape(grid_state.shape[0], -1).T
        times = np.asarray(sample.observation_times, dtype=np.float64)
        qoi = np.asarray(sample.qoi_observations, dtype=np.float64)
        if sample.input_values is None:
            raise ValueError(f"Sample {sample.sample_id} has no input_values")
        inputs = np.asarray(sample.input_values, dtype=np.float64)
        if states.shape[1] != times.shape[0]:
            raise ValueError(f"{sample.sample_id}: states {states.shape} do not match times {times.shape}")
        index = np.arange(0, times.shape[0], time_stride, dtype=int)
        if index[-1] != times.shape[0] - 1:
            index = np.concatenate([index, np.asarray([times.shape[0] - 1], dtype=int)])
        payloads.append(
            SamplePayload(
                sample_id=sample.sample_id,
                times=times[index],
                qoi=qoi[index],
                inputs=inputs[index],
                states=states[::state_dof_stride, :][:, index],
                source_npz_path=source_path,
            )
        )
        if len(payloads) >= max_samples:
            break
    if not payloads:
        raise RuntimeError("No samples with saved full-order state_snapshots were found.")
    return payloads


def compute_state_mean_and_scale(payloads: list[SamplePayload], center: bool, scale_maxabs: bool) -> tuple[np.ndarray, float]:
    d = payloads[0].states.shape[0]
    mean = np.zeros(d, dtype=np.float64)
    count = 0
    if center:
        for payload in payloads:
            mean += np.sum(payload.states, axis=1)
            count += payload.states.shape[1]
        mean /= float(count)
    scale = 1.0
    if scale_maxabs:
        max_abs = 0.0
        for payload in payloads:
            centered = payload.states - mean[:, None]
            max_abs = max(max_abs, float(np.max(np.abs(centered))))
        if max_abs > 0.0:
            scale = max_abs
    return mean, scale


def transformed_states(payload: SamplePayload, mean: np.ndarray, scale: float) -> np.ndarray:
    return (payload.states - mean[:, None]) / scale


def compute_pod_basis(payloads: list[SamplePayload], rank: int, mean: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    d = payloads[0].states.shape[0]
    cov = np.zeros((d, d), dtype=np.float64)
    total_energy = 0.0
    for payload in payloads:
        x = transformed_states(payload, mean, scale)
        cov += x @ x.T
        total_energy += float(np.sum(x * x))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    if rank > eigvecs.shape[1]:
        raise ValueError(f"rank={rank} exceeds state dimension {eigvecs.shape[1]}")
    basis = eigvecs[:, :rank].copy()
    captured = float(np.sum(eigvals[:rank]) / total_energy) if total_energy > 0.0 else float("nan")
    return basis, eigvals, captured


def quadratic_feature_matrix(z: np.ndarray) -> np.ndarray:
    s = compressed_quadratic_dimension(z.shape[0])
    out = np.empty((s, z.shape[1]), dtype=np.float64)
    for j in range(z.shape[1]):
        out[:, j] = quadratic_features(z[:, j])
    return out


def add_ridge_blocks(matrix: np.ndarray, block_sizes: list[int], regs: list[float]) -> None:
    start = 0
    for size, reg in zip(block_sizes, regs):
        end = start + size
        if reg != 0.0:
            matrix[start:end, start:end] += float(reg) * np.eye(size)
        start = end


def fit_dynamics(
    payloads: list[SamplePayload],
    basis: np.ndarray,
    mean: np.ndarray,
    scale: float,
    dynamic_form: str,
    regs: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rank = basis.shape[1]
    input_dim = payloads[0].inputs.shape[1]
    s = compressed_quadratic_dimension(rank)
    if dynamic_form == "AHBc":
        width = rank + s + input_dim + 1
        block_sizes = [rank, s, input_dim, 1]
        block_regs = [regs[0], regs[1], regs[2], regs[3]]
    else:
        width = rank + input_dim + 1
        block_sizes = [rank, input_dim, 1]
        block_regs = [regs[0], regs[2], regs[3]]
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros((rank, width), dtype=np.float64)
    target_energy = 0.0
    residual_energy = 0.0
    for payload in payloads:
        x = transformed_states(payload, mean, scale)
        z = basis.T @ x
        dt = np.diff(payload.times)
        if not np.all(dt > 0.0):
            raise ValueError(f"{payload.sample_id} has non-increasing times")
        zdot = (z[:, 1:] - z[:, :-1]) / dt[None, :]
        zmid = 0.5 * (z[:, 1:] + z[:, :-1])
        bmid = 0.5 * (payload.inputs[1:] + payload.inputs[:-1]).T
        if dynamic_form == "AHBc":
            phi = np.vstack([zmid, quadratic_feature_matrix(zmid), bmid, np.ones((1, zmid.shape[1]))])
        else:
            phi = np.vstack([zmid, bmid, np.ones((1, zmid.shape[1]))])
        gram += phi @ phi.T
        rhs += zdot @ phi.T
        target_energy += float(np.sum(zdot * zdot))
    reg_gram = gram.copy()
    add_ridge_blocks(reg_gram, block_sizes, block_regs)
    theta = np.linalg.solve(reg_gram, rhs.T).T
    marker = 0
    a = theta[:, marker : marker + rank].copy()
    marker += rank
    if dynamic_form == "AHBc":
        h = theta[:, marker : marker + s].copy()
        marker += s
    else:
        h = np.zeros((rank, s), dtype=np.float64)
    b = theta[:, marker : marker + input_dim].copy()
    marker += input_dim
    c = theta[:, marker].copy()
    for payload in payloads:
        x = transformed_states(payload, mean, scale)
        z = basis.T @ x
        dt = np.diff(payload.times)
        zdot = (z[:, 1:] - z[:, :-1]) / dt[None, :]
        zmid = 0.5 * (z[:, 1:] + z[:, :-1])
        bmid = 0.5 * (payload.inputs[1:] + payload.inputs[:-1]).T
        pred = a @ zmid + h @ quadratic_feature_matrix(zmid) + b @ bmid + c[:, None]
        residual_energy += float(np.sum((pred - zdot) ** 2))
    rel_resid = float(np.sqrt(residual_energy / target_energy)) if target_energy > 0.0 else float("nan")
    return a, h, b, c, rel_resid


def fit_decoder(
    payloads: list[SamplePayload],
    basis: np.ndarray,
    mean: np.ndarray,
    scale: float,
    decoder_form: str,
    regs: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rank = basis.shape[1]
    output_dim = payloads[0].qoi.shape[1]
    s = compressed_quadratic_dimension(rank)
    width = rank + s + 1
    gram = np.zeros((width, width), dtype=np.float64)
    rhs = np.zeros((output_dim, width), dtype=np.float64)
    target_energy = 0.0
    residual_energy = 0.0
    for payload in payloads:
        z = basis.T @ transformed_states(payload, mean, scale)
        phi = np.vstack([z, quadratic_feature_matrix(z), np.ones((1, z.shape[1]))])
        q = payload.qoi.T
        gram += phi @ phi.T
        rhs += q @ phi.T
        target_energy += float(np.sum(q * q))
    reg_gram = gram.copy()
    add_ridge_blocks(reg_gram, [rank, s, 1], [regs[0], regs[1], regs[2]])
    theta = np.linalg.solve(reg_gram, rhs.T).T
    v1 = theta[:, :rank].copy()
    v2 = theta[:, rank : rank + s].copy()
    v0 = theta[:, rank + s].copy()
    if decoder_form == "V1v":
        v2[:, :] = 0.0
    for payload in payloads:
        z = basis.T @ transformed_states(payload, mean, scale)
        pred = v1 @ z + v2 @ quadratic_feature_matrix(z) + v0[:, None]
        q = payload.qoi.T
        residual_energy += float(np.sum((pred - q) ** 2))
    rel_resid = float(np.sqrt(residual_energy / target_energy)) if target_energy > 0.0 else float("nan")
    return v1, v2, v0, rel_resid


def save_oldgoam_parameters(
    output_dir: Path,
    rank: int,
    a: np.ndarray,
    h: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v0: np.ndarray,
    overwrite: bool,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mug_path = output_dir / f"mug_{rank}.npy"
    muf_path = output_dir / f"muf_{rank}.npy"
    if not overwrite and (mug_path.exists() or muf_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing {mug_path} or {muf_path}; pass --overwrite")
    mu_h = compressed_h_to_mu_h(h, rank)
    mug = np.concatenate([a.reshape(-1), mu_h.reshape(-1), b.reshape(-1), c.reshape(-1)])
    muf = np.concatenate([v1.reshape(-1), v2.reshape(-1), v0.reshape(-1)])
    np.save(mug_path, mug)
    np.save(muf_path, muf)
    return muf_path, mug_path


def main() -> None:
    args = parse_args()
    payloads = load_payloads(
        manifest_path=args.manifest_path,
        max_samples=args.max_samples,
        time_stride=args.time_stride,
        state_dof_stride=args.state_dof_stride,
        grid_state_root=args.grid_state_root,
    )
    rank = int(args.rank)
    state_dim = payloads[0].states.shape[0]
    input_dim = payloads[0].inputs.shape[1]
    output_dim = payloads[0].qoi.shape[1]
    mean, scale = compute_state_mean_and_scale(payloads, center=args.center_state, scale_maxabs=args.scale_state_maxabs)
    basis, eigvals, captured_energy = compute_pod_basis(payloads, rank=rank, mean=mean, scale=scale)
    a, h, b, c, dynamics_rel_resid = fit_dynamics(
        payloads,
        basis=basis,
        mean=mean,
        scale=scale,
        dynamic_form=args.dynamic_form,
        regs=(args.reg_a, args.reg_h, args.reg_b, args.reg_c),
    )
    v1, v2, v0, decoder_rel_resid = fit_decoder(
        payloads,
        basis=basis,
        mean=mean,
        scale=scale,
        decoder_form=args.decoder_form,
        regs=(args.reg_decoder_v1, args.reg_decoder_v2, args.reg_decoder_v0),
    )
    muf_path, mug_path = save_oldgoam_parameters(
        output_dir=args.output_dir,
        rank=rank,
        a=a,
        h=h,
        b=b,
        c=c,
        v1=v1,
        v2=v2,
        v0=v0,
        overwrite=bool(args.overwrite),
    )
    summary = {
        "pipeline": "full_order_state_opinf_initialization",
        "manifest_path": str(args.manifest_path),
        "grid_state_root": str(args.grid_state_root) if args.grid_state_root is not None else None,
        "output_dir": str(args.output_dir),
        "rank": rank,
        "sample_count": len(payloads),
        "sample_ids": [p.sample_id for p in payloads],
        "state_dimension_after_stride": int(state_dim),
        "input_dimension": int(input_dim),
        "output_dimension": int(output_dim),
        "time_count": int(payloads[0].times.shape[0]),
        "time_stride": int(args.time_stride),
        "state_dof_stride": int(args.state_dof_stride),
        "center_state": bool(args.center_state),
        "scale_state_maxabs": bool(args.scale_state_maxabs),
        "state_scale": float(scale),
        "pod_captured_energy": float(captured_energy),
        "pod_eigenvalues_first": [float(v) for v in eigvals[: min(20, eigvals.shape[0])]],
        "dynamic_form": args.dynamic_form,
        "decoder_form": args.decoder_form,
        "dynamics_relative_residual": float(dynamics_rel_resid),
        "decoder_relative_residual": float(decoder_rel_resid),
        "regularization": {
            "reg_a": float(args.reg_a),
            "reg_h": float(args.reg_h),
            "reg_b": float(args.reg_b),
            "reg_c": float(args.reg_c),
            "reg_decoder_v1": float(args.reg_decoder_v1),
            "reg_decoder_v2": float(args.reg_decoder_v2),
            "reg_decoder_v0": float(args.reg_decoder_v0),
        },
        "muf_path": str(muf_path),
        "mug_path": str(mug_path),
        "muf_length": int(output_dim * rank + output_dim * compressed_quadratic_dimension(rank) + output_dim),
        "mug_length": int(rank * rank + mu_h_dimension(rank) + rank * input_dim + rank),
    }
    summary_path = args.output_dir / f"full_order_opinf_summary_r{rank}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
