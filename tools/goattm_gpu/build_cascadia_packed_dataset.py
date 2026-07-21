from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


QOI_INDICES = np.arange(0, 150, 3, dtype=np.int64)


def manifest_paths(manifest_path: Path) -> tuple[list[str], list[Path], Path]:
    data = np.load(manifest_path, allow_pickle=True)
    ids = [str(x) for x in data["sample_ids"].tolist()]
    paths = [Path(str(x)) for x in data["sample_paths"].tolist()]
    root_str = str(data["root"].item()) if "root" in data.files else ""
    root = manifest_path.parent if root_str == "" else Path(root_str)
    abs_paths = [p if p.is_absolute() else root / p for p in paths]
    return ids, abs_paths, root


def load_raw_arrays(manifest_path: Path) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_ids, paths, _ = manifest_paths(manifest_path)
    n = len(paths)
    qoi = torch.empty((541, n, 50), dtype=torch.float64)
    active_input = torch.empty((18, n, 150), dtype=torch.float64)
    observation_times = None
    input_times = None
    for j, path in enumerate(paths):
        sample = np.load(path, allow_pickle=True)
        qoi[:, j, :] = torch.from_numpy(np.asarray(sample["qoi_observations"][:, QOI_INDICES], dtype=np.float64))
        active_input[:, j, :] = torch.from_numpy(np.asarray(sample["input_values"], dtype=np.float64))
        if observation_times is None:
            observation_times = torch.from_numpy(np.asarray(sample["observation_times"], dtype=np.float64))
            input_times = torch.from_numpy(np.asarray(sample["input_times"], dtype=np.float64))
    if observation_times is None or input_times is None:
        raise ValueError("empty manifest")
    return sample_ids, observation_times, input_times, qoi, active_input


def channel_stats(values: torch.Tensor, target_max_abs: float, eps: float = 1.0e-12) -> dict[str, torch.Tensor | float]:
    flat = values.reshape(-1, values.shape[-1])
    mean = flat.mean(dim=0)
    centered_max_abs = (flat - mean).abs().amax(dim=0)
    scale = centered_max_abs / float(target_max_abs)
    scale = torch.where(centered_max_abs < float(eps), torch.ones_like(scale), scale)
    return {"mean": mean, "scale": scale, "target_max_abs": float(target_max_abs)}


def normalize(values: torch.Tensor, stats: dict[str, torch.Tensor | float]) -> torch.Tensor:
    return (values - stats["mean"]) / stats["scale"]


def build_full_input(
    observation_times: torch.Tensor,
    input_times: torch.Tensor,
    active_input_norm: torch.Tensor,
) -> torch.Tensor:
    full = torch.zeros((observation_times.numel(), active_input_norm.shape[1], active_input_norm.shape[2]), dtype=torch.float64)
    for n, t_value in enumerate(observation_times.tolist()):
        if t_value <= 0.0:
            continue
        if t_value > float(input_times[-1]):
            continue
        if abs(t_value % 10.0) < 1.0e-12:
            idx = int(round(t_value / 10.0)) - 1
            full[n] = active_input_norm[idx]
        else:
            right = int(np.searchsorted(input_times.numpy(), t_value, side="right"))
            if right == 0:
                weight = t_value / float(input_times[0])
                full[n] = weight * active_input_norm[0]
            else:
                left = right - 1
                t0 = float(input_times[left])
                t1 = float(input_times[right])
                weight = (t_value - t0) / (t1 - t0)
                full[n] = (1.0 - weight) * active_input_norm[left] + weight * active_input_norm[right]
    return full


def pack_dataset(
    manifest_path: Path,
    output_path: Path,
    *,
    qoi_stats: dict[str, torch.Tensor | float] | None,
    input_stats: dict[str, torch.Tensor | float] | None,
    target_max_abs: float,
) -> tuple[dict[str, torch.Tensor | float], dict[str, torch.Tensor | float], dict[str, object]]:
    sample_ids, observation_times_physical, input_times_physical, qoi_raw, active_input_raw = load_raw_arrays(manifest_path)
    if qoi_stats is None:
        qoi_stats = channel_stats(qoi_raw, target_max_abs)
    if input_stats is None:
        input_stats = channel_stats(active_input_raw, target_max_abs)
    qoi_norm = normalize(qoi_raw, qoi_stats)
    active_input_norm = normalize(active_input_raw, input_stats)
    input_full = build_full_input(observation_times_physical, input_times_physical, active_input_norm)
    observation_times = observation_times_physical / observation_times_physical[-1]
    payload = {
        "sample_ids": sample_ids,
        "observation_times": observation_times,
        "input_times": observation_times.clone(),
        "qoi": qoi_norm,
        "input_values": input_full,
        "qoi_stats": qoi_stats,
        "input_stats": input_stats,
        "metadata": {
            "manifest": str(manifest_path),
            "sample_count": len(sample_ids),
            "qoi_indices": QOI_INDICES.tolist(),
            "qoi_shape": list(qoi_norm.shape),
            "input_shape": list(input_full.shape),
            "physical_step_size_seconds": float(observation_times_physical[1] - observation_times_physical[0]),
            "normalized_step_size": float(observation_times[1] - observation_times[0]),
            "post_source_input_max_abs": float(input_full[observation_times_physical > input_times_physical[-1]].abs().amax()),
            "qoi_normalized_max_abs": float(qoi_norm.abs().amax()),
            "active_input_normalized_max_abs": float(active_input_norm.abs().amax()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return qoi_stats, input_stats, payload["metadata"]


def make_pod_opinf_initializer(
    packed_train_path: Path,
    output_path: Path,
    *,
    latent_dim: int,
    ridge: float,
    stability_shift: float,
) -> dict[str, object]:
    data = torch.load(packed_train_path, map_location="cpu")
    q = data["qoi"].to(torch.float64)
    p = data["input_values"].to(torch.float64)
    t = data["observation_times"].to(torch.float64)
    dt = float(t[1] - t[0])
    flat_q = q.reshape(-1, q.shape[-1])
    covariance = flat_q.T @ flat_q / float(flat_q.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals, descending=True)
    basis = eigvecs[:, order[:latent_dim]].contiguous()
    z = torch.einsum("tnq,qr->tnr", q, basis)
    z_mid = 0.5 * (z[:-1] + z[1:])
    dz = (z[1:] - z[:-1]) / dt
    p_mid = 0.5 * (p[:-1] + p[1:])
    ones = torch.ones((*z_mid.shape[:-1], 1), dtype=torch.float64)
    features = torch.cat((z_mid, p_mid, ones), dim=-1).reshape(-1, latent_dim + p.shape[-1] + 1)
    targets = dz.reshape(-1, latent_dim)
    normal = features.T @ features
    rhs = features.T @ targets
    eye = torch.eye(normal.shape[0], dtype=torch.float64)
    coeff = torch.linalg.solve(normal + float(ridge) * eye, rhs)
    a_matrix = coeff[:latent_dim].T.contiguous()
    b_matrix = coeff[latent_dim : latent_dim + p.shape[-1]].T.contiguous()
    c_vector = coeff[-1].contiguous()

    eig_a = torch.linalg.eigvals(a_matrix).real
    max_real_before = float(eig_a.max())
    shift = max(0.0, max_real_before + float(stability_shift))
    if shift > 0.0:
        a_matrix = a_matrix - shift * torch.eye(latent_dim, dtype=torch.float64)
    max_real_after = float(torch.linalg.eigvals(a_matrix).real.max())

    quad_dim = latent_dim * (latent_dim + 1) // 2
    free_dim = sum(1 for a in range(latent_dim) for b in range(latent_dim) for _c in range(b, latent_dim) if a > b)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        a_matrix=a_matrix.numpy(),
        b_matrix=b_matrix.numpy(),
        c_vector=c_vector.numpy(),
        decoder_template_v0=np.zeros(q.shape[-1], dtype=np.float64),
        decoder_template_v1=basis.numpy(),
        decoder_template_v2=np.zeros((q.shape[-1], quad_dim), dtype=np.float64),
        mu_h=np.zeros(free_dim, dtype=np.float64),
        pod_eigenvalues=eigvals[order].numpy(),
        pod_basis=basis.numpy(),
        ridge=np.array(float(ridge)),
        stability_shift=np.array(float(stability_shift)),
        max_real_before_shift=np.array(max_real_before),
        max_real_after_shift=np.array(max_real_after),
    )
    return {
        "path": str(output_path),
        "latent_dim": int(latent_dim),
        "ridge": float(ridge),
        "stability_shift": float(stability_shift),
        "max_real_before_shift": max_real_before,
        "max_real_after_shift": max_real_after,
        "free_h_dim": int(free_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack Cascadia train/test manifests into reusable .pt tensors.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-max-abs", type=float, default=0.9)
    parser.add_argument("--pod-latent-dim", type=int, default=50)
    parser.add_argument("--opinf-ridge", type=float, default=1.0e-6)
    parser.add_argument("--stability-shift", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "train_8192_packed.pt"
    test_path = output_dir / "test_5000_packed.pt"
    q_stats, p_stats, train_meta = pack_dataset(
        Path(args.train_manifest).expanduser().resolve(),
        train_path,
        qoi_stats=None,
        input_stats=None,
        target_max_abs=args.target_max_abs,
    )
    _, _, test_meta = pack_dataset(
        Path(args.test_manifest).expanduser().resolve(),
        test_path,
        qoi_stats=q_stats,
        input_stats=p_stats,
        target_max_abs=args.target_max_abs,
    )
    init_meta = make_pod_opinf_initializer(
        train_path,
        output_dir / f"pod_opinf_r{int(args.pod_latent_dim)}_initial_parameters.npz",
        latent_dim=int(args.pod_latent_dim),
        ridge=float(args.opinf_ridge),
        stability_shift=float(args.stability_shift),
    )
    summary = {
        "train_packed": str(train_path),
        "test_packed": str(test_path),
        "train_metadata": train_meta,
        "test_metadata": test_meta,
        "initializer": init_meta,
    }
    (output_dir / "packed_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
