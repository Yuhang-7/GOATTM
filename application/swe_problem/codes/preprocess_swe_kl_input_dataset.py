from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.data import NpzQoiSample, NpzSampleManifest, save_npz_qoi_sample, save_npz_sample_manifest  # noqa: E402


INPUT_ROOT_ENV_VAR = "SWE_ORIGINAL_DATA_ROOT"
DEFAULT_INPUT_ROOT = Path("/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/original_data")
DEFAULT_OUTPUT_ROOT = THIS_FILE.parents[1] / "data" / "processed_data_kl_m200"
DEFAULT_KL_COMPONENTS = 200
DEFAULT_GRID_SIZE = 128
DEFAULT_BATCH_SIZE = 128
DEFAULT_KL_X_MAX = 40.0
L_KM = 100.0


@dataclass(frozen=True)
class OriginalSample:
    path: Path
    sample_id: str
    sensor_locations: np.ndarray
    xi: np.ndarray
    yi: np.ndarray
    ti: np.ndarray
    sigma_i: np.ndarray
    hi: np.ndarray
    qoi_times: np.ndarray
    qoi_values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SWE original samples to GOATTM NPZ samples using KL-projected "
            "time-dependent final-uplift KL coefficients m(t) as the latent ODE input."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Directory containing original sample_*/sample_*.npz files. Defaults to "
            f"${INPUT_ROOT_ENV_VAR}, or {DEFAULT_INPUT_ROOT} if unset."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--kl-components", type=int, default=DEFAULT_KL_COMPONENTS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument(
        "--kl-x-max",
        type=float,
        default=DEFAULT_KL_X_MAX,
        help="Use only grid points with x <= kl_x_max when learning/projecting the uplift KL input.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of Gaussian source fields assembled per batch while building the KL snapshot matrix.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Storage dtype for the source-field snapshot matrix before the final eigensolve.",
    )
    return parser.parse_args()


def default_input_root() -> Path:
    env_value = os.environ.get(INPUT_ROOT_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_INPUT_ROOT


def discover_original_samples(input_root: Path, limit: int | None) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    sample_paths = sorted(input_root.glob("sample_*/*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample_*/sample_*.npz files found under {input_root}")
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"--limit must be positive, got {limit}")
        sample_paths = sample_paths[:limit]
    return sample_paths


def _load_vector(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"Missing required field '{key}'")
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 1:
        raise ValueError(f"Field '{key}' must be rank-1, got {value.shape}")
    return value


def _load_matrix(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"Missing required field '{key}'")
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"Field '{key}' must be rank-2, got {value.shape}")
    return value


def load_original_sample(path: Path) -> OriginalSample:
    with np.load(path, allow_pickle=True) as data:
        sensor_locations = _load_matrix(data, "sensor_locations")
        xi = _load_vector(data, "xi")
        yi = _load_vector(data, "yi")
        ti = _load_vector(data, "Ti")
        sigma_i = _load_vector(data, "sigma_i")
        hi = _load_vector(data, "Hi")
        qoi_times = _load_vector(data, "qoi_times")
        qoi_values = _load_matrix(data, "qoi_values")
    if not all(arr.shape == xi.shape for arr in (yi, ti, sigma_i, hi)):
        raise ValueError(f"{path} has inconsistent Gaussian source vector lengths")
    if qoi_values.shape[1] != qoi_times.shape[0]:
        raise ValueError(
            f"{path} has qoi_values shape {qoi_values.shape}, expected second axis to match "
            f"qoi_times length {qoi_times.shape[0]}"
        )
    if not np.all(np.diff(qoi_times) > 0.0):
        raise ValueError(f"{path} has non-increasing qoi_times")
    return OriginalSample(
        path=path,
        sample_id=path.stem,
        sensor_locations=sensor_locations,
        xi=xi,
        yi=yi,
        ti=ti,
        sigma_i=sigma_i,
        hi=hi,
        qoi_times=qoi_times,
        qoi_values=qoi_values,
    )


def make_grid(grid_size: int, kl_x_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if not (0.0 < kl_x_max <= L_KM):
        raise ValueError(f"kl_x_max must satisfy 0 < kl_x_max <= {L_KM}, got {kl_x_max}")
    x = np.linspace(0.0, L_KM, grid_size, dtype=np.float64)
    y = np.linspace(0.0, L_KM, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    x_flat_full = xx.reshape(-1)
    y_flat_full = yy.reshape(-1)
    kl_mask = x_flat_full <= kl_x_max
    return x, y, x_flat_full[kl_mask], y_flat_full[kl_mask], kl_mask


def gaussian_source_field(
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    xi: float,
    yi: float,
    sigma_i: float,
    hi: float,
) -> np.ndarray:
    r2 = (x_flat - xi) ** 2 + (y_flat - yi) ** 2
    return hi * np.exp(-r2 / (2.0 * sigma_i**2))


def source_count_per_sample(samples: list[OriginalSample]) -> int:
    first_count = samples[0].xi.shape[0]
    for sample in samples:
        if sample.xi.shape[0] != first_count:
            raise ValueError(f"{sample.path} has {sample.xi.shape[0]} Gaussian sources; expected {first_count}")
    return int(first_count)


def final_uplift_field(
    sample: OriginalSample,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
) -> np.ndarray:
    field = np.zeros(x_flat.shape[0], dtype=np.float64)
    for source_index in range(sample.xi.shape[0]):
        field += gaussian_source_field(
            x_flat=x_flat,
            y_flat=y_flat,
            xi=sample.xi[source_index],
            yi=sample.yi[source_index],
            sigma_i=sample.sigma_i[source_index],
            hi=sample.hi[source_index],
        )
    return field


def build_final_uplift_snapshot_matrix(
    samples: list[OriginalSample],
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    source_count_per_sample(samples)
    matrix = np.empty((len(samples), x_flat.shape[0]), dtype=dtype)
    for row, sample in enumerate(samples):
        matrix[row] = final_uplift_field(sample, x_flat, y_flat).astype(dtype, copy=False)
    return matrix


def compute_centered_kl_basis(source_matrix: np.ndarray, component_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if component_count <= 0:
        raise ValueError(f"component_count must be positive, got {component_count}")
    if component_count > min(source_matrix.shape):
        raise ValueError(
            f"component_count={component_count} exceeds min snapshot dimension {min(source_matrix.shape)}"
        )
    mean_field = np.asarray(source_matrix.mean(axis=0), dtype=np.float64)
    centered = np.asarray(source_matrix, dtype=np.float64) - mean_field[None, :]
    # Method of snapshots: solve the smaller sample covariance eigenproblem.
    gram = centered @ centered.T
    eigenvalues_small, eigenvectors_small = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues_small)[::-1]
    eigenvalues_small = eigenvalues_small[order]
    eigenvectors_small = eigenvectors_small[:, order]
    singular_values_all = np.sqrt(np.maximum(eigenvalues_small, 0.0))
    nonzero = singular_values_all > (np.finfo(np.float64).eps * max(centered.shape) * singular_values_all[0])
    if np.count_nonzero(nonzero) < component_count:
        raise ValueError(
            f"Only {np.count_nonzero(nonzero)} nonzero KL modes are available; requested {component_count}."
        )
    singular_values = singular_values_all[:component_count]
    eigenvectors_small = eigenvectors_small[:, :component_count]
    modes = (eigenvectors_small.T @ centered) / singular_values[:, None]
    eigenvalues = eigenvalues_small / max(source_matrix.shape[0] - 1, 1)
    cumulative_energy = np.cumsum(singular_values_all**2) / np.sum(singular_values_all**2)
    return mean_field, modes, eigenvalues[:component_count], cumulative_energy


def project_final_uplift(
    sample: OriginalSample,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    mean_field: np.ndarray,
    kl_modes: np.ndarray,
) -> np.ndarray:
    field = final_uplift_field(sample, x_flat, y_flat)
    return (field - mean_field) @ kl_modes.T


def project_individual_sources_on_final_basis(
    sample: OriginalSample,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    kl_modes: np.ndarray,
) -> np.ndarray:
    coefficients = np.empty((sample.xi.shape[0], kl_modes.shape[0]), dtype=np.float64)
    for source_index in range(sample.xi.shape[0]):
        field = gaussian_source_field(
            x_flat=x_flat,
            y_flat=y_flat,
            xi=sample.xi[source_index],
            yi=sample.yi[source_index],
            sigma_i=sample.sigma_i[source_index],
            hi=sample.hi[source_index],
        )
        coefficients[source_index] = field @ kl_modes.T
    return coefficients


def _mollifier(times: np.ndarray, ti: np.ndarray) -> np.ndarray:
    tau = np.clip(times[:, None] / ti[None, :], 0.0, 1.0)
    return 6.0 * tau**5 - 15.0 * tau**4 + 10.0 * tau**3


def build_time_dependent_kl_input_values(
    input_times: np.ndarray,
    source_coefficients: np.ndarray,
    mean_coefficients: np.ndarray,
    ti: np.ndarray,
) -> np.ndarray:
    temporal_weights = _mollifier(input_times, ti)
    return temporal_weights @ source_coefficients - mean_coefficients[None, :]


def save_processed_samples(
    samples: list[OriginalSample],
    output_root: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    mean_field: np.ndarray,
    kl_modes: np.ndarray,
    eigenvalues: np.ndarray,
    cumulative_energy: np.ndarray,
    input_root: Path,
    kl_mask: np.ndarray,
    kl_x_max: float,
) -> NpzSampleManifest:
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    rel_paths: list[Path] = []
    sample_ids: list[str] = []
    for index, sample in enumerate(samples):
        final_uplift_coefficients = project_final_uplift(
            sample=sample,
            x_flat=x_flat,
            y_flat=y_flat,
            mean_field=mean_field,
            kl_modes=kl_modes,
        )
        source_coefficients = project_individual_sources_on_final_basis(
            sample=sample,
            x_flat=x_flat,
            y_flat=y_flat,
            kl_modes=kl_modes,
        )
        mean_coefficients = mean_field @ kl_modes.T
        observation_times = np.concatenate([[0.0], sample.qoi_times], axis=0)
        input_values = build_time_dependent_kl_input_values(
            input_times=observation_times,
            source_coefficients=source_coefficients,
            mean_coefficients=mean_coefficients,
            ti=sample.ti,
        )
        qoi_observations = np.vstack(
            [
                np.zeros((1, sample.qoi_values.shape[0]), dtype=np.float64),
                sample.qoi_values.T.astype(np.float64, copy=False),
            ]
        )
        processed = NpzQoiSample(
            sample_id=sample.sample_id,
            observation_times=observation_times,
            u0=qoi_observations[0].copy(),
            qoi_observations=qoi_observations,
            input_times=observation_times.copy(),
            input_values=input_values,
            metadata={
                "dataset_kind": "swe_sensor_qoi",
                "input_mode": "kl_projected_time_dependent_gaussian_uplift",
                "input_feature_names": np.asarray([f"kl_uplift_coeff_{j:04d}" for j in range(kl_modes.shape[0])]),
                "original_npz_path": str(sample.path),
                "sensor_locations": sample.sensor_locations,
                "xi": sample.xi,
                "yi": sample.yi,
                "Ti": sample.ti,
                "sigma_i": sample.sigma_i,
                "Hi": sample.hi,
                "source_count": int(sample.xi.shape[0]),
                "kl_component_count": int(kl_modes.shape[0]),
                "kl_grid_size": int(x_grid.shape[0]),
                "kl_final_uplift_coefficients": final_uplift_coefficients,
                "kl_source_coefficients": source_coefficients,
                "prepended_zero_initial_qoi": 1,
            },
        )
        rel_path = Path("samples") / f"{sample.sample_id}.npz"
        save_npz_qoi_sample(samples_root / rel_path.name, processed)
        rel_paths.append(rel_path)
        sample_ids.append(sample.sample_id)
        if (index + 1) % 100 == 0 or index == len(samples) - 1:
            print(f"  wrote {index + 1}/{len(samples)} processed KL samples", flush=True)
    manifest = NpzSampleManifest(root_dir=output_root, sample_paths=tuple(rel_paths), sample_ids=tuple(sample_ids))
    save_npz_sample_manifest(output_root / "manifest.npz", manifest)
    np.savez_compressed(
        output_root / "kl_basis_m200.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        kl_x_flat=x_flat,
        kl_y_flat=y_flat,
        kl_mask=kl_mask,
        kl_x_max=float(kl_x_max),
        mean_field=mean_field,
        kl_modes=kl_modes,
        eigenvalues=eigenvalues,
        cumulative_energy=cumulative_energy,
        input_root=str(input_root),
    )
    return manifest


def write_summary(
    output_root: Path,
    input_root: Path,
    samples: list[OriginalSample],
    manifest: NpzSampleManifest,
    eigenvalues: np.ndarray,
    cumulative_energy: np.ndarray,
    grid_size: int,
    kl_components: int,
    kl_mask: np.ndarray,
    kl_x_max: float,
) -> None:
    first = samples[0]
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "manifest_path": str(output_root / "manifest.npz"),
        "samples_root": str(output_root / "samples"),
        "sample_count": len(samples),
        "sample_id_first": manifest.sample_ids[0],
        "sample_id_last": manifest.sample_ids[-1],
        "observation_count_per_sample": int(first.qoi_times.shape[0] + 1),
        "qoi_output_dimension": int(first.qoi_values.shape[0]),
        "input_parameter_dimension": int(kl_components),
        "input_mode": "kl_projected_time_dependent_gaussian_uplift",
        "kl_component_count": int(kl_components),
        "kl_grid_size": int(grid_size),
        "kl_basis_path": str(output_root / "kl_basis_m200.npz"),
        "kl_x_max": float(kl_x_max),
        "kl_point_count": int(np.count_nonzero(kl_mask)),
        "kl_energy_at_m": float(cumulative_energy[kl_components - 1]),
        "kl_eigenvalue_first": float(eigenvalues[0]),
        "kl_eigenvalue_m": float(eigenvalues[kl_components - 1]),
        "prepended_zero_initial_qoi": True,
        "workflow_warning": (
            "This is the corrected SWE input pipeline: for each sample, the five Gaussian uplift "
            "sources are summed at final time, centered KL modes are learned from these final total "
            "uplift fields, each individual source is projected onto that final-uplift KL basis, "
            "and each sample stores time-dependent input_values m(t) obtained by weighting those "
            "source coefficients with the original temporal mollifiers. Do not use "
            "the legacy uplift_parameters processed_data for scientific SWE comparisons."
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    input_root = default_input_root() if args.input_root is None else args.input_root.expanduser()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sample_paths = discover_original_samples(input_root, args.limit)
    print(f"Loading {len(sample_paths)} original SWE samples from {input_root}")
    samples = [load_original_sample(path.resolve()) for path in sample_paths]
    x_grid, y_grid, x_flat, y_flat, kl_mask = make_grid(args.grid_size, args.kl_x_max)
    dtype = np.dtype(args.dtype)
    print(
        f"Assembling {len(samples)} final total Gaussian uplift fields on a {args.grid_size}x{args.grid_size} grid "
        f"restricted to x <= {args.kl_x_max} km ({x_flat.shape[0]} points)"
    )
    source_matrix = build_final_uplift_snapshot_matrix(samples, x_flat, y_flat, dtype)
    print(f"Final-uplift snapshot matrix shape: {source_matrix.shape}, dtype={source_matrix.dtype}")
    print(f"Computing centered KL basis with m={args.kl_components}")
    mean_field, kl_modes, eigenvalues, cumulative_energy = compute_centered_kl_basis(source_matrix, args.kl_components)
    print(f"KL energy at m={args.kl_components}: {cumulative_energy[args.kl_components - 1]:.8f}")
    print(f"Writing corrected GOATTM processed dataset to {output_root}")
    manifest = save_processed_samples(
        samples=samples,
        output_root=output_root,
        x_grid=x_grid,
        y_grid=y_grid,
        x_flat=x_flat,
        y_flat=y_flat,
        mean_field=mean_field,
        kl_modes=kl_modes,
        eigenvalues=eigenvalues,
        cumulative_energy=cumulative_energy,
        input_root=input_root,
        kl_mask=kl_mask,
        kl_x_max=args.kl_x_max,
    )
    write_summary(
        output_root=output_root,
        input_root=input_root,
        samples=samples,
        manifest=manifest,
        eigenvalues=eigenvalues,
        cumulative_energy=cumulative_energy,
        grid_size=args.grid_size,
        kl_components=args.kl_components,
        kl_mask=kl_mask,
        kl_x_max=args.kl_x_max,
    )
    print(f"Done. Manifest: {output_root / 'manifest.npz'}")


if __name__ == "__main__":
    main()
