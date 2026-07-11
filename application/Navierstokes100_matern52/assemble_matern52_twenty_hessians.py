from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=/absolute/path.npz")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name must be nonempty")
    return name, Path(raw_path).expanduser().resolve()


def regularizations_from_config(config_path: Path):
    from goattm.problems import (
        DecoderTikhonovRegularization,
        DynamicsTikhonovRegularization,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder_values = config["decoder_regularization"]
    dynamics_values = config["dynamics_regularization"]
    decoder_regularization = DecoderTikhonovRegularization(
        coeff_v1=float(decoder_values["coeff_v1"]),
        coeff_v2=float(decoder_values["coeff_v2"]),
        coeff_v0=float(decoder_values["coeff_v0"]),
    )
    dynamics_regularization = DynamicsTikhonovRegularization(
        coeff_a=float(dynamics_values["coeff_a"]),
        coeff_s=float(dynamics_values.get("coeff_s", 0.0)),
        coeff_w=float(dynamics_values.get("coeff_w", 0.0)),
        coeff_mu_h=float(dynamics_values["coeff_mu_h"]),
        coeff_b=float(dynamics_values["coeff_b"]),
        coeff_c=float(dynamics_values["coeff_c"]),
    )
    return config, decoder_regularization, dynamics_regularization


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble five Hessians for energy/general dynamics at two checkpoints."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=56)
    parser.add_argument("--sample-cap", type=int, default=0)
    parser.add_argument("--max-columns", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))

    from mpi4py import MPI

    from goattm.analysis.hessian_landscape import (
        HessianLandscapeConfig,
        build_five_curvature_operators,
        energy_to_general_dynamics,
        load_checkpoint_models,
    )
    from goattm.data.npz_dataset import load_npz_sample_manifest
    from goattm.runtime.distributed import DistributedContext

    world = MPI.COMM_WORLD
    world_rank = world.Get_rank()
    world_size = world.Get_size()
    checkpoints = list(args.checkpoint)
    if len(checkpoints) != 2:
        raise ValueError(f"Exactly two checkpoints are required, got {len(checkpoints)}")
    if world_size % len(checkpoints) != 0:
        raise ValueError("MPI world size must be divisible by checkpoint count")

    ranks_per_checkpoint = world_size // len(checkpoints)
    checkpoint_index = world_rank // ranks_per_checkpoint
    checkpoint_name, checkpoint_path = checkpoints[checkpoint_index]
    checkpoint_comm = world.Split(color=checkpoint_index, key=world_rank)
    checkpoint_rank = checkpoint_comm.Get_rank()
    checkpoint_size = checkpoint_comm.Get_size()
    group_size = int(args.group_size)
    if group_size <= 0 or checkpoint_size % group_size != 0:
        raise ValueError(
            f"group_size={group_size} must divide checkpoint communicator size {checkpoint_size}"
        )
    group_index = checkpoint_rank // group_size
    group_comm = checkpoint_comm.Split(color=group_index, key=checkpoint_rank)
    group_rank = group_comm.Get_rank()
    group_count = checkpoint_size // group_size
    context = DistributedContext.from_comm(group_comm)

    output_dir = args.output_dir.resolve()
    point_dir = output_dir / checkpoint_name
    parts_dir = point_dir / "parts"
    if world_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_rank == 0:
        point_dir.mkdir(parents=True, exist_ok=True)
        parts_dir.mkdir(parents=True, exist_ok=True)
    world.Barrier()

    config_data, decoder_regularization, dynamics_regularization = (
        regularizations_from_config(args.config.resolve())
    )
    manifest = load_npz_sample_manifest(args.manifest.resolve())
    if int(args.sample_cap) > 0:
        manifest = manifest.subset_by_indices(
            list(range(min(int(args.sample_cap), len(manifest))))
        )
    decoder, energy_dynamics = load_checkpoint_models(checkpoint_path)
    general_dynamics = energy_to_general_dynamics(energy_dynamics)
    landscape_config = HessianLandscapeConfig(
        max_dt=float(config_data["solver"]["max_dt"]),
        time_integrator=str(config_data["time_integrator"]),
        solve_root=0,
    )

    if checkpoint_rank == 0:
        print(
            json.dumps(
                {
                    "event": "checkpoint_start",
                    "checkpoint": checkpoint_name,
                    "path": str(checkpoint_path),
                    "world_size": world_size,
                    "checkpoint_size": checkpoint_size,
                    "group_size": group_size,
                    "group_count": group_count,
                    "sample_count": len(manifest),
                    "max_dt": landscape_config.max_dt,
                }
            ),
            flush=True,
        )

    point_summaries: list[dict[str, object]] = []
    for dynamics_form, dynamics in (
        ("energy", energy_dynamics),
        ("general", general_dynamics),
    ):
        operators = build_five_curvature_operators(
            manifest=manifest,
            dynamics=dynamics,
            decoder=decoder,
            config=landscape_config,
            dynamics_form=dynamics_form,
            decoder_regularization=decoder_regularization,
            dynamics_regularization=dynamics_regularization,
            context=context,
        )
        for case in operators:
            dimension = int(case.dimension)
            assembled_dimension = (
                dimension
                if int(args.max_columns) <= 0
                else min(dimension, int(args.max_columns))
            )
            case_name = f"{dynamics_form}_{case.name}"
            final_matrix_path = point_dir / f"{case_name}.matrix.npz"
            final_summary_path = point_dir / f"{case_name}.json"
            part_path = parts_dir / f"{case_name}.group_{group_index:03d}.npz"
            columns = np.arange(
                group_index, assembled_dimension, group_count, dtype=np.int64
            )
            if args.resume and part_path.exists():
                if group_rank == 0:
                    print(
                        json.dumps(
                            {
                                "event": "part_resume",
                                "checkpoint": checkpoint_name,
                                "case": case_name,
                                "group": group_index,
                            }
                        ),
                        flush=True,
                    )
            else:
                local_matrix = (
                    np.zeros((assembled_dimension, columns.size), dtype=np.float64)
                    if group_rank == 0
                    else None
                )
                direction = np.zeros(dimension, dtype=np.float64)
                action_times = np.zeros(columns.size, dtype=np.float64)
                case_start = time.perf_counter()
                for local_column, global_column in enumerate(columns):
                    direction.fill(0.0)
                    direction[int(global_column)] = 1.0
                    action_start = time.perf_counter()
                    action = np.asarray(case.operator @ direction, dtype=np.float64)
                    action_times[local_column] = time.perf_counter() - action_start
                    if group_rank == 0 and local_matrix is not None:
                        local_matrix[:, local_column] = action[:assembled_dimension]
                    if group_rank == 0 and (
                        local_column < 2
                        or (local_column + 1) % 25 == 0
                        or local_column + 1 == columns.size
                    ):
                        print(
                            json.dumps(
                                {
                                    "event": "column",
                                    "checkpoint": checkpoint_name,
                                    "case": case_name,
                                    "group": group_index,
                                    "local_column": local_column + 1,
                                    "local_column_count": int(columns.size),
                                    "global_column": int(global_column),
                                    "seconds": float(action_times[local_column]),
                                }
                            ),
                            flush=True,
                        )
                if group_rank == 0 and local_matrix is not None:
                    atomic_savez(
                        part_path,
                        columns=columns,
                        matrix=local_matrix,
                        action_times=action_times,
                        elapsed_seconds=np.asarray(
                            [time.perf_counter() - case_start], dtype=np.float64
                        ),
                    )

            checkpoint_comm.Barrier()
            if checkpoint_rank == 0 and not (args.resume and final_matrix_path.exists()):
                matrix = np.zeros(
                    (assembled_dimension, assembled_dimension), dtype=np.float64
                )
                group_elapsed = []
                all_action_times = []
                for part_group in range(group_count):
                    data = np.load(
                        parts_dir / f"{case_name}.group_{part_group:03d}.npz"
                    )
                    part_columns = np.asarray(data["columns"], dtype=np.int64)
                    matrix[:, part_columns] = np.asarray(data["matrix"], dtype=np.float64)
                    group_elapsed.append(float(data["elapsed_seconds"][0]))
                    all_action_times.extend(
                        np.asarray(data["action_times"], dtype=np.float64).tolist()
                    )
                asymmetry = float(np.linalg.norm(matrix - matrix.T))
                matrix_norm = float(np.linalg.norm(matrix))
                symmetric_matrix = 0.5 * (matrix + matrix.T)
                atomic_savez(final_matrix_path, hessian=symmetric_matrix)
                summary = {
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": str(checkpoint_path),
                    "case": case_name,
                    "curvature": case.name,
                    "dynamics_form": dynamics_form,
                    "full_dimension": dimension,
                    "assembled_dimension": assembled_dimension,
                    "sample_count": len(manifest),
                    "max_dt": landscape_config.max_dt,
                    "prepare_seconds": case.prepare_seconds,
                    "group_count": group_count,
                    "group_size": group_size,
                    "wall_seconds": max(group_elapsed) if group_elapsed else 0.0,
                    "action_seconds_median": float(np.median(all_action_times)),
                    "action_seconds_min": float(np.min(all_action_times)),
                    "action_seconds_max": float(np.max(all_action_times)),
                    "raw_frobenius_norm": matrix_norm,
                    "raw_asymmetry_norm": asymmetry,
                    "raw_relative_asymmetry": asymmetry / max(1.0, matrix_norm),
                    "trace": float(np.trace(symmetric_matrix)),
                    "matrix_path": str(final_matrix_path),
                }
                final_summary_path.write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                point_summaries.append(summary)
                print(json.dumps({"event": "case_done", **summary}), flush=True)
            checkpoint_comm.Barrier()

    if checkpoint_rank == 0:
        point_manifest = {
            "checkpoint": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "cases": point_summaries,
        }
        (point_dir / "manifest.json").write_text(
            json.dumps(point_manifest, indent=2), encoding="utf-8"
        )
    world.Barrier()
    if world_rank == 0:
        combined = {
            "checkpoints": [name for name, _ in checkpoints],
            "checkpoint_count": len(checkpoints),
            "hessian_count_per_checkpoint": 10,
            "total_hessian_count": 20,
            "world_size": world_size,
            "ranks_per_checkpoint": ranks_per_checkpoint,
            "group_size": group_size,
            "sample_cap": int(args.sample_cap),
            "max_columns": int(args.max_columns),
            "output_dir": str(output_dir),
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(combined, indent=2), encoding="utf-8"
        )
        print(json.dumps({"event": "run_done", **combined}), flush=True)


if __name__ == "__main__":
    main()
