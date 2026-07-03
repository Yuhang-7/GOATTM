#!/usr/bin/env python3
"""Export saved distributed CG1 velocity states onto a fixed physical grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gmsh
import numpy as np
from basix.ufl import element
from dolfinx import geometry
from dolfinx.fem import Function, functionspace
from dolfinx.io import gmshio
from mpi4py import MPI


L = 2.2
H = 0.41
CYLINDER_X = 0.2
CYLINDER_Y = 0.2
CYLINDER_R = 0.05
GDIM = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--ranks-per-case", type=int, default=2)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mesh-path", type=Path, default=Path("cylinder.msh"))
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--margin", type=float, default=1.0e-8)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def build_grid(nx: int, ny: int, margin: float) -> np.ndarray:
    xs = np.linspace(0.0 + margin, L - margin, nx)
    ys = np.linspace(0.0 + margin, H - margin, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    radius2 = (points[:, 0] - CYLINDER_X) ** 2 + (points[:, 1] - CYLINDER_Y) ** 2
    keep = radius2 > (CYLINDER_R + margin) ** 2
    return points[keep]


def owned_point_indices(mesh, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_3d = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
    points_3d[:, :2] = points_xy
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points_3d)
    colliding = geometry.compute_colliding_cells(mesh, candidates, points_3d)

    owned_indices = []
    owned_cells = []
    for j in range(points_xy.shape[0]):
        links = colliding.links(j)
        if len(links) > 0:
            owned_indices.append(j)
            owned_cells.append(links[0])
    return np.asarray(owned_indices, dtype=np.int64), np.asarray(owned_cells, dtype=np.int32)


def gather_grid_values(comm, local_indices: np.ndarray, local_values: np.ndarray, npoints: int) -> np.ndarray | None:
    gathered_indices = comm.gather(local_indices, root=0)
    gathered_values = comm.gather(local_values, root=0)
    if comm.rank != 0:
        return None

    values = np.full((npoints, 2), np.nan, dtype=np.float64)
    for indices, vals in zip(gathered_indices, gathered_values):
        if len(indices) == 0:
            continue
        values[indices, :] = vals

    missing = np.where(~np.isfinite(values[:, 0]))[0]
    if missing.size:
        raise RuntimeError(f"{missing.size} grid points were not owned by any rank")
    return values


def export_one_case(
    caseid: int,
    mesh,
    vproj,
    uproj: Function,
    comm,
    grid_points: np.ndarray,
    local_point_indices: np.ndarray,
    local_cells: np.ndarray,
    source_root: Path,
    output_root: Path,
    dtype: np.dtype,
) -> None:
    source_path = source_root / f"output_{caseid:06d}.npz"
    if not source_path.exists():
        if comm.rank == 0:
            print(f"skip case {caseid}: missing {source_path}", flush=True)
        return

    data = np.load(source_path)
    if int(data["state_saved"]) != 1:
        if comm.rank == 0:
            print(f"skip case {caseid}: state_saved=0", flush=True)
        return

    partitions = np.asarray(data["state_vector_partition_sizes"], dtype=np.int64)
    state_snapshots = np.asarray(data["state_snapshots"], dtype=np.float64)
    if partitions.size != comm.size:
        raise RuntimeError(f"case {caseid}: saved with {partitions.size} ranks, now running with {comm.size}")

    # Match the solver's saved layout exactly. The original solver used
    # ``uproj.vector.array`` here, which stores owned entries without ghosts.
    local_size = uproj.vector.array.size
    if local_size != int(partitions[comm.rank]):
        raise RuntimeError(
            f"case {caseid}: rank {comm.rank} local size {local_size} != saved partition {partitions[comm.rank]}"
        )
    start = int(np.sum(partitions[: comm.rank]))
    stop = start + local_size

    nt = state_snapshots.shape[1]
    if comm.rank == 0:
        grid_state = np.empty((nt, grid_points.shape[0], 2), dtype=dtype)
    else:
        grid_state = None

    local_points_3d = np.zeros((local_point_indices.shape[0], 3), dtype=np.float64)
    local_points_3d[:, :2] = grid_points[local_point_indices]

    for k in range(nt):
        uproj.vector.array[:] = state_snapshots[start:stop, k]
        uproj.x.scatter_forward()
        if local_point_indices.size:
            local_values = uproj.eval(local_points_3d, local_cells)
        else:
            local_values = np.empty((0, 2), dtype=np.float64)
        values = gather_grid_values(comm, local_point_indices, local_values, grid_points.shape[0])
        if comm.rank == 0:
            grid_state[k, :, :] = values.astype(dtype, copy=False)

    if comm.rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"grid_state_{caseid:06d}.npz"
        tmp_path = output_root / f".grid_state_{caseid:06d}.npz.tmp"
        payload = {
            "case_id": np.asarray(caseid, dtype=np.int64),
            "grid_points": grid_points.astype(np.float64),
            "grid_state": grid_state,
            "observation_times": np.asarray(data["observation_times"], dtype=np.float64),
            "source_npz_path": np.asarray(str(source_path)),
            "state_kind": np.asarray("cg1_velocity_grid"),
            "nx": np.asarray(args.nx, dtype=np.int64),
            "ny": np.asarray(args.ny, dtype=np.int64),
        }
        with open(tmp_path, "wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(tmp_path, output_path)
        print(f"wrote {output_path}", flush=True)


args = parse_args()
overall_comm = MPI.COMM_WORLD
overall_rank = overall_comm.rank

if overall_comm.size % args.ranks_per_case != 0:
    if overall_rank == 0:
        print(f"MPI size {overall_comm.size} must be divisible by ranks-per-case {args.ranks_per_case}", flush=True)
    overall_comm.Abort(2)

group_id = overall_rank // args.ranks_per_case
group_count = overall_comm.size // args.ranks_per_case
case_comm = overall_comm.Split(group_id, overall_rank)

gmsh.initialize()
mesh, _, _ = gmshio.read_from_msh(str(args.mesh_path), case_comm, 0, GDIM)
v_cg1 = element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
vproj = functionspace(mesh, v_cg1)
uproj = Function(vproj)

grid_points = build_grid(args.nx, args.ny, args.margin)
local_point_indices, local_cells = owned_point_indices(mesh, grid_points)

if overall_rank == 0:
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "state_kind": "cg1_velocity_grid",
        "nx": args.nx,
        "ny": args.ny,
        "npoints": int(grid_points.shape[0]),
        "feature_dimension": int(2 * grid_points.shape[0]),
        "domain": {"L": L, "H": H, "cylinder_x": CYLINDER_X, "cylinder_y": CYLINDER_Y, "cylinder_r": CYLINDER_R},
        "source_root": str(args.source_root),
    }
    (args.output_root / "grid_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

dtype = np.float32 if args.dtype == "float32" else np.float64
caseid = args.start + group_id
while caseid < args.end:
    export_one_case(
        caseid,
        mesh,
        vproj,
        uproj,
        case_comm,
        grid_points,
        local_point_indices,
        local_cells,
        args.source_root,
        args.output_root,
        dtype,
    )
    caseid += group_count

case_comm.Barrier()
