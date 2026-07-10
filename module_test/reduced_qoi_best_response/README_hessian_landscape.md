# Four-Case Hessian Landscape Driver

This note documents the reusable entry point for the four Gauss-Newton Hessian
operators used in the VarPro/energy-constraint landscape experiments.

The driver lives in:

```text
src/goattm/analysis/hessian_landscape.py
```

It builds four matrix-free operators:

| case | dynamics | decoder treatment |
| --- | --- | --- |
| `general_joint_gn` | dense unconstrained `h_matrix` | joint decoder variables |
| `general_varpro_gn` | dense unconstrained `h_matrix` | VarPro Schur GN |
| `energy_joint_gn` | energy-preserving `mu_h` | joint decoder variables |
| `energy_varpro_gn` | energy-preserving `mu_h` | VarPro Schur GN |

All four operators are Gauss-Newton/linearized least-squares Hessian actions.
They are not exact reduced Hessians.

## CLI

Run from the repository root:

```bash
PYTHONPATH=src python -m goattm.analysis.hessian_landscape \
  --manifest-path application/Navierstokes100_matern52/data/processed_data_qoi5_input40/normalized/train_manifest.npz \
  --checkpoint-path path/to/checkpoints/best.npz \
  --output-dir path/to/hessian_landscape \
  --max-dt 0.02 \
  --time-integrator lagged_midpoint \
  --k 8 \
  --which LA,SA
```

MPI usage:

```bash
PYTHONPATH=src mpiexec -n 64 python -m goattm.analysis.hessian_landscape \
  --manifest-path path/to/train_manifest.npz \
  --checkpoint-path path/to/checkpoints/best.npz \
  --output-dir path/to/hessian_landscape \
  --max-dt 0.02 \
  --k 8 \
  --which LA,SA
```

Each rank owns its manifest shard through `DistributedContext`. The eigensolver
must run on all ranks at once because each matrix-vector product calls MPI
collectives.

## Outputs

Only `solve_root` writes:

```text
summary.json
eigenvalues.npz
```

`summary.json` stores timings, residual norms, dimensions, and matvec counts.
`eigenvalues.npz` stores arrays named like:

```text
general_varpro_gn_LA_eigenvalues
general_varpro_gn_LA_residual_norms
```

Eigenvectors are not saved by default. Add `--store-eigenvectors` only for
small runs.

## Python API

```python
from goattm.analysis.hessian_landscape import (
    HessianLandscapeConfig,
    run_four_hessian_landscape_from_checkpoint,
)

result = run_four_hessian_landscape_from_checkpoint(
    manifest_path="path/to/train_manifest.npz",
    checkpoint_path="path/to/checkpoints/best.npz",
    output_dir="path/to/hessian_landscape",
    config=HessianLandscapeConfig(max_dt=0.02, k=8, which_values=("LA", "SA")),
)
```

For custom experiments, call `run_four_hessian_landscape(...)` directly with an
already constructed `dynamics` and `decoder`.
