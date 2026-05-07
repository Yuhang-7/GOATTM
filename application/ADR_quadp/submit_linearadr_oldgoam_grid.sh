#!/usr/bin/env bash
# Login-node submitter for the LinearADR oldGOAM rerun set.
# Submits the three prepared Slurm jobs: n=16/32/64 serial, n=128, n=256.

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  application/ADR_quadp/submit_linearadr_oldgoam_grid.sh [--dry-run]

Prepared jobs:
  ntrain = 16, 32, 64: 1 node, 64 MPI ranks, serial inside one Slurm job
  ntrain = 128:         1 node, 64 MPI ranks
  ntrain = 256:         2 nodes, 64 tasks per node, 128 MPI ranks

All jobs use ranks 12,14,16; RK4; reg_f=1e-7; reg_g=1e-9; no spectral penalty.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"

jobs=(
  "${APP_ROOT}/submit_linearadr_oldgoam_n16_32_64.slurm"
  "${APP_ROOT}/submit_linearadr_oldgoam_n128.slurm"
  "${APP_ROOT}/submit_linearadr_oldgoam_n256.slurm"
)

for job in "${jobs[@]}"; do
  if [[ ! -f "${job}" ]]; then
    echo "Missing job script: ${job}" >&2
    exit 2
  fi
done

for job in "${jobs[@]}"; do
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q --test-only %q
' "${SBATCH_BIN}" "${job}"
    "${SBATCH_BIN}" --test-only "${job}"
  else
    "${SBATCH_BIN}" "${job}"
  fi
done
