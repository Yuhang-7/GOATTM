#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: application/ADR_quadp/submit_linearadr_oldgoam_test_grid.sh [--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "${REPO_ROOT}"
jobs=(
  "${APP_ROOT}/submit_linearadr_oldgoam_n16_32_64_test.slurm"
  "${APP_ROOT}/submit_linearadr_oldgoam_n128_test.slurm"
  "${APP_ROOT}/submit_linearadr_oldgoam_n256_test.slurm"
)
for job in "${jobs[@]}"; do
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q --test-only %q
' "${SBATCH_BIN}" "${job}"
    "${SBATCH_BIN}" --test-only "${job}"
  else
    "${SBATCH_BIN}" --parsable "${job}"
  fi
done
