#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
JOB_SCRIPT="${REPO_ROOT}/application/Navier_Stokes/submit_rerun_oldgoam_test.slurm"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: application/Navier_Stokes/submit_navier_stokes_oldgoam_test.sh [--dry-run]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "${REPO_ROOT}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q --test-only %q
' "${SBATCH_BIN}" "${JOB_SCRIPT}"
  "${SBATCH_BIN}" --test-only "${JOB_SCRIPT}"
else
  "${SBATCH_BIN}" --parsable "${JOB_SCRIPT}"
fi
