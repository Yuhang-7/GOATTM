#!/usr/bin/env bash
# Login-node submitter for the Navier-Stokes oldGOAM baseline rerun.

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/Navier_Stokes"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
JOB_SCRIPT="${APP_ROOT}/submit_rerun_oldgoam.slurm"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  application/Navier_Stokes/submit_navier_stokes_oldgoam_baseline.sh [--dry-run]

Prepared job:
  ntrain = 896, ntest = 104
  latent ranks = 12,14,16
  1 node, default script allocation, RK4, reg_f=1e-7, reg_g=1e-9, no spectral penalty
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
if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Missing job script: ${JOB_SCRIPT}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q --test-only %q
' "${SBATCH_BIN}" "${JOB_SCRIPT}"
  "${SBATCH_BIN}" --test-only "${JOB_SCRIPT}"
else
  "${SBATCH_BIN}" "${JOB_SCRIPT}"
fi
