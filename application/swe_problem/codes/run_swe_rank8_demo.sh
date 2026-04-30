#!/usr/bin/env bash
# Run a small SWE GOATTM optimization demo.
#
# Defaults:
#   64 samples total, 48 train / 16 test
#   latent rank 8
#   RK4 time integration
#   max_dt = 1/600
#
# Usage:
#   bash application/swe_problem/codes/run_swe_rank8_demo.sh [extra python args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_PROBLEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_PROBLEM_ROOT}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
MPI_RANKS="${MPI_RANKS:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fenicsx-clean}"
GOATTM_SKIP_CONDA_ACTIVATE="${GOATTM_SKIP_CONDA_ACTIVATE:-0}"

MANIFEST_PATH="${MANIFEST_PATH:-${SWE_PROBLEM_ROOT}/data/processed_data/manifest.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${SWE_PROBLEM_ROOT}/outputs/swe_rank8_rk4_demo}"
SAMPLE_COUNT="${SAMPLE_COUNT:-64}"
NTRAIN="${NTRAIN:-48}"
NTEST="${NTEST:-16}"
LATENT_RANK="${LATENT_RANK:-8}"
MAX_DT="${MAX_DT:-0.0016666666666666668}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
OPTIMIZER="${OPTIMIZER:-lbfgs}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
NORMALIZATION_TARGET_MAX_ABS="${NORMALIZATION_TARGET_MAX_ABS:-0.9}"
LBFGS_MAXCOR="${LBFGS_MAXCOR:-20}"
LBFGS_FTOL="${LBFGS_FTOL:-1e-12}"
LBFGS_GTOL="${LBFGS_GTOL:-1e-8}"
LBFGS_MAXLS="${LBFGS_MAXLS:-30}"
BFGS_GTOL="${BFGS_GTOL:-1e-6}"
BFGS_C1="${BFGS_C1:-1e-4}"
BFGS_C2="${BFGS_C2:-0.9}"
BFGS_XRTOL="${BFGS_XRTOL:-1e-7}"

activate_conda_env() {
  if [[ "${GOATTM_SKIP_CONDA_ACTIVATE}" == "1" ]]; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    if [[ -x "/work2/08667/yuuuhang/anaconda3/bin/conda" ]]; then
      set +u
      source "/work2/08667/yuuuhang/anaconda3/bin/activate" "/work2/08667/yuuuhang/anaconda3/envs/${CONDA_ENV_NAME}"
      set -u
      return 0
    fi
    echo "conda was not found. Set GOATTM_SKIP_CONDA_ACTIVATE=1 if the environment is already active." >&2
    exit 2
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  set +u
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  set -u
}

activate_conda_env
mkdir -p "${OUTPUT_DIR}"

PY_ARGS=(
  "${SWE_PROBLEM_ROOT}/codes/run_swe_rank8_demo.py"
  --manifest-path "${MANIFEST_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --sample-count "${SAMPLE_COUNT}"
  --ntrain "${NTRAIN}"
  --ntest "${NTEST}"
  --latent-rank "${LATENT_RANK}"
  --max-dt "${MAX_DT}"
  --time-integrator "${TIME_INTEGRATOR}"
  --optimizer "${OPTIMIZER}"
  --max-iterations "${MAX_ITERATIONS}"
  --normalization-target-max-abs "${NORMALIZATION_TARGET_MAX_ABS}"
  --lbfgs-maxcor "${LBFGS_MAXCOR}"
  --lbfgs-ftol "${LBFGS_FTOL}"
  --lbfgs-gtol "${LBFGS_GTOL}"
  --lbfgs-maxls "${LBFGS_MAXLS}"
  --bfgs-gtol "${BFGS_GTOL}"
  --bfgs-c1 "${BFGS_C1}"
  --bfgs-c2 "${BFGS_C2}"
  --bfgs-xrtol "${BFGS_XRTOL}"
)

echo "Running SWE rank-8 demo from ${REPO_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "MPI ranks: ${MPI_RANKS}"
echo "Integrator: ${TIME_INTEGRATOR}, max_dt=${MAX_DT}"

if [[ "${MPI_RANKS}" == "1" ]]; then
  "${PYTHON_BIN}" "${PY_ARGS[@]}" "$@"
else
  "${MPIRUN_BIN}" -n "${MPI_RANKS}" "${PYTHON_BIN}" "${PY_ARGS[@]}" "$@"
fi
