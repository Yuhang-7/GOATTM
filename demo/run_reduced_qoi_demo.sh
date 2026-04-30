#!/usr/bin/env bash
# Run the configurable GOATTM reduced-QoI demo using the fenicsx-clean conda
# environment by default.
#
# Usage:
#   bash demo/run_reduced_qoi_demo.sh [NTRAIN] [NTEST] [extra python args...]
#
# Examples:
#   bash demo/run_reduced_qoi_demo.sh
#   bash demo/run_reduced_qoi_demo.sh 10 10
#   bash demo/run_reduced_qoi_demo.sh 100 100 --max-iterations 50
#   LATENT_RANK=10 bash demo/run_reduced_qoi_demo.sh 10 10
#   MPI_RANKS=20 LATENT_RANK=6 OPTIMIZER=adam bash demo/run_reduced_qoi_demo.sh 100 100
#
# By default MPI_RANKS uses the currently available launcher allocation when
# possible; otherwise it falls back to NTRAIN.
#
# Default behavior in this wrapper:
# - activates conda env fenicsx-clean unless GOATTM_SKIP_CONDA_ACTIVATE=1
# - writes outputs under demo/outputs unless OUTPUT_DIR is set

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NTRAIN="${1:-10}"
NTEST="${2:-10}"
if [[ $# -ge 1 ]]; then
  shift
fi
if [[ $# -ge 1 ]]; then
  shift
fi

detect_available_ranks() {
  if [[ -n "${SLURM_NTASKS:-}" ]]; then
    echo "${SLURM_NTASKS}"
    return 0
  fi
  if [[ -n "${OMPI_COMM_WORLD_SIZE:-}" ]]; then
    echo "${OMPI_COMM_WORLD_SIZE}"
    return 0
  fi
  if [[ -n "${PMI_SIZE:-}" ]]; then
    echo "${PMI_SIZE}"
    return 0
  fi
  echo "${NTRAIN}"
}

MPI_RANKS="${MPI_RANKS:-$(detect_available_ranks)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
LATENT_RANK="${LATENT_RANK:-10}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
MAX_DT="${MAX_DT:-0.01}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-implicit_midpoint}"
NORMALIZATION_TARGET_MAX_ABS="${NORMALIZATION_TARGET_MAX_ABS:-0.9}"
SEED="${SEED:-20260428}"
LBFGS_MAXCOR="${LBFGS_MAXCOR:-20}"
LBFGS_FTOL="${LBFGS_FTOL:-1e-12}"
LBFGS_GTOL="${LBFGS_GTOL:-1e-8}"
LBFGS_MAXLS="${LBFGS_MAXLS:-30}"
BFGS_GTOL="${BFGS_GTOL:-1e-6}"
BFGS_C1="${BFGS_C1:-1e-4}"
BFGS_C2="${BFGS_C2:-0.9}"
BFGS_XRTOL="${BFGS_XRTOL:-1e-7}"
OPINF_REG_W="${OPINF_REG_W:-1e-4}"
OPINF_REG_H="${OPINF_REG_H:-1e-4}"
OPINF_REG_B="${OPINF_REG_B:-1e-4}"
OPINF_REG_C="${OPINF_REG_C:-1e-6}"
DECODER_REG_V1="${DECODER_REG_V1:-1e-7}"
DECODER_REG_V2="${DECODER_REG_V2:-1e-7}"
DECODER_REG_V0="${DECODER_REG_V0:-1e-7}"
DYNAMICS_REG_S="${DYNAMICS_REG_S:-1e-4}"
DYNAMICS_REG_W="${DYNAMICS_REG_W:-1e-4}"
DYNAMICS_REG_MU_H="${DYNAMICS_REG_MU_H:-1e-4}"
DYNAMICS_REG_B="${DYNAMICS_REG_B:-1e-4}"
DYNAMICS_REG_C="${DYNAMICS_REG_C:-1e-4}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fenicsx-clean}"
GOATTM_SKIP_CONDA_ACTIVATE="${GOATTM_SKIP_CONDA_ACTIVATE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/demo/outputs/reduced_qoi_optimization_demo}"

if ! [[ "${NTRAIN}" =~ ^[0-9]+$ ]] || [[ "${NTRAIN}" -le 0 ]]; then
  echo "NTRAIN must be a positive integer, got '${NTRAIN}'." >&2
  exit 2
fi
if ! [[ "${NTEST}" =~ ^[0-9]+$ ]] || [[ "${NTEST}" -le 0 ]]; then
  echo "NTEST must be a positive integer, got '${NTEST}'." >&2
  exit 2
fi
if ! [[ "${MPI_RANKS}" =~ ^[0-9]+$ ]] || [[ "${MPI_RANKS}" -le 0 ]]; then
  echo "MPI_RANKS must be a positive integer, got '${MPI_RANKS}'." >&2
  exit 2
fi
if ! [[ "${LATENT_RANK}" =~ ^[0-9]+$ ]] || [[ "${LATENT_RANK}" -le 0 ]]; then
  echo "LATENT_RANK must be a positive integer, got '${LATENT_RANK}'." >&2
  exit 2
fi
if ! [[ "${MAX_ITERATIONS}" =~ ^[0-9]+$ ]] || [[ "${MAX_ITERATIONS}" -le 0 ]]; then
  echo "MAX_ITERATIONS must be a positive integer, got '${MAX_ITERATIONS}'." >&2
  exit 2
fi

activate_conda_env() {
  if [[ "${GOATTM_SKIP_CONDA_ACTIVATE}" == "1" ]]; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    if [[ -x "/work2/08667/yuuuhang/anaconda3/bin/conda" ]]; then
      # Match the existing shared Anaconda install on this system.
      source "/work2/08667/yuuuhang/anaconda3/bin/activate" "/work2/08667/yuuuhang/anaconda3/envs/${CONDA_ENV_NAME}"
      return 0
    fi
    echo "conda was not found and fallback activate path is unavailable." >&2
    echo "Set GOATTM_SKIP_CONDA_ACTIVATE=1 if the environment is already active." >&2
    exit 2
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
}

activate_conda_env
mkdir -p "${OUTPUT_DIR}"

PY_ARGS=(
  "${REPO_ROOT}/demo/run_reduced_qoi_demo.py"
  --ntrain "${NTRAIN}"
  --ntest "${NTEST}"
  --latent-rank "${LATENT_RANK}"
  --optimizer "${OPTIMIZER}"
  --max-iterations "${MAX_ITERATIONS}"
  --max-dt "${MAX_DT}"
  --time-integrator "${TIME_INTEGRATOR}"
  --normalization-target-max-abs "${NORMALIZATION_TARGET_MAX_ABS}"
  --seed "${SEED}"
  --lbfgs-maxcor "${LBFGS_MAXCOR}"
  --lbfgs-ftol "${LBFGS_FTOL}"
  --lbfgs-gtol "${LBFGS_GTOL}"
  --lbfgs-maxls "${LBFGS_MAXLS}"
  --bfgs-gtol "${BFGS_GTOL}"
  --bfgs-c1 "${BFGS_C1}"
  --bfgs-c2 "${BFGS_C2}"
  --bfgs-xrtol "${BFGS_XRTOL}"
  --opinf-reg-w "${OPINF_REG_W}"
  --opinf-reg-h "${OPINF_REG_H}"
  --opinf-reg-b "${OPINF_REG_B}"
  --opinf-reg-c "${OPINF_REG_C}"
  --decoder-reg-v1 "${DECODER_REG_V1}"
  --decoder-reg-v2 "${DECODER_REG_V2}"
  --decoder-reg-v0 "${DECODER_REG_V0}"
  --dynamics-reg-s "${DYNAMICS_REG_S}"
  --dynamics-reg-w "${DYNAMICS_REG_W}"
  --dynamics-reg-mu-h "${DYNAMICS_REG_MU_H}"
  --dynamics-reg-b "${DYNAMICS_REG_B}"
  --dynamics-reg-c "${DYNAMICS_REG_C}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  PY_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

echo "Running GOATTM demo from ${REPO_ROOT}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "MPI ranks: ${MPI_RANKS}"
echo "Output dir: ${OUTPUT_DIR}"

"${MPIRUN_BIN}" -n "${MPI_RANKS}" "${PYTHON_BIN}" "${PY_ARGS[@]}" "$@"
