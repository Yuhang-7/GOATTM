#!/bin/bash
#SBATCH --job-name=goattm_demo
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yuhang/Myresearch/GOATTM/demo/batchout/%x-%j.out
#SBATCH --error=/storage/yuhang/Myresearch/GOATTM/demo/batchout/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/.conda/envs/fenicsx-env/bin}"
PY_BIN="${PY_BIN:-${CONDA_BIN_DIR}/python3}"
IBRUN_BIN="${IBRUN_BIN:-ibrun}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NTRAIN="${NTRAIN:-112}"
NTEST="${NTEST:-112}"
MPI_RANKS="${MPI_RANKS:-${SLURM_NTASKS}}"
LATENT_RANK="${LATENT_RANK:-4}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
MAX_DT="${MAX_DT:-0.01}"
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

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_goattm_demo_ntrain${NTRAIN}_ntest${NTEST}_r${LATENT_RANK}_${OPTIMIZER}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/demo/outputs/slurm_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"
mkdir -p "${OUTPUT_DIR}"

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
if [[ "${MPI_RANKS}" -gt "${SLURM_NTASKS}" ]]; then
  echo "MPI_RANKS=${MPI_RANKS} exceeds allocated SLURM_NTASKS=${SLURM_NTASKS}." >&2
  exit 2
fi

echo "Job start: $(date)"
echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PY_BIN}"
echo "Launcher: ${IBRUN_BIN}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "MPI_RANKS: ${MPI_RANKS}"
echo "NTRAIN: ${NTRAIN}"
echo "NTEST: ${NTEST}"
echo "LATENT_RANK: ${LATENT_RANK}"
echo "OPTIMIZER: ${OPTIMIZER}"
echo "NORMALIZATION_TARGET_MAX_ABS: ${NORMALIZATION_TARGET_MAX_ABS}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

export PYTHON_BIN="${PY_BIN}"
export MPIRUN_BIN="${IBRUN_BIN}"
export OUTPUT_DIR
export MPI_RANKS
export LATENT_RANK
export OPTIMIZER
export MAX_ITERATIONS
export MAX_DT
export NORMALIZATION_TARGET_MAX_ABS
export SEED
export LBFGS_MAXCOR
export LBFGS_FTOL
export LBFGS_GTOL
export LBFGS_MAXLS
export BFGS_GTOL
export BFGS_C1
export BFGS_C2
export BFGS_XRTOL
export OPINF_REG_W
export OPINF_REG_H
export OPINF_REG_B
export OPINF_REG_C
export DECODER_REG_V1
export DECODER_REG_V2
export DECODER_REG_V0
export DYNAMICS_REG_S
export DYNAMICS_REG_W
export DYNAMICS_REG_MU_H
export DYNAMICS_REG_B
export DYNAMICS_REG_C

bash "${REPO_ROOT}/demo/run_reduced_qoi_demo.sh" "${NTRAIN}" "${NTEST}" "$@"

echo "Job end: $(date)"
echo "Final output directory: ${OUTPUT_DIR}"
