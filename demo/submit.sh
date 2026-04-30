#!/bin/bash
#SBATCH --job-name=goattm_demo
#SBATCH --partition=spr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=56
#SBATCH --time=00:10:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/demo/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/demo/batchout/%x-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/anaconda3/envs/fenicsx-clean/bin}"
PY_BIN="${PY_BIN:-${CONDA_BIN_DIR}/python3}"
MPIRUN_BIN="${MPIRUN_BIN:-ibrun}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NTRAIN="${NTRAIN:-448}"
NTEST="${NTEST:-112}"
MPI_RANKS="${MPI_RANKS:-${SLURM_NTASKS}}"
LATENT_RANK_START="${LATENT_RANK_START:-8}"
LATENT_RANK_END="${LATENT_RANK_END:-10}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
MAX_ITERATIONS="${MAX_ITERATIONS:-4000}"
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

RUN_STEM="${RUN_STEM:-$(date +%Y%m%d_%H%M%S)_goattm_demo_ntrain${NTRAIN}_ntest${NTEST}_${OPTIMIZER}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/demo/outputs/slurm_runs}"
mkdir -p "${OUTPUT_ROOT}"

CONDA_BASE=/work2/08667/yuuuhang/anaconda3
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/fenicsx-clean

#export LD_PRELOAD="$(echo "${LD_PRELOAD:-}" | tr ':' '\n' | grep -v xalt | tr '\n' ':' | sed 's/:$//')"
#export LD_LIBRARY_PATH="${TACC_IMPI_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

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
if ! [[ "${LATENT_RANK_START}" =~ ^[0-9]+$ ]] || [[ "${LATENT_RANK_START}" -le 0 ]]; then
  echo "LATENT_RANK_START must be a positive integer, got '${LATENT_RANK_START}'." >&2
  exit 2
fi
if ! [[ "${LATENT_RANK_END}" =~ ^[0-9]+$ ]] || [[ "${LATENT_RANK_END}" -le 0 ]]; then
  echo "LATENT_RANK_END must be a positive integer, got '${LATENT_RANK_END}'." >&2
  exit 2
fi
if [[ "${LATENT_RANK_END}" -lt "${LATENT_RANK_START}" ]]; then
  echo "LATENT_RANK_END=${LATENT_RANK_END} must be >= LATENT_RANK_START=${LATENT_RANK_START}." >&2
  exit 2
fi
if [[ "${MPI_RANKS}" -gt "${SLURM_NTASKS}" ]]; then
  echo "MPI_RANKS=${MPI_RANKS} exceeds allocated SLURM_NTASKS=${SLURM_NTASKS}." >&2
  exit 2
fi

echo "Job start: $(date)"
echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PY_BIN}"
echo "Launcher: ${MPIRUN_BIN}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "MPI_RANKS: ${MPI_RANKS}"
echo "NTRAIN: ${NTRAIN}"
echo "NTEST: ${NTEST}"
echo "LATENT_RANK range: ${LATENT_RANK_START}..${LATENT_RANK_END}"
echo "OPTIMIZER: ${OPTIMIZER}"
echo "TIME_INTEGRATOR: ${TIME_INTEGRATOR}"
echo "NORMALIZATION_TARGET_MAX_ABS: ${NORMALIZATION_TARGET_MAX_ABS}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"

export PYTHON_BIN="${PY_BIN}"
export MPIRUN_BIN
export MPI_RANKS
export OPTIMIZER
export MAX_ITERATIONS
export MAX_DT
export TIME_INTEGRATOR
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

for LATENT_RANK in $(seq "${LATENT_RANK_START}" "${LATENT_RANK_END}"); do
  RUN_TAG="${RUN_STEM}_r${LATENT_RANK}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
  mkdir -p "${OUTPUT_DIR}"
  export LATENT_RANK
  export OUTPUT_DIR

  echo "----------------------------------------"
  echo "Starting LATENT_RANK=${LATENT_RANK} at $(date)"
  echo "RUN_TAG: ${RUN_TAG}"
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"

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
    --output-dir "${OUTPUT_DIR}"
  )

  echo "Running GOATTM demo from ${REPO_ROOT}"
  echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
  echo "MPI ranks: ${MPI_RANKS}"
  echo "Output dir: ${OUTPUT_DIR}"

  "${MPIRUN_BIN}" "${PY_BIN}" "${PY_ARGS[@]}" "$@"

  echo "Finished LATENT_RANK=${LATENT_RANK} at $(date)"
done

echo "Job end: $(date)"
echo "Output root: ${OUTPUT_ROOT}"
