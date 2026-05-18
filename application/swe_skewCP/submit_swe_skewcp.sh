#!/bin/bash
#SBATCH --job-name=goattm_swe_skewcp
#SBATCH --partition=spr
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=112
#SBATCH --time=36:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT}"
SWE_SKEWCP_ROOT="${REPO_ROOT}/application/swe_skewCP"
cd "${REPO_ROOT}"

CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin}"
PY_BIN="${PY_BIN:-${CONDA_BIN_DIR}/python3}"
MPIRUN_BIN="${MPIRUN_BIN:-ibrun}"
CONDA_BASE="${CONDA_BASE:-/work2/08667/yuuuhang/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-GOATTM311}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/work2/08667/yuuuhang/stampede3/envs/${CONDA_ENV_NAME}}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MANIFEST_PATH="${MANIFEST_PATH:-/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data/manifest.npz}"
NTRAIN="${NTRAIN:-1120}"
NTEST="${NTEST:-560}"
SAMPLE_COUNT="${SAMPLE_COUNT:-$((NTRAIN + NTEST))}"
MPI_RANKS="${MPI_RANKS:-${SLURM_NTASKS}}"
LATENT_RANK="${LATENT_RANK:-100}"
SKEW_CP_RANK="${SKEW_CP_RANK:-30}"
OPTIMIZER="${OPTIMIZER:-gradient_descent}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-none}"
SKEW_CP_INIT_SCALE="${SKEW_CP_INIT_SCALE:-1e-4}"
SKEW_CP_TARGET_H_RATIO="${SKEW_CP_TARGET_H_RATIO:-1e-3}"
SKEW_CP_ZERO_INIT="${SKEW_CP_ZERO_INIT:-0}"
MAX_DT="${MAX_DT:-0.0002}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
NORMALIZATION_TARGET_MAX_ABS="${NORMALIZATION_TARGET_MAX_ABS:-0.9}"
QOI_STRIDE="${QOI_STRIDE:-1}"
DECODER_FORM="${DECODER_FORM:-V1v}"
LATENT_EMBEDDING_MODE="${LATENT_EMBEDDING_MODE:-qoi_augmentation}"
LATENT_EMBEDDING_AUGMENTATION_SEED="${LATENT_EMBEDDING_AUGMENTATION_SEED:-20260507}"
LATENT_EMBEDDING_AUGMENTATION_SCALE="${LATENT_EMBEDDING_AUGMENTATION_SCALE:-0.1}"
OPINF_REG_W="${OPINF_REG_W:-1e-4}"
OPINF_REG_H="${OPINF_REG_H:-1e-4}"
OPINF_REG_B="${OPINF_REG_B:-1e-4}"
OPINF_REG_C="${OPINF_REG_C:-1e-6}"
DECODER_REG_V1="${DECODER_REG_V1:-1e-7}"
DECODER_REG_V2="${DECODER_REG_V2:-1e-7}"
DECODER_REG_V0="${DECODER_REG_V0:-1e-7}"
DYNAMICS_REG_A="${DYNAMICS_REG_A:-1e-6}"
DYNAMICS_REG_SKEW_CP="${DYNAMICS_REG_SKEW_CP:-1e-4}"
DYNAMICS_REG_B="${DYNAMICS_REG_B:-1e-4}"
DYNAMICS_REG_C="${DYNAMICS_REG_C:-1e-4}"
LBFGS_MAXCOR="${LBFGS_MAXCOR:-20}"
LBFGS_FTOL="${LBFGS_FTOL:-1e-12}"
LBFGS_GTOL="${LBFGS_GTOL:-1e-8}"
LBFGS_MAXLS="${LBFGS_MAXLS:-30}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
TEST_EVERY="${TEST_EVERY:-1}"
KEEP_ITERATION_CHECKPOINTS="${KEEP_ITERATION_CHECKPOINTS:-1}"

RUN_STEM="${RUN_STEM:-$(date +%Y%m%d_%H%M%S)_swe_skewcp_ntrain${NTRAIN}_ntest${NTEST}_${OPTIMIZER}}"
RUN_TAG="${RUN_STEM}_r${LATENT_RANK}_R${SKEW_CP_RANK}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SWE_SKEWCP_ROOT}/outputs/slurm_runs}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SWE_SKEWCP_ROOT}/batchout"

source "${CONDA_BASE}/bin/activate" "${CONDA_ENV_PREFIX}"

print_numba_status() {
  "${PY_BIN}" - <<'PY'
import os

disabled = os.environ.get("GOATTM_DISABLE_NUMBA", "").strip().lower() in {"1", "true", "yes", "on"}
print(f"GOATTM_DISABLE_NUMBA: {os.environ.get('GOATTM_DISABLE_NUMBA', 'unset')}")
try:
    import numba
except Exception as exc:
    print(f"Numba import: unavailable ({type(exc).__name__}: {exc})")
    print("Numba acceleration: disabled")
else:
    print(f"Numba import: available ({numba.__version__})")
    if disabled:
        print("Numba acceleration: disabled by GOATTM_DISABLE_NUMBA")
    else:
        print("Numba acceleration: enabled")
PY
}

if ! [[ "${NTRAIN}" =~ ^[0-9]+$ ]] || [[ "${NTRAIN}" -le 0 ]]; then
  echo "NTRAIN must be a positive integer, got '${NTRAIN}'." >&2
  exit 2
fi
if ! [[ "${NTEST}" =~ ^[0-9]+$ ]] || [[ "${NTEST}" -le 0 ]]; then
  echo "NTEST must be a positive integer, got '${NTEST}'." >&2
  exit 2
fi
if ! [[ "${SAMPLE_COUNT}" =~ ^[0-9]+$ ]] || [[ "${SAMPLE_COUNT}" -le 1 ]]; then
  echo "SAMPLE_COUNT must be an integer greater than 1, got '${SAMPLE_COUNT}'." >&2
  exit 2
fi
if [[ $((NTRAIN + NTEST)) -ne "${SAMPLE_COUNT}" ]]; then
  echo "SAMPLE_COUNT=${SAMPLE_COUNT} must equal NTRAIN+NTEST=$((NTRAIN + NTEST))." >&2
  exit 2
fi
if ! [[ "${MPI_RANKS}" =~ ^[0-9]+$ ]] || [[ "${MPI_RANKS}" -le 0 ]]; then
  echo "MPI_RANKS must be a positive integer, got '${MPI_RANKS}'." >&2
  exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" ]] && [[ "${MPI_RANKS}" -gt "${SLURM_NTASKS}" ]]; then
  echo "MPI_RANKS=${MPI_RANKS} exceeds allocated SLURM_NTASKS=${SLURM_NTASKS}." >&2
  exit 2
fi
if ! [[ "${LATENT_RANK}" =~ ^[0-9]+$ ]] || [[ "${LATENT_RANK}" -le 0 ]]; then
  echo "LATENT_RANK must be a positive integer, got '${LATENT_RANK}'." >&2
  exit 2
fi
if ! [[ "${SKEW_CP_RANK}" =~ ^[0-9]+$ ]] || [[ "${SKEW_CP_RANK}" -le 0 ]]; then
  echo "SKEW_CP_RANK must be a positive integer, got '${SKEW_CP_RANK}'." >&2
  exit 2
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Processed manifest not found: ${MANIFEST_PATH}" >&2
  echo "Run the SWE preprocessing workflow first." >&2
  exit 1
fi

echo "Job start: $(date)"
echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PY_BIN}"
print_numba_status
echo "Launcher: ${MPIRUN_BIN}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_NNODES: ${SLURM_NNODES:-unset}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-unset}"
echo "MPI_RANKS: ${MPI_RANKS}"
echo "MANIFEST_PATH: ${MANIFEST_PATH}"
echo "NTRAIN: ${NTRAIN}"
echo "NTEST: ${NTEST}"
echo "SAMPLE_COUNT: ${SAMPLE_COUNT}"
echo "LATENT_RANK: ${LATENT_RANK}"
echo "SKEW_CP_RANK: ${SKEW_CP_RANK}"
echo "OPTIMIZER: ${OPTIMIZER}"
echo "TIME_INTEGRATOR: ${TIME_INTEGRATOR}"
echo "MAX_DT: ${MAX_DT}"
echo "MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "QOI_STRIDE: ${QOI_STRIDE}"
echo "DECODER_FORM: ${DECODER_FORM}"
echo "GRADIENT_CLIP_NORM: ${GRADIENT_CLIP_NORM}"
echo "SKEW_CP_INIT_SCALE: ${SKEW_CP_INIT_SCALE}"
echo "SKEW_CP_TARGET_H_RATIO: ${SKEW_CP_TARGET_H_RATIO}"
echo "SKEW_CP_ZERO_INIT: ${SKEW_CP_ZERO_INIT}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "CHECKPOINT_EVERY: ${CHECKPOINT_EVERY}"
echo "TEST_EVERY: ${TEST_EVERY}"
echo "KEEP_ITERATION_CHECKPOINTS: ${KEEP_ITERATION_CHECKPOINTS}"

PY_ARGS=(
  "${SWE_SKEWCP_ROOT}/codes/run_swe_skewcp_smoke.py"
  --manifest-path "${MANIFEST_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --sample-count "${SAMPLE_COUNT}"
  --ntrain "${NTRAIN}"
  --ntest "${NTEST}"
  --latent-rank "${LATENT_RANK}"
  --skew-cp-rank "${SKEW_CP_RANK}"
  --max-dt "${MAX_DT}"
  --time-integrator "${TIME_INTEGRATOR}"
  --optimizer "${OPTIMIZER}"
  --max-iterations "${MAX_ITERATIONS}"
  --learning-rate "${LEARNING_RATE}"
  --skew-cp-init-scale "${SKEW_CP_INIT_SCALE}"
  --skew-cp-target-h-ratio "${SKEW_CP_TARGET_H_RATIO}"
  --skew-cp-zero-init "${SKEW_CP_ZERO_INIT}"
  --latent-embedding-mode "${LATENT_EMBEDDING_MODE}"
  --latent-embedding-augmentation-seed "${LATENT_EMBEDDING_AUGMENTATION_SEED}"
  --latent-embedding-augmentation-scale "${LATENT_EMBEDDING_AUGMENTATION_SCALE}"
  --normalization-target-max-abs "${NORMALIZATION_TARGET_MAX_ABS}"
  --qoi-stride "${QOI_STRIDE}"
  --decoder-form "${DECODER_FORM}"
  --opinf-reg-w "${OPINF_REG_W}"
  --opinf-reg-h "${OPINF_REG_H}"
  --opinf-reg-b "${OPINF_REG_B}"
  --opinf-reg-c "${OPINF_REG_C}"
  --decoder-reg-v1 "${DECODER_REG_V1}"
  --decoder-reg-v2 "${DECODER_REG_V2}"
  --decoder-reg-v0 "${DECODER_REG_V0}"
  --dynamics-reg-a "${DYNAMICS_REG_A}"
  --dynamics-reg-skew-cp "${DYNAMICS_REG_SKEW_CP}"
  --dynamics-reg-b "${DYNAMICS_REG_B}"
  --dynamics-reg-c "${DYNAMICS_REG_C}"
  --lbfgs-maxcor "${LBFGS_MAXCOR}"
  --lbfgs-ftol "${LBFGS_FTOL}"
  --lbfgs-gtol "${LBFGS_GTOL}"
  --lbfgs-maxls "${LBFGS_MAXLS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --test-every "${TEST_EVERY}"
  --keep-iteration-checkpoints "${KEEP_ITERATION_CHECKPOINTS}"
)

case "${GRADIENT_CLIP_NORM,,}" in
  ""|"none"|"null"|"off"|"false"|"0")
    ;;
  *)
    PY_ARGS+=(--gradient-clip-norm "${GRADIENT_CLIP_NORM}")
    ;;
esac

"${MPIRUN_BIN}" -n "${MPI_RANKS}" "${PY_BIN}" "${PY_ARGS[@]}" "$@"

echo "Job end: $(date)"
echo "Output dir: ${OUTPUT_DIR}"
