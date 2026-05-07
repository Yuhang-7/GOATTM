#!/bin/bash
#SBATCH --job-name=linearadr_oldgoam
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --time=12:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
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

MANIFEST_PATH="${MANIFEST_PATH:-${APP_ROOT}/data/processed_data/linearadr_all_samples_stride1/manifest.npz}"
NTRAIN="${NTRAIN:-896}"
NTEST="${NTEST:-104}"
SAMPLE_COUNT="${SAMPLE_COUNT:-$((NTRAIN + NTEST))}"
MPI_RANKS="${MPI_RANKS:-${SLURM_NTASKS:-112}}"
LATENT_RANK_SEQUENCE="${LATENT_RANK_SEQUENCE:-12 14 16}"

DATASET_NAME="${DATASET_NAME:-linearadr}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
MAX_DT="${MAX_DT:-0.01}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
DYNAMIC_FORM="${DYNAMIC_FORM:-AHBc}"
DECODER_FORM="${DECODER_FORM:-V1V2v}"
NORMALIZATION_TARGET_MAX_ABS="${NORMALIZATION_TARGET_MAX_ABS:-0.9}"
LATENT_EMBEDDING_MODE="${LATENT_EMBEDDING_MODE:-qoi_augmentation}"
LATENT_EMBEDDING_AUGMENTATION_SEED="${LATENT_EMBEDDING_AUGMENTATION_SEED:-12345}"
LATENT_EMBEDDING_AUGMENTATION_SCALE="${LATENT_EMBEDDING_AUGMENTATION_SCALE:-0.1}"

LBFGS_MAXCOR="${LBFGS_MAXCOR:-20}"
LBFGS_FTOL="${LBFGS_FTOL:-1e-12}"
LBFGS_GTOL="${LBFGS_GTOL:-1e-8}"
LBFGS_MAXLS="${LBFGS_MAXLS:-30}"
BFGS_GTOL="${BFGS_GTOL:-1e-6}"
BFGS_C1="${BFGS_C1:-1e-4}"
BFGS_C2="${BFGS_C2:-0.9}"
BFGS_XRTOL="${BFGS_XRTOL:-1e-7}"

OPINF_REG_W="${OPINF_REG_W:-1e-9}"
OPINF_REG_H="${OPINF_REG_H:-1e-9}"
OPINF_REG_B="${OPINF_REG_B:-1e-9}"
OPINF_REG_C="${OPINF_REG_C:-1e-9}"
DECODER_REG_V1="${DECODER_REG_V1:-1e-7}"
DECODER_REG_V2="${DECODER_REG_V2:-1e-7}"
DECODER_REG_V0="${DECODER_REG_V0:-1e-7}"
DYNAMICS_REG_A="${DYNAMICS_REG_A:-1e-9}"
DYNAMICS_REG_S="${DYNAMICS_REG_S:-0.0}"
DYNAMICS_REG_W="${DYNAMICS_REG_W:-0.0}"
DYNAMICS_REG_MU_H="${DYNAMICS_REG_MU_H:-1e-9}"
DYNAMICS_REG_B="${DYNAMICS_REG_B:-1e-9}"
DYNAMICS_REG_C="${DYNAMICS_REG_C:-1e-9}"
DYNAMICS_REG_SPECTRAL_ABSCISSA="${DYNAMICS_REG_SPECTRAL_ABSCISSA:-0.0}"
DYNAMICS_REG_SPECTRAL_ALPHA="${DYNAMICS_REG_SPECTRAL_ALPHA:-0.0}"

RUN_STEM="${RUN_STEM:-$(date +%Y%m%d_%H%M%S)_linearadr_oldgoam_ntrain${NTRAIN}_ntest${NTEST}_${OPTIMIZER}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/outputs/linearadr_oldgoam_regf1e-7_regg1e-9}"
mkdir -p "${APP_ROOT}/batchout" "${OUTPUT_ROOT}"

if [[ -f "${CONDA_BASE}/bin/activate" ]]; then
  source "${CONDA_BASE}/bin/activate" "${CONDA_ENV_PREFIX}"
fi

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

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -le 0 ]]; then
    echo "${name} must be a positive integer, got '${value}'." >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a nonnegative integer, got '${value}'." >&2
    exit 2
  fi
}

require_positive_int NTRAIN "${NTRAIN}"
require_nonnegative_int NTEST "${NTEST}"
require_positive_int SAMPLE_COUNT "${SAMPLE_COUNT}"
require_positive_int MPI_RANKS "${MPI_RANKS}"
if [[ $((NTRAIN + NTEST)) -ne "${SAMPLE_COUNT}" ]]; then
  echo "SAMPLE_COUNT=${SAMPLE_COUNT} must equal NTRAIN+NTEST=$((NTRAIN + NTEST))." >&2
  exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" && "${MPI_RANKS}" -gt "${SLURM_NTASKS}" ]]; then
  echo "MPI_RANKS=${MPI_RANKS} exceeds allocated SLURM_NTASKS=${SLURM_NTASKS}." >&2
  exit 2
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Processed manifest not found: ${MANIFEST_PATH}" >&2
  echo "Run application/ADR_quadp/run_prepare_linearadr_dataset.sh first." >&2
  exit 1
fi

read -r -a LATENT_RANKS <<< "${LATENT_RANK_SEQUENCE}"
if [[ "${#LATENT_RANKS[@]}" -eq 0 ]]; then
  echo "LATENT_RANK_SEQUENCE must contain at least one integer rank." >&2
  exit 2
fi
for LATENT_RANK in "${LATENT_RANKS[@]}"; do
  require_positive_int LATENT_RANK "${LATENT_RANK}"
done

"${PY_BIN}" - "${MANIFEST_PATH}" "${SAMPLE_COUNT}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from goattm.data import load_npz_sample_manifest

manifest_path = Path(sys.argv[1])
sample_count = int(sys.argv[2])
manifest = load_npz_sample_manifest(manifest_path)
if len(manifest) < sample_count:
    raise SystemExit(f"Manifest has {len(manifest)} samples, requested {sample_count}.")
print(f"Manifest sample count: {len(manifest)}; requested prefix: {sample_count}")
PY

echo "Job start: $(date)"
echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PY_BIN}"
print_numba_status
echo "Launcher: ${MPIRUN_BIN}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_NNODES: ${SLURM_NNODES:-unset}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-unset}"
echo "MPI_RANKS: ${MPI_RANKS}"
echo "OLDGOAM_MODE: enabled (--oldgoam)"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "MANIFEST_PATH: ${MANIFEST_PATH}"
echo "NTRAIN: ${NTRAIN}"
echo "NTEST: ${NTEST}"
echo "SAMPLE_COUNT: ${SAMPLE_COUNT}"
echo "LATENT_RANK_SEQUENCE: ${LATENT_RANK_SEQUENCE}"
echo "DYNAMIC_FORM: ${DYNAMIC_FORM}"
echo "DECODER_FORM: ${DECODER_FORM}"
echo "OPTIMIZER: ${OPTIMIZER}"
echo "TIME_INTEGRATOR: ${TIME_INTEGRATOR}"
echo "LATENT_EMBEDDING_MODE: ${LATENT_EMBEDDING_MODE}"
echo "LATENT_EMBEDDING_AUGMENTATION_SEED: ${LATENT_EMBEDDING_AUGMENTATION_SEED}"
echo "LATENT_EMBEDDING_AUGMENTATION_SCALE: ${LATENT_EMBEDDING_AUGMENTATION_SCALE}"
echo "MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "reg_f(decoder): ${DECODER_REG_V1}, ${DECODER_REG_V2}, ${DECODER_REG_V0}"
echo "reg_g(dynamics): A=${DYNAMICS_REG_A}, H=${DYNAMICS_REG_MU_H}, B=${DYNAMICS_REG_B}, c=${DYNAMICS_REG_C}"
echo "spectral penalty: ${DYNAMICS_REG_SPECTRAL_ABSCISSA}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"

for LATENT_RANK in "${LATENT_RANKS[@]}"; do
  RUN_TAG="${RUN_STEM}_r${LATENT_RANK}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
  mkdir -p "${OUTPUT_DIR}"

  echo "----------------------------------------"
  echo "Starting LATENT_RANK=${LATENT_RANK} at $(date)"
  echo "RUN_TAG: ${RUN_TAG}"
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"

  "${MPIRUN_BIN}" -n "${MPI_RANKS}" "${PY_BIN}" application/common/run_manifest_oldgoam.py \
    --dataset-name "${DATASET_NAME}" \
    --manifest-path "${MANIFEST_PATH}" \
    --sample-count "${SAMPLE_COUNT}" \
    --ntrain "${NTRAIN}" \
    --ntest "${NTEST}" \
    --latent-rank "${LATENT_RANK}" \
    --dynamic-form "${DYNAMIC_FORM}" \
    --decoder-form "${DECODER_FORM}" \
    --oldgoam \
    --latent-embedding-mode "${LATENT_EMBEDDING_MODE}" \
    --latent-embedding-augmentation-seed "${LATENT_EMBEDDING_AUGMENTATION_SEED}" \
    --latent-embedding-augmentation-scale "${LATENT_EMBEDDING_AUGMENTATION_SCALE}" \
    --max-dt "${MAX_DT}" \
    --time-integrator "${TIME_INTEGRATOR}" \
    --optimizer "${OPTIMIZER}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --normalization-target-max-abs "${NORMALIZATION_TARGET_MAX_ABS}" \
    --lbfgs-maxcor "${LBFGS_MAXCOR}" --lbfgs-ftol "${LBFGS_FTOL}" --lbfgs-gtol "${LBFGS_GTOL}" --lbfgs-maxls "${LBFGS_MAXLS}" \
    --bfgs-gtol "${BFGS_GTOL}" --bfgs-c1 "${BFGS_C1}" --bfgs-c2 "${BFGS_C2}" --bfgs-xrtol "${BFGS_XRTOL}" \
    --opinf-reg-w "${OPINF_REG_W}" --opinf-reg-h "${OPINF_REG_H}" --opinf-reg-b "${OPINF_REG_B}" --opinf-reg-c "${OPINF_REG_C}" \
    --decoder-reg-v1 "${DECODER_REG_V1}" --decoder-reg-v2 "${DECODER_REG_V2}" --decoder-reg-v0 "${DECODER_REG_V0}" \
    --dynamics-reg-a "${DYNAMICS_REG_A}" --dynamics-reg-s "${DYNAMICS_REG_S}" --dynamics-reg-w "${DYNAMICS_REG_W}" \
    --dynamics-reg-mu-h "${DYNAMICS_REG_MU_H}" --dynamics-reg-b "${DYNAMICS_REG_B}" --dynamics-reg-c "${DYNAMICS_REG_C}" \
    --dynamics-reg-spectral-abscissa "${DYNAMICS_REG_SPECTRAL_ABSCISSA}" \
    --dynamics-reg-spectral-alpha "${DYNAMICS_REG_SPECTRAL_ALPHA}" \
    --output-dir "${OUTPUT_DIR}"

  echo "Finished LATENT_RANK=${LATENT_RANK} at $(date)"
done

echo "Job end: $(date)"
echo "Output root: ${OUTPUT_ROOT}"
