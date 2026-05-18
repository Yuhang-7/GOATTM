#!/usr/bin/env bash
# Login-node smoke test for SWE skewCP training on the tuckerTT branch.
# This script intentionally runs a tiny case and should not be submitted with sbatch.

set -euo pipefail

REPO_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT"
APP_ROOT="${REPO_ROOT}/application/swe_skewCP"
PY_BIN="${PY_BIN:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python}"
CONDA_BASE="${CONDA_BASE:-/work2/08667/yuuuhang/anaconda3}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311}"

MANIFEST_PATH="${MANIFEST_PATH:-/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data/manifest.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${APP_ROOT}/outputs/login_smoke}"
SAMPLE_COUNT="${SAMPLE_COUNT:-3}"
NTRAIN="${NTRAIN:-2}"
NTEST="${NTEST:-1}"
LATENT_RANK="${LATENT_RANK:-32}"
SKEW_CP_RANK="${SKEW_CP_RANK:-8}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
MAX_DT="${MAX_DT:-0.0016666666666666668}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
OPTIMIZER="${OPTIMIZER:-gradient_descent}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"
SKEW_CP_INIT_SCALE="${SKEW_CP_INIT_SCALE:-1e-4}"
SKEW_CP_ZERO_INIT="${SKEW_CP_ZERO_INIT:-1}"
LATENT_EMBEDDING_MODE="${LATENT_EMBEDDING_MODE:-qoi_augmentation}"
QOI_STRIDE="${QOI_STRIDE:-100}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GOATTM_DISABLE_NUMBA="${GOATTM_DISABLE_NUMBA:-1}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${CONDA_BASE}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/bin/activate" "${CONDA_ENV_PREFIX}"
fi

echo "SWE skewCP login smoke test"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PY_BIN}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "latent_rank=${LATENT_RANK}, skew_cp_rank=${SKEW_CP_RANK}, qoi_stride=${QOI_STRIDE}, optimizer=${OPTIMIZER}, max_iterations=${MAX_ITERATIONS}"

"${PY_BIN}" "${APP_ROOT}/codes/run_swe_skewcp_smoke.py" \
  --manifest-path "${MANIFEST_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-count "${SAMPLE_COUNT}" \
  --ntrain "${NTRAIN}" \
  --ntest "${NTEST}" \
  --latent-rank "${LATENT_RANK}" \
  --skew-cp-rank "${SKEW_CP_RANK}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --max-dt "${MAX_DT}" \
  --time-integrator "${TIME_INTEGRATOR}" \
  --optimizer "${OPTIMIZER}" \
  --learning-rate "${LEARNING_RATE}" \
  --gradient-clip-norm "${GRADIENT_CLIP_NORM}" \
  --skew-cp-init-scale "${SKEW_CP_INIT_SCALE}" \
  --skew-cp-zero-init "${SKEW_CP_ZERO_INIT}" \
  --latent-embedding-mode "${LATENT_EMBEDDING_MODE}" \
  --qoi-stride "${QOI_STRIDE}" \
  "$@"
