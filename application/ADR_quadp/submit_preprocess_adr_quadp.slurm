#!/bin/bash
#SBATCH --job-name=goattm_adr_pre
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
GOAM_CLEAN_ROOT="${GOAM_CLEAN_ROOT:-/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin}"
PY_BIN="${PY_BIN:-${CONDA_BIN_DIR}/python}"

DEFAULT_TRAIN_ROOT="${GOAM_CLEAN_ROOT}/Example/ADR_LinearQoI/tmpdirectory/ADR_LinearQoI_quadp_opinf_trainsize_112_threads_112"
DEFAULT_TEST_ROOT="${GOAM_CLEAN_ROOT}/Example/ADR_LinearQoI/tmpdirectory/ADR_LinearQoI_quadp_validate_trainsize_112_threads_112"
DEFAULT_RAW_NPZ_ROOT="${APP_ROOT}/data/raw_npz/train112_test20"

TRAIN_ROOT="${TRAIN_ROOT:-${DEFAULT_RAW_NPZ_ROOT}}"
TEST_ROOT="${TEST_ROOT:-${TRAIN_ROOT}}"
TRAIN_COUNT="${TRAIN_COUNT:-112}"
TEST_COUNT="${TEST_COUNT:-20}"
QOI_STRIDE="${QOI_STRIDE:-1}"
TIME_MODE="${TIME_MODE:-physical}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/data/processed_data/train${TRAIN_COUNT}_test${TEST_COUNT}}"

mkdir -p "${APP_ROOT}/batchout"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "Job start: $(date)"
echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PY_BIN}"
echo "GOAM_CLEAN_ROOT: ${GOAM_CLEAN_ROOT}"
echo "TRAIN_ROOT: ${TRAIN_ROOT}"
echo "TEST_ROOT: ${TEST_ROOT}"
echo "TRAIN_COUNT: ${TRAIN_COUNT}"
echo "TEST_COUNT: ${TEST_COUNT}"
echo "Fallback original GOAM train root: ${DEFAULT_TRAIN_ROOT}"
echo "Fallback original GOAM test root: ${DEFAULT_TEST_ROOT}"
echo "QOI_STRIDE: ${QOI_STRIDE}"
echo "TIME_MODE: ${TIME_MODE}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"

"${PY_BIN}" "${APP_ROOT}/codes/preprocess_adr_quadp_npz_dataset.py" \
  --train-root "${TRAIN_ROOT}" \
  --test-root "${TEST_ROOT}" \
  --train-count "${TRAIN_COUNT}" \
  --test-count "${TEST_COUNT}" \
  --qoi-stride "${QOI_STRIDE}" \
  --time-mode "${TIME_MODE}" \
  --output-root "${OUTPUT_ROOT}"

echo "Job finished: $(date)"
