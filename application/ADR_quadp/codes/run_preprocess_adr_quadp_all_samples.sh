#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
PYTHON="${PYTHON:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python}"

: "${ADR_RAW_NPZ_ROOT:=${APP_ROOT}/data/raw_npz/all_samples}"
: "${QOI_STRIDE:=1}"
: "${ADR_PROCESSED_DATA_ROOT:=${APP_ROOT}/data/processed_data/all_samples_stride${QOI_STRIDE}}"
: "${TIME_MODE:=physical}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON}" "${APP_ROOT}/codes/preprocess_adr_quadp_npz_dataset.py" \
  --train-root "${ADR_RAW_NPZ_ROOT}" \
  --no-test \
  --qoi-stride "${QOI_STRIDE}" \
  --time-mode "${TIME_MODE}" \
  --output-root "${ADR_PROCESSED_DATA_ROOT}" \
  "$@"
