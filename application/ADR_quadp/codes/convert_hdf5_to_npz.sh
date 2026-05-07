#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
GOAM_CLEAN_ROOT="${GOAM_CLEAN_ROOT:-/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean}"
PYTHON="${HDF5_PY_BIN:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python}"

INPUT_FILE="${INPUT_FILE:-${GOAM_CLEAN_ROOT}/Example/ADR_LinearQoI/dataset/ADR_LinearQoI.hdf5}"
TRAIN_COUNT="${TRAIN_COUNT:-112}"
TEST_COUNT="${TEST_COUNT:-20}"
SAMPLE_START="${SAMPLE_START:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-${TRAIN_COUNT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/data/raw_npz/train${TRAIN_COUNT}_test${TEST_COUNT}}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

exec "${PYTHON}" "${REPO_ROOT}/src/goattm/preprocess/hdf5_to_goam_npz.py" \
  --input-file "${INPUT_FILE}" --output-root "${OUTPUT_ROOT}" \
  --train-count "${TRAIN_COUNT}" --test-count "${TEST_COUNT}" \
  --sample-start "${SAMPLE_START}" --chunk-size "${CHUNK_SIZE}" "$@"
