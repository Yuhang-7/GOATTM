#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_PROBLEM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SWE_PROBLEM_ROOT}/../.." && pwd)"

: "${SWE_ORIGINAL_DATA_ROOT:=/storage/yuhang/swedata/originaldata/swe_data_2026_510510}"
: "${SWE_PROCESSED_DATA_ROOT:=${SWE_PROBLEM_ROOT}/data/processed_data}"
: "${SWE_INPUT_MODE:=uplift_parameters}"

cd "${REPO_ROOT}"
python "${SWE_PROBLEM_ROOT}/codes/preprocess_swe_npz_dataset.py" \
  --input-root "${SWE_ORIGINAL_DATA_ROOT}" \
  --output-root "${SWE_PROCESSED_DATA_ROOT}" \
  --input-mode "${SWE_INPUT_MODE}" \
  "$@"
