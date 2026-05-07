#!/usr/bin/env bash
# Prepare the canonical LinearADR dataset for GOATTM training.
# Intended for an interactive compute node, not for sbatch submission.

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
GOAM_CLEAN_ROOT="${GOAM_CLEAN_ROOT:-/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean}"
PYTHON="${PYTHON:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python}"

INPUT_FILE="${INPUT_FILE:-${GOAM_CLEAN_ROOT}/Example/LinearADR/dataset/linearADR.hdf5}"
RAW_ROOT="${RAW_ROOT:-${APP_ROOT}/data/raw_npz/linearadr_all_samples}"
QOI_STRIDE="${QOI_STRIDE:-1}"
TIME_MODE="${TIME_MODE:-physical}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${APP_ROOT}/data/processed_data/linearadr_all_samples_stride${QOI_STRIDE}}"
TOTAL_SAMPLE_COUNT="${TOTAL_SAMPLE_COUNT:-1104}"
FORCE="${FORCE:-0}"

export GOATTM_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

load_hdf5_module_if_needed() {
  if command -v h5dump >/dev/null 2>&1; then
    return
  fi
  if command -v module >/dev/null 2>&1; then
    module load hdf5/1.14.4
  fi
  if ! command -v h5dump >/dev/null 2>&1; then
    echo "h5dump was not found. Try: module load hdf5/1.14.4" >&2
    exit 2
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 2
  fi
}

cd "${REPO_ROOT}"
load_hdf5_module_if_needed
require_file "${INPUT_FILE}"
require_file "${PYTHON}"

echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON}"
echo "Input HDF5: ${INPUT_FILE}"
echo "Raw NPZ root: ${RAW_ROOT}"
echo "Processed root: ${PROCESSED_ROOT}"
echo "QOI stride: ${QOI_STRIDE}"
echo "Total samples: ${TOTAL_SAMPLE_COUNT}"
echo "Start: $(date)"

if [[ "${FORCE}" == "1" || ! -f "${RAW_ROOT}/0.npz" || ! -f "${RAW_ROOT}/summary.json" ]]; then
  echo ""
  echo "[1/2] Converting LinearADR HDF5 to canonical raw NPZ..."
  "${PYTHON}" "${REPO_ROOT}/src/goattm/preprocess/hdf5_to_goam_npz.py" \
    --input-file "${INPUT_FILE}" \
    --output-root "${RAW_ROOT}" \
    --train-count "${TOTAL_SAMPLE_COUNT}" \
    --test-count 0 \
    --sample-start 0 \
    --chunk-size "${TOTAL_SAMPLE_COUNT}" \
    --engine auto
else
  echo ""
  echo "[1/2] Raw NPZ already exists; skipping HDF5 conversion. Set FORCE=1 to rebuild."
fi

if [[ "${FORCE}" == "1" || ! -f "${PROCESSED_ROOT}/manifest.npz" || ! -f "${PROCESSED_ROOT}/summary.json" ]]; then
  echo ""
  echo "[2/2] Converting raw NPZ to GOATTM processed manifest..."
  "${PYTHON}" "${APP_ROOT}/codes/preprocess_adr_quadp_npz_dataset.py" \
    --train-root "${RAW_ROOT}" \
    --no-test \
    --qoi-stride "${QOI_STRIDE}" \
    --time-mode "${TIME_MODE}" \
    --output-root "${PROCESSED_ROOT}"
else
  echo ""
  echo "[2/2] Processed manifest already exists; skipping preprocess. Set FORCE=1 to rebuild."
fi

"${PYTHON}" - "${PROCESSED_ROOT}/manifest.npz" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from goattm.data import load_npz_sample_manifest

manifest_path = Path(sys.argv[1])
manifest = load_npz_sample_manifest(manifest_path)
first = manifest.root_dir / manifest.sample_paths[0]
last = manifest.root_dir / manifest.sample_paths[-1]
print(f"Manifest: {manifest_path}")
print(f"Sample count: {len(manifest)}")
print(f"First sample exists: {first.exists()} -> {first}")
print(f"Last sample exists: {last.exists()} -> {last}")
PY

echo ""
echo "Summary:"
cat "${PROCESSED_ROOT}/summary.json"
echo ""
echo "Done: $(date)"
