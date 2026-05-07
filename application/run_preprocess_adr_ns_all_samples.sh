#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
PYTHON="${PYTHON:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python}"

export GOATTM_REPO_ROOT="${REPO_ROOT}"
export PYTHON
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${REPO_ROOT}"

echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON}"
echo "Start: $(date)"

echo ""
echo "[1/2] Preprocessing ADR all-samples dataset..."
bash application/ADR_quadp/codes/run_preprocess_adr_quadp_all_samples.sh

echo ""
echo "[2/2] Preprocessing Navier-Stokes all-samples dataset..."
bash application/Navier_Stokes/codes/run_preprocess_navier_stokes_all_samples.sh

echo ""
echo "Done: $(date)"
echo "ADR manifest: application/ADR_quadp/data/processed_data/all_samples_stride1/manifest.npz"
echo "NS manifest:  application/Navier_Stokes/data/processed_data/all_samples_stride1/manifest.npz"
