#!/bin/bash
#SBATCH --job-name=ns_h5_to_npz
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navier_Stokes/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navier_Stokes/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/Navier_Stokes"
GOAM_CLEAN_ROOT="${GOAM_CLEAN_ROOT:-/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin}"

module load hdf5/1.14.4

export HDF5_PY_BIN="${HDF5_PY_BIN:-${CONDA_BIN_DIR}/python}"
export INPUT_FILE="${INPUT_FILE:-${GOAM_CLEAN_ROOT}/Example/Navier_Stokes_Re=100/dataset/latent_data_1000_Re=100.hdf5}"
export TRAIN_COUNT="${TRAIN_COUNT:-896}"
export TEST_COUNT="${TEST_COUNT:-104}"
export SAMPLE_START="${SAMPLE_START:-0}"
export CHUNK_SIZE="${CHUNK_SIZE:-${TRAIN_COUNT}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/data/raw_npz/train${TRAIN_COUNT}_test${TEST_COUNT}}"

mkdir -p "${APP_ROOT}/batchout"

echo "Job start: $(date)"
echo "INPUT_FILE: ${INPUT_FILE}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "TRAIN_COUNT: ${TRAIN_COUNT}"
echo "TEST_COUNT: ${TEST_COUNT}"
echo "HDF5_PY_BIN: ${HDF5_PY_BIN}"
echo "h5dump: $(command -v h5dump)"

"${APP_ROOT}/codes/convert_hdf5_to_npz.sh" "$@"

echo "Job finished: $(date)"
