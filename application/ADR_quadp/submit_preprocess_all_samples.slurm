#!/bin/bash
#SBATCH --job-name=adr_pre_all
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin}"
export PYTHON="${PYTHON:-${CONDA_BIN_DIR}/python}"

export ADR_RAW_NPZ_ROOT="${ADR_RAW_NPZ_ROOT:-${APP_ROOT}/data/raw_npz/all_samples}"
export QOI_STRIDE="${QOI_STRIDE:-1}"
export TIME_MODE="${TIME_MODE:-physical}"
export ADR_PROCESSED_DATA_ROOT="${ADR_PROCESSED_DATA_ROOT:-${APP_ROOT}/data/processed_data/all_samples_stride${QOI_STRIDE}}"

mkdir -p "${APP_ROOT}/batchout"

echo "Job start: $(date)"
echo "Repository root: ${REPO_ROOT}"
echo "Python: ${PYTHON}"
echo "ADR_RAW_NPZ_ROOT: ${ADR_RAW_NPZ_ROOT}"
echo "ADR_PROCESSED_DATA_ROOT: ${ADR_PROCESSED_DATA_ROOT}"
echo "QOI_STRIDE: ${QOI_STRIDE}"
echo "TIME_MODE: ${TIME_MODE}"

"${APP_ROOT}/codes/run_preprocess_adr_quadp_all_samples.sh" "$@"

echo "Job finished: $(date)"
