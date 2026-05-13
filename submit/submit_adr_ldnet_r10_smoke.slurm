#!/bin/bash
#SBATCH --job-name=adr_ldnet_r10
#SBATCH --partition=skx-dev
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --output=./batchout/adr_ldnet_r10_%j.out
#SBATCH --error=./batchout/adr_ldnet_r10_%j.err

set -euo pipefail

REPO_DIR="/work2/08667/yuuuhang/stampede3/GOATTM_simplemamba"
PY_BIN="/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python"
MANIFEST="/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR/data/processed_data_n16_test200/manifest.npz"
OUT_DIR="${REPO_DIR}/application/ADR_simplemamba/outputs/r10_smoke_${SLURM_JOB_ID}"

export PYTHONPATH="${REPO_DIR}/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_DIR}"
mkdir -p batchout application/ADR_simplemamba/outputs

echo "Start: $(date)"
echo "Running ADR LDNet r=10 smoke"
ibrun -n 10 "${PY_BIN}" application/ADR_simplemamba/run_adr_ldnet_demo.py \
  --manifest-path "${MANIFEST}" \
  --latent-rank 10 \
  --train-samples 16 \
  --test-samples 20 \
  --max-time-steps 121 \
  --ssm-state-dim 32 \
  --outer-cycles 5 \
  --dynamics-steps 5 \
  --decoder-steps 10 \
  --output-dir "${OUT_DIR}"
echo "Done: $(date)"
