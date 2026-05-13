#!/bin/bash
#SBATCH --job-name=ldnet_smoke
#SBATCH --partition=skx-dev
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --output=./batchout/ldnet_smoke_%j.out
#SBATCH --error=./batchout/ldnet_smoke_%j.err

set -euo pipefail

REPO_DIR="/work2/08667/yuuuhang/stampede3/GOATTM_simplemamba"
PY_BIN="/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python"

export PYTHONPATH="${REPO_DIR}/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_DIR}"
mkdir -p batchout demo/ldnet_synthetic_output

echo "Start: $(date)"
echo "Python: ${PY_BIN}"
"${PY_BIN}" -c 'import sys, torch, mpi4py, scipy; print(sys.version); print("torch", torch.__version__); print("mpi4py", mpi4py.__version__); print("scipy", scipy.__version__)'

echo "Running Taylor test"
"${PY_BIN}" module_test/ldnet_taylor/run_ldnet_taylor_test.py

echo "Running 10-rank LDNet demo"
ibrun -n 10 "${PY_BIN}" demo/run_ldnet_synthetic_demo.py \
  --outer-cycles 8 \
  --dynamics-steps 8 \
  --decoder-steps 12 \
  --train-samples 100 \
  --test-samples 40 \
  --time-steps 41 \
  --ssm-state-dim 24

echo "Done: $(date)"
