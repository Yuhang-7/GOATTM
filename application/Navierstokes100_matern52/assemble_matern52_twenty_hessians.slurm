#!/bin/bash
#SBATCH -J matern52_20hess
#SBATCH -p spr
#SBATCH -N 4
#SBATCH --ntasks-per-node=112
#SBATCH -t 12:00:00
#SBATCH -o /work2/08667/yuuuhang/stampede3/GOATTM_varpro_research/outputs/matern52_twenty_hessians/%x-%j.out
#SBATCH -e /work2/08667/yuuuhang/stampede3/GOATTM_varpro_research/outputs/matern52_twenty_hessians/%x-%j.err

set -euo pipefail

REPO=/work2/08667/yuuuhang/stampede3/GOATTM_varpro_research
PYTHON=/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python
EXPERIMENT=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/outputs/savedoldgoam_h0_timeunit_r18_r20_strong_lagged_midpoint_numba_224r/20260601_161515_strong_A1e6H1e8_lmp_numba224_ntrain896_ntest448_r20/ns100_matern52_qoi5_input40_savedoldgoam_r20_n1344_20260601_161515_strong_A1e6H1e8_lmp_numba224_ntrain896_ntest448_r20
RUN_DIR=${EXPERIMENT}/runs/ns100_matern52_qoi5_input40_savedoldgoam_bfgs_lagged_midpoint_r20_ntrain896_ntest448_20260601_161554_78f44bf1
OUTPUT=${OUTPUT_DIR:-${REPO}/outputs/matern52_twenty_hessians/full_mid500_best_dt0025}

mkdir -p "${OUTPUT}"
cd "${REPO}"

export PYTHONPATH=${REPO}/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "start=$(date --iso-8601=seconds)"
echo "repo=${REPO}"
echo "output=${OUTPUT}"
echo "nodes=${SLURM_NNODES} tasks=${SLURM_NTASKS}"

ibrun -n 448 "${PYTHON}" \
  application/Navierstokes100_matern52/assemble_matern52_twenty_hessians.py \
  --repo "${REPO}" \
  --manifest "${EXPERIMENT}/oldgoam_latent_train_manifest.npz" \
  --config "${RUN_DIR}/config.json" \
  --checkpoint "mid_iter500=${RUN_DIR}/checkpoints/iter_000500.npz" \
  --checkpoint "best=${RUN_DIR}/checkpoints/best.npz" \
  --output-dir "${OUTPUT}" \
  --group-size 56 \
  --resume

echo "finish=$(date --iso-8601=seconds)"
