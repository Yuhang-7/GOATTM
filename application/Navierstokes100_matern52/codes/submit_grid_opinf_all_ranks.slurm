#!/bin/bash
#SBATCH -J ns100_m52_opinf_ranks
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p spr
#SBATCH -t 02:00:00
#SBATCH --array=0-12
#SBATCH -o /work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/logs/grid_opinf_rank_%A_%a.out
#SBATCH -e /work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/logs/grid_opinf_rank_%A_%a.err

set -eo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

APP=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52
GOATTM=/work2/08667/yuuuhang/stampede3/GOATTM
PYTHON=/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python
SCRIPT=${APP}/codes/sweep_grid_opinf_regularizers_ns100_matern52.py

RANKS=(4 6 8 10 12 14 16 18 20 24 28 32 40)
RANK=${RANKS[${SLURM_ARRAY_TASK_ID}]}

mkdir -p ${APP}/logs
cd ${GOATTM}

${PYTHON} -u ${SCRIPT} \
  --manifest-path ${APP}/data/processed_data_qoi5_input40/normalized/train_manifest.npz \
  --grid-state-root /scratch/08667/yuuuhang/NS_matern_input/grid_state_re100_matern52_nx80_ny24_448 \
  --output-root ${APP}/initial/AHBc_gridstate_all_ranks_regB10 \
  --rank ${RANK} \
  --max-samples 112 \
  --train-samples 96 \
  --time-stride 2 \
  --score-mode one_step \
  --reg-a-list 1e-8 \
  --reg-h-list 1e0 \
  --reg-b-list 1e1 \
  --reg-c-list 1e-6 \
  --reg-decoder-list 1e-4 \
  --save-best
