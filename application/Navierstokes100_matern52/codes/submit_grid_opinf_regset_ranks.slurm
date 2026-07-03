#!/bin/bash
#SBATCH -J ns100_m52_regsets
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p spr
#SBATCH -t 02:00:00
#SBATCH --array=0-90
#SBATCH -o /work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/logs/grid_opinf_regset_%A_%a.out
#SBATCH -e /work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/logs/grid_opinf_regset_%A_%a.err

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

REG_NAMES=(
  baseline_B10
  B1
  B100
  other_down100
  other_down10
  other_up10
  Hsmall_B10
  Hlarge_B10
)

REG_A=(1e-8 1e-8 1e-8 1e-10 1e-9 1e-7 1e-8 1e-8)
REG_H=(1e0 1e0 1e0 1e-2 1e-1 1e1 1e-2 1e2)
REG_B=(1e1 1e0 1e2 1e1 1e1 1e1 1e1 1e1)
REG_C=(1e-6 1e-6 1e-6 1e-8 1e-7 1e-5 1e-6 1e-6)
REG_D=(1e-4 1e-4 1e-4 1e-6 1e-5 1e-3 1e-4 1e-4)

NRANKS=${#RANKS[@]}
RANK_INDEX=$((SLURM_ARRAY_TASK_ID % NRANKS))
REG_INDEX=$((SLURM_ARRAY_TASK_ID / NRANKS))

RANK=${RANKS[${RANK_INDEX}]}
REG_NAME=${REG_NAMES[${REG_INDEX}]}

mkdir -p ${APP}/logs
cd ${GOATTM}

echo "rank=${RANK} reg_set=${REG_NAME} reg_a=${REG_A[${REG_INDEX}]} reg_h=${REG_H[${REG_INDEX}]} reg_b=${REG_B[${REG_INDEX}]} reg_c=${REG_C[${REG_INDEX}]} reg_decoder=${REG_D[${REG_INDEX}]}"

${PYTHON} -u ${SCRIPT} \
  --manifest-path ${APP}/data/processed_data_qoi5_input40/normalized/train_manifest.npz \
  --grid-state-root /scratch/08667/yuuuhang/NS_matern_input/grid_state_re100_matern52_nx80_ny24_448 \
  --output-root ${APP}/initial/AHBc_gridstate_reg_sweep_sets/${REG_NAME} \
  --rank ${RANK} \
  --max-samples 112 \
  --train-samples 96 \
  --time-stride 2 \
  --score-mode one_step \
  --reg-a-list ${REG_A[${REG_INDEX}]} \
  --reg-h-list ${REG_H[${REG_INDEX}]} \
  --reg-b-list ${REG_B[${REG_INDEX}]} \
  --reg-c-list ${REG_C[${REG_INDEX}]} \
  --reg-decoder-list ${REG_D[${REG_INDEX}]} \
  --save-best
