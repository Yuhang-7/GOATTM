#!/bin/bash
#SBATCH -J ns100_m52_regsets_pack
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p spr
#SBATCH -t 04:00:00
#SBATCH -o /work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/logs/grid_opinf_regset_packed_%j.out
#SBATCH -e /work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/logs/grid_opinf_regset_packed_%j.err

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
TASK_LOG_DIR=${APP}/logs/grid_opinf_regset_packed_${SLURM_JOB_ID}
MAX_PARALLEL=8

RANKS=(4 6 8 10 12 14 16 18 20 24 28 32 40)

REG_NAMES=(
  baseline_B10
  B1
  B100
  other_down100
  other_down10
  other_up10
  Hsmall_B10
)

REG_A=(1e-8 1e-8 1e-8 1e-10 1e-9 1e-7 1e-8)
REG_H=(1e0 1e0 1e0 1e-2 1e-1 1e1 1e-2)
REG_B=(1e1 1e0 1e2 1e1 1e1 1e1 1e1)
REG_C=(1e-6 1e-6 1e-6 1e-8 1e-7 1e-5 1e-6)
REG_D=(1e-4 1e-4 1e-4 1e-6 1e-5 1e-3 1e-4)

mkdir -p ${APP}/logs ${TASK_LOG_DIR}
cd ${GOATTM}

run_case() {
  local rank=$1
  local reg_index=$2
  local reg_name=${REG_NAMES[${reg_index}]}
  local tag=r${rank}_${reg_name}
  local outlog=${TASK_LOG_DIR}/${tag}.out
  local errlog=${TASK_LOG_DIR}/${tag}.err

  {
    echo "rank=${rank} reg_set=${reg_name} reg_a=${REG_A[${reg_index}]} reg_h=${REG_H[${reg_index}]} reg_b=${REG_B[${reg_index}]} reg_c=${REG_C[${reg_index}]} reg_decoder=${REG_D[${reg_index}]}"
    ${PYTHON} -u ${SCRIPT} \
      --manifest-path ${APP}/data/processed_data_qoi5_input40/normalized/train_manifest.npz \
      --grid-state-root /scratch/08667/yuuuhang/NS_matern_input/grid_state_re100_matern52_nx80_ny24_448 \
      --output-root ${APP}/initial/AHBc_gridstate_reg_sweep_sets/${reg_name} \
      --rank ${rank} \
      --max-samples 112 \
      --train-samples 96 \
      --time-stride 2 \
      --score-mode one_step \
      --reg-a-list ${REG_A[${reg_index}]} \
      --reg-h-list ${REG_H[${reg_index}]} \
      --reg-b-list ${REG_B[${reg_index}]} \
      --reg-c-list ${REG_C[${reg_index}]} \
      --reg-decoder-list ${REG_D[${reg_index}]} \
      --save-best
  } >"${outlog}" 2>"${errlog}"
}

active=0
for reg_index in "${!REG_NAMES[@]}"; do
  for rank in "${RANKS[@]}"; do
    run_case "${rank}" "${reg_index}" &
    active=$((active + 1))
    if [[ ${active} -ge ${MAX_PARALLEL} ]]; then
      wait -n
      active=$((active - 1))
    fi
  done
done

wait

echo "Completed packed reg-set sweep. Per-task logs: ${TASK_LOG_DIR}"
