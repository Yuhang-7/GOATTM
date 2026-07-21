#!/usr/bin/env bash
set -eo pipefail

source /global/u2/y/yuuuhang/miniforge3/etc/profile.d/conda.sh
conda activate quad_goattm
cd /global/homes/y/yuuuhang/quad_goattm

PREV=/global/homes/y/yuuuhang/quad_goattm/outputs/cascadia_packed8192_r120_dissskewA20_shift1_energytucker40tt40_maskcross2000_adam300_lbfgs_wolfe_reset10_restart1_80steps_node1024
DATA=/pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train16384_r50
OUT=/global/homes/y/yuuuhang/quad_goattm/outputs/cascadia_packed16384_r120_dissskewA20_shift1_energytucker40tt40_maskcross2000_from8192step70_lbfgs_wolfe_reset10_80steps_node1024
mkdir -p "$OUT"

nohup torchrun --standalone --nproc_per_node=4 tools/train_cascadia_packed.py \
  --train-packed "$DATA/train_16384_packed.pt" \
  --test-packed "$DATA/test_5000_packed.pt" \
  --initializer "$DATA/pod_opinf_r50_initial_parameters.npz" \
  --load-checkpoint "$PREV/checkpoint_latest.pt" \
  --output-dir "$OUT" \
  --latent-dim 120 \
  --linear-a dissipative_skew \
  --a-rank 20 \
  --a-damping-init 0.1 \
  --a-damping-shift 1.0 \
  --quadratic energy_tucker \
  --h-reduced-rank 40 \
  --h-tt-rank 40 \
  --decoder-quadratic-mode masked_cross \
  --decoder-cross-terms 2000 \
  --decoder-mask-seed 20260705 \
  --optimizer lbfgs \
  --optimizer-steps 80 \
  --max-iter 5 \
  --history-size 20 \
  --lr 0.25 \
  --line-search strong_wolfe \
  --reset-lbfgs-interval 10 \
  --validation-interval 10 \
  --validation-chunk-samples 512 \
  --normal-chunk-size 4096 \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid"
echo "detached_pid=$(cat "$OUT/pid")"
echo "output_dir=$OUT"
