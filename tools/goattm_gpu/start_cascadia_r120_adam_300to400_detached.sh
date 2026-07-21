#!/usr/bin/env bash
set -eo pipefail

source /global/u2/y/yuuuhang/miniforge3/etc/profile.d/conda.sh
conda activate quad_goattm
cd /global/homes/y/yuuuhang/quad_goattm

OUT=/global/homes/y/yuuuhang/quad_goattm/outputs/cascadia_packed8192_r120_dissskewA20_shift1_energytucker40tt40_maskcross2000_adam300to400_lr3e-3_node1024
CKPT=/global/homes/y/yuuuhang/quad_goattm/outputs/cascadia_packed8192_r120_dissskewA20_shift1_energytucker40tt40_maskcross2000_adam300_lr3e-3_node1024/checkpoint.pt
mkdir -p "$OUT"

nohup torchrun --standalone --nproc_per_node=4 tools/train_cascadia_packed.py \
  --train-packed /pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train8192_r50/train_8192_packed.pt \
  --test-packed /pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train8192_r50/test_5000_packed.pt \
  --initializer /pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train8192_r50/pod_opinf_r50_initial_parameters.npz \
  --load-checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --latent-dim 120 \
  --linear-a dissipative_skew --a-rank 20 --a-damping-init 0.1 --a-damping-shift 1.0 \
  --quadratic energy_tucker --h-reduced-rank 40 --h-tt-rank 40 \
  --decoder-quadratic-mode masked_cross --decoder-cross-terms 2000 --decoder-mask-seed 20260705 \
  --optimizer adam --optimizer-steps 100 --lr 3.0e-3 \
  --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1.0e-8 --adam-weight-decay 0.0 \
  --validation-interval 10 --validation-chunk-samples 512 --normal-chunk-size 4096 \
  > "$OUT/run.log" 2>&1 &

echo $! > "$OUT/pid"
echo "$OUT"
