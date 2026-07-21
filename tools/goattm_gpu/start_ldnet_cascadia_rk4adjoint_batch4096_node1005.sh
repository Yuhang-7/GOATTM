#!/usr/bin/env bash
set -eo pipefail

source /global/u2/y/yuuuhang/miniforge3/etc/profile.d/conda.sh
conda activate quad_goattm
cd /global/homes/y/yuuuhang/quad_goattm

OUT=/pscratch/sd/y/yuuuhang/cascadia_goattm_runs/ldnet_neural_decoder_podinit_rk4adjoint_batch4096_armijo8_node1005_20260705
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --standalone --nproc_per_node=4 tools/train_ldnet_cascadia.py \
  --train /pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train16384_r50_accum_displacement/train_16384_packed.pt \
  --test /pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train16384_r50_accum_displacement/test_5000_packed.pt \
  --initializer /pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train16384_r50_accum_displacement/pod_opinf_r120_dt1_from_r50_initial_parameters.npz \
  --output "$OUT" \
  --latent-dim 120 \
  --h-rank 50 \
  --h-tt-rank 50 \
  --encoder-hidden 128,128 \
  --decoder-hidden 128,128 \
  --train-sample-limit 0 \
  --test-sample-limit 2048 \
  --validation-sample-limit 4096 \
  --batch-size 1024 \
  --accumulation-steps 1 \
  --gradient-mode rk4_adjoint \
  --optimizer armijo \
  --steps 8 \
  --validation-interval 2 \
  --fd-check \
  --fd-sample-limit 2 \
  --fd-max-time-steps 16 \
  --fd-epsilons 1e-4,3e-5,1e-5 \
  --armijo-initial-alpha 1 \
  --armijo-max-alpha 1 \
  --armijo-grow 2 \
  --armijo-shrink 0.5 \
  --armijo-trust 3e-5 \
  --armijo-min-norm 1 \
  --decode-chunk-size 32768 \
  --device cuda \
  --dtype float64 \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/launcher_pid"
echo "launcher_pid=$(cat "$OUT/launcher_pid")"
echo "output_dir=$OUT"
