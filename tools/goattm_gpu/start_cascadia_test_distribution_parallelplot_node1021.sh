#!/usr/bin/env bash
set -eo pipefail

source /global/u2/y/yuuuhang/miniforge3/etc/profile.d/conda.sh
conda activate quad_goattm
cd /global/homes/y/yuuuhang/quad_goattm

OUT=/global/homes/y/yuuuhang/quad_goattm/outputs/cascadia_test_distribution_latest_16384_lbfgs_node1021_parallelplot
mkdir -p "$OUT"
python -m py_compile tools/evaluate_cascadia_test_distribution.py

nohup torchrun --standalone --nproc_per_node=4 tools/evaluate_cascadia_test_distribution.py \
  --test-packed /pscratch/sd/y/yuuuhang/cascadia_region_data_goattm_npz_input18_rank150/packed/40bin_test125_train16384_r50/test_5000_packed.pt \
  --checkpoint /global/homes/y/yuuuhang/quad_goattm/outputs/cascadia_packed16384_r120_dissskewA20_shift1_energytucker40tt40_maskcross2000_from8192step70_lbfgs_wolfe_reset10_80steps_node1024/checkpoint_latest.pt \
  --output-dir "$OUT" \
  --chunk-samples 128 \
  --sample-count 50 \
  --qois-per-sample 20 \
  --seed 20260705 \
  --picard-iters 2 \
  --normal-chunk-size 4096 \
  --a-damping-shift 1.0 \
  --parallel-plot \
  --write-pngs \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid"
echo "pid=$(cat "$OUT/pid")"
echo "output_dir=$OUT"
