#!/bin/bash
set -euo pipefail

cd /work2/08667/yuuuhang/stampede3/GOATTM
mkdir -p application/ADR/batchout application/ADR/outputs/saved_oldgoam_regf1e-7_regg1e-9

submit_one() {
  local ntrain="$1"
  local nodes="$2"
  local ranks="$3"
  local manifest="application/ADR/data/processed_data_n${ntrain}_test200/manifest.npz"
  local initdir="application/ADR/data/old_goam_initializers/ADR_quadp_size=${ntrain}_testsize=200_validate/AHBc"
  local sample_count=$((ntrain + 200))
  local job="adr_savedinit_n${ntrain}"

  sbatch \
    --job-name="${job}" \
    --partition=spr \
    --nodes="${nodes}" \
    --ntasks-per-node=64 \
    --time=18:00:00 \
    --output="/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR/batchout/%x-%j.out" \
    --error="/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR/batchout/%x-%j.err" \
    --wrap="set -euo pipefail
cd /work2/08667/yuuuhang/stampede3/GOATTM
export PYTHONPATH=/work2/08667/yuuuhang/stampede3/GOATTM/src:\${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
for r in 12 14 16; do
  echo Starting saved-oldGOAM ADR n=${ntrain} r=\$r at \$(date)
  ibrun -n ${ranks} /work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python application/ADR/codes/run_manifest_oldgoam_saved_init.py \
    --dataset-name adr_quadp \
    --manifest-path ${manifest} \
    --sample-count ${sample_count} \
    --ntrain ${ntrain} \
    --ntest 200 \
    --latent-rank \$r \
    --initial-value-dir ${initdir} \
    --output-dir application/ADR/outputs/saved_oldgoam_regf1e-7_regg1e-9 \
    --optimizer bfgs \
    --max-iterations 20000 \
    --max-dt 0.002 \
    --time-integrator rk4 \
    --decoder-reg-v1 1e-7 \
    --decoder-reg-v2 1e-7 \
    --decoder-reg-v0 1e-7 \
    --dynamics-reg-a 1e-9 \
    --dynamics-reg-mu-h 1e-9 \
    --dynamics-reg-b 1e-9 \
    --dynamics-reg-c 1e-9 \
    --dynamics-reg-spectral-abscissa 0.0 \
    --dynamics-reg-spectral-alpha 0.0
  echo Finished saved-oldGOAM ADR n=${ntrain} r=\$r at \$(date)
done"
}

submit_one 16 1 64
submit_one 32 1 64
submit_one 64 1 64
submit_one 128 1 64
submit_one 256 2 128
