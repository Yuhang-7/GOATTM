#!/usr/bin/env bash
# Submit 24-hour SWE skewCP production jobs using the reviewed V1v, dt=2e-4 setting.

set -euo pipefail

JOB_SCRIPT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/submit_swe_skewcp.sh"
SBATCH_BIN="sbatch"
SUBMIT_DELAY_SECONDS="1"
MAX_DT="0.0002"
DECODER_FORM="V1v"
NTRAIN="1120"
NTEST="560"
SAMPLE_COUNT="1680"
OUTPUT_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/outputs/production_24h_dt2e-4_v1v"
BATCHOUT_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/submit/production_24h_dt2e-4_v1v/batchout"

CASES=(
  "100 30"
  "64 20"
  "30 15"
)

mkdir -p "${OUTPUT_ROOT}" "${BATCHOUT_ROOT}"

for CASE_SPEC in "${CASES[@]}"; do
  read -r LATENT_RANK SKEW_CP_RANK <<< "${CASE_SPEC}"
  JOB_NAME="swe24_r${LATENT_RANK}_R${SKEW_CP_RANK}"
  RUN_STEM="$(date +%Y%m%d_%H%M%S)_swe_24h_r${LATENT_RANK}_R${SKEW_CP_RANK}_dt2e-4_v1v"
  CMD=(
    env
    LATENT_RANK="${LATENT_RANK}"
    SKEW_CP_RANK="${SKEW_CP_RANK}"
    MAX_DT="${MAX_DT}"
    DECODER_FORM="${DECODER_FORM}"
    NTRAIN="${NTRAIN}"
    NTEST="${NTEST}"
    SAMPLE_COUNT="${SAMPLE_COUNT}"
    OUTPUT_ROOT="${OUTPUT_ROOT}"
    RUN_STEM="${RUN_STEM}"
    "${SBATCH_BIN}"
    --job-name="${JOB_NAME}"
    --time=24:00:00
    --output="${BATCHOUT_ROOT}/%x-%j.out"
    --error="${BATCHOUT_ROOT}/%x-%j.err"
    "${JOB_SCRIPT}"
  )
  printf 'Submitting SWE 24h LATENT_RANK=%s SKEW_CP_RANK=%s NTRAIN=%s NTEST=%s MAX_DT=%s DECODER_FORM=%s\n' "${LATENT_RANK}" "${SKEW_CP_RANK}" "${NTRAIN}" "${NTEST}" "${MAX_DT}" "${DECODER_FORM}"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
  sleep "${SUBMIT_DELAY_SECONDS}"
done

# Why submit this task:
# Submit 24-hour SWE skewCP production jobs with the same 1120/560 split, dt=2e-4, and V1v decoder, while leaving the 36-hour base script available for later full reruns.
