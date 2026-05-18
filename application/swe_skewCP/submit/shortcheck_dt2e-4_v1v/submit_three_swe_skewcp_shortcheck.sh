#!/usr/bin/env bash
# Submit 30-minute short-check SWE skewCP jobs before the 36-hour production rerun.

set -euo pipefail

SWE_SKEWCP_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP"
JOB_SCRIPT="${SWE_SKEWCP_ROOT}/submit_swe_skewcp.sh"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SUBMIT_DELAY_SECONDS="${SUBMIT_DELAY_SECONDS:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_DT="${MAX_DT:-0.0002}"
DECODER_FORM="${DECODER_FORM:-V1v}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SWE_SKEWCP_ROOT}/outputs/shortcheck_dt2e-4_v1v}"
BATCHOUT_ROOT="${BATCHOUT_ROOT:-${SWE_SKEWCP_ROOT}/submit/shortcheck_dt2e-4_v1v/batchout}"

CASES=(
  "100 30"
  "64 20"
  "30 15"
)

mkdir -p "${OUTPUT_ROOT}" "${BATCHOUT_ROOT}"

for CASE_SPEC in "${CASES[@]}"; do
  read -r LATENT_RANK SKEW_CP_RANK <<< "${CASE_SPEC}"
  JOB_NAME="swe_short_r${LATENT_RANK}_R${SKEW_CP_RANK}"
  RUN_STEM="$(date +%Y%m%d_%H%M%S)_swe_shortcheck_r${LATENT_RANK}_R${SKEW_CP_RANK}_dt2e-4"
  CMD=(
    env
    LATENT_RANK="${LATENT_RANK}"
    SKEW_CP_RANK="${SKEW_CP_RANK}"
    MAX_DT="${MAX_DT}"
    DECODER_FORM="${DECODER_FORM}"
    OUTPUT_ROOT="${OUTPUT_ROOT}"
    RUN_STEM="${RUN_STEM}"
    "${SBATCH_BIN}"
    --job-name="${JOB_NAME}"
    --time=00:30:00
    --output="${BATCHOUT_ROOT}/%x-%j.out"
    --error="${BATCHOUT_ROOT}/%x-%j.err"
    "${JOB_SCRIPT}"
  )
  printf 'Submitting short-check LATENT_RANK=%s SKEW_CP_RANK=%s MAX_DT=%s DECODER_FORM=%s\n' "${LATENT_RANK}" "${SKEW_CP_RANK}" "${MAX_DT}" "${DECODER_FORM}"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: not submitted."
  else
    "${CMD[@]}"
    sleep "${SUBMIT_DELAY_SECONDS}"
  fi
done

# Why submit this task:
# Submit short 30-minute SWE skewCP checks with the same 1120/560 split, dt=2e-4, and V1v decoder before launching the 36-hour production jobs.
