#!/usr/bin/env bash
# Login-node submitter for the three SWE skewCP comparison jobs.
# Do not run until the submit scripts have been reviewed.

set -euo pipefail

SWE_SKEWCP_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP"
JOB_SCRIPT="${SWE_SKEWCP_ROOT}/submit_swe_skewcp.sh"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SUBMIT_DELAY_SECONDS="${SUBMIT_DELAY_SECONDS:-1}"
MAX_DT="${MAX_DT:-0.0002}"
DRY_RUN="${DRY_RUN:-0}"

CASES=(
  "100 30"
  "64 20"
  "30 15"
)

for CASE_SPEC in "${CASES[@]}"; do
  read -r LATENT_RANK SKEW_CP_RANK <<< "${CASE_SPEC}"
  JOB_NAME="swe_skewcp_r${LATENT_RANK}_R${SKEW_CP_RANK}"
  RUN_STEM="$(date +%Y%m%d_%H%M%S)_swe_skewcp_r${LATENT_RANK}_R${SKEW_CP_RANK}_dt2e-4"
  CMD=(
    env
    LATENT_RANK="${LATENT_RANK}"
    SKEW_CP_RANK="${SKEW_CP_RANK}"
    MAX_DT="${MAX_DT}"
    RUN_STEM="${RUN_STEM}"
    "${SBATCH_BIN}"
    --job-name="${JOB_NAME}"
    "${JOB_SCRIPT}"
  )
  printf 'Submitting case LATENT_RANK=%s SKEW_CP_RANK=%s MAX_DT=%s\n' "${LATENT_RANK}" "${SKEW_CP_RANK}" "${MAX_DT}"
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
