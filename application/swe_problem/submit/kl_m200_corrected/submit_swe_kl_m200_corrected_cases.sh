#!/usr/bin/env bash
# Submit corrected SWE KL-input reruns.
#
# Intent:
#   Use processed_data_kl_m200, whose input_values are time-dependent KL
#   coefficients of the Gaussian seafloor uplift, not the legacy 25-parameter
#   Gaussian packet.

set -euo pipefail

MANIFEST_PATH="/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data_kl_m200/manifest.npz"
NTRAIN="1344"
NTEST="384"
SAMPLE_COUNT="1728"
TIME_LIMIT="${TIME_LIMIT:-36:00:00}"
NODES="${NODES:-6}"
TASKS_PER_NODE="${TASKS_PER_NODE:-112}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
MAX_DT_DENSE="${MAX_DT_DENSE:-0.0016666666666666668}"
MAX_DT_SKEW="${MAX_DT_SKEW:-0.00025}"
TEST_EVERY="${TEST_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"

GOATTM_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM"
TUCKER_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT"
DENSE_JOB_SCRIPT="${GOATTM_ROOT}/application/swe_problem/submit_swe.sh"
SKEW_JOB_SCRIPT="${TUCKER_ROOT}/application/swe_skewCP/submit_swe_skewcp.sh"
DENSE_OUTPUT_ROOT="${GOATTM_ROOT}/application/swe_problem/outputs/kl_m200_corrected"
SKEW_OUTPUT_ROOT="${TUCKER_ROOT}/application/swe_skewCP/outputs/kl_m200_corrected"
DENSE_BATCHOUT="${GOATTM_ROOT}/application/swe_problem/batchout/kl_m200_corrected"
SKEW_BATCHOUT="${TUCKER_ROOT}/application/swe_skewCP/batchout/kl_m200_corrected"

mkdir -p "${DENSE_OUTPUT_ROOT}" "${SKEW_OUTPUT_ROOT}" "${DENSE_BATCHOUT}" "${SKEW_BATCHOUT}"

python3 - <<'PY'
from pathlib import Path
import numpy as np

manifest_path = Path("/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data_kl_m200/manifest.npz")
if not manifest_path.exists():
    raise SystemExit(f"missing manifest: {manifest_path}")
manifest = np.load(manifest_path, allow_pickle=True)
sample_path = manifest_path.parent / manifest["sample_paths"][0]
sample = np.load(sample_path, allow_pickle=True)
input_shape = sample["input_values"].shape
qoi_shape = sample["qoi_observations"].shape
mode = str(sample["input_mode"]) if "input_mode" in sample.files else "missing"
rowdiff = float(np.max(np.abs(sample["input_values"] - sample["input_values"][0])))
print(f"Validated corrected SWE KL dataset: {manifest_path}")
print(f"  first sample: {sample_path}")
print(f"  input_mode: {mode}")
print(f"  input_values shape: {input_shape}")
print(f"  qoi_observations shape: {qoi_shape}")
print(f"  input max rowdiff: {rowdiff:.6e}")
if input_shape != (1501, 200):
    raise SystemExit(f"expected input_values shape (1501, 200), got {input_shape}")
if qoi_shape != (1501, 30):
    raise SystemExit(f"expected qoi_observations shape (1501, 30), got {qoi_shape}")
if "kl_projected" not in mode or "gaussian_uplift" not in mode:
    raise SystemExit(f"unexpected input_mode: {mode}")
if rowdiff <= 0.0:
    raise SystemExit("input_values are not time-dependent; refusing to submit")
PY

submit_dense() {
  local latent_rank="$1"
  local job_name="swe_klm200_dense_r${latent_rank}"
  local run_stem
  run_stem="$(date +%Y%m%d_%H%M%S)_swe_klm200_dense_r${latent_rank}"
  echo "Submitting dense-H baseline r=${latent_rank}"
  env \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    NTRAIN="${NTRAIN}" \
    NTEST="${NTEST}" \
    SAMPLE_COUNT="${SAMPLE_COUNT}" \
    LATENT_RANK_SEQUENCE="${latent_rank}" \
    OPTIMIZER="bfgs" \
    MAX_ITERATIONS="${MAX_ITERATIONS}" \
    MAX_DT="${MAX_DT_DENSE}" \
    TIME_INTEGRATOR="rk4" \
    OUTPUT_ROOT="${DENSE_OUTPUT_ROOT}" \
    RUN_STEM="${run_stem}" \
    sbatch --parsable \
      --job-name="${job_name}" \
      --nodes="${NODES}" \
      --ntasks-per-node="${TASKS_PER_NODE}" \
      --time="${TIME_LIMIT}" \
      --output="${DENSE_BATCHOUT}/%x-%j.out" \
      --error="${DENSE_BATCHOUT}/%x-%j.err" \
      "${DENSE_JOB_SCRIPT}"
}

submit_skew() {
  local latent_rank="$1"
  local skew_rank="$2"
  local job_name="swe_klm200_r${latent_rank}_R${skew_rank}"
  local run_stem
  run_stem="$(date +%Y%m%d_%H%M%S)_swe_klm200_skewcp_r${latent_rank}_R${skew_rank}"
  echo "Submitting skewCP r=${latent_rank}, R=${skew_rank}"
  env \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    NTRAIN="${NTRAIN}" \
    NTEST="${NTEST}" \
    SAMPLE_COUNT="${SAMPLE_COUNT}" \
    LATENT_RANK="${latent_rank}" \
    SKEW_CP_RANK="${skew_rank}" \
    OPTIMIZER="lbfgs" \
    MAX_ITERATIONS="${MAX_ITERATIONS}" \
    MAX_DT="${MAX_DT_SKEW}" \
    TIME_INTEGRATOR="rk4" \
    DECODER_FORM="V1V2v" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    TEST_EVERY="${TEST_EVERY}" \
    KEEP_ITERATION_CHECKPOINTS="0" \
    OUTPUT_ROOT="${SKEW_OUTPUT_ROOT}" \
    RUN_STEM="${run_stem}" \
    sbatch --parsable \
      --job-name="${job_name}" \
      --nodes="${NODES}" \
      --ntasks-per-node="${TASKS_PER_NODE}" \
      --time="${TIME_LIMIT}" \
      --output="${SKEW_BATCHOUT}/%x-%j.out" \
      --error="${SKEW_BATCHOUT}/%x-%j.err" \
      "${SKEW_JOB_SCRIPT}"
}

submit_dense 20
submit_dense 30
submit_skew 40 40
submit_skew 40 80
submit_skew 50 50
submit_skew 50 100
submit_skew 60 60
submit_skew 60 120

echo "Submitted corrected SWE KL-input cases at $(date)."
