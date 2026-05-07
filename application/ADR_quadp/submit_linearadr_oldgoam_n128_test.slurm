#!/bin/bash
#SBATCH --job-name=linearadr_n128_test
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=01:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
MANIFEST_PATH="${MANIFEST_PATH:-${APP_ROOT}/data/processed_data/linearadr_all_samples_stride1/manifest.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/outputs/linearadr_oldgoam_regf1e-7_regg1e-9}"
NTRAIN="${NTRAIN:-128}"
NTEST="${NTEST:-104}"
SAMPLE_COUNT="${SAMPLE_COUNT:-$((NTRAIN + NTEST))}"
LATENT_RANK_SEQUENCE="${LATENT_RANK_SEQUENCE:-12 14 16}"
MPI_RANKS="${MPI_RANKS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
MAX_DT="${MAX_DT:-0.01}"
RUN_STEM="${RUN_STEM:-$(date +%Y%m%d_%H%M%S)_linearadr_oldgoam_ntrain${NTRAIN}_ntest${NTEST}}"

cd "${REPO_ROOT}"
env MANIFEST_PATH="${MANIFEST_PATH}" NTRAIN="${NTRAIN}" NTEST="${NTEST}" SAMPLE_COUNT="${SAMPLE_COUNT}" \
  LATENT_RANK_SEQUENCE="${LATENT_RANK_SEQUENCE}" MPI_RANKS="${MPI_RANKS}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_STEM="${RUN_STEM}" MAX_ITERATIONS="${MAX_ITERATIONS}" TIME_INTEGRATOR="${TIME_INTEGRATOR}" \
  OPTIMIZER="${OPTIMIZER}" MAX_DT="${MAX_DT}" bash application/ADR_quadp/submit_linearadr_oldgoam.sh

# Test-copy submitted on 2026-05-03. Intent: full-case wiring check; only Slurm time and MAX_ITERATIONS differ from the formal script.
