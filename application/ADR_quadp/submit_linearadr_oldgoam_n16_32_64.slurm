#!/bin/bash
#SBATCH --job-name=linearadr_n16_32_64
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/ADR_quadp/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/ADR_quadp"
MANIFEST_PATH="${MANIFEST_PATH:-${APP_ROOT}/data/processed_data/linearadr_all_samples_stride1/manifest.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/outputs/linearadr_oldgoam_regf1e-7_regg1e-9}"
LATENT_RANK_SEQUENCE="${LATENT_RANK_SEQUENCE:-12 14 16}"
NTEST="${NTEST:-104}"
MPI_RANKS="${MPI_RANKS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
MAX_DT="${MAX_DT:-0.01}"

cd "${REPO_ROOT}"
echo "LinearADR oldGOAM serial small-data job start: $(date)"
echo "NTRAIN sequence: 16 32 64"
echo "LATENT_RANK_SEQUENCE: ${LATENT_RANK_SEQUENCE}"
echo "MPI_RANKS: ${MPI_RANKS}"
echo "TIME_INTEGRATOR: ${TIME_INTEGRATOR}"

for NTRAIN in 16 32 64; do
  SAMPLE_COUNT=$((NTRAIN + NTEST))
  RUN_STEM="$(date +%Y%m%d_%H%M%S)_linearadr_oldgoam_ntrain${NTRAIN}_ntest${NTEST}"
  echo "Starting NTRAIN=${NTRAIN} at $(date)"
  env MANIFEST_PATH="${MANIFEST_PATH}" NTRAIN="${NTRAIN}" NTEST="${NTEST}" SAMPLE_COUNT="${SAMPLE_COUNT}" \
    LATENT_RANK_SEQUENCE="${LATENT_RANK_SEQUENCE}" MPI_RANKS="${MPI_RANKS}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_STEM="${RUN_STEM}" MAX_ITERATIONS="${MAX_ITERATIONS}" TIME_INTEGRATOR="${TIME_INTEGRATOR}" \
    OPTIMIZER="${OPTIMIZER}" MAX_DT="${MAX_DT}" bash application/ADR_quadp/submit_linearadr_oldgoam.sh
  echo "Finished NTRAIN=${NTRAIN} at $(date)"
done

echo "LinearADR oldGOAM serial small-data job end: $(date)"
