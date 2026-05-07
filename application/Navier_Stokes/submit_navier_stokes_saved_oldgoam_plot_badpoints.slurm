#!/bin/bash
#SBATCH --job-name=ns_plot_badpt
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --time=18:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navier_Stokes/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navier_Stokes/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/Navier_Stokes"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin}"
PY_BIN="${PY_BIN:-${CONDA_BIN_DIR}/python3}"
MPIRUN_BIN="${MPIRUN_BIN:-ibrun}"

MANIFEST_PATH="${MANIFEST_PATH:-${APP_ROOT}/data/processed_data/all_samples_stride1/manifest.npz}"
INITIAL_VALUE_DIR="${INITIAL_VALUE_DIR:-/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100/initial/AHBc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/outputs/ns_plot_badpoint_saved_oldgoam_reruns}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MPI_RANKS="${MPI_RANKS:-${SLURM_NTASKS:-112}}"

MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
MAX_DT="${MAX_DT:-0.01}"
DECODER_FORM="${DECODER_FORM:-V1V2v}"
WEIGHTF="${WEIGHTF:-1e-4}"
G_STAB_COEFF="${G_STAB_COEFF:-1e-8}"
SPECTRAL_ALPHA="${SPECTRAL_ALPHA:-0.0}"

BFGS_GTOL="${BFGS_GTOL:-1e-6}"
BFGS_C1="${BFGS_C1:-1e-4}"
BFGS_C2="${BFGS_C2:-0.9}"
BFGS_XRTOL="${BFGS_XRTOL:-1e-7}"

case "${CASE_GROUP_ID:-0}" in
  0)
    CASES=$'896 104 15 1e-6 1e-6 r15_ds896_coeff1e-6\n896 104 14 1e-7 1e-7 r14_ds896_coeff1e-7'
    ;;
  1)
    CASES=$'896 104 12 1e-5 1e-5 r12_ds896_coeff1e-5\n448 52 15 1e-5 1e-5 r15_ds448_coeff1e-5'
    ;;
  2)
    CASES=$'448 52 14 1e-5 1e-5 r14_ds448_coeff1e-5\n448 52 14 0 0 r14_ds448_coeff0'
    ;;
  3)
    CASES=$'448 52 14 1e-4 1e-4 r14_ds448_coeff1e-4\n448 52 11 1e-6 1e-6 r11_ds448_coeff1e-6'
    ;;
  4)
    CASES=$'224 26 10 1e-6 1e-6 r10_ds224_coeff1e-6\n112 13 12 1e-6 1e-6 r12_ds112_coeff1e-6'
    ;;
  5)
    CASES=$'896 104 14 1e-7 1e-7 r14_ds896_coeff1e-7'
    ;;
  6)
    CASES=$'448 52 15 1e-5 1e-5 r15_ds448_coeff1e-5\n448 52 14 0 0 r14_ds448_coeff0\n448 52 11 1e-6 1e-6 r11_ds448_coeff1e-6'
    ;;
  7)
    CASES=$'112 13 12 1e-6 1e-6 r12_ds112_coeff1e-6'
    ;;
  8)
    CASES=$'896 104 12 1e-5 1e-5 r12_ds896_coeff1e-5_xrtol0_rerun'
    ;;
  *) echo "Unknown CASE_GROUP_ID=${CASE_GROUP_ID:-unset}" >&2; exit 2 ;;
esac

mkdir -p "${APP_ROOT}/batchout" "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Processed manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

echo "Array task start: $(date)"
echo "CASE_GROUP_ID=${CASE_GROUP_ID:-0}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "INITIAL_VALUE_DIR=${INITIAL_VALUE_DIR}"
echo "CASES:"
printf '%s\n' "${CASES}"

while read -r NTRAIN NTEST LATENT_RANK COEFF_LABEL COEFF_VALUE CASE_TAG; do
  [[ -z "${NTRAIN}" ]] && continue
  SAMPLE_COUNT=$((NTRAIN + NTEST))
  DECODER_REG=$(awk -v n="${NTRAIN}" -v w="${WEIGHTF}" 'BEGIN { printf "%.17g", n*w }')
  DYNAMICS_REG=$(awk -v n="${SAMPLE_COUNT}" -v w="${G_STAB_COEFF}" 'BEGIN { printf "%.17g", n*w }')
  SPECTRAL_REG=$(awk -v n="${NTRAIN}" -v w="${COEFF_VALUE}" 'BEGIN { printf "%.17g", n*w }')
  RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/mfAHBc_coeff${COEFF_LABEL}_greg${G_STAB_COEFF}_ds${NTRAIN}/${RUN_STAMP}_${CASE_TAG}"
  mkdir -p "${RUN_OUTPUT_ROOT}"

  if [[ ! -f "${INITIAL_VALUE_DIR}/mug_${LATENT_RANK}.npy" || ! -f "${INITIAL_VALUE_DIR}/muf_${LATENT_RANK}.npy" ]]; then
    echo "Missing saved oldGOAM NS initialization for rank ${LATENT_RANK} in ${INITIAL_VALUE_DIR}" >&2
    exit 2
  fi

  echo "Case start: $(date)"
  echo "CASE_TAG=${CASE_TAG}"
  echo "NTRAIN=${NTRAIN} NTEST=${NTEST} SAMPLE_COUNT=${SAMPLE_COUNT} RANK=${LATENT_RANK}"
  echo "plot coeff=${COEFF_VALUE}; scaled spectral reg=NTRAIN*coeff=${SPECTRAL_REG}; alpha=${SPECTRAL_ALPHA}"
  echo "greg=${G_STAB_COEFF}; scaled dynamics reg=Ns*greg=${DYNAMICS_REG}"
  echo "decoder reg=NTRAIN*${WEIGHTF}=${DECODER_REG}"
  echo "OUTPUT=${RUN_OUTPUT_ROOT}"

  "${MPIRUN_BIN}" -n "${MPI_RANKS}" "${PY_BIN}" application/Navier_Stokes/codes/run_manifest_navier_stokes_saved_init.py \
    --dataset-name navier_stokes_re100 \
    --manifest-path "${MANIFEST_PATH}" \
    --sample-count "${SAMPLE_COUNT}" \
    --ntrain "${NTRAIN}" \
    --ntest "${NTEST}" \
    --latent-rank "${LATENT_RANK}" \
    --decoder-form "${DECODER_FORM}" \
    --initial-value-dir "${INITIAL_VALUE_DIR}" \
    --max-dt "${MAX_DT}" \
    --time-integrator "${TIME_INTEGRATOR}" \
    --optimizer "${OPTIMIZER}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --decoder-reg-v1 "${DECODER_REG}" --decoder-reg-v2 "${DECODER_REG}" --decoder-reg-v0 "${DECODER_REG}" \
    --dynamics-reg-a "${DYNAMICS_REG}" --dynamics-reg-mu-h "${DYNAMICS_REG}" --dynamics-reg-b "${DYNAMICS_REG}" --dynamics-reg-c "${DYNAMICS_REG}" \
    --dynamics-reg-spectral-abscissa "${SPECTRAL_REG}" \
    --dynamics-reg-spectral-alpha "${SPECTRAL_ALPHA}" \
    --bfgs-gtol "${BFGS_GTOL}" --bfgs-c1 "${BFGS_C1}" --bfgs-c2 "${BFGS_C2}" --bfgs-xrtol "${BFGS_XRTOL}" \
    --output-dir "${RUN_OUTPUT_ROOT}" \
    --run-stamp "${RUN_STAMP}_${CASE_TAG}" \
    < /dev/null

  echo "Case finished: $(date)"
done <<< "${CASES}"

echo "Array task finished: $(date)"

# Prepared on 2026-05-05. Intent: rerun NS plotting bad/sample points from the eigenvalue-penalty sweep, two cases per Slurm task, using saved oldGOAM muf/mug initialization and the plotting regularization convention:
# decoder reg = Ntrain*1e-4, dynamics reg = (Ntrain+Ntest)*1e-8, spectral reg = Ntrain*plot_coeff.
