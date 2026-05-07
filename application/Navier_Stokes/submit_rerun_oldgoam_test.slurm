#!/bin/bash
#SBATCH --job-name=ns_oldgoam_test
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --time=01:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navier_Stokes/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/Navier_Stokes/batchout/%x-%j.err

set -euo pipefail

REPO_ROOT="${GOATTM_REPO_ROOT:-/work2/08667/yuuuhang/stampede3/GOATTM}"
APP_ROOT="${REPO_ROOT}/application/Navier_Stokes"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin}"
PY_BIN="${PY_BIN:-${CONDA_BIN_DIR}/python3}"
MPIRUN_BIN="${MPIRUN_BIN:-ibrun}"

NTRAIN="${NTRAIN:-896}"
NTEST="${NTEST:-104}"
SAMPLE_COUNT="${SAMPLE_COUNT:-$((NTRAIN + NTEST))}"
LATENT_RANK_SEQUENCE="${LATENT_RANK_SEQUENCE:-12 14 16}"
MPI_RANKS="${MPI_RANKS:-${SLURM_NTASKS:-112}}"
MANIFEST_PATH="${MANIFEST_PATH:-${APP_ROOT}/data/processed_data/all_samples_stride1/manifest.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/outputs/oldgoam_regf1e-7_regg1e-9}"
RUN_STEM="${RUN_STEM:-$(date +%Y%m%d_%H%M%S)_ns_oldgoam_ntrain${NTRAIN}_ntest${NTEST}}"

MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
OPTIMIZER="${OPTIMIZER:-bfgs}"
TIME_INTEGRATOR="${TIME_INTEGRATOR:-rk4}"
MAX_DT="${MAX_DT:-0.01}"
DYNAMIC_FORM="${DYNAMIC_FORM:-AHBc}"
DECODER_FORM="${DECODER_FORM:-V1V2v}"
NORMALIZATION_TARGET_MAX_ABS="${NORMALIZATION_TARGET_MAX_ABS:-0.9}"
LATENT_EMBEDDING_MODE="${LATENT_EMBEDDING_MODE:-qoi_augmentation}"
LATENT_EMBEDDING_AUGMENTATION_SEED="${LATENT_EMBEDDING_AUGMENTATION_SEED:-12345}"
LATENT_EMBEDDING_AUGMENTATION_SCALE="${LATENT_EMBEDDING_AUGMENTATION_SCALE:-0.1}"

OPINF_REG_W="${OPINF_REG_W:-1e-9}"
OPINF_REG_H="${OPINF_REG_H:-1e-9}"
OPINF_REG_B="${OPINF_REG_B:-1e-9}"
OPINF_REG_C="${OPINF_REG_C:-1e-9}"
DECODER_REG_V1="${DECODER_REG_V1:-1e-7}"
DECODER_REG_V2="${DECODER_REG_V2:-1e-7}"
DECODER_REG_V0="${DECODER_REG_V0:-1e-7}"
DYNAMICS_REG_A="${DYNAMICS_REG_A:-1e-9}"
DYNAMICS_REG_MU_H="${DYNAMICS_REG_MU_H:-1e-9}"
DYNAMICS_REG_B="${DYNAMICS_REG_B:-1e-9}"
DYNAMICS_REG_C="${DYNAMICS_REG_C:-1e-9}"
DYNAMICS_REG_SPECTRAL_ABSCISSA="${DYNAMICS_REG_SPECTRAL_ABSCISSA:-0.0}"
DYNAMICS_REG_SPECTRAL_ALPHA="${DYNAMICS_REG_SPECTRAL_ALPHA:-0.0}"

BFGS_GTOL="${BFGS_GTOL:-1e-6}"
BFGS_C1="${BFGS_C1:-1e-4}"
BFGS_C2="${BFGS_C2:-0.9}"
BFGS_XRTOL="${BFGS_XRTOL:-1e-7}"

mkdir -p "${APP_ROOT}/batchout" "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Processed manifest not found: ${MANIFEST_PATH}" >&2
  echo "Run submit_convert_hdf5_to_npz.slurm and submit_preprocess_navier_stokes.slurm first." >&2
  exit 1
fi

read -r -a LATENT_RANKS <<< "${LATENT_RANK_SEQUENCE}"
echo "Job start: $(date)"
echo "MANIFEST_PATH: ${MANIFEST_PATH}"
echo "NTRAIN/NTEST: ${NTRAIN}/${NTEST}"
echo "LATENT_RANK_SEQUENCE: ${LATENT_RANK_SEQUENCE}"
echo "reg_f(decoder): ${DECODER_REG_V1}, ${DECODER_REG_V2}, ${DECODER_REG_V0}"
echo "reg_g(dynamics): A=${DYNAMICS_REG_A}, H=${DYNAMICS_REG_MU_H}, B=${DYNAMICS_REG_B}, c=${DYNAMICS_REG_C}"
echo "spectral penalty: ${DYNAMICS_REG_SPECTRAL_ABSCISSA}"
echo "LATENT_EMBEDDING_MODE: ${LATENT_EMBEDDING_MODE}"
echo "TIME_INTEGRATOR: ${TIME_INTEGRATOR}"

for LATENT_RANK in "${LATENT_RANKS[@]}"; do
  RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/${RUN_STEM}_r${LATENT_RANK}"
  mkdir -p "${RUN_OUTPUT_ROOT}"
  echo "Starting NS oldGOAM rank ${LATENT_RANK}: $(date)"
  "${MPIRUN_BIN}" -n "${MPI_RANKS}" "${PY_BIN}" application/common/run_manifest_oldgoam.py \
    --dataset-name navier_stokes_re100 \
    --manifest-path "${MANIFEST_PATH}" \
    --sample-count "${SAMPLE_COUNT}" \
    --ntrain "${NTRAIN}" \
    --ntest "${NTEST}" \
    --latent-rank "${LATENT_RANK}" \
    --dynamic-form "${DYNAMIC_FORM}" \
    --decoder-form "${DECODER_FORM}" \
    --oldgoam \
    --latent-embedding-mode "${LATENT_EMBEDDING_MODE}" \
    --latent-embedding-augmentation-seed "${LATENT_EMBEDDING_AUGMENTATION_SEED}" \
    --latent-embedding-augmentation-scale "${LATENT_EMBEDDING_AUGMENTATION_SCALE}" \
    --max-dt "${MAX_DT}" \
    --time-integrator "${TIME_INTEGRATOR}" \
    --optimizer "${OPTIMIZER}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --normalization-target-max-abs "${NORMALIZATION_TARGET_MAX_ABS}" \
    --opinf-reg-w "${OPINF_REG_W}" --opinf-reg-h "${OPINF_REG_H}" --opinf-reg-b "${OPINF_REG_B}" --opinf-reg-c "${OPINF_REG_C}" \
    --decoder-reg-v1 "${DECODER_REG_V1}" --decoder-reg-v2 "${DECODER_REG_V2}" --decoder-reg-v0 "${DECODER_REG_V0}" \
    --dynamics-reg-a "${DYNAMICS_REG_A}" --dynamics-reg-mu-h "${DYNAMICS_REG_MU_H}" --dynamics-reg-b "${DYNAMICS_REG_B}" --dynamics-reg-c "${DYNAMICS_REG_C}" \
    --dynamics-reg-spectral-abscissa "${DYNAMICS_REG_SPECTRAL_ABSCISSA}" \
    --dynamics-reg-spectral-alpha "${DYNAMICS_REG_SPECTRAL_ALPHA}" \
    --bfgs-gtol "${BFGS_GTOL}" --bfgs-c1 "${BFGS_C1}" --bfgs-c2 "${BFGS_C2}" --bfgs-xrtol "${BFGS_XRTOL}" \
    --output-dir "${RUN_OUTPUT_ROOT}"
  echo "Finished NS oldGOAM rank ${LATENT_RANK}: $(date)"
done

echo "Job finished: $(date)"

# Test-copy submitted on 2026-05-03. Intent: full-case wiring check; only Slurm time and MAX_ITERATIONS differ from the formal script.
