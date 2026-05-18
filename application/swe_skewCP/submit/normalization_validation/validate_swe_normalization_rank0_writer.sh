#!/bin/bash
#SBATCH --job-name=swe_normchk
#SBATCH --partition=spr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --time=00:20:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/submit/normalization_validation/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/submit/normalization_validation/batchout/%x-%j.err

set -euo pipefail

export GOATTM_REPO_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT"
export CONDA_BIN_DIR="/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin"
export PY_BIN="/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python3"
export MPIRUN_BIN="ibrun"
export CONDA_BASE="/work2/08667/yuuuhang/anaconda3"
export CONDA_ENV_NAME="GOATTM311"
export CONDA_ENV_PREFIX="/work2/08667/yuuuhang/stampede3/envs/GOATTM311"
export MANIFEST_PATH="/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data/manifest.npz"

export NTRAIN="224"
export NTEST="112"
export SAMPLE_COUNT="336"
export MPI_RANKS="224"
export LATENT_RANK="30"
export SKEW_CP_RANK="15"
export OPTIMIZER="gradient_descent"
export MAX_ITERATIONS="1"
export LEARNING_RATE="1e-4"
export GRADIENT_CLIP_NORM="1.0"
export SKEW_CP_INIT_SCALE="1e-4"
export SKEW_CP_ZERO_INIT="1"
export MAX_DT="0.0002"
export TIME_INTEGRATOR="rk4"
export NORMALIZATION_TARGET_MAX_ABS="0.9"
export QOI_STRIDE="1"
export DECODER_FORM="V1v"
export LATENT_EMBEDDING_MODE="qoi_augmentation"
export LATENT_EMBEDDING_AUGMENTATION_SEED="20260507"
export LATENT_EMBEDDING_AUGMENTATION_SCALE="0.1"
export OPINF_REG_W="1e-4"
export OPINF_REG_H="1e-4"
export OPINF_REG_B="1e-4"
export OPINF_REG_C="1e-6"
export DECODER_REG_V1="1e-7"
export DECODER_REG_V2="1e-7"
export DECODER_REG_V0="1e-7"
export DYNAMICS_REG_A="1e-6"
export DYNAMICS_REG_SKEW_CP="1e-4"
export DYNAMICS_REG_B="1e-4"
export DYNAMICS_REG_C="1e-4"
export LBFGS_MAXCOR="20"
export LBFGS_FTOL="1e-12"
export LBFGS_GTOL="1e-8"
export LBFGS_MAXLS="30"

export RUN_STEM="swe_normalization_rank0_writer_${SLURM_JOB_ID:-manual}"
export OUTPUT_ROOT="/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/outputs/normalization_validation/rank0_writer_${SLURM_JOB_ID:-manual}"

mkdir -p "/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/submit/normalization_validation/batchout"

"/work2/08667/yuuuhang/stampede3/GOATTM_tuckerTT/application/swe_skewCP/submit_swe_skewcp.sh"

train_count="$(find "${OUTPUT_ROOT}" -path "*/opinf_abc_init/normalized/samples/train/*.npz" | wc -l | tr -d ' ')"
test_count="$(find "${OUTPUT_ROOT}" -path "*/opinf_abc_init/normalized/samples/test/*.npz" | wc -l | tr -d ' ')"
echo "Normalized train file count: ${train_count}"
echo "Normalized test file count: ${test_count}"

if [[ "${train_count}" -ne "${NTRAIN}" ]]; then
  echo "Expected ${NTRAIN} normalized train files, found ${train_count}." >&2
  exit 3
fi
if [[ "${test_count}" -ne "${NTEST}" ]]; then
  echo "Expected ${NTEST} normalized test files, found ${test_count}." >&2
  exit 3
fi

# Why submit this task:
# This short two-node job verifies the SWE normalization fix under the same
# Slurm/MPI materialization path that failed in production. It should complete
# with exactly NTRAIN train files and NTEST test files in opinf_abc_init/normalized
# before the failed production SWE jobs are resubmitted.
