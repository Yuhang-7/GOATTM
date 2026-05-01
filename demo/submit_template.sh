#!/bin/bash
#SBATCH --job-name=NS_ds896_c1e-6_r15
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --time=24:00:00
#SBATCH --output=./batchout/%x-%j.out
#SBATCH --error=./batchout/%x-%j.err

set -euo pipefail

cd /work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100_New

CONDA_BIN_DIR="/work2/08667/yuuuhang/.conda/envs/fenicsx-env/bin"
PY_BIN="${CONDA_BIN_DIR}/python3"

export PATH=/work2/08667/yuuuhang/stampede3/GOAM:${PATH}
export PYTHONPATH=/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean:${PYTHONPATH:-}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NTRAIN=896
NS=1000
RANK=15
DYNAMIC_FORM="AHBc"
DECODER_FORM="V1V2v"
COEFF_MUA_STABLE="1e-6"
G_STAB_COEFF="1e-8"
THREADS="${SLURM_NTASKS}"
TMP_TAG="Navier_Stokes_Re_100_Ns_${NS}_ntrain_${NTRAIN}_threads_${THREADS}"
TMP_DIR="./tmpdirectory/${TMP_TAG}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_simple_ds896_coeff1e-6_r15"
OUTPUT_DIR="./output_data/${DYNAMIC_FORM}_${DECODER_FORM}_coeff${COEFF_MUA_STABLE}_greg${G_STAB_COEFF}_ds${NTRAIN}/${RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"

echo "Job start: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Assuming preprocess was already run locally."
echo "Expected tmpdirectory: ${TMP_DIR}"

if [[ ! -d "${TMP_DIR}/train" ]] || [[ ! -d "${TMP_DIR}/test" ]]; then
  echo "Missing preprocessed tmpdirectory: ${TMP_DIR}" >&2
  echo "Run the local preprocess script first." >&2
  exit 1
fi

ibrun "${PY_BIN}" ./main.py \
  --Ns "${NS}" \
  --Ntrain "${NTRAIN}" \
  -r "${RANK}" \
  -t "${THREADS}" \
  -Re 100 \
  --coeff_muA_stable "${COEFF_MUA_STABLE}" \
  --g_stab_coeff "${G_STAB_COEFF}" \
  --output_dir "${OUTPUT_DIR}" \
  --dynamic-form "${DYNAMIC_FORM}" \
  --decoder-form "${DECODER_FORM}"

"${PY_BIN}" ./parse_results.py --output_dir "${OUTPUT_DIR}"

echo "Job end: $(date)"
