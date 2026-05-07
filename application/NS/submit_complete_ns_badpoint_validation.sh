#!/bin/bash
#SBATCH --job-name=ns_badpoint_validation
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --time=03:00:00
#SBATCH --output=/work2/08667/yuuuhang/stampede3/GOATTM/application/NS/batchout/%x-%j.out
#SBATCH --error=/work2/08667/yuuuhang/stampede3/GOATTM/application/NS/batchout/%x-%j.err

set -euo pipefail

GOAM_NS_ROOT="${GOAM_NS_ROOT:-/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100_New}"
CONDA_SH="${CONDA_SH:-/work2/08667/yuuuhang/.conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fenicsx-env}"
CONDA_BIN_DIR="${CONDA_BIN_DIR:-/work2/08667/yuuuhang/.conda/envs/fenicsx-env/bin}"
PY_BIN="${PY_BIN:-${CONDA_BIN_DIR}/python}"
TASKS="${TASKS:-112}"
NS="${NS:-1000}"
RE="${RE:-100}"
MODELFORM="${MODELFORM:-AHBc}"
RANK="${RANK:-15}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PATH=/work2/08667/yuuuhang/stampede3/GOAM:${PATH}
export PYTHONPATH=/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${GOAM_NS_ROOT}"
mkdir -p batchout

run_case() {
  local case_id="$1"
  local ntrain="$2"
  local logfile_dir="$3"

  echo "============================================================================"
  echo "CASE: ${case_id}"
  echo "Start: $(date)"
  echo "LOGFILE_DIR=${logfile_dir}"
  echo "RANK=${RANK} NS=${NS} NTRAIN=${ntrain} NTEST=$((NS - ntrain)) TASKS=${TASKS}"
  echo "============================================================================"

  if [[ ! -d "${logfile_dir}" ]]; then
    echo "Missing logfile directory: ${logfile_dir}" >&2
    exit 1
  fi
  if [[ ! -d "${logfile_dir}/muf" || ! -d "${logfile_dir}/mug" ]]; then
    echo "Missing muf/mug parameter directories under: ${logfile_dir}" >&2
    exit 1
  fi

  echo "[step] preprocess_all for NTRAIN=${ntrain}"
  "${PY_BIN}" ./preprocess_all.py \
    --Ns "${NS}" \
    --Ntrain "${ntrain}" \
    --threads "${TASKS}" \
    --problemname "Navier_Stokes_Re_100"

  echo "[step] analyze validation convergence"
  ibrun "${PY_BIN}" ./analyze_validation_convergence.py \
    --logfile_dir "${logfile_dir}" \
    --rank "${RANK}" \
    --Ns "${NS}" \
    --Ntrain "${ntrain}" \
    --Re "${RE}" \
    --modelform "${MODELFORM}"

  local csv_path="${logfile_dir}/validation_convergence_r${RANK}.csv"
  if [[ ! -f "${csv_path}" ]]; then
    echo "Expected validation CSV was not produced: ${csv_path}" >&2
    exit 1
  fi
  echo "[ok] produced ${csv_path}"
  tail -5 "${csv_path}"
  echo "End: $(date)"
}

run_case \
  "coeff1e-5_ds448_r15_apr24" \
  448 \
  "/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100_New/output_data/mfAHBc_coeff1e-5_greg1e-8_ds448/20260424_032916_simple_ds448_coeff1e-5_r15/r=15_form=AHBc/logfile/2026-04-24-03-30-17"

run_case \
  "coeff1e-6_ds896_r15_apr24" \
  896 \
  "/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100_New/output_data/mfAHBc_coeff1e-6_greg1e-8_ds896/20260424_033711_simple_ds896_coeff1e-6_r15/r=15_form=AHBc/logfile/2026-04-24-03-38-00"

echo "All NS badpoint validation backfill cases completed at $(date)."
