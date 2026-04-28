#!/usr/bin/env bash
# Run the configurable GOATTM reduced-QoI demo.
#
# Usage:
#   bash demo/run_reduced_qoi_demo.sh [NTRAIN] [NTEST] [extra python args...]
#
# Examples:
#   bash demo/run_reduced_qoi_demo.sh
#   bash demo/run_reduced_qoi_demo.sh 10 10
#   bash demo/run_reduced_qoi_demo.sh 100 100 --max-iterations 50
#   LATENT_RANK=10 bash demo/run_reduced_qoi_demo.sh 10 10
#   MPI_RANKS=20 LATENT_RANK=6 OPTIMIZER=adam bash demo/run_reduced_qoi_demo.sh 100 100
#
# By default MPI_RANKS=NTRAIN, so each train case gets one rank. Override
# MPI_RANKS when the train set is larger than the available MPI slots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NTRAIN="${1:-10}"
NTEST="${2:-10}"
if [[ $# -ge 1 ]]; then
  shift
fi
if [[ $# -ge 1 ]]; then
  shift
fi

MPI_RANKS="${MPI_RANKS:-${NTRAIN}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
LATENT_RANK="${LATENT_RANK:-4}"
OPTIMIZER="${OPTIMIZER:-lbfgs}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
MAX_DT="${MAX_DT:-0.01}"
SEED="${SEED:-20260428}"
LBFGS_MAXCOR="${LBFGS_MAXCOR:-20}"
LBFGS_FTOL="${LBFGS_FTOL:-1e-12}"
LBFGS_GTOL="${LBFGS_GTOL:-1e-8}"
LBFGS_MAXLS="${LBFGS_MAXLS:-30}"
OPINF_REG_W="${OPINF_REG_W:-1e-4}"
OPINF_REG_H="${OPINF_REG_H:-1e-4}"
OPINF_REG_B="${OPINF_REG_B:-1e-4}"
OPINF_REG_C="${OPINF_REG_C:-1e-6}"
DECODER_REG_V1="${DECODER_REG_V1:-1e-7}"
DECODER_REG_V2="${DECODER_REG_V2:-1e-7}"
DECODER_REG_V0="${DECODER_REG_V0:-1e-7}"
DYNAMICS_REG_S="${DYNAMICS_REG_S:-1e-4}"
DYNAMICS_REG_W="${DYNAMICS_REG_W:-1e-4}"
DYNAMICS_REG_MU_H="${DYNAMICS_REG_MU_H:-1e-4}"
DYNAMICS_REG_B="${DYNAMICS_REG_B:-1e-4}"
DYNAMICS_REG_C="${DYNAMICS_REG_C:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if ! [[ "${NTRAIN}" =~ ^[0-9]+$ ]] || [[ "${NTRAIN}" -le 0 ]]; then
  echo "NTRAIN must be a positive integer, got '${NTRAIN}'." >&2
  exit 2
fi
if ! [[ "${NTEST}" =~ ^[0-9]+$ ]] || [[ "${NTEST}" -le 0 ]]; then
  echo "NTEST must be a positive integer, got '${NTEST}'." >&2
  exit 2
fi
if ! [[ "${MPI_RANKS}" =~ ^[0-9]+$ ]] || [[ "${MPI_RANKS}" -le 0 ]]; then
  echo "MPI_RANKS must be a positive integer, got '${MPI_RANKS}'." >&2
  exit 2
fi
if ! [[ "${LATENT_RANK}" =~ ^[0-9]+$ ]] || [[ "${LATENT_RANK}" -le 0 ]]; then
  echo "LATENT_RANK must be a positive integer, got '${LATENT_RANK}'." >&2
  exit 2
fi
if ! [[ "${MAX_ITERATIONS}" =~ ^[0-9]+$ ]] || [[ "${MAX_ITERATIONS}" -le 0 ]]; then
  echo "MAX_ITERATIONS must be a positive integer, got '${MAX_ITERATIONS}'." >&2
  exit 2
fi

PY_ARGS=(
  "${REPO_ROOT}/demo/run_reduced_qoi_demo.py"
  --ntrain "${NTRAIN}"
  --ntest "${NTEST}"
  --latent-rank "${LATENT_RANK}"
  --optimizer "${OPTIMIZER}"
  --max-iterations "${MAX_ITERATIONS}"
  --max-dt "${MAX_DT}"
  --seed "${SEED}"
  --lbfgs-maxcor "${LBFGS_MAXCOR}"
  --lbfgs-ftol "${LBFGS_FTOL}"
  --lbfgs-gtol "${LBFGS_GTOL}"
  --lbfgs-maxls "${LBFGS_MAXLS}"
  --opinf-reg-w "${OPINF_REG_W}"
  --opinf-reg-h "${OPINF_REG_H}"
  --opinf-reg-b "${OPINF_REG_B}"
  --opinf-reg-c "${OPINF_REG_C}"
  --decoder-reg-v1 "${DECODER_REG_V1}"
  --decoder-reg-v2 "${DECODER_REG_V2}"
  --decoder-reg-v0 "${DECODER_REG_V0}"
  --dynamics-reg-s "${DYNAMICS_REG_S}"
  --dynamics-reg-w "${DYNAMICS_REG_W}"
  --dynamics-reg-mu-h "${DYNAMICS_REG_MU_H}"
  --dynamics-reg-b "${DYNAMICS_REG_B}"
  --dynamics-reg-c "${DYNAMICS_REG_C}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  PY_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

exec "${MPIRUN_BIN}" -n "${MPI_RANKS}" "${PYTHON_BIN}" "${PY_ARGS[@]}" "$@"
