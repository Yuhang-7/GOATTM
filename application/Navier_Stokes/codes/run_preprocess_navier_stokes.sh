#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python}"

exec "${PYTHON}" "${SCRIPT_DIR}/preprocess_navier_stokes_npz_dataset.py" "$@"
