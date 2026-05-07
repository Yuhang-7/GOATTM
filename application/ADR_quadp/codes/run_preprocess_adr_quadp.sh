#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/work2/08667/yuuuhang/stampede3/envs/GOATTM311/bin/python}"

exec "${PYTHON}" "${SCRIPT_DIR}/preprocess_adr_quadp_npz_dataset.py" "$@"
