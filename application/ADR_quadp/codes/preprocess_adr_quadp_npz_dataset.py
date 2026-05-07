from __future__ import annotations

import os
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goattm.preprocess.goam_chunk_npz import GoamChunkNpzDefaults, run_cli  # noqa: E402


GOAM_CLEAN_ROOT = Path(os.environ.get("GOAM_CLEAN_ROOT", "/work2/08667/yuuuhang/stampede3/GOAM_clean/goam_clean"))
APP_ROOT = THIS_FILE.parents[1]
DEFAULT_TRAIN_ROOT = (
    GOAM_CLEAN_ROOT
    / "Example"
    / "ADR_LinearQoI"
    / "tmpdirectory"
    / "ADR_LinearQoI_quadp_opinf_trainsize_112_threads_112"
)
DEFAULT_TEST_ROOT = (
    GOAM_CLEAN_ROOT
    / "Example"
    / "ADR_LinearQoI"
    / "tmpdirectory"
    / "ADR_LinearQoI_quadp_validate_trainsize_112_threads_112"
)
DEFAULT_OUTPUT_ROOT = APP_ROOT / "data" / "processed_data"


def main() -> None:
    run_cli(
        GoamChunkNpzDefaults(
            train_root=DEFAULT_TRAIN_ROOT,
            test_root=DEFAULT_TEST_ROOT,
            output_root=DEFAULT_OUTPUT_ROOT,
            dataset_kind="adr_quadp_goam_chunk_npz",
            sample_prefix="adr_quadp",
            qoi_stride=1,
        ),
        description="Convert GOAM ADR_quadp chunk .npz files into GOATTM manifest format.",
    )


if __name__ == "__main__":
    main()
