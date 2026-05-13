from __future__ import annotations

import runpy
from pathlib import Path


def test_ldnet_taylor_script_runs() -> None:
    script = Path(__file__).resolve().parents[1] / "module_test" / "ldnet_taylor" / "run_ldnet_taylor_test.py"
    runpy.run_path(str(script), run_name="__main__")
