from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "module_test" / "timer_overhead" / "run_phase_timing_worker.py"
OUTPUT_DIR = ROOT / "module_test" / "output_plots" / "timer_overhead"


def run_worker(output_json: Path, disable_numba: bool) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    if disable_numba:
        env["GOATTM_DISABLE_NUMBA"] = "1"
    else:
        env.pop("GOATTM_DISABLE_NUMBA", None)

    command = [sys.executable, str(WORKER), "--output-json", str(output_json), "--sample-count", "100"]
    completed = subprocess.run(command, env=env, cwd=str(ROOT), capture_output=True, text=True, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enabled_path = OUTPUT_DIR / "phase_timing_numba_enabled.json"
    disabled_path = OUTPUT_DIR / "phase_timing_numba_disabled.json"

    enabled = run_worker(enabled_path, disable_numba=False)
    disabled = run_worker(disabled_path, disable_numba=True)

    compare: dict[str, object] = {
        "sample_count": enabled["sample_count"],
        "enabled_wall_seconds": enabled["wall_seconds"],
        "disabled_wall_seconds": disabled["wall_seconds"],
        "wall_speedup_ratio_enabled_over_disabled": (
            float(enabled["wall_seconds"]) / float(disabled["wall_seconds"])
            if float(disabled["wall_seconds"]) > 0.0
            else None
        ),
        "phase_compare": {},
    }

    enabled_phases = enabled["phase_summary"]
    disabled_phases = disabled["phase_summary"]
    for phase_name in enabled_phases.keys():
        enabled_phase = enabled_phases[phase_name]
        disabled_phase = disabled_phases[phase_name]
        if enabled_phase is None or disabled_phase is None:
            compare["phase_compare"][phase_name] = {
                "enabled": enabled_phase,
                "disabled": disabled_phase,
            }
            continue

        enabled_total = float(enabled_phase["total_seconds"])
        disabled_total = float(disabled_phase["total_seconds"])
        compare["phase_compare"][phase_name] = {
            "enabled_total_seconds": enabled_total,
            "disabled_total_seconds": disabled_total,
            "enabled_call_count": int(enabled_phase["call_count"]),
            "disabled_call_count": int(disabled_phase["call_count"]),
            "enabled_average_seconds": float(enabled_phase["average_seconds"]),
            "disabled_average_seconds": float(disabled_phase["average_seconds"]),
            "enabled_over_disabled_ratio": enabled_total / disabled_total if disabled_total > 0.0 else None,
        }

    compare_path = OUTPUT_DIR / "phase_timing_numba_compare.json"
    compare_path.write_text(json.dumps(compare, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(compare, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
