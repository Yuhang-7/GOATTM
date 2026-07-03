#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/work2/08667/yuuuhang/stampede3/GOATTM/application/Navierstokes100_matern52/initial/AHBc_gridstate_reg_sweep_sets")

rows = []
for path in sorted(ROOT.glob("*/sweep_summary_r*.json")):
    data = json.loads(path.read_text())
    best = data["best"]
    rows.append(
        {
            "regset": path.parent.name,
            "rank": int(data["rank"]),
            "score": float(best["score"]),
            "dyn_valid": float(best["dynamics_valid_relative_residual"]),
            "dec_valid": float(best["decoder_valid_relative_residual"]),
            "reg_a": float(best["reg_a"]),
            "reg_h": float(best["reg_h"]),
            "reg_b": float(best["reg_b"]),
            "reg_c": float(best["reg_c"]),
            "reg_decoder": float(best["reg_decoder"]),
        }
    )

rows = sorted(rows, key=lambda r: (r["rank"], r["score"]))
csv_path = ROOT / "regset_score_summary.csv"
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

best_by_rank = []
for rank in sorted({row["rank"] for row in rows}):
    best = min([row for row in rows if row["rank"] == rank], key=lambda row: row["score"])
    best_by_rank.append(best)

best_path = ROOT / "regset_best_by_rank.json"
best_path.write_text(json.dumps(best_by_rank, indent=2) + "\n")

print(f"rows {len(rows)}")
print(f"csv {csv_path}")
print(f"best_json {best_path}")
print("best by rank:")
for row in best_by_rank:
    print(
        row["rank"],
        row["regset"],
        f"score={row['score']:.9g}",
        f"dyn={row['dyn_valid']:.9g}",
        f"dec={row['dec_valid']:.9g}",
    )
print("best counts:", dict(Counter(row["regset"] for row in best_by_rank)))
