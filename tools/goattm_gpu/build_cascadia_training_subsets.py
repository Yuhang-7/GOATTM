from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rupture_id_text(value: int) -> str:
    return f"{int(value):06d}"


def write_id_list(path: Path, ids: list[int]) -> None:
    path.write_text("\n".join(rupture_id_text(i) for i in ids) + "\n", encoding="utf-8")


def save_manifest(path: Path, records: list[dict[str, object]], root: str) -> None:
    np.savez(
        path,
        sample_ids=np.asarray([str(item["sample_id"]) for item in records], dtype=object),
        sample_paths=np.asarray([str(item["sample_path"]) for item in records], dtype=object),
        root=np.array(root),
    )


def evenly_spaced_bins(active_bins: list[int], count: int) -> set[int]:
    if count <= 0:
        return set()
    if count > len(active_bins):
        raise ValueError("cannot choose more bins than active bins")
    selected = set()
    m = len(active_bins)
    for i in range(count):
        selected.add(active_bins[int(np.floor((i + 0.5) * m / count))])
    if len(selected) != count:
        for b in active_bins:
            selected.add(b)
            if len(selected) == count:
                break
    return selected


def allocate_counts(capacities: dict[int, int], target_total: int) -> dict[int, int]:
    total_capacity = sum(capacities.values())
    if target_total > total_capacity:
        raise ValueError(f"target_total={target_total} exceeds total capacity={total_capacity}")
    remaining = int(target_total)
    active = sorted(capacities)
    allocation = {b: 0 for b in active}
    while active:
        base = remaining // len(active)
        saturated = [b for b in active if capacities[b] <= base]
        if not saturated:
            remainder = remaining - base * len(active)
            extra_bins = evenly_spaced_bins(active, remainder)
            for b in active:
                allocation[b] = base + (1 if b in extra_bins else 0)
            break
        for b in saturated:
            allocation[b] = capacities[b]
            remaining -= capacities[b]
        active = [b for b in active if b not in saturated]
    if sum(allocation.values()) != target_total:
        raise RuntimeError("allocation did not hit target total")
    return allocation


def select_evenly(sorted_ids: list[int], count: int) -> list[int]:
    if count > len(sorted_ids):
        raise ValueError("cannot select more ids than available")
    if count == len(sorted_ids):
        return list(sorted_ids)
    if count == 0:
        return []
    indices = np.floor((np.arange(count) + 0.5) * len(sorted_ids) / count).astype(int)
    selected = [sorted_ids[int(i)] for i in indices]
    if len(set(selected)) != count:
        raise RuntimeError("even selection produced duplicate ids")
    return selected


def bin_for_rupture_id(rid: int, bins: int, bin_size: int) -> int | None:
    if 0 <= rid < bins * bin_size:
        return rid // bin_size
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested Cascadia training subsets from fixed train split.")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--targets", type=int, nargs="+", default=[2048 * 4, 2048 * 8])
    args = parser.parse_args()

    split_dir = Path(args.split_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = np.load(split_dir / "train_manifest.npz", allow_pickle=True)
    root = str(train_manifest["root"].item()) if "root" in train_manifest.files else ""
    records = []
    for sample_id, sample_path in zip(train_manifest["sample_ids"].tolist(), train_manifest["sample_paths"].tolist()):
        rid = int(str(sample_id))
        b = bin_for_rupture_id(rid, int(args.bins), int(args.bin_size))
        records.append(
            {
                "sample_id": str(sample_id),
                "sample_path": str(sample_path),
                "rupture_id": rid,
                "bin": b,
            }
        )
    records.sort(key=lambda item: int(item["rupture_id"]))
    by_id = {int(item["rupture_id"]): item for item in records}
    by_bin: dict[int, list[int]] = {b: [] for b in range(int(args.bins))}
    outside_ids = []
    for item in records:
        rid = int(item["rupture_id"])
        if item["bin"] is None:
            outside_ids.append(rid)
        else:
            by_bin[int(item["bin"])].append(rid)
    for ids in by_bin.values():
        ids.sort()
    capacities = {b: len(ids) for b, ids in by_bin.items()}

    targets = sorted(set(int(t) for t in args.targets), reverse=True)
    largest_target = targets[0]
    largest_counts = allocate_counts(capacities, largest_target)
    largest_selected_by_bin = {
        b: select_evenly(by_bin[b], largest_counts[b])
        for b in range(int(args.bins))
    }

    summaries = {}
    selected_sets: dict[int, set[int]] = {}
    for target in targets:
        if target == largest_target:
            selected_by_bin = largest_selected_by_bin
            counts = largest_counts
        else:
            counts = allocate_counts({b: len(largest_selected_by_bin[b]) for b in largest_selected_by_bin}, target)
            selected_by_bin = {
                b: select_evenly(largest_selected_by_bin[b], counts[b])
                for b in range(int(args.bins))
            }
        selected_ids = sorted(rid for ids in selected_by_bin.values() for rid in ids)
        selected_sets[target] = set(selected_ids)
        selected_records = [by_id[rid] for rid in selected_ids]
        prefix = f"train_{target}"
        save_manifest(output_dir / f"{prefix}_manifest.npz", selected_records, root)
        write_id_list(output_dir / f"{prefix}_ruptures.txt", selected_ids)
        bin_rows = []
        for b in range(int(args.bins)):
            ids = selected_by_bin[b]
            bin_rows.append(
                {
                    "bin": b,
                    "capacity": capacities[b],
                    "selected_count": len(ids),
                    "selected_first": ids[0] if ids else "",
                    "selected_last": ids[-1] if ids else "",
                }
            )
        with (output_dir / f"{prefix}_bin_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bin_rows)
        summaries[target] = {
            "target": target,
            "selected_count": len(selected_ids),
            "manifest": f"{prefix}_manifest.npz",
            "rupture_list": f"{prefix}_ruptures.txt",
            "bin_summary": f"{prefix}_bin_summary.csv",
            "min_selected_per_bin": min(len(selected_by_bin[b]) for b in selected_by_bin),
            "max_selected_per_bin": max(len(selected_by_bin[b]) for b in selected_by_bin),
            "bin_counts": {str(b): len(selected_by_bin[b]) for b in range(int(args.bins))},
        }

    with (output_dir / "training_subset_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["rupture_id", "sample_id", "bin", "sample_path"] + [f"in_train_{t}" for t in sorted(targets)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rid in sorted(by_id):
            item = by_id[rid]
            row = {
                "rupture_id": rupture_id_text(rid),
                "sample_id": item["sample_id"],
                "bin": "outside_40_bins" if item["bin"] is None else item["bin"],
                "sample_path": item["sample_path"],
            }
            for target in sorted(targets):
                row[f"in_train_{target}"] = int(rid in selected_sets[target])
            writer.writerow(row)

    smaller_targets = sorted(targets)
    nested_checks = {}
    for small, large in zip(smaller_targets, smaller_targets[1:]):
        nested_checks[f"{small}_subset_of_{large}"] = selected_sets[small].issubset(selected_sets[large])

    payload = {
        "source_split_dir": str(split_dir),
        "root": root,
        "bins": int(args.bins),
        "bin_size": int(args.bin_size),
        "training_eligible_count": len(records),
        "training_eligible_inside_40_bins": sum(capacities.values()),
        "outside_40_bins": [rupture_id_text(rid) for rid in sorted(outside_ids)],
        "targets": summaries,
        "nested_checks": nested_checks,
        "selection_policy": (
            "Counts are allocated by deterministic water-filling across 40 bins. "
            "Within each bin, rupture ids are selected by midpoint-of-strata spacing. "
            "Smaller training sets are selected as nested subsets of larger training sets."
        ),
    }
    (output_dir / "training_subset_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        r"\documentclass[11pt]{ctexart}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\title{Cascadia Training Subsets}",
        r"\author{}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{规则}",
        "从固定的 training-eligible set 中再取两个训练子集：8192 和 16384。",
        "抽取只在 40 个 rupture-id bin 内进行；40 个 bin 之外的 rupture 不进入这两个固定 training subset。",
        "每个目标大小先按 bin 做 deterministic water-filling，尽量均匀，同时不超过该 bin 的可用 training 数。",
        "在每个 bin 内，对按 id 升序排列的 training ruptures 使用 midpoint-of-strata 方式均匀抽取。",
        "8192 子集被构造成 16384 子集的 nested subset。",
        r"\section{计数}",
        rf"training-eligible total: {len(records)}.",
        rf"inside 40 bins: {sum(capacities.values())}.",
        rf"outside 40 bins: \texttt{{{', '.join(rupture_id_text(rid) for rid in sorted(outside_ids)) if outside_ids else 'none'}}}.",
        r"\section{Subsets}",
        r"\begin{longtable}{rrrr}",
        r"\toprule",
        r"target & selected & min per bin & max per bin \\",
        r"\midrule",
        r"\endhead",
    ]
    for target in sorted(summaries):
        s = summaries[target]
        lines.append(
            f"{target} & {s['selected_count']} & {s['min_selected_per_bin']} & {s['max_selected_per_bin']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\section{Files}",
            r"\begin{itemize}",
        ]
    )
    for target in sorted(summaries):
        lines.append(rf"\item \texttt{{train\_{target}\_manifest.npz}} and \texttt{{train\_{target}\_ruptures.txt}}.")
    lines.extend(
        [
            r"\item \texttt{training\_subset\_assignments.csv}: membership flags for all training-eligible ruptures.",
            r"\item \texttt{training\_subset\_summary.json}: machine-readable summary.",
            r"\end{itemize}",
            r"\end{document}",
        ]
    )
    (output_dir / "training_subset_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "training_eligible_count": len(records),
        "inside_40_bins": sum(capacities.values()),
        "outside_40_bins": payload["outside_40_bins"],
        "targets": {str(k): summaries[k]["selected_count"] for k in summaries},
        "nested_checks": nested_checks,
    }, indent=2))
    print("wrote", output_dir)


if __name__ == "__main__":
    main()
