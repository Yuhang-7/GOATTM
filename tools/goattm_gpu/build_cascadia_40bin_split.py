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


def save_manifest(path: Path, sample_ids: list[str], sample_paths: list[str], root: str) -> None:
    np.savez(
        path,
        sample_ids=np.asarray(sample_ids, dtype=object),
        sample_paths=np.asarray(sample_paths, dtype=object),
        root=np.array(root),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Cascadia 40-bin test125 split.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--test-per-bin", type=int, default=125)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = np.load(manifest_path, allow_pickle=True)
    root = str(manifest["root"].item()) if "root" in manifest.files else str(manifest_path.parent)
    if root == "":
        root = str(manifest_path.parent)
    raw_ids = [str(x) for x in manifest["sample_ids"].tolist()]
    raw_paths = [str(x) for x in manifest["sample_paths"].tolist()]
    records = []
    for sample_id, sample_path in zip(raw_ids, raw_paths):
        rid = int(sample_id)
        records.append({"sample_id": sample_id, "rupture_id": rid, "sample_path": sample_path})
    records.sort(key=lambda item: item["rupture_id"])

    by_id = {item["rupture_id"]: item for item in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate rupture ids found in manifest")

    test_ids: set[int] = set()
    bin_rows = []
    for bin_index in range(int(args.bins)):
        lo = bin_index * int(args.bin_size)
        hi = lo + int(args.bin_size) - 1
        bin_ids = sorted(rid for rid in by_id if lo <= rid <= hi)
        if len(bin_ids) < int(args.test_per_bin):
            raise ValueError(
                f"bin {bin_index} has only {len(bin_ids)} considered ruptures; "
                f"need {args.test_per_bin}"
            )
        selected = bin_ids[: int(args.test_per_bin)]
        test_ids.update(selected)
        bin_rows.append(
            {
                "bin": bin_index,
                "range_start": lo,
                "range_end": hi,
                "considered_count": len(bin_ids),
                "test_count": len(selected),
                "train_count": len(bin_ids) - len(selected),
                "test_first": selected[0],
                "test_last": selected[-1],
                "considered_first": bin_ids[0],
                "considered_last": bin_ids[-1],
            }
        )

    considered_ids = sorted(by_id)
    testing_ids = sorted(test_ids)
    training_ids = sorted(rid for rid in considered_ids if rid not in test_ids)
    outside_ids = sorted(
        rid
        for rid in considered_ids
        if rid < 0 or rid >= int(args.bins) * int(args.bin_size)
    )

    train_records = [by_id[rid] for rid in training_ids]
    test_records = [by_id[rid] for rid in testing_ids]
    save_manifest(
        output_dir / "train_manifest.npz",
        [item["sample_id"] for item in train_records],
        [item["sample_path"] for item in train_records],
        root,
    )
    save_manifest(
        output_dir / "test_manifest.npz",
        [item["sample_id"] for item in test_records],
        [item["sample_path"] for item in test_records],
        root,
    )

    write_id_list(output_dir / "considered_ruptures.txt", considered_ids)
    write_id_list(output_dir / "testing_ruptures.txt", testing_ids)
    write_id_list(output_dir / "training_ruptures.txt", training_ids)

    with (output_dir / "split_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rupture_id", "sample_id", "bin", "split", "sample_path"],
        )
        writer.writeheader()
        for rid in considered_ids:
            if 0 <= rid < int(args.bins) * int(args.bin_size):
                bin_label = rid // int(args.bin_size)
            else:
                bin_label = "outside_40_bins"
            item = by_id[rid]
            writer.writerow(
                {
                    "rupture_id": rupture_id_text(rid),
                    "sample_id": item["sample_id"],
                    "bin": bin_label,
                    "split": "test" if rid in test_ids else "train",
                    "sample_path": item["sample_path"],
                }
            )

    with (output_dir / "bin_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)

    summary = {
        "source_manifest": str(manifest_path),
        "root": root,
        "bins": int(args.bins),
        "bin_size": int(args.bin_size),
        "test_per_bin": int(args.test_per_bin),
        "considered_count": len(considered_ids),
        "testing_count": len(testing_ids),
        "training_count": len(training_ids),
        "outside_40_bins": [rupture_id_text(rid) for rid in outside_ids],
        "bin_summary": bin_rows,
        "files": {
            "train_manifest": "train_manifest.npz",
            "test_manifest": "test_manifest.npz",
            "split_assignments": "split_assignments.csv",
            "bin_summary": "bin_summary.csv",
            "considered_ruptures": "considered_ruptures.txt",
            "testing_ruptures": "testing_ruptures.txt",
            "training_ruptures": "training_ruptures.txt",
        },
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        r"\documentclass[11pt]{ctexart}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{amsmath}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\title{Cascadia 40-bin Train/Test Split}",
        r"\author{}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{规则}",
        "我们只考虑 source manifest 中已经有 QoI 的 rupture。按 rupture id 使用 40 个 bin：",
        r"\[",
        r"\mathrm{bin}\ b = \{1000b,\ldots,1000b+999\},\qquad b=0,\ldots,39.",
        r"\]",
        "在每个 bin 内，把已经有 QoI 的 rupture 按 id 升序排列，取前 125 个作为 testing data。",
        "这 5000 个 testing samples 在 learning 和 normalization 中都不能使用。剩下所有 considered ruptures 都可作为 training data。",
        r"\section{计数}",
        rf"source manifest: \path{{{manifest_path}}}.",
        rf"considered ruptures: {len(considered_ids)}.",
        rf"testing ruptures: {len(testing_ids)}.",
        rf"training-eligible ruptures: {len(training_ids)}.",
        rf"outside the 40 bins but considered: \texttt{{{', '.join(rupture_id_text(rid) for rid in outside_ids) if outside_ids else 'none'}}}.",
        r"\section{输出文件}",
        r"\begin{itemize}",
        r"\item \texttt{train\_manifest.npz}: learning 和 normalization 可以使用的 training manifest。",
        r"\item \texttt{test\_manifest.npz}: 固定 testing manifest，learning 不能使用。",
        r"\item \texttt{split\_assignments.csv}: 每个 considered rupture 的 split 标记。",
        r"\item \texttt{considered\_ruptures.txt}, \texttt{testing\_ruptures.txt}, \texttt{training\_ruptures.txt}: 三个 id list。",
        r"\item \texttt{bin\_summary.csv}: 每个 bin 的 considered/test/train 计数。",
        r"\end{itemize}",
        r"\section{Bin Summary}",
        r"\begin{longtable}{rrrrrrrr}",
        r"\toprule",
        r"bin & start & end & considered & test & train & test first & test last \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in bin_rows:
        lines.append(
            f"{row['bin']} & {row['range_start']} & {row['range_end']} & "
            f"{row['considered_count']} & {row['test_count']} & {row['train_count']} & "
            f"{row['test_first']} & {row['test_last']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{document}"])
    (output_dir / "split_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("considered_count", "testing_count", "training_count", "outside_40_bins")}, indent=2))
    print("wrote", output_dir)


if __name__ == "__main__":
    main()
