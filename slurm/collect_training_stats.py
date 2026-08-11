#!/usr/bin/env python3
"""Collect local/local-upper STATS lines into a long-form CSV.

Each stdout file holds up to two STATS lines: the first is the n=1 query,
the second the n=10 query (which may be absent).

Usage: collect_training_stats.py OUT.csv DATASET [DATASET ...]
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path("/u/ianchen3/community_search/output")
STATS = re.compile(r"\(et_size / gt_size\)=(\S+) \(gt_vol / et_vol\)=(\S+)")


def stats_lines(path):
    return STATS.findall(path.read_text())


def main(out_path, datasets):
    rows = []
    for dataset in datasets:
        d = ROOT / dataset / "training-local-steiner-shell"
        for method in ("local", "local-upper"):
            pattern = re.compile(rf"^{method}-stdout-rep(\d+)\.txt$")
            for f in d.iterdir():
                m = pattern.match(f.name)
                if not m:
                    continue
                rep = int(m.group(1))
                for n, (size_ratio, vol_ratio) in zip([1, 10], stats_lines(f)):
                    rows.append((dataset, rep, n, method, float(size_ratio), 1 / float(vol_ratio)))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "rep", "query_size", "method", "et_size_gt_size", "et_vol_gt_vol"])
        w.writerows(sorted(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
