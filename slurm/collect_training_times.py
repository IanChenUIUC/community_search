#!/usr/bin/env python3
"""Collect training-local-steiner-shell timings into a long-form CSV.

Usage: collect_training_times.py OUT.csv DATASET [DATASET ...]
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path("/u/ianchen3/community_search/output")
SIZES = [1, 10]


def mytime_wall(path):
    for line in path.read_text().splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"no wall_s in {path}")


def querytimes(path):
    with open(path) as f:
        return [float(row["wall_s"]) for row in csv.DictReader(f)]


def main(out_path, datasets):
    rows = []
    for dataset in datasets:
        d = ROOT / dataset / "training-local-steiner-shell"
        coredecomp = mytime_wall(d / "timing-coredecomp.txt")
        offline = mytime_wall(d / "timing-shellstruct-offline.txt")

        for n in SIZES:
            for method, base in (("shell", coredecomp + offline), ("steiner", coredecomp)):
                stem = "shellstruct" if method == "shell" else "steiner"
                for rep, t in enumerate(querytimes(d / f"{stem}-querytimes-n{n}.csv")):
                    rows.append((dataset, rep, n, method, base + t))

            for method in ("local", "local-upper"):
                pattern = re.compile(rf"^{method}-querytimes-n{n}-rep(\d+)\.csv$")
                for f in d.iterdir():
                    m = pattern.match(f.name)
                    if m:
                        rows.append((dataset, int(m.group(1)), n, method, querytimes(f)[0]))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "rep", "query_size", "method", "wall_s"])
        w.writerows(sorted(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
