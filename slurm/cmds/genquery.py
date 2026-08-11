#!/usr/bin/env python3
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

INDPTR    = "${csr-format.indptr}"
INDICES   = "${csr-format.indices}"
QUERYBASE = "${querybase}"
TIMING    = "${timing}"
MYTIME    = "${mytime}".split()
GENQUERY  = ["${genq}/.venv/bin/python", "${genq}/gen_query.py"]

SIZES   = [int(s) for s in "${sizes}".split()]
BATCHES = [int(b) for b in "${batches}".split()]

# The labels name timing files the collectors read, so they cannot be derived.
LABELS = {
    (1, 1): "p1",   (1, 10): "p2a",  (1, 100): "p2b",
    (5, 1): "p3a",  (10, 1): "p3b",  (20, 1): "p3c",
    (5, 10): "p4a", (10, 10): "p4b", (20, 10): "p4c",
    (5, 100): "p4d", (10, 100): "p4e", (20, 100): "p4f",
}
CELLS = [(LABELS[(s, b)], s, b) for s in SIZES for b in BATCHES]

tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs("${icebug}/.venv/bin/python")

for label, size, batch in CELLS:
    subprocess.run(
        [*MYTIME, "-o", f"{TIMING}-{label}.txt", "--", *GENQUERY,
         INDPTR, INDICES, f"{QUERYBASE}-n{size}-b{batch}-r20",
         "-n", str(size), "-b", str(batch), "-r", "20"],
        check=True,
    )
