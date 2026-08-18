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

tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs("${icebug}/.venv/bin/python")
subprocess.run(["vmtouch", "-t", INDPTR, INDICES], check=True)

# One invocation for every cell: the per-cell graph build dominates the sampling.
cells = []
for size in SIZES:
    for batch in BATCHES:
        cells += ["--cell", f"{size}:{batch}:{QUERYBASE}-n{size}-b{batch}-r20"]

subprocess.run([*MYTIME, "-o", f"{TIMING}.txt", "--", *GENQUERY,
                INDPTR, INDICES, "-r", "20", *cells],
               check=True)
