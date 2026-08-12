#!/usr/bin/env python3
import itertools
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

SHELL     = "${testing-par-shellstruct-offline.shell}"
QUERYBASE = "${genquery.querybase}"
DIR       = "${dir}"
TIMING    = "${timing}"
MYTIME    = "${mytime}".split()
PYTHON    = "${pycs}/.venv/bin/python"
SHELLQ    = "${pycs}/query_shellstruct.py"

REPS    = [int(r) for r in "${genquery.reps}".split()]
SIZES   = [int(s) for s in "${genquery.sizes}".split()]
BATCHES = [int(b) for b in "${genquery.batches}".split()]

tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(PYTHON)
subprocess.run(["vmtouch", "-t", f"{SHELL}.components.feather", f"{SHELL}.tree.feather"],
               check=True)

for size, batch, rep in itertools.product(SIZES, BATCHES, REPS):
    cell = f"n{size}-b{batch}-rep{rep}"
    subprocess.run(
        [*MYTIME, "-o", f"{TIMING}-{cell}.txt", "--", PYTHON, SHELLQ,
         SHELL, f"{QUERYBASE}-n{size}-b{batch}-r20/query{rep}.csv",
         f"{DIR}/querytimes-{cell}.csv", "-b", str(batch)],
        check=True,
    )
