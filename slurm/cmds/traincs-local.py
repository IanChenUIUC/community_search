#!/usr/bin/env python3
import subprocess
import sys
import tempfile

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

INDPTR    = "${csr-format.indptr}"
INDICES   = "${csr-format.indices}"
QUERYBASE = "${traincs-genquery.querybase}"
DIR       = "${dir}"
TIMING    = "${timing}"
MYTIME    = "${mytime}".split()
ICEBUG    = "${icebug}"
PYTHON    = "${icebug}/.venv/bin/python"
VARIANT   = "${variant}"
REP       = int("${rep}")
SIZE      = "${size}"

tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(PYTHON)

subprocess.run(["vmtouch", "-t", INDPTR, INDICES], check=True)

batch = f"{QUERYBASE}-n{SIZE}-b20-r1/query0.csv"
with open(batch) as f:
    query_row = f.read().splitlines()[REP]

with tempfile.NamedTemporaryFile("w", suffix=".csv") as q:
    q.write(query_row + "\n")
    q.flush()

    cmd = [PYTHON, f"{ICEBUG}/query_local.py", INDPTR, INDICES, q.name,
           f"{DIR}/{VARIANT}-querytimes-n{SIZE}-rep{REP}.csv"]
    if VARIANT == "local-upper":
        cmd.append("-u")

    subprocess.run(
        [*MYTIME, "-o", f"{TIMING}-{VARIANT}-n{SIZE}-rep{REP}.txt", "--", *cmd],
        check=True,
    )
