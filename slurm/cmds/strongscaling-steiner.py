#!/usr/bin/env python3
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

INDPTR    = "${csr-format.indptr}"
INDICES32 = "${csr-format.indices32}"
CORES     = "${strongscaling-core-decomp.cores}"
QUERYBASE = "${strongscaling-genquery.querybase}"
BATCH     = "${strongscaling-genquery.batch}"
NREPS     = int("${strongscaling-genquery.nreps}")
DIR       = "${dir}"
TIMING    = "${timing}"
THREADS   = "${threads}"
MYTIME    = "${mytime}".split()
PYTHON    = "${pycs}/.venv/bin/python"
STEINER   = "${pycs}/query_steiner.py"

tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(PYTHON)
subprocess.run(["vmtouch", "-t", INDPTR, INDICES32, CORES], check=True)

for rep in range(NREPS):
    subprocess.run(
        [*MYTIME, "-o", f"{TIMING}-rep{rep}.txt", "--", PYTHON, STEINER,
         INDPTR, INDICES32, CORES, f"{QUERYBASE}/query{rep}.csv",
         f"{DIR}/steiner-querytimes-t{THREADS}-rep{rep}.csv",
         "-t", THREADS, "-b", BATCH],
        check=True,
    )
