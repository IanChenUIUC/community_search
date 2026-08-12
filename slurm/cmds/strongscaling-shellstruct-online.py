#!/usr/bin/env python3
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

SHELL     = "${strongscaling-shellstruct.shell}"
QUERYBASE = "${strongscaling-genquery.querybase}"
BATCH     = "${strongscaling-genquery.batch}"
NREPS     = int("${strongscaling-genquery.nreps}")
DIR       = "${dir}"
TIMING    = "${timing}"
MYTIME    = "${mytime}".split()
PYTHON    = "${pycs}/.venv/bin/python"
SHELLQ    = "${pycs}/query_shellstruct.py"

tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(PYTHON)
subprocess.run(["vmtouch", "-t", f"{SHELL}.components.feather", f"{SHELL}.tree.feather"],
               check=True)

for rep in range(NREPS):
    subprocess.run(
        [*MYTIME, "-o", f"{TIMING}-rep{rep}.txt", "--", PYTHON, SHELLQ,
         SHELL, f"{QUERYBASE}/query{rep}.csv",
         f"{DIR}/shellstruct-online-querytimes-rep{rep}.csv", "-b", BATCH],
        check=True,
    )
