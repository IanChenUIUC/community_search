#!/usr/bin/env python3
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

INDPTR   = "${csr-format.indptr}"
INDICES  = "${csr-format.indices}"
CORES    = "${cores}"
DIR      = "${dir}"
TIMING   = "${timing}"
MYTIME   = "${mytime}".split()
COLDWARM = "${coldwarm}"
PYTHON   = "${coldwarm}/.venv/bin/python"

CORENESS = f"{COLDWARM}/coreness.py"
QUERY    = f"{COLDWARM}/query.py"


def timed(label, *argv):
    return [*MYTIME, "-o", f"{TIMING}-{label}.txt", "--", *argv]


def run(label, *argv):
    subprocess.run(timed(label, *argv), check=True)


# The simult phases measure contention, so the pair must overlap.
def run_together(*jobs):
    procs = [subprocess.Popen(timed(label, *argv)) for label, *argv in jobs]
    for p in procs:
        if p.wait() != 0:
            raise SystemExit(f"cold-warm: {p.args} failed")


def evict_inputs():
    subprocess.run(["vmtouch", "-e", INDPTR, INDICES], check=True)


def evict_cores():
    subprocess.run(["sync"], check=True)
    subprocess.run(["vmtouch", "-e", CORES], check=False)


tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(PYTHON)

## cold
evict_inputs()
run("coldcores", PYTHON, CORENESS, INDPTR, INDICES, CORES)

evict_inputs()
evict_cores()
run("coldquery", PYTHON, QUERY, INDPTR, INDICES, CORES)

## simultaneous
evict_inputs()
run_together(
    ("simult1coreness", PYTHON, CORENESS, INDPTR, INDICES, f"{DIR}/simult1-cores.npy"),
    ("simult2coreness", PYTHON, CORENESS, INDPTR, INDICES, f"{DIR}/simult2-cores.npy"),
)

evict_inputs()
evict_cores()
run_together(
    ("simult1query", PYTHON, QUERY, INDPTR, INDICES, CORES),
    ("simult2query", PYTHON, QUERY, INDPTR, INDICES, CORES),
)

## warm
subprocess.run(["vmtouch", "-t", INDPTR, INDICES], check=True)
run("warmcoreness", PYTHON, CORENESS, INDPTR, INDICES, CORES)

subprocess.run(["vmtouch", "-t", INDPTR, INDICES, CORES], check=True)
run("warmquery", PYTHON, QUERY, INDPTR, INDICES, CORES)
