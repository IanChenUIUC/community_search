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
THREADS  = "${threads}"
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


def coreness(output):
    return (PYTHON, CORENESS, INDPTR, INDICES, output, "--threads", THREADS)


def query():
    return (PYTHON, QUERY, INDPTR, INDICES, CORES, "--threads", THREADS)


def evict_inputs():
    subprocess.run(["vmtouch", "-e", INDPTR, INDICES], check=True)


def evict_cores():
    subprocess.run(["sync"], check=True)
    subprocess.run(["vmtouch", "-e", CORES], check=False)


def touch_inputs():
    subprocess.run(["vmtouch", "-t", INDPTR, INDICES], check=True)


def touch_cores():
    subprocess.run(["vmtouch", "-t", CORES], check=True)


tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(PYTHON)

## cold
evict_inputs()
run("cold-serial-coreness", *coreness(CORES))

evict_inputs()
evict_cores()
run("cold-serial-query", *query())

evict_inputs()
run_together(
    ("cold-simult1-coreness", *coreness(f"{DIR}/simult1-cores.npy")),
    ("cold-simult2-coreness", *coreness(f"{DIR}/simult2-cores.npy")),
)

evict_inputs()
evict_cores()
run_together(
    ("cold-simult1-query", *query()),
    ("cold-simult2-query", *query()),
)

## warm
touch_inputs()
run("warm-serial-coreness", *coreness(CORES))

touch_inputs()
touch_cores()
run("warm-serial-query", *query())

touch_inputs()
run_together(
    ("warm-simult1-coreness", *coreness(f"{DIR}/simult1-cores.npy")),
    ("warm-simult2-coreness", *coreness(f"{DIR}/simult2-cores.npy")),
)

touch_inputs()
touch_cores()
run_together(
    ("warm-simult1-query", *query()),
    ("warm-simult2-query", *query()),
)
