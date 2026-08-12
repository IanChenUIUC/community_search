#!/usr/bin/env python3
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

INDPTR    = "${csr-format.indptr}"
INDICES   = "${csr-format.indices}"
QUERYBASE = "${traincs-genquery.querybase}"
CORES     = "${traincd-icebug-core-decomp.cores}"
SHELL     = "${shell}"
DIR       = "${dir}"
TIMING    = "${timing}"
MYTIME    = "${mytime}".split()
ICEBUG    = "${icebug}"
PYTHON    = "${icebug}/.venv/bin/python"
THREADS   = "${slurm.cpus}"

SIZES = [int(s) for s in "${traincs-genquery.sizes}".split()]


def timed(label, *argv):
    subprocess.run([*MYTIME, "-o", f"{TIMING}-{label}.txt", "--", *argv], check=True)


def query(size):
    return f"{QUERYBASE}-n{size}-b20-r1/query0.csv"


tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(PYTHON)

subprocess.run(["vmtouch", "-t", INDPTR, INDICES, CORES], check=True)

timed("shellstruct-offline", PYTHON, f"{ICEBUG}/build_shellstruct.py",
      INDPTR, INDICES, SHELL, CORES, "--threads", THREADS)

for size in SIZES:
    timed(f"shellstruct-online-n{size}", PYTHON, f"{ICEBUG}/query_shellstruct.py",
          INDPTR, INDICES, f"{SHELL}.components.feather", f"{SHELL}.tree.feather",
          query(size), f"{DIR}/shellstruct-querytimes-n{size}.csv")

for size in SIZES:
    timed(f"steiner-n{size}", PYTHON, f"{ICEBUG}/query_steiner.py",
          INDPTR, INDICES, CORES, query(size),
          f"{DIR}/steiner-querytimes-n{size}.csv")
