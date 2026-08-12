#!/usr/bin/env python3
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

INDPTR    = "${csr-format.indptr}"
INDICES   = "${csr-format.indices}"
INDICES32 = "${csr-format.indices32}"
CORES     = "${stat-icebug-core-decomp.cores}"
QUERY     = "${query}"
DIR       = "${dir}"
TIMING    = "${timing}"
MYTIME    = "${mytime}".split()
ICEBUG    = "${icebug}"
PYCS      = "${pycs}"
GENQ      = "${genq}"
IB_PY     = "${icebug}/.venv/bin/python"
PY_PY     = "${pycs}/.venv/bin/python"


def timed(label, *argv):
    subprocess.run([*MYTIME, "-o", f"{TIMING}-{label}.txt", "--", *argv], check=True)


tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(IB_PY)

timed("genquery", f"{GENQ}/.venv/bin/python", f"{GENQ}/py_v_ib.py",
      INDPTR, INDICES, QUERY)

## icebug
subprocess.run(["vmtouch", "-t", INDPTR, INDICES], check=True)

timed("icebug_build_shell", IB_PY, f"{ICEBUG}/build_shellstruct.py",
      INDPTR, INDICES, f"{DIR}/icebug_shell", CORES)

timed("icebug_query_shell", IB_PY, f"{ICEBUG}/query_shellstruct.py",
      INDPTR, INDICES,
      f"{DIR}/icebug_shell.components.feather", f"{DIR}/icebug_shell.tree.feather",
      QUERY, f"{DIR}/icebug_shell_querytimes.csv")

timed("icebug_query_steiner", IB_PY, f"{ICEBUG}/query_steiner.py",
      INDPTR, INDICES, CORES, QUERY, f"{DIR}/icebug_steiner_querytimes.csv")

## python
subprocess.run([PY_PY, f"{PYCS}/warmup_jit.py"], check=True)
subprocess.run(["vmtouch", "-t", INDPTR, INDICES32], check=True)

timed("python_build_shell", PY_PY, f"{PYCS}/build_shellstruct.py",
      INDPTR, INDICES32, f"{DIR}/python_shell", CORES)

timed("python_query_shell", PY_PY, f"{PYCS}/query_shellstruct.py",
      f"{DIR}/python_shell", QUERY, f"{DIR}/python_shell_querytimes.csv", "-b", "1")

timed("python_query_steiner", PY_PY, f"{PYCS}/query_steiner.py",
      INDPTR, INDICES32, CORES, QUERY,
      f"{DIR}/python_steiner_querytimes.csv", "-t", "1", "-b", "1")
