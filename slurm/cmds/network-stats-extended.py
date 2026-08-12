import csv
import os
import re
import subprocess
import sys

sys.path.insert(0, "${root}/slurm/cmds")
from _driver import tee_streams, use_pyarrow_libs

INDPTR_PQ  = "${csr-format.indptr_pq}"
COMPONENTS = "${stat-icebug-shellstruct.components}"
TREE       = "${stat-icebug-shellstruct.tree}"
GBBS_BIN   = "${gbbs-format.bin}"
KCORE      = "${gbbs}/bazel-bin/benchmarks/KCore/JulienneDBS17/KCore_main"
STATS      = "${stats}"
CORESIZES  = "${coresizes}"
TIMING     = "${timing}"
MYTIME     = "${mytime}".split()
ICEBUG     = "${icebug}"
IB_PY      = "${icebug}/.venv/bin/python"
THREADS    = "${slurm.cpus}"

tee_streams("${stdout}", "${stderr}")
use_pyarrow_libs(IB_PY)
os.environ["PARLAY_NUM_THREADS"] = THREADS

subprocess.run(["vmtouch", "-t", GBBS_BIN], check=True)
kcore = subprocess.run([KCORE, "-s", "-b", "-rounds", "1", GBBS_BIN],
                       capture_output=True, text=True, check=True)
print(kcore.stdout, end="")
print(kcore.stderr, end="", file=sys.stderr)

reported = re.search(r"rho = (\d+) k_\{max\} = (\d+)", kcore.stdout)
if not reported:
    sys.exit("no peeling complexity line in the KCore output")
rho, kmax = reported.group(1), reported.group(2)

subprocess.run(["vmtouch", "-t", INDPTR_PQ, COMPONENTS, TREE], check=True)
subprocess.run([*MYTIME, "-o", TIMING, "--",
                IB_PY, f"{ICEBUG}/network-stats-extended.py",
                INDPTR_PQ, COMPONENTS, TREE, STATS, CORESIZES], check=True)

with open(STATS, "a", newline="") as f:
    csv.writer(f).writerows([["degeneracy", kmax], ["peeling_complexity", rho]])
