import json
import math
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = ROOT / "slurm" / "pipeline.toml"
LOG = ROOT / "slurm" / ".pipeline" / "run.jsonl"
OUTPUT = ROOT / "output"
ANALYSIS = ROOT / "analysis"

MYTIME_KEYS = ("exit_code", "wall_s", "user_s", "sys_s", "cpu_pct", "peak_rss_kb",
               "peak_rss_anon_kb", "peak_rss_file_kb", "peak_rss_tree_kb",
               "peak_minflt", "peak_majflt")

RANGE = re.compile(r"^(-?\d+)\.\.(-?\d+)$")
OOM = {"OUT_OF_MEMORY"}
TIMEOUT = {"TIMEOUT", "DEADLINE"}


def expand_ranges(v):
    if isinstance(v, dict):
        return {k: expand_ranges(x) for k, x in v.items()}
    if isinstance(v, list):
        return [expand_ranges(x) for x in v]
    if isinstance(v, str) and (m := RANGE.fullmatch(v.strip())):
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return v


def load_spec(path):
    with open(path, "rb") as f:
        return expand_ranges(tomllib.load(f))


def task_states(path):
    """Node name -> {state, elapsed, max_rss, job_id}.

    `nodes` appears only on submit/force records and `tasks` only on reconcile ones, so
    the two are folded separately per unit and joined by index position. Unit names are
    chunked (`testing-steiner:bitcoin`), so several units contribute disjoint nodes.
    """
    nodes, latest = {}, {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("nodes"):
                nodes[r["unit"]] = r["nodes"]
            latest[r["unit"]] = r

    states = {}
    for unit, names in nodes.items():
        record = latest.get(unit, {})
        tasks = record.get("tasks") or {}
        for i, name in enumerate(names):
            task = tasks.get(str(i), {})
            states[name] = {"state": task.get("state", record.get("state")),
                            "elapsed": task.get("elapsed", record.get("elapsed")),
                            "max_rss": task.get("max_rss", record.get("max_rss")),
                            "job_id": record.get("job_id")}
    return states


def read_mytime(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None

    stats = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key in MYTIME_KEYS:
            stats[key] = float(value)
    return stats


def row_status(cell, mytime, task):
    """One of ok / oom / timeout / failed / absent.

    The two sources answer different questions, so both are needed: sacct knows what
    *kind* of ending a task had, which an exit code cannot express, while the mytime
    exit code is per-step and so is the only thing that can speak for one stage of a
    job that runs several. The exit code therefore decides ok-ness and sacct supplies
    the kind. A missing mytime file under a COMPLETED task is `absent` on purpose: the
    job claimed success and left no artifact.
    """
    state = (task or {}).get("state")
    kind = ("ok" if state == "COMPLETED" else
            "oom" if state in OOM else
            "timeout" if state in TIMEOUT else
            "absent" if state is None else "failed")

    if mytime is None:
        return "absent" if kind == "ok" else kind

    ok = mytime.get("exit_code") == 0
    if ok != (kind == "ok"):
        print(f"warning: {cell}: mytime exit_code={mytime.get('exit_code'):.0f} "
              f"disagrees with task state {state} (job {(task or {}).get('job_id')})",
              file=sys.stderr)
    if ok:
        return "ok"
    return "failed" if kind == "ok" else kind


def report(recipe, **axes):
    shape = " x ".join(f"{n} {name}" for name, n in axes.items())
    cells = math.prod(axes.values())
    print(f"reading {recipe}: {shape} = {cells} cells")
    return cells
