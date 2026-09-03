import json
import pathlib
import re
import sys
import tomllib

import pandas as pd

MYTIME_KEYS = ("exit_code", "wall_s", "user_s", "sys_s", "cpu_pct", "peak_rss_kb",
               "peak_rss_anon_kb", "peak_rss_file_kb", "peak_rss_tree_kb",
               "peak_minflt", "peak_majflt")

RANGE = re.compile(r"^(-?\d+)\.\.(-?\d+)$")
GULLO_QUERY = re.compile(r"query (\d+): size=(\d+), time=(\d+)ms")
OOM = {"OUT_OF_MEMORY"}
TIMEOUT = {"TIMEOUT", "DEADLINE"}

NO_QUERY = object()


def expand_ranges(v):
    """Rewrite "a..b" strings into inclusive int lists, anywhere in the spec."""
    if isinstance(v, dict):
        return {k: expand_ranges(x) for k, x in v.items()}
    if isinstance(v, list):
        return [expand_ranges(x) for x in v]
    if isinstance(v, str) and (m := RANGE.fullmatch(v.strip())):
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return v


def load_spec(path):
    """The pipeline.toml as a dict, with every range already expanded."""
    with open(path, "rb") as f:
        return expand_ranges(tomllib.load(f))


def task_states(path):
    """Map each node name in run.jsonl to its latest {state, elapsed, max_rss, job_id}."""
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
    """The mytime key=value file as floats, or None if it was never written."""
    path = pathlib.Path(path)
    if not path.exists():
        return None

    stats = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key in MYTIME_KEYS:
            stats[key] = float(value)
    return stats


def read_mytimes(path):
    """The appended mytime records in order, one dict each; `exit_code` opens a new record."""
    path = pathlib.Path(path)
    if not path.exists():
        return []

    records = []
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key == "exit_code":
            records.append({})
        if key in MYTIME_KEYS and records:
            records[-1][key] = float(value)
    return records


def row_status(cell, mytime, task):
    """One of ok / oom / timeout / failed / absent for a cell; warns on a source disagreement."""
    state = (task or {}).get("state")
    kind = ("ok" if state == "COMPLETED" else
            "oom" if state in OOM else
            "timeout" if state in TIMEOUT else
            "absent" if state is None else "failed")

    if mytime is None:
        return "absent" if kind == "ok" else kind

    ok = mytime.get("exit_code") == 0
    if ok != (kind == "ok"):
        print(f"warning: {cell}: mytime exit_code={mytime.get('exit_code')} "
              f"disagrees with task state {state} (job {(task or {}).get('job_id')})",
              file=sys.stderr)
    if ok:
        return "ok"
    return "failed" if kind == "ok" else kind


def emit(rows, key, mytime, status, query_s=NO_QUERY):
    """Append one long-format row per mytime stat, plus query_s for stages that have one."""
    for stat in MYTIME_KEYS:
        rows.append([*key, stat, (mytime or {}).get(stat), status])
    if query_s is not NO_QUERY:
        rows.append([*key, "query_s", query_s, status])


def emit_shared(rows, key, shared, stage_at):
    """Repeat a cell's prerequisite stages onto it, so one filter on the cell keys gives
    its whole cost and each stage keeps its own status. `stage_at` is the stage column's
    position in `key`, which differs per collector."""
    for stage, mytime, status in shared:
        emit(rows, [*key[:stage_at], stage, *key[stage_at + 1:]], mytime, status)


def read_stage(out, states, network, recipe, filename, node):
    """A prerequisite stage's mytime and status, ready to sit in a `shared` list."""
    mytime = read_mytime(out / network / recipe / filename)
    return mytime, row_status(node, mytime, states.get(node))


def querytimes(path, package):
    """Total query seconds for a cell: the "all" row for pycs, the row sum for icebug."""
    rows = querytimes_rows(path)
    if rows is None:
        return None
    return rows["all"] if package == "pycs" else sum(rows.values())


def querytimes_rows(path):
    """Per-query wall seconds keyed by int index, plus "all" for pycs files."""
    path = pathlib.Path(path)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "index" not in df.columns:
        return dict(enumerate(df["wall_s"]))
    index = [i if i == "all" else int(i) for i in df["index"]]
    return dict(zip(index, df["wall_s"]))


def csk_timing(path):
    """Total query seconds from csk's `queryid,ms` timing.log."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    return sum(int(line.split(",")[1]) for line in path.read_text().splitlines()
               if line.strip()) / 1000


def gullo_timing(path):
    """Total query seconds from gullo's per-query `time=Nms` log lines."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    times = [int(m.group(3)) for m in GULLO_QUERY.finditer(path.read_text())]
    return sum(times) / 1000 if times else None

