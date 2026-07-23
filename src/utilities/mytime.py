#!/usr/bin/env python3
"""
mytime.py - track wall time, CPU time, and multi-process peak memory.
Usage: mytime.py [-o OUT] [-a] [-i MS] [--pss] -- <command> [args...]
Output: key=value lines
  wall_s            : wall-clock seconds
  user_s            : user CPU seconds, summed over the child tree (getrusage)
  sys_s             : system CPU seconds, summed over the child tree (getrusage)
  cpu_pct           : 100 * (user_s + sys_s) / wall_s  (~cpus*100)
  peak_rss_kb       : getrusage RUSAGE_CHILDREN ru_maxrss -- the single LARGEST
                      process in the tree (NOT a sum); kept for continuity.
  peak_rss_anon_kb  : polled high-water of sum(RssAnon) over the process tree
                      (private per process -> no double-count across workers)
  peak_rss_file_kb  : polled high-water of max(RssFile) over the tree
                      (the shared mmap'd graph -> counted once)
  peak_rss_tree_kb  : polled high-water of (sum RssAnon + max RssFile) -- the
                      multi-process footprint number
  peak_minflt/majflt: polled high-water of summed page faults over the tree
  peak_pss_kb       : (only with --pss) sum(Pss) over the tree, via smaps_rollup
  peak_anon_kb      : (only with --pss) sum(Anonymous) over the tree, via smaps_rollup

Note that --pss adds smaps_rollup sampling, which walks page tables and can be expensive.
"""

from pathlib import Path
import os
import subprocess
import sys
import time
import resource

import click


def _ppid_of(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        return int(raw[raw.rfind(")") + 2 :].split()[1])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None


def _descendants(root):
    """root + all its descendant PIDs, from a single /proc scan."""
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        ppid = _ppid_of(int(entry))
        if ppid is not None:
            children.setdefault(ppid, []).append(int(entry))
    seen, stack = [], [root]
    while stack:
        p = stack.pop()
        seen.append(p)
        stack.extend(children.get(p, ()))
    return seen


def parse_status_rss(pid):
    """RssAnon, RssFile in kB from /proc/<pid>/status (O(1) counters)."""
    anon = file = None
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    anon = int(line.split()[1])
                elif line.startswith("RssFile:"):
                    file = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return anon, file


def parse_smaps_rollup(pid):
    """Anonymous, Pss in kB from /proc/<pid>/smaps_rollup (page-table walk)."""
    anon, pss = None, None
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Anonymous:"):
                    anon = int(line.split()[1])
                elif line.startswith("Pss:"):
                    pss = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return anon, pss


def parse_stat_faults(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        fields = raw[raw.rfind(")") + 2 :].split()
        return int(fields[7]), int(fields[9])  # minflt, majflt
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None, None


PATH_ONLY = click.Path(dir_okay=False, path_type=Path)


@click.command()
@click.option("-o", "--output", type=PATH_ONLY)
@click.option("-a", "--append", is_flag=True)
@click.option(
    "-i",
    "--interval",
    type=float,
    default=50.0,
    help="sampling interval in milliseconds (default 50)",
)
@click.option(
    "--pss",
    is_flag=True,
    help="also sample PSS/Anonymous via smaps_rollup (expensive page walk)",
)
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=False)
def main(output, append, interval, pss, cmd):
    if not cmd:
        raise click.UsageError("missing command")

    out = sys.stdout if output is None else output.open("a" if append else "w")

    peak_anon_sum = 0  # sum(RssAnon) over tree
    peak_file_max = 0  # max(RssFile) over tree
    peak_tree = 0  # peak of (anon_sum + file_max)
    peak_pss = 0  # sum(Pss), only with --pss
    peak_smaps_anon = 0  # sum(Anonymous), only with --pss
    peak_minflt = 0
    peak_majflt = 0

    start = time.monotonic()
    proc = subprocess.Popen(cmd)

    def sample():
        nonlocal peak_anon_sum, peak_file_max, peak_tree
        nonlocal peak_pss, peak_smaps_anon, peak_minflt, peak_majflt
        anon_sum = file_max = minflt = majflt = pss_sum = smaps_anon = 0
        for pid in _descendants(proc.pid):
            a, fl = parse_status_rss(pid)
            if a is not None:
                anon_sum += a
            if fl is not None:
                file_max = max(file_max, fl)
            mn, mj = parse_stat_faults(pid)
            if mn is not None:
                minflt += mn
            if mj is not None:
                majflt += mj
            if pss:
                sa, sp = parse_smaps_rollup(pid)
                if sa is not None:
                    smaps_anon += sa
                if sp is not None:
                    pss_sum += sp
        peak_anon_sum = max(peak_anon_sum, anon_sum)
        peak_file_max = max(peak_file_max, file_max)
        peak_tree = max(peak_tree, anon_sum + file_max)
        peak_minflt = max(peak_minflt, minflt)
        peak_majflt = max(peak_majflt, majflt)
        if pss:
            peak_pss = max(peak_pss, pss_sum)
            peak_smaps_anon = max(peak_smaps_anon, smaps_anon)

    sleep_s = interval / 1000.0
    while True:
        sample()
        ret = proc.poll()
        if ret is not None:
            sample()
            break
        time.sleep(sleep_s)

    elapsed = time.monotonic() - start

    # CPU time from getrusage sums the whole waited-for child tree; ru_maxrss,
    # however, is the peak of the single largest process, not a tree total.
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss_kb = ru.ru_maxrss
    user_s, sys_s = ru.ru_utime, ru.ru_stime
    cpu_pct = 100 * (user_s + sys_s) / elapsed if elapsed > 0 else 0.0

    print(f"exit_code={ret}", file=out)
    print(f"wall_s={elapsed:.3f}", file=out)
    print(f"user_s={user_s:.3f}", file=out)
    print(f"sys_s={sys_s:.3f}", file=out)
    print(f"cpu_pct={cpu_pct:.0f}", file=out)
    print(f"peak_rss_kb={peak_rss_kb}", file=out)
    print(f"peak_rss_anon_kb={peak_anon_sum}", file=out)
    print(f"peak_rss_file_kb={peak_file_max}", file=out)
    print(f"peak_rss_tree_kb={peak_tree}", file=out)
    print(f"peak_minflt={peak_minflt}", file=out)
    print(f"peak_majflt={peak_majflt}", file=out)
    if pss:
        print(f"peak_pss_kb={peak_pss}", file=out)
        print(f"peak_anon_kb={peak_smaps_anon}", file=out)
    out.flush()

    sys.exit(ret)


if __name__ == "__main__":
    main()
