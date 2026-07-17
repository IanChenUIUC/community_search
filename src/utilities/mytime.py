#!/usr/bin/env python3
"""
mytime.py - track peak RSS, anonymous memory
Usage: memtrack.py <command> [args...]
Output: key=value lines
  peak_rss_kb   : kernel high-water mark via getrusage (same as /usr/bin/time -v)
  peak_anon_kb  : polling high-water mark of Anonymous from smaps_rollup
"""

from pathlib import Path
import subprocess
import sys
import time
import resource

import click

PAGE_SIZE_KB = resource.getpagesize() // 1024


def parse_smaps_rollup(pid):
    anon, pss = None, None
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Anonymous:"):
                    anon = int(line.split()[1])  # kB
                elif line.startswith("Pss:"):
                    pss = int(line.split()[1])  # kB
    except (FileNotFoundError, ProcessLookupError):
        pass
    return anon, pss


def parse_stat_faults(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        after_comm = raw[raw.rfind(")") + 2 :]
        fields = after_comm.split()
        return int(fields[7]), int(fields[9])  # minflt, majflt
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None, None


PATH_ONLY = click.Path(dir_okay=False, path_type=Path)


@click.command()
@click.option("-o", "--output", type=PATH_ONLY)
@click.option("-a", "--append", is_flag=True)
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=False)
def main(output, append, cmd):
    if not cmd:
        raise click.UsageError("missing command")

    if output is None:
        out = sys.stdout
    else:
        out = output.open("a" if append else "w")

    peak_anon = 0
    peak_pss = 0
    peak_minflt = 0
    peak_majflt = 0

    start = time.monotonic()
    proc = subprocess.Popen(cmd)

    def sample():
        nonlocal peak_anon, peak_pss, peak_minflt, peak_majflt
        anon, pss = parse_smaps_rollup(proc.pid)
        minflt, majflt = parse_stat_faults(proc.pid)
        if anon is not None:
            peak_anon = max(peak_anon, anon)
        if pss is not None:
            peak_pss = max(peak_pss, pss)
        if minflt is not None:
            peak_minflt = max(peak_minflt, minflt)
        if majflt is not None:
            peak_majflt = max(peak_majflt, majflt)

    while True:
        sample()
        ret = proc.poll()
        if ret is not None:
            sample()
            break
        time.sleep(0.05)

    elapsed = time.monotonic() - start

    # kernel-maintained HWM, same source as /usr/bin/time -v
    peak_rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    print(f"exit_code={ret}", file=out)
    print(f"wall_s={elapsed:.3f}", file=out)
    print(f"peak_rss_kb={peak_rss_kb}", file=out)
    print(f"peak_pss={peak_pss}", file=out)
    print(f"peak_anon_kb={peak_anon}", file=out)
    print(f"peak_minflt={peak_minflt}", file=out)
    print(f"peak_majflt={peak_majflt}", file=out)
    out.flush()

    sys.exit(ret)


if __name__ == "__main__":
    main()
