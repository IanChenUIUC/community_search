#!/usr/bin/env python3
"""
mytime.py - track peak RSS, anonymous memory
Usage: memtrack.py <command> [args...]
Output: key=value lines
  peak_rss_kb   : kernel high-water mark via getrusage (same as /usr/bin/time -v)
  peak_anon_kb  : polling high-water mark of Anonymous from smaps_rollup
"""

import subprocess
import sys
import time
import resource

PAGE_SIZE_KB = resource.getpagesize() // 1024


def parse_smaps_rollup(pid):
    anon = None
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Anonymous:"):
                    anon = int(line.split()[1])  # kB
                    break
    except (FileNotFoundError, ProcessLookupError):
        pass
    return anon


def parse_stat_faults(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
        after_comm = raw[raw.rfind(")") + 2 :]
        fields = after_comm.split()
        return int(fields[7]), int(fields[9])  # minflt, majflt
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None, None


def main():
    cmd = sys.argv[1:]
    if not cmd:
        sys.exit("usage: memtrack.py <command> [args...]")

    peak_anon = 0
    peak_minflt = 0
    peak_majflt = 0

    start = time.monotonic()
    proc = subprocess.Popen(cmd)

    def sample():
        nonlocal peak_anon, peak_minflt, peak_majflt
        anon = parse_smaps_rollup(proc.pid)
        minflt, majflt = parse_stat_faults(proc.pid)
        if anon is not None:
            peak_anon = max(peak_anon, anon)
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

    print(f"exit_code={ret}")
    print(f"wall_s={elapsed:.3f}")
    print(f"peak_rss_kb={peak_rss_kb}")
    print(f"peak_anon_kb={peak_anon}")

    sys.exit(ret)


if __name__ == "__main__":
    main()
