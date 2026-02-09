import os
import subprocess
import sys
from pathlib import Path

from filelock import FileLock

ENV_ID = "SLURM_ARRAY_TASK_ID"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} tasks", file=sys.stderr)
        return

    tasks = Path(sys.argv[1])
    lock = tasks.with_name(f"{tasks.name}.lock")

    if ENV_ID not in os.environ:
        print("[ERR]: not in slurm job array.", file=sys.stderr)
        return

    with FileLock(lock), tasks.open() as f:
        lines = f.readlines()

    numb = int(os.environ[ENV_ID])
    if len(lines) <= numb:
        print("[ERR]: no task to complete.", file=sys.stderr)
        return

    command = lines[numb].strip()
    process = subprocess.run(command, shell=True, executable="/bin/bash")
    if process.returncode != 0:
        print(f"[ERR]: {command}", file=sys.stderr)
    else:
        print(f"[OK]: {command}", file=sys.stderr)


if __name__ == "__main__":
    main()
