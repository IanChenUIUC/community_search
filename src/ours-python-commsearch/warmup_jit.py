import click
import numpy as np

from commsearch import ShellStruct, SteinerKCore


@click.command()
def main():
    """Pre-compile the njit kernels on a tiny synthetic graph (no real graph)"""
    for dt in (np.uint32, np.uint64):
        SteinerKCore.warmup(np.dtype(dt))
    ShellStruct.warmup()


if __name__ == "__main__":
    main()
