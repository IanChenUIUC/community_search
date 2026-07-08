# For fun: generating data for figure 3 from
# Malvestio, I., Cardillo, A. & Masuda, N. Interplay between k-core and community
# structure in complex networks. Sci Rep 10, 14702 (2020). https://doi.org/10.1038/s41598-020-71426-8

from pathlib import Path

import click
import numpy as np
from shell import load_shell


@click.command()
@click.option("--shell", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def main(shell, output):
    shell = load_shell(shell)

    comm_sizes = np.diff(shell.nodes.indptr)
    ncomm = np.bincount(shell.coreness)
    nvert = np.bincount(shell.coreness, weights=comm_sizes)
    values = np.arange(len(ncomm))

    df = np.column_stack((values, ncomm, nvert))

    out = Path(output)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savetxt(out, df, fmt="%d", delimiter=",", header="core,ncomm,nvert", comments="")


if __name__ == "__main__":
    main()
