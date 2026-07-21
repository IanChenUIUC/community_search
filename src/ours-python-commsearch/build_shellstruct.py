import sys
import pathlib
import time

import click
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as pf

from commsearch import Graph, ShellStruct


@click.command()
@click.argument("graph", type=click.Path())
@click.argument("format", type=click.STRING)
@click.argument("coredecomp", type=click.Path(dir_okay=False, exists=True))
@click.argument("shell_base_path", type=click.Path(dir_okay=False))
def main(graph, format, coredecomp, shell_base_path):
    cores = pl.read_csv(coredecomp, has_header=False, new_columns=["node_id", "core"])
    cores.sort("node_id")
    scores = cores.get_column("core").to_numpy()

    g = Graph.load(graph, format)
    shell = ShellStruct.build(g, scores)

    components_path = pathlib.Path(shell_base_path).with_suffix(".components.feather")
    tree_path = pathlib.Path(shell_base_path).with_suffix(".tree.feather")
    shell.save(components_path, tree_path)


if __name__ == "__main__":
    main()
