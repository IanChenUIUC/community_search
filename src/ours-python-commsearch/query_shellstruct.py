import sys
import pathlib
import time

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

from commsearch import Graph, ShellStruct, SteinerKCore


@click.command()
@click.argument("shell_base_path", type=click.Path())
@click.argument("queries_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def main(graph, format, coredecomp, shell_base_path, queries_path, output):
    components_path = pathlib.Path(shell_base_path).with_suffix(".components.feather")
    tree_path = pathlib.Path(shell_base_path).with_suffix(".tree.feather")
    shell = ShellStruct.load(components_path, tree_path)
    shell.warmup()

    with open(queries_path) as f:
        queries = [np.fromstring(line, sep=",") for line in f.readlines()]

    timing = []  # wall_s
    for query in queries:
        start = time.perf_counter()
        _ = shell.expand_one_community(query)
        end = time.perf_counter()
        timing.append(end - start)

    df = pd.DataFrame(timing, columns=["wall_s"])
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
