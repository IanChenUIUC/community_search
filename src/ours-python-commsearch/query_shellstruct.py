import sys
import pathlib
import time

import click
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

from commsearch import Graph, ShellStruct, SteinerKCore


@click.command()
@click.argument("shell_base_path", type=click.Path())
@click.argument("queries_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("-b", "--max_batch_size", type=int, default=1)
def main(shell_base_path, queries_path, output, max_batch_size):
    components_path = pathlib.Path(shell_base_path).with_suffix(".components.feather")
    tree_path = pathlib.Path(shell_base_path).with_suffix(".tree.feather")
    shell = ShellStruct.load(components_path, tree_path)
    shell.warmup()

    with open(queries_path) as f:
        queries = [np.fromstring(line, sep=",") for line in f.readlines()]

    start = time.perf_counter()
    shell.run_parallel(queries, num_threads=1, max_batch_size=max_batch_size)
    end = time.perf_counter()

    timing = shell.timing
    index = [str(x) for x in range(len(timing))]
    timing.append(end - start)
    index.append("all")

    df = pl.DataFrame({"index": index, "wall_s": timing})
    df.write_csv(output)


if __name__ == "__main__":
    main()
