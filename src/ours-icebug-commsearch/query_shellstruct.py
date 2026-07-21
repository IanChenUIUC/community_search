import sys
import pathlib
import time

import click
import networkit as nk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq


def _read_column(path: str, column: str) -> pa.Array:
    if path.endswith(".feather"):
        return pf.read_table(path, memory_map=True)[column].chunk(0)
    elif path.endswith(".parquet"):
        return pq.read_table(path)[column].combine_chunks()

    print(f"Files must be .parquet or .feather, got {path}", file=sys.stderr)
    sys.exit(1)


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("components_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tree_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("queries_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def main(indptr_path, indices_path, components_path, tree_path, queries_path, output):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    shell = nk.scd.ShellStruct(graph)
    shell.load(components_path, tree_path)

    with open(queries_path) as f:
        queries = [set(map(int, line.split(","))) for line in f.readlines()]

    timing = []  # wall_s
    for query in queries:
        start = time.perf_counter()
        _ = shell.expandOneCommunity(query)
        end = time.perf_counter()
        timing.append(end - start)

    df = pd.DataFrame(timing, columns=["wall_s"])
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
