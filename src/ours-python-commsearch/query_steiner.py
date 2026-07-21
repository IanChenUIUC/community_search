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
@click.argument("coredecomp", type=click.Path(dir_okay=False, exists=True))
@click.argument("queries_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def main(indptr_path, indices_path, coredecomp, queries_path, output):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = Graph.from_csr(indptr, indices)
    cores = pl.read_csv(coredecomp, has_header=False, new_columns=["node_id", "core"])
    cores.sort("node_id")
    scores = cores.get_column("core").to_numpy()
    steiner = SteinerKCore(graph, scores)

    with open(queries_path) as f:
        queries = [np.fromstring(line, sep=",") for line in f.readlines()]

    timing = []  # wall_s
    for query in queries:
        start = time.perf_counter()
        _ = steiner.expand_one_community(query)
        end = time.perf_counter()
        timing.append(end - start)

    df = pl.DataFrame({"wall_s": timing})
    df.write_csv(output, index=False)


if __name__ == "__main__":
    main()
