import sys
import time

import click
import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

import networkit as nk
from networkit.scd import SteinerKCore


def _read_column(path: str, column: str) -> pa.Array:
    if path.endswith(".feather"):
        return pf.read_table(path)[column].chunk(0)
    elif path.endswith(".parquet"):
        return pq.read_table(path)[column].combine_chunks()

    print(f"Files must be .parquet or .feather, got {path}", file=sys.stderr)
    sys.exit(1)


def gen_queries(cores):
    vals, idx = np.unique(cores, return_index=True)
    return idx[-5:]


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("coreness_path", type=click.Path(exists=True, dir_okay=False))
def main(indptr_path, indices_path, coreness_path):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)
    cores = np.load(coreness_path, mmap_mode="r")

    steiner = SteinerKCore(graph, cores)
    queries = gen_queries(cores)

    nk.engineering.setNumberOfThreads(1)

    start = time.perf_counter()
    steiner.run(queries)
    end = time.perf_counter()
    print(end - start)


if __name__ == "__main__":
    main()
