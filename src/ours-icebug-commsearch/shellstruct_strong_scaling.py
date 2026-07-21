import sys
import time
import pathlib
import resource

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

import networkit as nk
from networkit.centrality import CoreDecomposition
from networkit.scd import ShellStruct


def _read_column(path: str, column: str) -> pa.Array:
    if path.endswith(".feather"):
        return pf.read_table(path, memory_map=True)[column].chunk(0)
    elif path.endswith(".parquet"):
        return pq.read_table(path)[column].combine_chunks()

    print(f"Files must be .parquet or .feather, got {path}", file=sys.stderr)
    sys.exit(1)


def run(graph, num_threads, data):
    nk.engineering.setNumberOfThreads(num_threads)

    start_cd = time.perf_counter()
    cores = np.array(CoreDecomposition(graph).run().scores(), dtype=np.uint64)
    end_cd = time.perf_counter()

    start_shell = time.perf_counter()
    shell = nk.scd.ShellStruct(graph)
    shell.build(cores)
    end_shell = time.perf_counter()

    ru = resource.getrusage(resource.RUSAGE_SELF)
    mem = ru.ru_maxrss
    # [threads, coredecomp time, shellstruct time, memory]
    data.append([num_threads, end_cd - start_cd, end_shell - start_shell, mem])


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
def main(indptr_path, indices_path, output_file):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    # columns = [threads, coredecomp time, shellstruct time, mem]
    data = []
    for num_threads in [1, 2, 4, 8, 16, 32, 64]:
        print(f"running with {num_threads=}")
        run(graph, num_threads, data)

    df = pd.DataFrame(data, columns=["threads", "coredecomp", "shellstruct", "mem"])
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
