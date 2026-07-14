import sys
import pathlib

import click
import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

import networkit as nk
from networkit.centrality import CoreDecomposition


def _read_column(path: str, column: str) -> pa.Array:
    if path.endswith(".feather"):
        return pf.read_table(path)[column].chunk(0)
    elif path.endswith(".parquet"):
        return pq.read_table(path)[column].combine_chunks()

    print(f"Files must be .parquet or .feather, got {path}", file=sys.stderr)
    sys.exit(1)


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(exists=False, dir_okay=False))
def main(indptr_path, indices_path, output):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)
    scores = np.array(CoreDecomposition(graph).run().scores(), dtype=np.uint64)
    pathlib.Path(output).parent.mkdir(exist_ok=True, parents=True)
    np.save(output, scores)


if __name__ == "__main__":
    main()
