import sys
import pathlib

import click
import networkit as nk
import numpy as np
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
@click.argument("outfile", type=click.Path(dir_okay=False))
def main(indptr_path, indices_path, outfile):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    deg = nk.centrality.DegreeCentrality(graph).run().scores()
    valid = np.flatnonzero(deg >= np.quantile(deg, 0.99))  # top 1%

    rng = np.random.default_rng(1234)
    queries = []
    for _ in range(50):
        queries.append(rng.choice(valid, 1))
    for _ in range(50):
        queries.append(rng.choice(valid, 10, replace=False))

    with open(outfile, "w") as f:
        f.writelines(",".join(map(str, q)) + "\n" for q in queries)


if __name__ == "__main__":
    main()
