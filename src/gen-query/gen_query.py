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
@click.argument("out_directory", type=click.Path(file_okay=False))
@click.option("-n", "--query_size", type=int, default=1)
@click.option("-b", "--batch_size", type=int, default=1)
@click.option("-r", "--reps", type=int, default=1)
def main(indptr_path, indices_path, out_directory, query_size, batch_size, reps):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    deg = nk.centrality.DegreeCentrality(graph).run().scores()
    valid = np.flatnonzero(deg >= np.quantile(deg, 0.99))  # top 1%

    rng = np.random.default_rng(1234)
    query_batches = []
    for rep in range(reps):
        batch = []
        for _ in range(batch_size):
            query = rng.choice(valid, size=query_size, replace=False)
            batch.append(query)
        query_batches.append(batch)

    out_directory = pathlib.Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)
    for rep, batch in enumerate(query_batches):
        with open(out_directory / f"query{rep}.csv", "w") as f:
            f.writelines(",".join(map(str, query)) + "\n" for query in batch)


if __name__ == "__main__":
    main()
