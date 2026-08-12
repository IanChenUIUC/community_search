import sys

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
    rng = np.random.default_rng(1234)

    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    cc = nk.components.ParallelConnectedComponents(graph).run()
    deg = nk.centrality.DegreeCentrality(graph).run().scores()
    valid = np.flatnonzero(deg >= np.quantile(deg, 0.99))  # top 1%

    def single_query(size):
        for _ in range(1_000):
            query = rng.choice(valid, size, replace=False)
            labels = [cc.componentOfNode(q) for q in query]
            if all(label == labels[0] for label in labels):
                return query
        raise RuntimeError("single query failed after 1000 trials")

    queries = []
    for _ in range(20):
        queries.append(single_query(1))
    for _ in range(20):
        queries.append(single_query(5))

    with open(outfile, "w") as f:
        f.writelines(",".join(map(str, q)) + "\n" for q in queries)


if __name__ == "__main__":
    main()
