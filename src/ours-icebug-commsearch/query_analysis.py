import itertools as it
import pathlib
import sys

import tqdm
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


def min_deg(graph, comm):
    subg = nk.graphtools.subgraphFromNodes(graph, comm)
    return min(subg.degree(u) for u in subg.iterNodes())


def coreness_queries(graph, shell, scores, size, threshold=0.99, num=100, seed=1234):
    valid = np.flatnonzero(scores >= np.quantile(scores, threshold))
    rng = np.random.default_rng(seed)

    coreness = []
    for _ in range(num):
        query = rng.choice(valid, size)
        coreness.append(shell.score(query))

    return coreness


@click.command()
@click.argument("network")
def main(network):
    indptr = _read_column(f"../../input/{network}.indptr.feather", "indptr")
    indices = _read_column(f"../../input/{network}.indices.feather", "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    shell = nk.scd.ShellStruct(graph)
    shell.load(f"{network}.components.parquet", f"{network}.tree.parquet")

    centrality = pd.read_csv(f"{network}.centrality.csv")
    centrality.head()

    columns = ["network", "centrality", "size", "threshold", "cores"]
    data = []

    cent = ["coreness", "c_coef", "pagerank", "degree"]
    sizes = [1, 2, 3, 5, 10, 25, 100, 300]
    thresholds = [0, 0.9, 0.99, 0.999, 0.9999]
    for c, s, t in tqdm.tqdm(list(it.product(cent, sizes, thresholds))):
        cores = coreness_queries(graph, shell, centrality[c].to_numpy(), s, t)
        for value in cores:
            data.append([network, c, s, t, value])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{network}.query_analysis.csv", index=False)


if __name__ == "__main__":
    main()
