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


def coreness_queries(shell, cc, rng, valid, size, num=100):
    def single_query(size):
        for _ in range(1_000):
            query = rng.choice(valid, size, replace=False)
            labels = [cc.componentOfNode(q) for q in query]
            if all(label == labels[0] for label in labels):
                return query
        raise RuntimeError("single query failed after 1000 trials")

    return [shell.score(single_query(size)) for _ in range(num)]


@click.command()
@click.argument("network")
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("components_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tree_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("centrality_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_csv", type=click.Path(dir_okay=False))
def main(network, indptr_path, indices_path, components_path, tree_path,
         centrality_csv, output_csv):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    rng = np.random.default_rng(1234)
    cc = nk.components.ParallelConnectedComponents(graph).run()

    shell = nk.scd.ShellStruct(graph)
    shell.load(components_path, tree_path)

    centrality = pd.read_csv(centrality_csv)

    columns = ["network", "centrality", "size", "threshold", "cores"]
    data = []

    cent = ["coreness", "c_coef", "pagerank", "degree"]
    sizes = [1, 5, 10, 20]
    thresholds = [0, 0.9, 0.99, 0.999]
    for c, s, t in tqdm.tqdm(list(it.product(cent, sizes, thresholds))):
        scores = centrality[c].to_numpy()
        valid = np.flatnonzero(scores >= np.quantile(scores, t))
        if len(valid) < s:
            print(f"skipping {c} size={s} threshold={t}: only {len(valid)} candidates",
                  file=sys.stderr)
            continue
        for value in coreness_queries(shell, cc, rng, valid, s):
            data.append([network, c, s, t, value])

    df = pd.DataFrame(data, columns=columns)
    pathlib.Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
