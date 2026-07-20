import sys
import pathlib

import click
import networkit as nk
import numpy as np
import polars as pl
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
def main(indptr_path, indices_path):
    indptr = _read_column(indptr_path, "indptr")
    indices = _read_column(indices_path, "indices")

    n, m = len(indptr) - 1, len(indices) // 2
    graph = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)
    print(f"n = {graph.numberOfNodes()}, m = {graph.numberOfEdges()}", flush=True)

    lcc = nk.centrality.LocalClusteringCoefficient(graph, turbo=True)
    scores = np.mean(lcc.run().scores())
    print("average clustering coeff =", float(scores), flush=True)
    del lcc

    cc = nk.components.ConnectedComponents(graph).run()
    giant = cc.extractLargestConnectedComponent(graph, compactGraph=True)
    giant_n, giant_m = giant.numberOfNodes(), giant.numberOfEdges()
    print(f"giant component: n = {giant_n} m = {giant_m}", flush=True)
    del cc

    # ed = nk.distance.EffectiveDiameter(giant, ratio=0.9)
    # print("effective diameter =", f"{ed.run().getEffectiveDiameter():.3f}", flush=True)
    # del ed


if __name__ == "__main__":
    main()
