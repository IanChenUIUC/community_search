from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp


def find_kcore(g, cores, repr, nbrs, q, k, n):
    visited = np.zeros(n, dtype=np.uint8)
    output = []
    stack = []

    visited[repr[q]] = 1
    stack.append(repr[q])

    while len(stack) > 0:
        v = stack.pop()
        output.append(repr[v])

        if repr[v] != q and repr[v] in nbrs:
            adj = nbrs[repr[v]]
            del nbrs[repr[v]]
        else:
            adj = g.indices[g.indptr[v] : g.indptr[v + 1]]
        repr[v] = q

        for u in adj:
            if visited[repr[u]]:
                continue

            visited[repr[u]] = 1
            if cores[u] < k:
                nbrs[q].append(repr[u])
            else:
                stack.append(repr[u])

    return output


@click.group()
def kcore():
    pass


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def index(edgelist, output):
    import networkit as nk

    graph = nk.graphio.EdgeListReader("\t", 0).read(edgelist)
    core = nk.centrality.CoreDecomposition(graph).run()

    data = [[node, int(core.score(node))] for node in range(graph.numberOfNodes())]
    df = pd.DataFrame(data, columns=["node", "core"])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, header=False, sep="\t")


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--index", required=True, type=click.Path(exists=True))
@click.option("--nodelist", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def search(edgelist, index, nodelist, outputdir):
    edges = np.loadtxt(edgelist, dtype=np.int64)
    n = int(np.max(edges) + 1)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(2 * len(edges), dtype=np.int8)
    g = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    cores = pd.read_csv(index, sep="\t", header=None, names=["node", "core"])
    cores = cores.sort_values("node")["core"].to_numpy()

    query_df = pd.read_csv(nodelist, sep=" ", header=None, names=["q", "k"])
    default_k = pd.Series(cores[query_df["q"].values], index=query_df.index)
    query_df["k"] = query_df["k"].fillna(default_k)
    query_df.sort_values(by="k", ascending=False, inplace=True)

    repr = np.arange(n, dtype=np.int64)
    nbrs = defaultdict(list)
    components = {}

    def print_kcore(q, k, outfile):
        if cores[q] < k:
            return

        stack = find_kcore(g, cores, repr, nbrs, q, k, n)
        comp = set()
        # redundant = 0
        while stack:
            v = stack.pop()
            other = components[v] if v in components and v not in comp else {v}
            # redundant += len(comp.intersection(other))
            comp.update(other)

        # print(f"{q=} {len(comp)=} {redundant=}")

        components[q] = list(comp)
        outfile.write("\n".join(map(str, comp)))

    for _, query in query_df.iterrows():
        q = int(query["q"])
        k = int(query["k"])

        outpath = Path(outputdir) / f"{q}/kcore_k{k}.txt"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open("w") as outfile:
            print_kcore(q, k, outfile)


if __name__ == "__main__":
    kcore()
