from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp


def find_kcore(graph, cores, repr, nbrs, q, k, n):
    visited = np.zeros(n, dtype=np.uint8)
    output = []
    stack = []

    def extend(v):
        remove = []
        for k1, adj in nbrs[v].items():
            if k1 < k:
                continue

            adj = np.array(adj, dtype=np.int64)
            adj = adj[visited[adj] == 0]
            visited[adj] = 1
            stack.extend(adj)
            remove.append(k1)
        for k1 in remove:
            del nbrs[v][k1]

    q0 = q if repr[q] == -1 else repr[q]
    visited[q0] = 1
    stack.append(q0)

    while len(stack) > 0:
        v = stack.pop()

        if repr[v] != -1:
            output.append(repr[v])
            extend(repr[v])
            continue

        repr[v] = q
        output.append(v)

        # adj = graph.induces[graph.indptr[v], graph.indptr[v + 1]]
        # adj = adj[visited[adj] == 0]
        # assert np.all(repr[adj] == -1)
        # ...

        for i in range(graph.indptr[v], graph.indptr[v + 1]):
            u = graph.indices[i]
            u0 = u if repr[u] == -1 else repr[u]

            if visited[u0]:
                continue

            visited[u0] = 1
            if cores[u0] < k:
                assert repr[u] == -1
                nbrs[q][cores[u0]].append(u0)
            else:
                stack.append(u0)

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

    repr = np.full(n, -1, dtype=np.int64)
    nbrs = {q: defaultdict(list) for q in query_df["q"]}
    components = {}

    def print_kcore(q, k, outfile):
        if cores[q] < k:
            return

        contracted = find_kcore(g, cores, repr, nbrs, q, k, n)
        comp = []
        for v in contracted:
            if repr[v] != q:
                comp.extend(components[repr[v]])
            else:
                comp.append(v)

        print(f"{q=} {len(comp)=} {len(set(comp))=}")

        components[q] = comp
        outfile.write("\n".join(map(str, components[q])))

    for _, query in query_df.iterrows():
        q = int(query["q"])
        k = int(query["k"])

        outpath = Path(outputdir) / f"{q}/kcore_k{k}.txt"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open("w") as outfile:
            print_kcore(q, k, outfile)


if __name__ == "__main__":
    kcore()
