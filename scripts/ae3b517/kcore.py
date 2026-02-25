from pathlib import Path

import click
import networkit as nk
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp

ADJACENCY = numba.types.ListType(numba.types.int64)
FRONTIER_PARTITION = numba.types.DictType(numba.types.int64, ADJACENCY)


@numba.njit
def ds_find(q: int, ds: np.ndarray):
    root = q
    while ds[root] >= 0:
        root = ds[root]
    while q != root:
        parent = ds[q]
        ds[q] = root
        q = parent
    return root


@numba.njit
def ds_union(q1: int, q2: int, ds: np.ndarray):
    root1 = ds_find(q1)
    root2 = ds_find(q2)
    if root1 == root2:
        return
    if (-ds[root1]) < (-ds[root2]):
        root1, root2 = root2, root1
    ds[root1] += ds[root2]  # adding size
    ds[root2] = root1  # setting parents


@numba.njit
def find_kcore(indices, indptr, cores, repr, ds, nbrs, q, k, n):
    visited = np.zeros(n, dtype=np.uint8)
    stack = np.empty(n, dtype=np.int64)
    sidx = 0
    out = np.empty(n, dtype=np.int64)
    oidx = 0

    def extend(v):
        nonlocal sidx
        for k1, adj in nbrs[v].items():
            if k1 < k:
                continue
            for u in adj:
                if visited[u] == 0:
                    visited[u] = 1
                    stack[sidx] = u
                    sidx += 1

    if repr[q] != -1:
        extend(repr[q])
        q = repr[q]
        nbrs[q] = numba.typed.Dict.empty(numba.types.int64, ADJACENCY)
        out[oidx] = q
        oidx += 1
    else:
        stack[sidx] = q
        sidx += 1

    visited[q] = 1
    while sidx > 0:
        sidx -= 1
        v = stack[sidx]

        if repr[v] != -1 and visited[repr[v]] == 0:
            visited[repr[v]] = 1
            extend(repr[v])
            out[oidx] = repr[v]
            oidx += 1
        else:
            out[oidx] = v
            oidx += 1
            repr[v] = q

        for i in range(indptr[v], indptr[v + 1]):
            u = indices[i]
            if visited[u]:
                continue

            if cores[u] >= k:
                stack[sidx] = u
                sidx += 1
                visited[u] = 1
            else:
                k1 = cores[u]
                if k1 not in nbrs[q]:
                    nbrs[q][k1] = numba.typed.List.empty_list(numba.types.int64)
                # TODO: reduce redundancy here?
                nbrs[q][k1].append(u)

    return out[:oidx]


@click.group()
def kcore():
    pass


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def index(edgelist, output):
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
    ds = np.full(n, -1)  # union-by-size

    repr = np.full(n, -1, dtype=np.int64)
    nbrs = numba.typed.Dict.empty(numba.types.int64, FRONTIER_PARTITION)
    for q in query_df["q"]:
        nbrs[q] = numba.typed.Dict.empty(numba.types.int64, ADJACENCY)

    components = {}

    def print_kcore(q, k, outfile):
        if cores[q] < k:
            return

        components[q] = find_kcore(g.indices, g.indptr, cores, repr, ds, nbrs, q, k, n)
        comp = set(components[q])
        for q1 in components:
            if q1 in comp:
                comp = comp.union(components[q1])
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
