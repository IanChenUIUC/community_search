from pathlib import Path

import click
import networkit as nk

# import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp


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
    graph = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    cores = pd.read_csv(index, sep="\t", header=None, names=["node", "core"])
    cores = cores.sort_values("node")["core"].to_numpy()

    query_df = pd.read_csv(nodelist, sep=" ", header=None, names=["q", "k"])
    default_k = pd.Series(cores[query_df["q"].values], index=query_df.index)
    query_df["k"] = query_df["k"].fillna(default_k)
    query_df.sort_values(by="k", ascending=False, inplace=True)

    queries = np.unique(query_df["q"].to_numpy())
    num_queries = len(queries)
    query_map = dict(zip(queries, np.arange(num_queries)))
    ds = np.full(num_queries, -1)  # union-by-size

    def ds_find(q):
        q = query_map[q]

        root = q
        while ds[root] >= 0:
            assert root != ds[root]
            root = ds[root]
        while q != root:
            assert q != ds[q]
            parent = ds[q]
            ds[q] = root
            q = parent

        return queries[root]

    def ds_union(q1, q2):
        root1 = query_map[ds_find(q1)]
        root2 = query_map[ds_find(q2)]
        if root1 != root2:
            return
        if (-ds[root1]) < (-ds[root2]):
            root1, root2 = root2, root1
        ds[root1] += ds[root2]  # adding size
        ds[root2] = root1  # setting parents

    components = {}
    repr = np.full(n, -1, dtype=np.int64)
    nbrs = {q: {} for q in queries}

    def find_kcore(q, k):
        if cores[q] < k:
            return set()
        if repr[q] != -1:
            return {repr[q]}

        visited = set()
        stack = [q]
        while stack:
            v = stack.pop()
            if repr[v] != -1 and repr[v] not in visited:
                ds_union(q, repr[v])
                visited.add(repr[v])
                for k1 in range(k, cores[repr[v]]):
                    stack.extend(nbrs[repr[v]].get(k1, []))
            else:
                visited.add(v)
                repr[v] = q
            visited.add(v)

            for i in range(graph.indptr[v], graph.indptr[v + 1]):
                u = graph.indices[i]
                if u in visited or repr[u] != -1:
                    continue

                if cores[u] >= k:
                    stack.append(u)
                    visited.add(u)
                else:
                    k1 = cores[u]
                    if k1 not in nbrs[q]:
                        nbrs[q][k1] = []
                    nbrs[q][k1].append(u)

        return visited

    def print_kcore(q, k, outfile):
        components[q] = find_kcore(q, k)

        for c in components[q]:
            if c in query_map and ds_find(c) in components:
                outfile.write("\n".join(map(str, components[ds_find(c)])))
                outfile.write("\n")
            else:
                outfile.write(f"{c}\n")

    for _, query in query_df.iterrows():
        q = int(query["q"])
        k = int(query["k"])

        outpath = Path(outputdir) / f"{q}/kcore_k{k}.txt"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open("w") as outfile:
            print_kcore(q, k, outfile)


if __name__ == "__main__":
    kcore()
