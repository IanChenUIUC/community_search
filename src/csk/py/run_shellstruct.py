import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rmq import find_lca
from shell import build_shell, get_vertices, load_shell, save_shell


@click.group()
def kcore():
    pass


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def index(edgelist, output):
    from networkit.centrality import CoreDecomposition
    from networkit.graphio import EdgeListReader

    reader = EdgeListReader("\t", 0, continuous=False)
    graph = reader.read(edgelist)
    core = CoreDecomposition(graph).run()

    data = [[node, int(core.score(node))] for node in graph.iterNodes()]
    df = pd.DataFrame(data, columns=["node", "core"])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, header=False, sep="\t")


@kcore.command()
@click.option("--coreslist", required=True, type=click.Path(exists=True))
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def build(coreslist, edgelist, output):
    global start
    start = time.perf_counter()

    df_values = pd.read_csv(coreslist, sep="\\s+", header=None)
    vertices = df_values[0].values
    cores = df_values[1].values
    length = len(vertices)
    order = np.argsort(cores)
    rorder = np.argsort(order)
    vertices = vertices[order]
    cores = cores[order]

    df_edges = pd.read_csv(edgelist, sep="\\s+", header=None)
    rows = df_edges[0].values
    cols = df_edges[1].values
    rows, cols = rorder[rows], rorder[cols]
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    data2 = np.ones(len(rows2), dtype=np.bool_)
    graph = sp.csr_matrix((data2, (rows2, cols2)), shape=(length, length))
    graph = sp.triu(graph, format="csr")

    shell = build_shell(graph, vertices, cores)
    save_shell(shell, output)


@kcore.command()
@click.option("--shell_file", required=True, type=click.Path(exists=True))
@click.option("--queries", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def search(shell_file, queries, outputdir):
    shell = load_shell(shell_file)
    indexer = pd.Index(shell.vertices)

    mapping: dict[int, Path] = dict()  # mapping from community IDs to files
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)

    with open(queries) as f:
        for spec, querylist in zip(f, f):
            spec = spec.strip().split(" ")
            assert len(spec) == 1 or len(spec) == 2

            name = spec[0]
            kmin = 0 if len(spec) == 1 else int(spec[1])

            query = np.fromstring(querylist, sep=" ", dtype=np.int32)
            nodes = shell.assign[indexer.get_indexer(query)]
            lca = find_lca(shell.lca, nodes)
            kval = shell.coreness[lca]

            path = output / f"{name}_k{kval}.txt"
            if lca in mapping:
                if path.exists(follow_symlinks=False):
                    path.unlink()
                path.symlink_to(mapping[lca].name)
            elif kval >= kmin:
                mapping[lca] = path
                vertices = get_vertices(shell, lca)
                np.savetxt(path, vertices, fmt="%d")
            else:
                np.savetxt(path, np.array([]))


if __name__ == "__main__":
    kcore()
