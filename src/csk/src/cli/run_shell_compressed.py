from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..csk.shell import ShellStruct


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def shell_compressed_index(edgelist, output):
    from networkit.centrality import CoreDecomposition
    from networkit.graphio import EdgeListReader

    reader = EdgeListReader("\t", 0, continuous=False)
    graph = reader.read(edgelist)
    core = CoreDecomposition(graph).run()

    data = [[node, int(core.score(node))] for node in graph.iterNodes()]
    df = pd.DataFrame(data, columns=["node", "core"])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, header=False, sep="\t")


@click.command()
@click.option("--coreslist", required=True, type=click.Path(exists=True))
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def shell_compressed_build(coreslist, edgelist, output):
    df_values = pd.read_csv(coreslist, sep="\\s+", header=None)
    vertices, cores = df_values[0].to_numpy(), df_values[1].to_numpy()
    del df_values

    length = len(vertices)
    order = np.argsort(cores)
    rorder = np.argsort(order)
    vertices = vertices[order]
    cores = cores[order]

    df_edges = pd.read_csv(edgelist, sep="\\s+", header=None)
    rows = rorder[df_edges[0].to_numpy()]
    cols = rorder[df_edges[1].to_numpy()]
    del df_edges

    mask = rows > cols
    rows[mask], cols[mask] = cols[mask], rows[mask]
    data = np.ones_like(rows, dtype=np.bool_)
    graph = sp.csr_array((data, (rows, cols)), shape=(length, length))

    shell = ShellStruct.build(graph, vertices, cores)
    shell.save(output)


@click.command()
@click.option("--shell_file", required=True, type=click.Path(exists=True))
@click.option("--queries", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def shell_compressed_search(shell_file, queries, outputdir):
    shell = ShellStruct.load(shell_file)
    indexer = pd.Index(shell.vertices)

    mapping: dict[np.int32, Path] = dict()  # mapping from community IDs to files
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
            lca = shell.lca.find_lca(nodes)
            kval = shell.coreness[lca]

            path = output / f"{name}_k{kval}.txt"
            if lca in mapping:
                if path.exists(follow_symlinks=False):
                    path.unlink()
                path.symlink_to(mapping[lca].name)
            elif kval >= kmin:
                mapping[lca] = path
                vertices = shell.get_vertices(lca)
                np.savetxt(path, vertices, fmt="%d")
            else:
                np.savetxt(path, np.array([]))
