from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp

from csk.cli.common import get_queries

from ..algs.shell import ShellStruct


@click.command()
@click.option("--coreslist", required=True, type=click.Path(exists=True))
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def shell_compressed_index(coreslist, edgelist, output):
    cores = np.loadtxt(coreslist, dtype=np.int32)

    length = cores.size
    order = np.argsort(cores)
    rorder = np.argsort(order)
    vertices = order
    cores = cores[order]

    df_edges = pd.read_csv(edgelist, sep=",", header=None)
    rows = rorder[df_edges[0].to_numpy()]
    cols = rorder[df_edges[1].to_numpy()]
    del df_edges

    mask = rows > cols
    rows[mask], cols[mask] = cols[mask], rows[mask]
    data = np.ones_like(rows, dtype=np.bool_)
    graph = sp.csr_array((data, (rows, cols)), shape=(length, length))

    shell = ShellStruct.build(graph, vertices, rorder, cores)
    shell.save(output)


@click.command()
@click.option("--shell_file", required=True, type=click.Path(exists=True))
@click.option("--queries", required=True, type=click.Path(exists=True))
@click.option("--nodemap", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def shell_compressed_search(shell_file, queries, nodemap, outputdir):
    shell = ShellStruct.load(shell_file)
    mapping: dict[np.int32, Path] = dict()  # mapping from community IDs to files
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)

    for queryID, kmin, query in get_queries(queries, nodemap):
        lca = shell.lca.find_lca(shell.assign[shell.order[query]])
        kval = shell.coreness[lca]

        path = output / f"query{queryID}_k{kval}.txt"
        if lca in mapping:
            if path.exists(follow_symlinks=False):
                path.unlink()
            path.symlink_to(mapping[lca].name)
        elif kval >= kmin and shell.coreness[lca] > 0:
            vertices = shell.get_vertices(lca)
            np.savetxt(path, vertices, fmt="%d")
            mapping[lca] = path
        else:
            np.savetxt(path, np.array([]))
