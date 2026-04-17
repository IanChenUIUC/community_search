from pathlib import Path

import click
import numpy as np
import scipy.sparse as sp

from csk.algs.steiner import search
from csk.cli.common import get_queries


@click.command()
@click.option("--edges", required=True, type=click.Path(exists=True))
@click.option("--nodemap", required=True, type=click.Path(exists=True))
@click.option("--coreslist", required=True, type=click.Path(exists=True))
@click.option("--nodelist", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def steiner_search(edges, nodemap, coreslist, nodelist, outputdir):
    cores = np.loadtxt(coreslist, dtype=np.int32)
    mapping = np.loadtxt(nodemap, dtype=np.int32)
    length = cores.size

    outputs: dict[int, Path] = dict()  # mapping from community IDs to files
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)

    df_edges = np.loadtxt(edges, delimiter=",", dtype=np.int32).T
    rows = np.concat([df_edges[0], df_edges[1]])
    cols = np.concat([df_edges[1], df_edges[0]])
    data = np.ones_like(rows, dtype=np.bool_)
    graph = sp.csr_array((data, (rows, cols)), shape=(length, length))

    kmins, queries = [], []
    for _, kmin, query in get_queries(nodelist, nodemap):
        print(f"{query=} {cores[query]=}")
        kmins.append(kmin)
        queries.append(query)

    et_comms = search(graph, mapping, cores, queries)
    for comm in et_comms:
        print(comm.queryID, comm.coreness, comm.commID)

        path = output / f"query{comm.queryID}_k{comm.coreness}.txt"
        if comm.coreness < kmins[comm.queryID]:
            np.savetxt(path, np.array([]))
        elif comm.commID in outputs:
            if path.exists(follow_symlinks=False):
                path.unlink()
            path.symlink_to(outputs[comm.commID].name)
        else:
            outputs[comm.commID] = path
            np.savetxt(path, comm.vertices, fmt="%d")
