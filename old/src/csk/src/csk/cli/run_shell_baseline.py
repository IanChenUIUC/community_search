import pickle
from pathlib import Path

import click
import numpy as np

from csk.cli.common import get_queries

from ..algs.shell_baseline import AdvancedIndexBuilder


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def shell_baseline_index(edgelist, output):
    import networkit as nk

    graph = nk.graphio.EdgeListReader(",", 0, continuous=True).read(edgelist)
    indexer = AdvancedIndexBuilder(graph)
    indexer.build()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump({"indexer": indexer}, f, protocol=pickle.HIGHEST_PROTOCOL)


@click.command()
@click.option("--index", required=True, type=click.Path(exists=True))
@click.option("--nodelist", required=True, type=click.Path(exists=True))
@click.option("--nodemap", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def shell_baseline_search(index, nodelist, nodemap, outputdir):
    with open(index, "rb") as f:
        data = pickle.load(f)

    indexer: AdvancedIndexBuilder = data["indexer"]
    new_to_orig = np.loadtxt(nodemap, dtype=np.int32)

    for queryID, kmin, query in get_queries(nodelist, nodemap):
        resolved_k, component = indexer.find_kcore(query, kmin)
        dirname = f"query{queryID}"
        outpath = Path(outputdir) / f"{dirname}/kcore_k{resolved_k}.txt"
        outpath.parent.mkdir(parents=True, exist_ok=True)

        with outpath.open("w") as outfile:
            comm = np.sort(new_to_orig[component])
            np.savetxt(outfile, comm, fmt="%d")

        if (queryID + 1) % 50 == 0:
            print(f"Finished {queryID + 1} queries...")

    print("Search jobs completed.")
