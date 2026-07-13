import pathlib

import click
import pandas as pd

import networkit as nk


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
def stats(edgelist, output):
    graph = nk.graphio.EdgeListReader(",", 0, "s", continuous=True).read(edgelist)

    cores = nk.centrality.CoreDecomposition(graph).run().scores()
    lcc = nk.centrality.LocalClusteringCoefficient(graph).run().scores()
    pr = nk.centrality.PageRank(graph).run().scores()
    deg = nk.centrality.DegreeCentrality(graph).run().scores()

    pathlib.Path(output).parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(dict(coreness=cores, c_coef=lcc, pagerank=pr, degree=deg))
    df.to_csv(output, index=False)


if __name__ == "__main__":
    stats()
