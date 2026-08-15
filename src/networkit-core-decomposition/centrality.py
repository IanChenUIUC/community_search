import pathlib

import click
import pandas as pd

import networkit as nk


@click.command()
@click.option("--graph", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
def stats(graph, output):
    nk.engineering.setNumberOfThreads(16)

    print("reading graph", flush=True)
    gr = nk.readGraph(graph, nk.Format.NetworkitBinary)

    print("running core decomp", flush=True)
    cores = nk.centrality.CoreDecomposition(gr).run().scores()
    print("running lcc", flush=True)
    lcc = nk.centrality.LocalClusteringCoefficient(gr, turbo=True).run().scores()
    print("running pr", flush=True)
    pr = nk.centrality.PageRank(gr).run().scores()
    print("running deg", flush=True)
    deg = nk.centrality.DegreeCentrality(gr).run().scores()

    print("writing output", flush=True)
    pathlib.Path(output).parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(dict(coreness=cores, c_coef=lcc, pagerank=pr, degree=deg))
    df.to_csv(output, index=False)


if __name__ == "__main__":
    stats()
