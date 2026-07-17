from pathlib import Path

import click
import numpy as np
import pandas as pd

import networkit as nk
from networkit.centrality import CoreDecomposition


@click.command()
@click.option("--graph", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def coreness(graph, output):
    gr = nk.readGraph(graph, nk.Format.NetworkitBinary)
    core = CoreDecomposition(gr).run()

    nodes = np.arange(gr.numberOfNodes(), dtype=np.int32)
    scores = np.array(core.scores(), dtype=np.int32)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(dict(node=nodes, score=scores))
    df.to_csv(output, index=False, header=False)


if __name__ == "__main__":
    coreness()
