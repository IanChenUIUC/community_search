from pathlib import Path

import click
import numpy as np
import pandas as pd


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def coreness(edgelist, output):
    from networkit.centrality import CoreDecomposition
    from networkit.graphio import EdgeListReader

    graph = EdgeListReader(",", 0, "s", continuous=True).read(edgelist)
    core = CoreDecomposition(graph).run()

    nodes = np.arange(graph.numberOfNodes(), dtype=np.int32)
    scores = np.array(core.scores(), dtype=np.int32)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(dict(node=nodes, score=scores))
    df.to_csv(output, index=False, header=False)


if __name__ == "__main__":
    coreness()
