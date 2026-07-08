from pathlib import Path

import click
import numpy as np


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def coreness(edgelist, output):
    from networkit.centrality import CoreDecomposition
    from networkit.graphio import EdgeListReader

    graph = EdgeListReader(",", 0, continuous=True).read(edgelist)
    core = CoreDecomposition(graph).run()

    scores = np.array(core.scores(), dtype=np.int32)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, scores, fmt="%d")
