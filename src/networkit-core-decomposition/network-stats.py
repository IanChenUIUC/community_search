import click
import numpy as np

import networkit as nk


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
def stats(edgelist):
    graph = nk.graphio.EdgeListReader(",", 0, "s", continuous=True).read(edgelist)
    print(f"n = {graph.numberOfNodes()}, m = {graph.numberOfEdges()}", flush=True)

    lcc = nk.centrality.LocalClusteringCoefficient(graph)
    scores = np.mean(lcc.run().scores())
    print("average clustering coeff =", float(scores), flush=True)
    del lcc

    cc = nk.components.ConnectedComponents(graph).run()
    giant = cc.extractLargestConnectedComponent(graph, compactGraph=True)
    giant_n, giant_m = giant.numberOfNodes(), giant.numberOfEdges()
    print(f"giant component: n = {giant_n} m = {giant_m}", flush=True)
    del cc

    # ed = nk.distance.EffectiveDiameter(giant, ratio=0.9)
    # print("effective diameter =", f"{ed.run().getEffectiveDiameter():.3f}", flush=True)
    # del ed


if __name__ == "__main__":
    stats()
