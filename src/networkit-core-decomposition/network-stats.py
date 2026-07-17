import click
import numpy as np

import networkit as nk


@click.command()
@click.option("--graph", required=True, type=click.Path(exists=True))
def stats(graph):
    gr = nk.readGraph(graph, nk.Format.NetworkitBinary)
    print(f"n = {gr.numberOfNodes()}, m = {gr.numberOfEdges()}", flush=True)

    lcc = nk.centrality.LocalClusteringCoefficient(gr)
    scores = np.mean(lcc.run().scores())
    print("average clustering coeff =", float(scores), flush=True)
    del lcc

    cc = nk.components.ConnectedComponents(gr).run()
    giant = cc.extractLargestConnectedComponent(gr, compactGraph=True)
    giant_n, giant_m = giant.numberOfNodes(), giant.numberOfEdges()
    print(f"giant component: n = {giant_n} m = {giant_m}", flush=True)
    del cc

    # ed = nk.distance.EffectiveDiameter(giant, ratio=0.9)
    # print("effective diameter =", f"{ed.run().getEffectiveDiameter():.3f}", flush=True)
    # del ed


if __name__ == "__main__":
    stats()
